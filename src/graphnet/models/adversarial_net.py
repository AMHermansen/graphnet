from dataclasses import dataclass, field
from itertools import chain
from typing import Union, List, Type, Any, Dict, Optional, Sequence

import torch
from lion_pytorch import Lion
from torch.nn import BCELoss
from torch_geometric.utils import to_dense_batch

from graphnet.data import SQLiteDataModule
from graphnet.models.gnn import GNN, DynEdgeTITO
from graphnet.models.gnn.aggregator import StandardPooling, Aggregator
from graphnet.models.gnn.gnn import RawGNN
from graphnet.models.lightweight_model import LightweightTemplateModel
from graphnet.models.model import Model
from graphnet.models.task import BinaryClassificationTaskLogits, Task, \
    ZenithReconstructionWithKappa
from graphnet.training.loss_functions import BinaryCrossEntropyWithLogitsLoss, \
    VonMisesFisher2DLoss
from lightning import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn, Tensor
from torch.optim import SGD
from torch_geometric.data import Data
from torchmetrics import Accuracy, Metric, AUROC


@dataclass
class ARNConfig:
    adv_target: Optional[str] = "is_data"
    adv_weight: float = 1.0
    adv_node_weight: float = 1.0
    adv_latent_weight: float = 1.0
    opt1: Type[torch.optim.Optimizer] = Lion
    opt2: Type[torch.optim.Optimizer] = Lion
    lr1: Optional[float] = 3e-5
    lr2: Optional[float] = 3e-5
    opt1_kwargs: Optional[Dict] = field(default_factory=dict)
    opt2_kwargs: Optional[Dict] = field(default_factory=dict)
    adv_metrics: Optional[List[Type[Metric]]] = field(default_factory=lambda: [Accuracy, AUROC])


class AdversarialRegulatedNeuralNet(LightweightTemplateModel):
    def __init__(
            self,
            gnn_encoder: RawGNN,
            aggregator: Aggregator,
            tasks: Union[Task, List[Task]],
            config: Optional[ARNConfig] = None,
    ):
        super().__init__(name=__name__, class_name=self.__class__.__name__, tasks=tasks)

        self._gnn_encoder = gnn_encoder
        self._aggregator = aggregator
        self._args = self.set_default_if_none(config, ARNConfig())

        self._adversarial_task = BinaryClassificationTaskLogits(
            hidden_size=aggregator.nb_outputs,
            loss_function=BinaryCrossEntropyWithLogitsLoss(),
            target_labels=[self._args.adv_target],
        )
        self._adversarial_node_level_task = BinaryClassificationTaskLogits(
            hidden_size=gnn_encoder.nb_outputs,
            loss_function=BinaryCrossEntropyWithLogitsLoss(),
            target_labels=[f"_{self._args.adv_target}_node"],
        )

        self._setup_adversarial_metrics(self._args.adv_metrics)

        self.automatic_optimization = False

        self.info(f"Adversarial target is: {self._args.adv_target}")

    def _setup_adversarial_metrics(self, metric_list: List[Type[Metric]]):
        self.train_latent_metrics = nn.ModuleDict({})
        self.train_node_metrics = nn.ModuleDict({})
        self.val_latent_metrics = nn.ModuleDict({})
        self.val_node_metrics = nn.ModuleDict({})
        self.metric_names = []

        for metric in metric_list:
            self.metric_names.append(f"{metric.__name__}")
            for metric_dict in [
                self.train_latent_metrics,
                self.train_node_metrics,
                self.val_latent_metrics,
                self.val_node_metrics,
            ]:
                metric_dict[f"{metric.__name__}"] = metric(task="binary")

    def _get_metrics(self, train):
        latent_metrics = self.train_latent_metrics if train else self.val_latent_metrics
        node_metrics = self.train_node_metrics if train else self.val_node_metrics
        return latent_metrics, node_metrics

    def update_adv_metrics(
            self,
            node_pred: Tensor,
            latent_pred: torch.Tensor,
            data: Data,
            train: bool
    ) -> None:
        latent_metrics, node_metrics = self._get_metrics(train)

        for latent_metric in latent_metrics.values():
            latent_metric(latent_pred, data[self._args.adv_target])
        for node_metric in node_metrics.values():
            node_metric(node_pred, data[f"_{self._args.adv_target}_node"])

    def log_adv_metrics(
            self,
            data: Data,
            train: bool,
            **kwargs
    ):
        prefix_str = "train" if train else "val"
        latent_metrics, node_metrics = self._get_metrics(train)
        for name, metric in latent_metrics.items():
            self.log(
                f"{prefix_str}_latent_{name}",
                metric,  # noqa
                batch_size=self._get_batch_size(data),
                **kwargs
            )

        for name, metric in node_metrics.items():
            self.log(
                f"{prefix_str}_node_{name}",
                metric,  # noqa
                batch_size=self._get_batch_size(data),
                **kwargs
            )

    def _repeat_to_nodes(self, data: Data):
        adv_target = torch.stack(
            [data[label] for label in self._adversarial_task._target_labels], dim=1
        )
        dense_x, mask = to_dense_batch(data.x, data.batch)
        B, L, C = dense_x.shape

        data[f"_{self._args.adv_target}_node"] = adv_target.repeat((1, L, 1))[mask].view(-1, 1)
        return data

    def training_step(self, data: Data, batch_idx: int):
        data = self._preprocess_data(data)
        opt1, opt2 = self.optimizers()

        opt2.zero_grad()
        feature_graph = self._gnn_encoder(data)
        features = self._aggregator(feature_graph)
        feature_graph.detach()
        features.detach()

        (
            adversarial_latent_loss,
            adversarial_latent_pred,
            adversarial_node_loss,
            adversarial_node_pred
        ) = self._compute_adversarial_loss(
            data, feature_graph.encoded_x, features
        )

        adv_loss: torch.Tensor = (
                self._args.adv_latent_weight * adversarial_latent_loss
                + self._args.adv_node_weight * adversarial_node_loss
        )

        self.update_adv_metrics(
            adversarial_node_pred,
            adversarial_latent_pred,
            data,
            train=True,
        )

        adv_loss.backward()

        opt2.step()
        opt2.zero_grad()

        opt1.zero_grad()
        feature_graph = self._gnn_encoder(data)
        features = self._aggregator(feature_graph)
        in_distribution_features = self._crop_to_in_distribution(data, features)
        task_preds = [task(in_distribution_features) for task in self._tasks]

        task_loss: torch.Tensor = self._compute_loss(task_preds, data)

        (
            adversarial_latent_loss,
            adversarial_latent_pred,
            adversarial_node_loss,
            adversarial_node_pred
        ) = self._compute_adversarial_loss(
            data, feature_graph.encoded_x, features
        )

        adv_loss: torch.Tensor = (
                self._args.adv_latent_weight * adversarial_latent_loss
                + self._args.adv_node_weight * adversarial_node_loss
        )

        (task_loss - self._args.adv_weight * adv_loss).backward()
        # from icecream import ic
        # for module in self.modules():
        #     c = 0
        #     for param in module.parameters():
        #         if param.grad is not None:
        #             c += 1
        #         else:
        #             ic(f"No grad: {module.__class__}")
        #     if c > 0:
        #         ic(module.__class__)
        opt1.step()

        self.log_adv_metrics(
            data,
            train=True,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        self.log(
            "t_adv_latent",
            adversarial_latent_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "t_adv_node",
            adversarial_node_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "t_task",
            task_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        return {"train_adversarial_loss": adv_loss, "task_loss": task_loss}

    def _compute_adversarial_loss(self, data: Data, feature_graph: Tensor, features: Tensor):
        adversarial_latent_pred = self._adversarial_task(features)
        adversarial_node_pred = self._adversarial_node_level_task(feature_graph)
        adversarial_latent_loss = self._adversarial_task.compute_loss(
            adversarial_latent_pred, data)
        adversarial_node_loss = self._adversarial_node_level_task.compute_loss(
            adversarial_node_pred, data)
        return adversarial_latent_loss, adversarial_latent_pred, adversarial_node_loss, adversarial_node_pred

    def validation_step(self, data: Data, batch_idx: int):
        data = self._preprocess_data(data)
        feature_graph = self._gnn_encoder(data)
        features = self._aggregator(feature_graph)
        in_distribution_features = self._crop_to_in_distribution(data, features)
        task_preds = [task(in_distribution_features) for task in self._tasks]
        task_loss = self._compute_loss(task_preds, data)

        (
            adversarial_latent_loss,
            adversarial_latent_pred,
            adversarial_node_loss,
            adversarial_node_pred
        ) = self._compute_adversarial_loss(
            data, feature_graph.encoded_x, features
        )

        self.update_adv_metrics(
            adversarial_node_pred,
            adversarial_latent_pred,
            data,
            train=False,
        )
        # Log validation metrics
        self.log_adv_metrics(
            data,
            train=False,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        # Log losses
        self.log(
            "v_adv_latent",
            adversarial_latent_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "v_adv_node",
            adversarial_node_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "v_task",
            task_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return {
            "feature_graph": feature_graph,
            "features": features,
            "preds": task_preds,
            "adversarial_pred": adversarial_latent_pred,
            "adversarial_node_pred": adversarial_node_pred,
        }

    def predict_step(self, data: Data, batch_idx: int):
        data = self._preprocess_data(data)
        feature_graph = self._gnn_encoder(data)
        features = self._aggregator(feature_graph)
        task_preds = [task(features) for task in self._tasks]

        adversarial_latent_pred = self._adversarial_task(features)
        adversarial_node_pred = self._adversarial_node_level_task(feature_graph.encoded_x)

        return {
            "adv_latent_pred": adversarial_latent_pred,
            "adv_node_pred": adversarial_node_pred,
            "preds": task_preds,
            "latent_features": features,
            "node_features": feature_graph.encoded_x,
        }

    def _preprocess_data(self, data):
        data[self._args.adv_target] = data[self._args.adv_target].view(-1, 1)
        data = self._repeat_to_nodes(data)
        return data

    def _crop_to_in_distribution(self, data: Data, features: torch.Tensor) -> torch.Tensor:
        in_distribution = data[self._args.adv_target].view(-1) == 0
        in_distribution_features = features[in_distribution]
        for target in self.target_labels:
            data[target] = data[target][in_distribution]
        return in_distribution_features

    @property
    def target_labels(self) -> List[str]:
        """Return target label."""
        return [label for task in self._tasks for label in task._target_labels]

    def configure_optimizers(self):
        opt1 = self._args.opt1(
            [
                {"params": self._gnn_encoder.parameters()},
                {"params": self._aggregator.parameters()},
                {"params": self._tasks.parameters()},
            ],
            lr=self._args.lr1,
            **self._args.opt1_kwargs
        )
        opt2 = self._args.opt2(
            [
                {"params": self._adversarial_task.parameters()},
                {"params": self._adversarial_node_level_task.parameters()}
            ],
            lr=self._args.lr2,
            **self._args.opt2_kwargs
        )
        return [opt1, opt2]

    def _compute_loss(
        self, preds: List[Tensor], data: Data, verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        losses = [
            task.compute_loss(pred, data)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.mean(torch.stack(losses))


if __name__ == "__main__":
    adv_net = AdversarialRegulatedNeuralNet(
        (tito := DynEdgeTITO(7)),
        agg := StandardPooling(
            nb_inputs=tito.nb_outputs,
            nb_original_graph_features=tito.nb_inputs,
            global_pooling_schemes=["mean"],
            readout_mlp_layers=[1024, 256],
        ),
        tasks=ZenithReconstructionWithKappa(
            hidden_size=tito.nb_outputs,
            loss_function=VonMisesFisher2DLoss()
        ),
    )
    print(adv_net)
    print(adv_net.configure_optimizers())
    for param in adv_net.parameters():
        print(param.requires_grad)

