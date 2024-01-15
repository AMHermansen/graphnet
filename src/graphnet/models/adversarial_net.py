from dataclasses import dataclass, field
from typing import Union, List, Type, Any, Dict, Optional, Sequence

import torch
from graphnet.data import SQLiteDataModule
from graphnet.models.gnn import GNN, DynEdgeTITO
from graphnet.models.model import Model
from graphnet.models.task import BinaryClassificationTaskLogits, Task, \
    ZenithReconstructionWithKappa
from graphnet.training.loss_functions import BinaryCrossEntropyWithLogitsLoss, \
    VonMisesFisher2DLoss
from icecream import ic
from lightning import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn, Tensor
from torch.optim import SGD
from torch_geometric.data import Data
from torchmetrics import Accuracy


@dataclass
class ARNConfig:
    adv_target: Optional[str] = "is_data"
    adv_weight: float = 1.0
    opt1: Type[torch.optim.Optimizer] = SGD
    opt2: Type[torch.optim.Optimizer] = SGD
    lr1: Optional[float] = 0.01
    lr2: Optional[float] = 0.001
    opt1_kwargs: Optional[Dict] = field(default_factory=dict)
    opt2_kwargs: Optional[Dict] = field(default_factory=dict)


class AdversarialRegulatedNeuralNet(Model):
    def __init__(
            self,
            gnn_encoder: GNN,
            tasks: Union[Task, List[Task]],
            config: Optional[ARNConfig] = None,
    ):
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        tasks = [tasks] if isinstance(tasks, Task) else tasks

        self._gnn_encoder = gnn_encoder
        self._tasks = nn.ModuleList(tasks)
        self._args = config or ARNConfig()

        self._adversarial_task = BinaryClassificationTaskLogits(
            hidden_size=gnn_encoder.nb_outputs,
            loss_function=BinaryCrossEntropyWithLogitsLoss(),
            target_labels=[self._args.adv_target],
        )
        self.metrics = nn.ModuleDict(
            {
                "train_adv_acc": Accuracy(task="binary"),
                "val_adv_acc": Accuracy(task="binary"),
            }
        )
        self.automatic_optimization = False

        self.info(f"Adversarial target is: {self._args.adv_target}")

    def training_step(self, data: Data, batch_idx: int):
        data[self._args.adv_target] = data[self._args.adv_target].view(-1, 1)
        opt1, opt2 = self.optimizers()
        opt2.zero_grad()
        features = self._gnn_encoder(data)
        features.detach()
        adversarial_pred = self._adversarial_task(features)
        adversarial_loss = self._adversarial_task.compute_loss(adversarial_pred, data)
        self.metrics.train_adv_acc(adversarial_pred, data[self._args.adv_target])
        self.manual_backward(adversarial_loss)
        opt2.step()

        opt1.zero_grad()
        features = self._gnn_encoder(data)
        in_distribution_features = self._crop_to_in_distribution(data, features)
        task_preds = [task(in_distribution_features) for task in self._tasks]
        adversarial_pred = self._adversarial_task(features)

        task_loss = self._compute_loss(task_preds, data)
        adversarial_loss = self._adversarial_task.compute_loss(adversarial_pred, data)
        self.manual_backward(task_loss - self._args.adv_weight * adversarial_loss)
        opt1.step()

        self.log(
            "t_adv_acc",
            self.metrics.train_adv_acc,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "t_adv_l",
            adversarial_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "t_task_l",
            task_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return {"train_adversarial_loss": adversarial_loss, "task_loss": task_loss}

    def validation_step(self, data: Data, batch_idx: int):
        data[self._args.adv_target] = data[self._args.adv_target].view(-1, 1)
        features = self._gnn_encoder(data)
        in_distribution_features = self._crop_to_in_distribution(data, features)
        task_preds = [task(in_distribution_features) for task in self._tasks]
        adversarial_pred = self._adversarial_task(features)
        task_loss = self._compute_loss(task_preds, data)
        adversarial_loss = self._adversarial_task.compute_loss(adversarial_pred, data)

        self.metrics.val_adv_acc(adversarial_pred, data[self._args.adv_target])

        self.log(
            "v_adv_acc",
            self.metrics.val_adv_acc,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "v_adv_l",
            adversarial_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "v_task_l",
            task_loss,
            batch_size=self._get_batch_size(data),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return {
            "features": features,
            "task_preds": task_preds,
            "adversarial_pred": adversarial_pred,
        }

    def _crop_to_in_distribution(self, data, features):
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
                {"params": self._tasks.parameters()},
            ],
            lr=self._args.lr1,
            **self._args.opt1_kwargs
        )
        opt2 = self._args.opt2(
            [
                {"params": self._adversarial_task.parameters()},
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
        ZenithReconstructionWithKappa(hidden_size=tito.nb_outputs,
                                      loss_function=VonMisesFisher2DLoss()),
    )
    print(adv_net)
