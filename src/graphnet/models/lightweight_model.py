"""Class containing a lightweight implementation of a standard module."""
from logging import DEBUG
from typing import Optional, Dict, List, Union, Any, Tuple

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch_geometric.data import Data
from torchmetrics import Metric

from graphnet.models.gnn.aggregator import Aggregator
from graphnet.models.model import Model
from graphnet.models.gnn.gnn import GNN, StandardGNN, RawGNN
from graphnet.models.task import Task


class LightweightTemplateModel(Model):
    """Adds minimal convenience methods for trainable models."""
    def __init__(self, tasks: Union[Task, List[Task]], **kwargs):
        super().__init__(**kwargs)
        if isinstance(tasks, Task):
            tasks = [tasks]
        self._tasks = ModuleList(tasks)

    def _update_train_metrics(self, preds, target):
        for task, pred in zip(self._tasks, preds):
            task.compute_metrics(pred, target, train=True)

    def _update_val_metrics(self, preds, target):
        for task, pred in zip(self._tasks, preds):
            task.compute_metrics(pred, target, train=False)

    def _log_metrics(self, metrics, metrics_log, **kwargs):
        for (key, metric), (_, metric_log) in zip(metrics.items(), metrics_log.items()):
            if metric_log:
                self.log(
                    f"{key}",
                    metric,
                    **kwargs,
                )

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

    @property
    def val_metrics(self) -> Tuple[Dict[str, Metric], Dict[str, bool]]:
        metrics = {}
        metrics_log = {}
        for task in self._tasks:
            target_str = "_".join(task._target_labels)
            for key, metric in task.val_metrics.items():
                metrics[f"{key}_{target_str}"] = metric
            for key, metric_log in task.val_metrics_log.items():
                metrics_log[f"{key}_{target_str}"] = metric_log
        return metrics, metrics_log

    @property
    def train_metrics(self) -> Tuple[Dict[str, Metric], Dict[str, bool]]:
        metrics = {}
        metrics_log = {}
        for task in self._tasks:
            target_str = "_".join(task._target_labels)
            for key, metric in task.train_metrics.items():
                metrics[f"{key}_{target_str}"] = metric
            for key, metric_log in task.train_metrics_log.items():
                metrics_log[f"{key}_{target_str}"] = metric_log
        return metrics, metrics_log

    @property
    def target_labels(self) -> List[str]:
        """Return target label."""
        return [label for task in self._tasks for label in task._target_labels]

    @property
    def tasks(self) -> "ModuleList[Task]":
        """Return tasks."""
        return self._tasks

    @property
    def prediction_labels(self) -> List[str]:
        """Return prediction labels."""
        return [
            label for task in self._tasks for label in task._prediction_labels
        ]


class LightweightModel(LightweightTemplateModel):
    """A more lightweight version of StandardModel.

    More in line with the styleguide of Lightning.
    """

    def __init__(
        self,
        *,
        gnn: RawGNN,
        aggregator: Aggregator,
        tasks: Union[Task, List[Task]],
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        state_dict_path: Optional[str] = "",
    ) -> None:
        """Construct lightweight lightning model.

        Args:
            gnn: GNN backbone used.
            tasks: Which task the model is trained for.
            optimizer_class: Optimizer used to train.
            optimizer_kwargs: keyword-args for optimizer.
            scheduler_class: Learning rate scheduler for the optimizer.
            scheduler_kwargs: Scheduler keyword-args.
            scheduler_config: Remaining config for scheduler.
            state_dict_path: Path to state_dict_path to load the model weights.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__, tasks=tasks)

        # Member variable(s)
        self._gnn = StandardGNN(gnn, aggregator)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()

        if state_dict_path:
            self.load_state_dict(state_dict_path)

    def forward(self, data: Data) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        assert isinstance(data, Data)
        x = self._gnn(data)
        preds = [task(x) for task in self._tasks]
        return preds

    def training_step(
        self, train_batch: Data, batch_idx: int
    ) -> Dict[str, Any]:
        """Perform training step."""
        preds = self._shared_step(train_batch, batch_idx)
        loss = self._compute_loss(preds, train_batch)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self._log_metrics(
            *self.train_metrics,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return {"loss": loss, "preds": preds}

    def validation_step(
        self, val_batch: Data, batch_idx: int
    ) -> Dict[str, Any]:
        """Perform validation step."""
        preds = self._shared_step(val_batch, batch_idx)
        loss = self._compute_loss(preds, val_batch)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self._log_metrics(
            *self.val_metrics,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return {"loss": loss, "preds": preds}

    def test_step(self, test_batch: Data, batch_idx: int) -> Dict[str, Any]:
        """Perform test step."""
        preds = self._shared_step(test_batch, batch_idx)
        loss = self._compute_loss(preds, test_batch)
        self.log(
            "test_loss",
            loss,
            batch_size=self._get_batch_size(test_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss, "preds": preds}

    def predict_step(self, predict_batch: Data, batch_idx: int) -> Dict[str, Any]:
        encoded_graph = self._gnn._gnn(predict_batch)
        encoded_latent_space = self._gnn._aggregation(encoded_graph)
        preds = [task(encoded_latent_space) for task in self._tasks]
        return {
            "preds": preds,
            "latent_features": encoded_latent_space,
            "node_features": encoded_graph.encoded_x,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def _shared_step(
        self, batch: Data, batch_idx: int
    ) -> List[Union[Tensor, Data]]:
        """Perform shared step.

        Applies the forward pass and the following loss calculation,
        shared between the training and validation step.
        """
        preds = self._gnn(batch)  # noqa
        out = [task(preds) for task in self._tasks]
        return out

