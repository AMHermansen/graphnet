"""Lightning Transformer."""
from typing import Optional, Dict, Any, List, Tuple, Union

from icecream import ic
from torch_geometric.utils import to_dense_batch

import torch
from torch import nn
from torch_geometric.data import Data

from .model import Model
from .task import Task


class IceMixT(Model):
    """Masked Auto Encoder model."""

    def __init__(
        self,
        backbone: nn.Module,
        tasks: Union[Task, List[Task]],
        optimizer_class: type = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Initialize MAELitR.

        Args:
            extractor: Encoder extractor.
            model: Feature learner.
            tasks: List of Tasks for the model.
            optimizer_class: Which optimizer to use.
            optimizer_kwargs: Keyword arguments for the optimizer.
            scheduler_class: Which Scheduler to use.
            scheduler_kwargs: Keyword arguments for the scheduler.
            scheduler_config: Configuration for the scheduler.
        """
        super().__init__(class_name=self.__class__.__name__, name=__name__)
        if not isinstance(tasks, list):
            tasks = nn.ModuleList([tasks])

        # self.extractor = extractor

        self.backbone = backbone
        self._tasks = tasks

        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_config = scheduler_config or {}

    @property
    def tasks(self) -> "ModuleList[Task]":
        """Return tasks."""
        return self._tasks

    @property
    def target_labels(self) -> List[str]:
        """Return target label."""
        return [label for task in self._tasks for label in task._target_labels]

    @property
    def prediction_labels(self) -> List[str]:
        """Return prediction labels."""
        return [
            label for task in self._tasks for label in task._prediction_labels
        ]

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[torch.Tensor, Data]]:
        """Forward pass, chaining model components."""
        if isinstance(data, Data):
            data = [data]
        x_list = []
        for d in data:
            x = self.backbone(d)
            # x_orig, padding_mask = to_dense_batch(d.x, d.batch)
            # x = self.extractor(x_orig)
            # x = self.model(x, x_orig, padding_mask)[:, [0], :]
            x_list.append(x)
        x = torch.cat(x_list, dim=0)
        preds = [task(x) for task in self._tasks]
        return preds

    def training_step(
        self, train_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Dict[str, Any]:
        """Perform training step."""
        if isinstance(train_batch, Data):
            train_batch = [train_batch]
        preds = self(train_batch)
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
        return {"loss": loss, "preds": preds[0]}

    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Dict[str, Any]:
        """Perform validation step."""
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        preds = self(val_batch)
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
        return {"loss": loss, "preds": preds[0]}

    def test_step(self, test_batch: Union[Data, List[Data]], batch_idx: int) -> Dict[str, Any]:
        """Perform test step."""
        if isinstance(test_batch, Data):
            test_batch = [test_batch]
        preds = self(test_batch)
        loss = self._compute_loss(preds, test_batch)
        self.log(
            "test_loss",
            loss,
            batch_size=self._get_batch_size(test_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return {"loss": loss, "preds": preds[0]}

    def predict_step(
        self, predict_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Dict[str, Any]:
        if isinstance(predict_batch, Data):
            predict_batch = [predict_batch]

        preds = self(predict_batch)
        return {"preds": preds}

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

    def _compute_loss(
        self, preds: torch.Tensor, data: List[Data], verbose: bool = False
    ) -> torch.Tensor:
        """Compute and sum losses across tasks."""
        data_merged = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in data], dim=0)
        for task in self._tasks:
            if task._loss_weight is not None:
                data_merged[task._loss_weight] = torch.cat(
                    [d[task._loss_weight] for d in data], dim=0
                )

        losses = [
            task.compute_loss(pred.squeeze(1), data_merged)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))
