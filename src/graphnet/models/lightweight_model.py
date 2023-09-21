"""Class containing a lightweight implementation of a standard module."""
from typing import Optional, Dict, List, Union, Any

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch_geometric.data import Data

from .model import Model
from .gnn.gnn import GNN
from .graphs import GraphDefinition
from .task import Task
from graphnet.utilities.config import save_model_config


class LightweightModel(Model):
    """A more lightweight version of StandardModel.

    More in line with the styleguide of Lightning.
    """

    def __init__(
        self,
        *,
        graph_definition: GraphDefinition,
        gnn: GNN,
        tasks: Union[Task, List[Task]],
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct lightweight lightning model.

        Args:
            graph_definition: NOT USED HERE ONLY FOR OVERLY VERBOSE BOILERPLATE DRIVEN CODE DESIGN.
            gnn: GNN backbone used.
            tasks: Which task the model is trained for.
            optimizer_class: Optimizer used to train.
            optimizer_kwargs: keyword-args for optimizer.
            scheduler_class: Learning rate scheduler for the optimizer.
            scheduler_kwargs: Scheduler keyword-args.
            scheduler_config: Remaining config for scheduler.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, Task) for task in tasks)
        assert isinstance(gnn, GNN)

        # Member variable(s)
        self._graph_definition = graph_definition  # Preprocessing step only done in data, but here for config recon.
        self._gnn = gnn
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()

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

    def _shared_step(self, batch: Data, batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation,
        shared between the training and validation step.
        """
        preds = self._gnn(batch)  # noqa

        return preds

    def _compute_loss(
        self, preds: Tensor, data: Data, verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        losses = [
            task._compute_loss(pred, data)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.mean(torch.stack(losses))

    @staticmethod
    def _get_batch_size(data: Data) -> int:
        """Get batch size."""
        return torch.numel(torch.unique(data.batch))

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
