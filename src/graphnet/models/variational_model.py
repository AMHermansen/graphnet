from typing import Optional, Dict, List, Union, Any, Tuple

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.nn.functional import softplus
from torch.optim import Adam
from torch_geometric.data import Data
from torchmetrics import Metric

from graphnet.models.gnn.aggregator import Aggregator
from graphnet.models.lightweight_model import LightweightTemplateModel
from graphnet.models.model import Model
from graphnet.models.gnn.gnn import GNN, StandardGNN, RawGNN
from graphnet.models.task import Task


class VariationalModel(LightweightTemplateModel):
    """A more lightweight version of StandardModel.

    More in line with the styleguide of Lightning.
    """

    var_eps = 1e-8

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

    @staticmethod
    def reparametrize(dist):
        return dist.rsample()

    def training_step(
        self, train_batch: Data, batch_idx: int
    ) -> Dict[str, Any]:
        """Perform training step."""
        preds, loss_kl, mu, log_var = self._shared_step(train_batch, batch_idx)
        loss = self._compute_loss(preds, train_batch)
        self.log(
            "train_loss",
            loss+loss_kl,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "train_kl_loss",
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

        return {"loss": loss + loss_kl, "preds": preds}

    def validation_step(
        self, val_batch: Data, batch_idx: int
    ) -> Dict[str, Any]:
        """Perform validation step."""
        preds, loss_kl, mu, log_var = self._shared_step(val_batch, batch_idx)
        loss = self._compute_loss(preds, val_batch)
        self.log(
            "val_loss",
            loss+loss_kl,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_kl_loss",
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

        return {"loss": loss + loss_kl/20, "preds": preds, "mu_latent": mu, "logvar_latent": log_var}

    def test_step(self, test_batch: Data, batch_idx: int) -> Dict[str, Any]:
        """Perform test step."""
        preds, loss_kl, *_ = self._shared_step(test_batch, batch_idx)
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
        mu, logvar = torch.chunk(encoded_latent_space, 2, dim=-1)
        preds = [task(encoded_latent_space) for task in self._tasks]
        return {
            "preds": preds,
            "latent_features": encoded_latent_space,
            "node_features": encoded_graph.encoded_x,
            "latent_features_mu": mu,
            "latent_features_logvar": logvar,
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
    ) -> Tuple[List[Union[Tensor, Data]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform shared step.

        Applies the forward pass and the following loss calculation,
        shared between the training and validation step.
        """
        x = self._gnn(batch)  # noqa
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = softplus(logvar) + self.var_eps
        scale_tril = torch.diag_embed(scale)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        z = self.reparametrize(dist)
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        out = [task(z) for task in self._tasks]
        return out, loss_kl, mu, logvar
