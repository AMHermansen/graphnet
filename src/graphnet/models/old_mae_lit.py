"""Old code for Masked Auto Encoder."""
from typing import Optional, Dict, List, Any, Tuple, Type

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric import utils
from torch_geometric.data import Data

from graphnet.models import Model
from graphnet.models.bept import Extractor, DeepIceModelNoExtractor


class MAELit(Model):
    """Old MAE model."""

    def __init__(
        self,
        extractor: Optional[nn.Module] = None,
        model: Optional[nn.Module] = None,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        *,
        dim: int = 384,
        dim_base: int = 128,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
        mask_probability: float = 0.2,
        in_dim: int = 7,
        masking_type_probabilities: Optional[List[float]] = None,
    ) -> None:
        """Initialize old model.

        Args:
            extractor: Extractor module.
            model: Primary model.
            optimizer_class: Optimizer to use.
            optimizer_kwargs: Keyword arguments for the optimizer.
            scheduler_class: Scheduler to use.
            scheduler_kwargs: Keyword arguments for the scheduler.
            scheduler_config: Configuration for the scheduler.
            dim: Dimension of the model.
            dim_base: Dimension of the extractor.
            depth: Depth of the model.
            head_size: Number of features per attention head.
            depth_rel: Number of relative attention layers.
            n_rel: Last time when relative attention is updated.
            mask_probability: Probability to mask a node.
            in_dim: Number of features per pulse.
            masking_type_probabilities: Probability for each masking type. See https://arxiv.org/abs/1605.07648
        """
        super().__init__()
        self.extractor = extractor or Extractor(dim_base, dim)
        self.model = model or DeepIceModelNoExtractor(
            dim=dim,
            depth=depth,
            head_size=head_size,
            depth_rel=depth_rel,
            n_rel=n_rel,
        )
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_config = scheduler_config or {}
        self._mask_type_probability = masking_type_probabilities or [
            0.1,
            0.1,
            0.8,
        ]

        self.mask_token = nn.Parameter(torch.rand(1, 1, dim))

        self.masking_proportions = [1 - mask_probability] + [
            mask_probability * p for p in self._mask_type_probability
        ]
        self.masking_intervals = (
            torch.cumsum(torch.tensor(self.masking_proportions), dim=0)
            .numpy()
            .tolist()
        )

        self.out = nn.Linear(dim, in_dim)
        self.cls_out = nn.Linear(dim, 4)

        self.ae_loss = nn.SmoothL1Loss()
        self.ae_loss = nn.SmoothL1Loss()

    def training_step(self, data: Data, batch_idx: int) -> Dict[str, Any]:
        """Train step.

        Args:
            data: Data object.
            batch_idx: Batch index.

        Returns: Dictionary of various losses.
        """
        ae_loss, cls_loss, *_ = self._compute_losses(data)
        self.log(
            "train_ae",
            ae_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log("train_cls", cls_loss)
        return {
            "loss": ae_loss + cls_loss,
            "ae_loss": ae_loss,
            "cls_loss": cls_loss,
        }

    def validation_step(self, data: Data, batch_idx: int) -> Dict[str, Any]:
        """Validate step.

        Args:
            data: Data object.
            batch_idx: Batch index.

        Returns: Dictionary containing results.
        """
        (
            ae_loss,
            cls_loss,
            x_pred,
            x_true,
            mask,
            mask_type,
        ) = self._compute_losses(data)
        self.log(
            "val_ae",
            ae_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_cls",
            cls_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_loss",
            cls_loss + ae_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "mean_guess",
            self.ae_loss(
                torch.mean(x_true, dim=0).expand(x_true.shape), x_true
            ),
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return {
            "loss": ae_loss + cls_loss,
            "ae_loss": ae_loss,
            "cls_loss": cls_loss,
            "x_pred": x_pred,
            "x_true": x_true,
            "mask": mask,
            "mask_type": mask_type,
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
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_orig, mask = utils.to_dense_batch(data.x, data.batch)
        mask_type = self._create_masktype(
            x_orig.shape[:-1], self.masking_intervals, x_orig.device
        )
        self._apply_masking(x_orig, mask_type)
        x = self.extractor(x_orig)
        x = self.model(x, x_orig, mask)
        return x, mask_type, mask

    @staticmethod
    def _create_masktype(
        shape: Tuple[int],
        cum_mask_probabilities: List[float],
        device: str,
    ) -> torch.Tensor:
        mask_type = torch.rand(shape, device=device)
        mask_type[mask_type < cum_mask_probabilities[0]] = 0
        tot = len(cum_mask_probabilities)
        mask_type[mask_type > cum_mask_probabilities[-1]] = tot - 1
        for index, (p_low, p_high) in enumerate(
            zip(cum_mask_probabilities[:-1], cum_mask_probabilities[1:])
        ):
            mask_type[(mask_type >= p_low) & (mask_type < p_high)] = index + 1
        return mask_type

    def _apply_masking(self, x: torch.Tensor, mask_type: torch.Tensor) -> None:
        # mask_type: 0 - no mask, 1 - masked but original, 2 - masked and random, 3 - masked and mask_token
        x[mask_type == 2] = torch.normal(
            mean=torch.zeros_like(x[mask_type == 2]), std=0.2
        ).to(x.device)
        mask_token = self.mask_token.expand(x[mask_type == 3].shape[0], 7)
        x[mask_type == 3] = mask_token

    def _compute_losses(
        self, data: Data
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x_true = utils.to_dense_batch(data.x, data.batch)[0]
        x_pred, mask_type, mask = self._shared_step(data)
        x_cls, x_pred = x_pred[:, 0], x_pred[:, 1:]

        x_pred = self.out(x_pred)
        x_cls = self.cls_out(x_cls)

        x_true_masked = x_true[mask]
        x_pred_masked = x_pred[mask]
        mask_type_masked = mask_type[mask]
        x_true_full_masked = x_true_masked[mask_type_masked > 0]
        x_pred_full_masked = x_pred_masked[mask_type_masked > 0]
        ae_loss = self.ae_loss(x_pred_full_masked, x_true_full_masked)
        cls_loss = self.cls_loss(
            x_cls,
            torch.stack(
                [data.xq_mean, data.yq_mean, data.zq_mean, data.time_duration],
                dim=-1,
            ),
        )  # Make a better task for the cls token. ... Maybe predict total time span, total charge, other derived values.
        return ae_loss, cls_loss, x_pred, x_true, mask, mask_type
