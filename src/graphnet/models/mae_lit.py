"""Masked Auto Encoder model."""
from typing import Optional, Dict, Any, List, Tuple

from icecream import ic

from .bept import (
    DeepIceModelNoExtractor,
    Extractor,
    PositionExtractor,
    SinusoidalPosEmb,
    TransformerBlock,
)
import torch
from torch import nn
from torch_geometric import utils
from torch_geometric.data import Data

from .mae_loss import CenterRegressionLoss, MaskedSmoothL1Loss, CELoss
from .model import Model


class MAELitR(Model):
    """Masked Auto Encoder model."""

    def __init__(
        self,
        extractor: Optional[nn.Module] = None,
        model: Optional[nn.Module] = None,
        ae_loss: Optional[nn.Module] = None,
        cls_loss: Optional[nn.Module] = None,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        *,
        dim: int = 192,
        dim_base: int = 128,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        decoder_depth: int = 2,
        n_rel: int = 1,
        in_dim: int = 7,
        cls_loss_scale: float = 1.0,
    ) -> None:
        """Initialize MAELitR.

        Args:
            extractor: Encoder extractor.
            decode_extractor: Decoder extractor.
            model: Feature learner.
            ae_loss: Auto-Encoder loss function.
            cls_loss: Cls-token task loss function.
            optimizer_class: Which optimizer to use.
            optimizer_kwargs: Keyword arguments for the optimizer.
            scheduler_class: Which Scheduler to use.
            scheduler_kwargs: Keyword arguments for the scheduler.
            scheduler_config: Configuration for the scheduler.
            dim: Dimension of the model.
            dim_base: Base dimension for the extractors.
            depth: Number of layers in the model.
            head_size: Number of features per attention head.
            depth_rel: Number of relative attention layers.
            decoder_depth: Number of layers in the decoder.
            n_rel: Number of times to update relative attention.
            in_dim: Number of features per pulse.
            cls_loss_scale: Scale of the cls loss.
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
        self.cls_loss = cls_loss
        self.ae_loss = ae_loss

        if self.cls_loss is None and self.ae_loss is None:
            raise ValueError("Must provide at least one loss function.")

        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_config = scheduler_config or {}

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, dim // 2), requires_grad=True
        )  # Decoder dim

        self.decoder_depth = decoder_depth
        if decoder_depth > 0:
            self.decode_pos_embedder = SinusoidalPosEmb(dim // 2)
            self.encode_decode_proj = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim // 2)
            )
        else:
            self.decode_pos_embedder = nn.Identity()
            self.encode_decode_proj = nn.Identity()

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim // 2, num_heads=(dim // 2) // head_size
                )
                for i in range(decoder_depth)
            ]
        )

        self.cls_loss_scale = cls_loss_scale

        # self.save_hyperparameters()

    def _shared_step(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pred, mask, ids_restore = self.encode(data)
        cls = x_pred[:, 0, :]
        if self.decoder_depth:
            x_pred = self.decode(data, ids_restore, x_pred)
        _, padding_mask = utils.to_dense_batch(data.x, data.batch)
        return x_pred, mask & padding_mask, cls

    def training_step(self, data: Data, batch_idx: int) -> Dict[str, Any]:
        """Training step of the model.

        Args:
            data: Data object.
            batch_idx: Batch index.

        Returns: Dictionary containing various losses.
        """
        x_pred, mask, cls = self._shared_step(data)
        ae_loss, cls_loss, *_ = self._compute_loss(data, x_pred, mask, cls)
        self.log(
            "train_ae",
            ae_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "train_cls",
            cls_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return {
            "loss": ae_loss + cls_loss,
            "ae_loss": ae_loss,
            "cls_loss": cls_loss,
        }

    def on_train_epoch_end(self):
        if self.ae_loss is None:
            return
        for name, metric in self.ae_loss.metrics.items():
            self.log(
                f"train_{name}",
                metric.compute(),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
            metric.reset()

    def validation_step(
        self, data: Data, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validate step of the model.

        Args:
            data: Data object.
            batch_idx: Batch index.

        Returns:
            Dictionary containing various relevant quantities.
        """
        x_pred, mask, cls = self._shared_step(data)
        (
            ae_loss,
            cls_loss,
            cls_pred,
            cls_target,
            x_pred,
            x_true,
        ) = self._compute_loss(data, x_pred, mask, cls)
        self.log(
            "val_ae",
            ae_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_cls",
            cls_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_loss",
            cls_loss + ae_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return {
            "loss": (ae_loss + cls_loss).to(torch.float32),
            "ae_loss": ae_loss.to(torch.float32),
            "cls_loss": cls_loss.to(torch.float32),
            "x_pred": x_pred,
            "mask": mask.to(torch.float32),
            "cls_pred": cls_pred,
            "cls_target": cls_target,
        }

    def on_validation_epoch_end(self):
        if self.ae_loss is None:
            return
        for name, metric in self.ae_loss.metrics.items():
            self.log(
                f"val_{name}",
                metric.compute(),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
            metric.reset()

    def predict_step(
        self, data: Data, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step, used to predict on the predict set.

        Args:
            data: Data object.
            batch_idx: Batch index.

        Returns:
            Dictionary containing various relevant quantities.
        """
        x_pred, mask, cls = self._shared_step(data)
        (
            ae_loss,
            cls_loss,
            cls_pred,
            cls_target,
            x_pred,
            x_true,
        ) = self._compute_loss(data, x_pred, mask, cls)
        return {
            "loss": (ae_loss + cls_loss).to(torch.float32),
            "ae_loss": ae_loss.to(torch.float32),
            "cls_loss": cls_loss.to(torch.float32),
            "x_pred": x_pred,
            "x_true": x_true,
            "mask": mask.to(torch.bool),
            "cls_pred": cls_pred,
            "cls_target": cls_target,
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

    def encode(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the data.

        Args:
            data: Data object.

        Returns:
            Tuple of the encoded data, the mask, and the ids_restore.
        """
        x_orig, padding_mask = utils.to_dense_batch(data.x, data.batch)
        x = self.extractor(x_orig)
        x, ids_keep_mask, mask, ids_restore = self._apply_masking(x, data)
        x_orig_masked, *_ = self._apply_masking(x_orig, data)
        x = self.model(x, x_orig_masked, ids_keep_mask)
        return x, mask, ids_restore

    def decode(
        self, data: Data, ids_restore: torch.Tensor, encoded_features: torch.Tensor
    ) -> torch.Tensor:
        """Decode procedure for MAE.

        Args:
            data: Data object.
            ids_restore: Indices to restore the original order of the data.
            encoded_features: Encoded features.

        Returns:
            Decoded data in latent space.
        """
        projected_features = self.encode_decode_proj(encoded_features)

        x = self._undo_masking(data, projected_features)
        for blk in self.decoder_blocks:
            x = blk(x)

        return x

    def _compute_loss(
        self, data: Data, x_pred: torch.Tensor, mask: torch.Tensor, cls: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        x_true, padding_mask = utils.to_dense_batch(data.x, data.batch)
        x_pred = x_pred[:, 1:, :]

        # Loss from auto-encoder
        if self.ae_loss is None:
            ae_loss = torch.zeros(1, device=self.device)
            x_pred = {"pred_orig": x_pred}
            x_true = {"true_orig": x_true}
        else:
            ae_loss, x_pred, x_true = self.ae_loss(x_pred, x_true, mask, data)

        # Loss from CLS token
        if self.cls_loss is None:
            cls_loss = torch.zeros(1, device=self.device)
            cls_pred = torch.zeros(1, device=self.device)
            cls_target = torch.zeros(1, device=self.device)
        else:
            cls_loss, cls_pred, cls_target = self.cls_loss(cls, data, mask)
        return (
            ae_loss,
            self.cls_loss_scale * cls_loss,
            cls_pred,
            cls_target,
            x_pred,
            x_true,
        )

    def _apply_masking(
        self, x: torch.Tensor, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, F = x.shape
        ids_keep, ids_keep_mask = self._get_ids_keep_batch(data)
        ids_restore, ids_restore_mask = self._get_ids_restore_batch(data)
        mask = torch.ones([B, L], device=x.device)
        for i, n in enumerate(data.n_keep):
            mask[i, :n] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return (
            torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, F)
            ),
            ids_keep_mask,
            mask.to(bool),  # convert to bool
            ids_restore,
        )

    def _undo_masking(self, data: Data, x: torch.Tensor):
        ids_restore, _ = self._get_ids_restore_batch(data)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1] + 1, 1
        )
        x_cls = x[:, :1, :]
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x = torch.gather(
            x_no_cls,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )  # restore the original order
        x = x + self.decode_pos_embedder(
            torch.arange(x.shape[1], device=x.device)
        )
        x = torch.cat([x_cls, x_no_cls], dim=1)  # prepend cls token
        return x

    def _get_ids_keep_batch(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids_keep_batch = self.custom_repeat(data.n_keep)
        return utils.to_dense_batch(data.ids_keep, ids_keep_batch)

    def _get_ids_restore_batch(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return utils.to_dense_batch(data.ids_restore, data.batch)

    @staticmethod
    def custom_repeat(tensor: torch.Tensor) -> torch.Tensor:
        """Create batch indices from the number of nodes in each graph.

        Args:
            tensor: Number of nodes in each graph.

        Returns:
            Batch indices.
        """
        result: List[torch.Tensor] = []
        for i, n in enumerate(tensor):
            repeated_sequence = torch.full(
                (n,), i, dtype=torch.int64, device=tensor.device
            )
            result.append(repeated_sequence)
        return torch.cat(result)


class DOMAwareMAELit(MAELitR):
    def __init__(
            self,
            extractor: Optional[nn.Module] = None,
            model: Optional[nn.Module] = None,
            ae_loss: Optional[nn.Module] = None,
            cls_loss: Optional[nn.Module] = None,
            optimizer_class: type = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict] = None,
            scheduler_class: Optional[type] = None,
            scheduler_kwargs: Optional[Dict] = None,
            scheduler_config: Optional[Dict] = None,
            *,
            dim: int = 192,
            dim_base: int = 128,
            depth: int = 12,
            head_size: int = 32,
            depth_rel: int = 4,
            decoder_depth: int = 2,
            n_rel: int = 1,
            in_dim: int = 7,
            cls_loss_scale: float = 1.0,
    ):
        super().__init__(
        extractor,
        model,
        ae_loss,
        cls_loss,
        optimizer_class,
        optimizer_kwargs,
        scheduler_class,
        scheduler_kwargs,
        scheduler_config,
        dim=dim,
        dim_base=dim_base,
        depth=depth,
        head_size=head_size,
        depth_rel=depth_rel,
        decoder_depth=decoder_depth,
        n_rel=n_rel,
        in_dim=in_dim,
        cls_loss_scale=cls_loss_scale,
    )
        self.decode_extractor = PositionExtractor(dim_base, dim // 2)
        self.encode_decode_proj2 = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2)
        )

    def _undo_masking(self, data: Data, x: torch.Tensor):
        x = super()._undo_masking(data, x)
        x_orig, padding_mask = utils.to_dense_batch(data.x, data.batch)
        x_orig = x_orig[..., :3]
        # x_orig.shape
        embedded_x_orig = self.decode_extractor(x_orig)
        embedded_x_orig = torch.cat([x[:, :1, :], embedded_x_orig], dim=1)  # prepend cls token
        x = torch.cat([embedded_x_orig, x], dim=2)
        x = self.encode_decode_proj2(x)
        return x
