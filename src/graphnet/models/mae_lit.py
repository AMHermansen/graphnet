from typing import Optional, Dict, Any, List, Union

from icecream import ic

from .bept import DeepIceModelNoExtractor, Extractor, SinusoidalPosEmb, TransformerBlock
import torch
from torch import nn
from torch_geometric import utils
from torch_geometric.data import Data
from .model import Model


class CenterRegressionLoss(nn.Module):
    _out_dim = 4

    def __init__(self, dim: int = 384):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, self._out_dim)
        self.loss = nn.SmoothL1Loss()

    def forward(self, cls: torch.Tensor, data: Data):
        cls = self.proj(cls)
        return self.loss(cls, target := torch.stack([data.xq_mean, data.yq_mean, data.zq_mean, data.time_duration], dim=-1)), cls, target


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor, mask: torch.Tensor):
        return self.loss(x_pred[mask], x_true[mask]), x_pred, x_true


class MaskedChamferLoss(nn.Module):
    def __init__(self, p: Union[int, List[int]]):
        super().__init__()
        if isinstance(p, int):
            p = [p]
        self.p = p

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor, mask: torch.Tensor):
        B, L, F = x_pred.shape
        losses = []
        for b in range(B):
            for p in self.p:
                losses.append(self._single_row_chamfer(x_pred[b], x_true[b], mask[b], p))
        return torch.stack(losses).mean(), x_pred, x_true

    def _single_row_chamfer(self, x_pred, x_true, mask, p):
        x_pred = x_pred[mask]
        x_true = x_true[mask]
        cross_dist = torch.cdist(x_pred, x_true, p=p).pow(p)
        return (
                    cross_dist.min(dim=0)[0].mean()
                    + cross_dist.min(dim=1)[0].mean()
        ) / 2


class MAELitR(Model):
    def __init__(
        self,
        extractor=None,
        decode_extractor=None,
        model=None,
        ae_loss=None,
        cls_loss=None,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        *,
        dim: int = 384,
        dim_base: int = 128,
        depth: int = 12,
        use_checkpoint: bool = False,
        head_size: int = 32,
        depth_rel: int = 4,
        decoder_depth: int = 2,
        n_rel: int = 1,
        in_dim: int = 7,
        cls_loss_scale: float = 1.0
    ):
        super().__init__()
        self.extractor = extractor or Extractor(dim_base, dim)
        self.decode_extractor = decode_extractor or Extractor(dim_base, dim // 2)
        self.model = model or DeepIceModelNoExtractor(
            dim=dim,
            dim_base=dim_base,
            depth=depth,
            use_checkpoint=use_checkpoint,
            head_size=head_size,
            depth_rel=depth_rel,
            n_rel=n_rel
        )
        self.cls_loss = cls_loss or CenterRegressionLoss(dim // 2)
        self.ae_loss = ae_loss or MaskedSmoothL1Loss()

        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._scheduler_config = scheduler_config or {}

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim // 2), requires_grad=True)  # Decoder dim
        self.decode_pos_embedder = SinusoidalPosEmb(dim // 2)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=dim // 2, num_heads=dim // head_size)
                for i in range(decoder_depth)
            ]
        )
        self.decode_proj = nn.Sequential(
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, in_dim)
        )
        self.cls_loss_scale = cls_loss_scale

    def _shared_step(self, data: Data):
        x_pred, mask, ids_restore = self.encode(data)
        x_pred = self.decode(data, ids_restore)
        _, padding_mask = utils.to_dense_batch(data.x, data.batch)
        return x_pred, mask & padding_mask

    def training_step(self, data: Data, batch_idx: int):
        x_pred, mask = self._shared_step(data)
        ae_loss, cls_loss, *_ = self._compute_loss(data, x_pred, mask)
        self.log(
            'train_ae',
            ae_loss,
            batch_size=self._get_batch_size([data]),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
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
        return {"loss": ae_loss + cls_loss, "ae_loss": ae_loss, "cls_loss": cls_loss}

    def validation_step(self, data: Data, batch_idx: int) -> Dict[str, torch.Tensor]:
        x_pred, mask = self._shared_step(data)
        ae_loss, cls_loss, cls_pred, cls_target, x_pred, x_true = self._compute_loss(data, x_pred, mask)
        self.log(
            'val_ae',
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
            "x_pred": x_pred.to(torch.float32),
            "mask": mask.to(torch.float32),
            "cls_pred": cls_pred.to(torch.float32),
            "cls_target": cls_target.to(torch.float32),
        }

    def predict_step(self, data: Data, batch_idx: int) -> Dict[str, torch.Tensor]:
        x_pred, mask = self._shared_step(data)
        ae_loss, cls_loss, cls_pred, cls_target, x_pred, x_true = self._compute_loss(data, x_pred, mask)
        return {
            "loss": (ae_loss + cls_loss).to(torch.float32),
            "ae_loss": ae_loss.to(torch.float32),
            "cls_loss": cls_loss.to(torch.float32),
            "x_pred": x_pred.to(torch.float32),
            "x_true": x_true.to(torch.float32),
            "mask": mask.to(torch.bool),
            "cls_pred": cls_pred.to(torch.float32),
            "cls_target": cls_target.to(torch.float32),
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

    def encode(self, data):
        x_orig, padding_mask = utils.to_dense_batch(data.x, data.batch)
        x = self.extractor(x_orig)
        x, ids_keep_mask, mask, ids_restore = self._apply_masking(x, data)
        x_orig_masked, *_ = self._apply_masking(x_orig, data)
        x = self.model(x, x_orig_masked, ids_keep_mask)
        return x, mask, ids_restore

    def decode(self, data, ids_restore):
        x_orig, padding_mask = utils.to_dense_batch(data.x, data.batch)
        x = self.decode_extractor(x_orig)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1] + 1, 1)
        x_cls = x[:, :1, :]
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x = torch.gather(x_no_cls, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # restore the original order
        x = x + self.decode_pos_embedder(torch.arange(x.shape[1], device=x.device))

        x = torch.cat([x_cls, x_no_cls], dim=1)  # prepend cls token
        for blk in self.decoder_blocks:
            x = blk(x)

        return x

    def _compute_loss(self, data, x_pred, mask):
        x_true, padding_mask = utils.to_dense_batch(data.x, data.batch)
        cls, x_pred = x_pred[:, 0, :], x_pred[:, 1:, :]
        x_pred = self.decode_proj(x_pred)
        ae_loss, x_pred, x_true = self.ae_loss(x_pred, x_true, mask)
        cls_loss, cls_pred, cls_target = self.cls_loss(
            cls,
            data
        )
        return ae_loss, self.cls_loss_scale * cls_loss, cls_pred, cls_target, x_pred, x_true

    def _apply_masking(self, x, data):
        B, L, F = x.shape
        ids_keep, ids_keep_mask = self._get_ids_keep_batch(data)
        ids_restore, ids_restore_mask = self._get_ids_restore_batch(data)
        mask = torch.ones([B, L], device=x.device)
        for i, n in enumerate(data.n_keep):
            mask[i, :n] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return (
            torch.gather(
                x,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, F)
            ),
            ids_keep_mask,
            mask.to(bool),  # convert to bool
            ids_restore
        )

    def _get_ids_keep_batch(self, data: Data):
        ids_keep_batch = self.custom_repeat(data.n_keep)
        return utils.to_dense_batch(data.ids_keep, ids_keep_batch)

    def _get_ids_restore_batch(self, data: Data):
        return utils.to_dense_batch(data.ids_restore, data.batch)

    @staticmethod
    def custom_repeat(tensor: torch.Tensor):
        result: List[torch.Tensor] = []
        for i, n in enumerate(tensor):
            repeated_sequence = torch.full((n,), i, dtype=torch.int64, device=tensor.device)
            result.append(repeated_sequence)
        return torch.cat(result)


