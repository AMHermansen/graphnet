"""Module containing the losses for the Auto-Encoder."""
from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Protocol, Dict, Optional

import torch
from icecream import ic
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from .bept import TransformerBlock
from torchmetrics import Accuracy


class CenterRegressionLoss(nn.Module):
    """Loss for the center regression task, of cls_token.

    Regression towards the mean of the x, y, z coordinates and the time
    duration.
    """

    _out_dim = 4

    def __init__(self, dim: int = 384):
        """Initialize CenterRegressionLoss.

        Args:
            dim: Latent features.
        """
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, self._out_dim)
        self.loss = nn.SmoothL1Loss()

    def forward(
        self,
        cls: torch.Tensor,
        data: Data,
        ae_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the loss.

        Args:
            cls: cls_token, see https://arxiv.org/abs/1605.07648.
            data: Data object.

        Returns:
            Tuple containing the loss, the predictions and the targets.
        """
        cls = self.proj(cls)
        return (
            self.loss(
                cls,
                target := torch.stack(
                    [
                        data.xq_mean,
                        data.yq_mean,
                        data.zq_mean,
                        data.time_duration,
                    ],
                    dim=-1,
                ),
            ),
            cls,
            target,
        )


class ClsCELoss(nn.Module):
    def __init__(self, latent_dim: int = 192, num_classes: int = 5484):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.proj = nn.Linear(latent_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        cls: torch.Tensor,
        data: Data,
        ae_mask: torch.Tensor,
    ):
        sensor_id, _ = to_dense_batch(data.sensor_id, data.batch)
        sensor_id = sensor_id.to(torch.int64)
        cls = self.proj(cls)
        loss = self.compute_loss(cls, sensor_id, ae_mask)
        return loss, {"prediction_distribution": cls}, {"target": sensor_id}

    def compute_loss(
        self, cls: torch.Tensor, sensor_id: torch.Tensor, ae_mask: torch.Tensor
    ):
        loss = 0
        for cls_event, sensor_id_event, ae_mask_event in zip(
            cls, sensor_id, ae_mask
        ):
            sensor_id_event = sensor_id_event[ae_mask_event]
            cls_event = cls_event.repeat(sensor_id_event.shape[0], 1)
            loss += self.loss(cls_event, sensor_id_event)
        return loss / cls.shape[0]  # Normalize loss per event


class AELoss(nn.Module, ABC):
    """Base class for AE losses."""

    def __init__(self, latent_dim: int = 192, output_dim: int = 7):
        """Initialize AELoss.

        Args:
            latent_dim: Latent features.
            output_dim: Original dimension of the data.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(latent_dim, output_dim)
        self.metrics = nn.ModuleDict({})

    @abstractmethod
    def compute_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        pass

    @abstractmethod
    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass of Auto-Encoder Loss.

        Args:
            x_latent: Latent representation of the data.
            x_true: True (scaled) values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Tuple of loss, predictions and targets.
        """
        pass


class MaskedSmoothL1Loss(AELoss):
    """SmoothL1Loss, which only uses values from a mask."""

    def __init__(self, latent_dim: int = 192, output_dim: int = 7) -> None:
        """Initialize MaskedSmoothL1Loss.

        Args:
            latent_dim: Number of latent dimensions.
            output_dim: The original dimension of the data.
        """
        super().__init__(latent_dim, output_dim)
        self.loss = nn.SmoothL1Loss()

    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of Auto-Encoder Loss.

        Args:
            x_latent: Latent representation of the data.
            x_true: True (scaled) values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Tuple of loss, predictions and targets.
        """
        x_pred = self.proj(x_latent)
        return self.compute_loss(x_pred, x_true, mask, data), x_pred, x_true

    def compute_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        return self.loss(x_pred[mask], x_true[mask])


class MaskedChamferLoss(AELoss):
    """Loss function which uses the Chamfer distance."""

    def __init__(
        self,
        p: Union[int, List[int]],
        latent_dim: int = 192,
        output_dim: int = 7,
    ) -> None:
        """Initialize Masked Chamfer Loss.

        Args:
            p: p-norm raised to p to use. If a list is given, the losses for each p are averaged.
            latent_dim: Number of latent dimensions.
            output_dim: The original dimension of the data.
        """
        super().__init__()
        if isinstance(p, int):
            p = [p]
        self.p = p
        self._proj = nn.Linear(latent_dim, output_dim)

    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of Auto-Encoder Loss.

        Args:
            x_latent: Latent representation of the data.
            x_true: True (scaled) values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Tuple of loss, predictions and targets.
        """
        x_pred = self._proj(x_latent)

        loss = self.compute_loss(mask, x_pred, x_true, data)
        return loss, x_pred, x_true

    def compute_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        B, L, F = x_pred.shape
        losses = []
        for b in range(B):
            for p in self.p:
                losses.append(
                    self._single_row_chamfer(x_pred[b], x_true[b], mask[b], p)
                )
        return torch.stack(losses).mean()

    def _single_row_chamfer(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """Calculate the Chamfer distance for a single event in a batch."""
        x_pred = x_pred[mask]
        x_true = x_true[mask]
        cross_dist = torch.cdist(x_pred, x_true, p=p).pow(p)
        return (
            cross_dist.min(dim=0)[0].mean() + cross_dist.min(dim=1)[0].mean()
        ) / 2


class MaskedChamferWithUncertaintyLoss(MaskedChamferLoss):
    """Masked Chamfer with Gaussian uncertainty."""

    def __init__(
        self,
        p: Union[int, List[int]],
        eps: float = 1e-6,
        latent_dim: int = 192,
        input_dim: int = 7,
    ) -> None:
        """Initialize MaskedChamferWithUncertaintyLoss.

        Args:
            p: List of p-norms to use.
            eps: Epsilon to avoid numerical instability.
            latent_dim: Number of latent dimensions.
            input_dim: Number of original dimensions.
        """
        super().__init__(p, output_dim=input_dim + 1, latent_dim=latent_dim)
        self.eps = eps

    def _single_row_chamfer(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """Compute Chamfer distance with uncertainty, for a single event."""
        x_pred = x_pred[mask]
        x_true = x_true[mask]
        sigma = x_pred[:, -1]
        sigma = torch.nn.functional.relu(sigma) + self.eps
        x_pred = x_pred[:, :-1]
        cross_dist = torch.cdist(x_pred, x_true, p=p).pow(p)
        sigma_scaled_dist = cross_dist / sigma.unsqueeze(1)
        closest_pred_from_true = sigma_scaled_dist.argmin(dim=0)
        return (
            sigma_scaled_dist.min(dim=0)[0].mean()
            + sigma_scaled_dist.min(dim=1)[0].mean()
            - torch.log(sigma).mean()
            - torch.log(sigma[closest_pred_from_true]).mean()
        ) / 4


class CELoss(AELoss):
    """Cross entropy loss for classification."""

    def __init__(self, latent_dim=96, num_classes: int = 5484):
        """Initialize CELoss.

        Args:
            num_classes: Number of classes.
        """
        super().__init__(latent_dim=latent_dim, output_dim=num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.post_proj = nn.Softmax(-1)
        acc1 = Accuracy(task="multiclass", num_classes=num_classes)
        acc10 = Accuracy(task="multiclass", num_classes=num_classes, top_k=10)
        acc100 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=100
        )
        acc1000 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1000
        )

        self.metrics = nn.ModuleDict(
            {
                "acc1": acc1,
                "acc10": acc10,
                "acc100": acc100,
                "acc1000": acc1000,
            }
        )

    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass of Auto-Encoder Loss.

        Args:
            x_latent: Latent representation of the data.
            x_true: True (scaled) values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Tuple of loss, predictions and targets.
        """
        x_pred = self.proj(x_latent)
        x_pred_proba = self.post_proj(x_pred)
        loss = self.compute_loss(x_pred, x_true, mask, data)
        sensor_id, _ = to_dense_batch(data.sensor_id, data.batch)
        return loss, {"prediction": x_pred_proba}, {"target": sensor_id}

    def compute_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions, as probabilities.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        sensor_id, _ = to_dense_batch(data.sensor_id, data.batch)
        sensor_id = sensor_id.to(torch.int64)
        x_pred_masked = x_pred[mask]
        sensor_id_masked = sensor_id[mask]
        for _, metric in self.metrics.items():
            metric.update(x_pred_masked, sensor_id_masked)
        return self.loss(x_pred_masked, sensor_id_masked)


class AEIndexLossWithUncertainty(AELoss):
    """Loss function which uses Gaussian NLL on a single index."""

    def __init__(
        self, latent_dim: int = 192, target_index: int = 4, eps: float = 1e-6
    ) -> None:
        """Initialize AEIndexLossWithUncertainty.

        Args:
            latent_dim: Number of latent dimensions.
            target_index: Index to use for the loss.
            eps: Epsilon to avoid numerical instability.
        """
        super().__init__(latent_dim, output_dim=2)
        self.time_index = target_index
        self.eps = eps

    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass of Auto-Encoder Loss.

        Args:
            x_latent: Latent representation of the data.
            x_true: True (scaled) values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Tuple of loss, predictions and targets.
        """
        x_pred = self.proj(x_latent)
        return (
            self.compute_loss(x_pred, x_true, mask, data),
            {"prediction": x_pred},
            {"target": x_true},
        )

    def compute_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        target = x_true[mask][:, self.time_index]
        mu = x_pred[mask][:, 0]
        sigma = x_pred[mask][:, 1]
        sigma = torch.nn.functional.relu(sigma) + self.eps
        return (target - mu).pow(2) / (2 * sigma.pow(2)) + torch.log(sigma)


class MaskedNonPositionWithUncertaintyLoss(AELoss):
    """Loss function which uses Gaussian NLL on non position."""

    def __init__(self, latent_dim: int = 192, original_input_dim: int = 7):
        """Initialize MaskedNonPositionWithUncertaintyLoss.

        Args:
            latent_dim: Number of latent dimensions.
            original_input_dim: Number of dimensions of the original input.
        """
        super().__init__(latent_dim, output_dim=2 * (original_input_dim - 3))
        self.loss = nn.GaussianNLLLoss()

    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass of Auto-Encoder Loss.

        Args:
            x_latent: Latent representation of the data.
            x_true: True (scaled) values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Tuple of loss, predictions and targets.
        """
        x_pred = self.proj(x_latent)
        return (
            self.compute_loss(x_pred, x_true, mask, data),
            {"prediction": x_pred},
            {"target": x_true},
        )

    def compute_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        sigma = x_pred[mask][:, -1]
        sigma = torch.nn.functional.relu(sigma) + 1e-6
        x_pred = x_pred[mask][:, :-1]
        return self.loss(x_pred, x_true[mask], sigma**2)


class MultipleAELosses(nn.Module):
    """Class to combine multiple AE losses."""

    def __init__(self, losses: List[AELoss]) -> None:
        """Initialize MultipleAELosses.

        Args:
            losses: List of AE losses to use.
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.loss_names = [loss.__class__.__name__ for loss in losses]

    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass of Auto-Encoder Loss.

        Args:
            x_latent: Latent representation of the data.
            x_true: True (scaled) values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Tuple of loss, predictions and targets.
        """
        all_out = [
            loss.forward(x_latent, x_true, mask, data) for loss in self.losses
        ]
        losses: List[torch.Tensor] = [out[0] for out in all_out]
        predictions: Dict[str, torch.Tensor] = {
            f"{name}_pred": out[1]
            for name, out in zip(self.loss_names, all_out)
        }
        targets: Dict[str, torch.Tensor] = {
            f"{name}_target": out[2]
            for name, out in zip(self.loss_names, all_out)
        }

        return sum(losses) / len(losses), predictions, targets  # type: ignore

    def compute_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        all_losses: List[torch.Tensor] = [
            loss.compute_loss(x_pred, x_true, mask, data)
            for loss in self.losses
        ]
        return sum(all_losses) / len(all_losses)  # type: ignore


class CEGaussianCombinedLoss(nn.Module):
    """Cross entropy loss for Sensor with NLL on remaining features."""

    def __init__(
        self,
        sensor_dim: int = 5484,  # Change later
        remaining_features: int = 4,
        latent_dim: int = 192,
        sensor_embedding_dim: int = 64,
        eps: float = 1e-4,
        transformer: Optional[nn.ModuleList] = None,
    ):
        """Initialize CEGaussianCombinedLoss.

        Args:
            sensor_dim: Number of sensors.
            remaining_features: Number of remaining features.
            latent_dim: Number of latent dimensions.
            eps: Epsilon to avoid numerical instability.
            n_transformer_layers: Number of transformer layers
                to use after revealing sensor id.
        """
        # [(338.44000244140625, 463.7200012207031, 233.38999938964844)]
        super().__init__()
        self.sensor_dim = sensor_dim
        self.remaining_features = remaining_features
        self.latent_dim = latent_dim
        self.eps = eps
        self.transformer = transformer

        self.ce_loss = nn.CrossEntropyLoss()
        self.emb = nn.Embedding(sensor_dim, sensor_embedding_dim)
        self.proj_to_sensor = nn.Linear(latent_dim, sensor_dim)
        self.proj_to_remaining = nn.Linear(
            latent_dim + sensor_embedding_dim, 2 * remaining_features
        )

    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ):
        """Forward pass of Auto-Encoder Loss.

        Args:
            x_latent: Latent representation of the data.
            x_true: True (scaled) values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Tuple of loss, predictions and targets.
        """
        sensor_id, padding_mask = to_dense_batch(data.sensor_id, data.batch)
        x_sensor_pred_logits = self.proj_to_sensor(x_latent)
        sensor_id = sensor_id.to(torch.int64)
        ce_loss = self.ce_loss(
            x_sensor_pred_logits[mask], sensor_id[mask]
        )  # Maybe use Quantum Efficiency of DOMs as weights?
        x_remaining_pred = torch.cat([x_latent, self.emb(sensor_id)], dim=-1)
        if self.transformer is not None:
            for blk in self.transformer:
                x_remaining_pred = blk(
                    x_remaining_pred, key_padding_mask=padding_mask
                )

        x_remaining_pred = self.proj_to_remaining(x_remaining_pred)

        x_remaining_pred_var = (
            torch.nn.functional.relu(
                x_remaining_pred[..., self.remaining_features :]
            )
            + self.eps
        )
        x_remaining_pred = x_remaining_pred[..., : self.remaining_features]

        x_remaining_pred_var_masked = x_remaining_pred_var[mask]
        x_remaining_pred_masked = x_remaining_pred[mask]

        x_true_masked = x_true[mask]
        x_true_masked = x_true_masked[:, 3:]  # Remove position

        # nll_loss = (x_remaining_pred_masked - x_true_masked).pow(2) / (2 * x_remaining_pred_var_masked.pow(2)) + torch.log(x_remaining_pred_var_masked)
        # nll_loss = torch.nn.functional.gaussian_nll_loss(x_true_masked, x_remaining_pred_masked, x_remaining_pred_var_masked, eps=self.eps, reduction='mean')
        # nll_loss = torch.nn.functional.smooth_l1_loss(x_remaining_pred_masked.to(torch.float32), x_true_masked, reduction='mean'))
        return (
            ce_loss,
            {
                "prediction_rem": x_remaining_pred,
                "prediction_rem_var": x_remaining_pred_var,
                "prediction_sensor": x_sensor_pred_logits,
            },
            {"target_rem": x_true, "target_sensor": sensor_id},
        )


class GaussianTimeAE(AELoss):
    """Loss function which uses Gaussian NLL on time."""
    _time_index = 3

    def __init__(self, latent_dim: int = 192, eps=1e-6):
        """Initialize GaussianTimeAE.

        Args:
            latent_dim: Number of latent dimensions.
            eps: Epsilon to avoid numerical instability.
        """
        super().__init__(latent_dim, output_dim=2)
        self.eps = eps

    def compute_loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        target = x_true[mask][:, self._time_index].to(torch.float32)
        mu = x_pred[mask][:, 0].to(torch.float32)
        log_sigma = x_pred[mask][:, 1].to(torch.float32)

        loss = (0.5 * (target - mu).pow(2) * torch.exp(- 2 * log_sigma) + log_sigma)
        return loss.mean()

    def forward(
        self,
        x_latent: torch.Tensor,
        x_true: torch.Tensor,
        mask: torch.Tensor,
        data: Data,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass of Auto-Encoder Loss.

                Args:
                    x_latent: Latent representation of the data.
                    x_true: True (scaled) values.
                    mask: Masking from the MAE. True means hidden
                        during the encoding and only those are used.
                    data: Data object, containing information of the batch.

                Returns: Tuple of loss, predictions and targets.
                """
        x_pred = self.proj(x_latent)
        return self.compute_loss(x_pred, x_true, mask, data), x_pred, x_true


class TimeAE(AELoss):
    """Loss function which uses Gaussian NLL on time."""
    _time_index = 3

    def __init__(self, latent_dim: int = 192):
        """Initialize GaussianTimeAE.

        Args:
            latent_dim: Number of latent dimensions.
            eps: Epsilon to avoid numerical instability.
        """
        super().__init__(latent_dim, output_dim=1)

    def compute_loss(
            self,
            x_pred: torch.Tensor,
            x_true: torch.Tensor,
            mask: torch.Tensor,
            data: Data,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            x_pred: Predictions.
            x_true: True values.
            mask: Masking from the MAE. True means hidden
                during the encoding and only those are used.
            data: Data object, containing information of the batch.

        Returns: Reduced loss.
        """
        target = x_true[mask][:, self._time_index].to(torch.float32)
        pred = x_pred[mask][:, 0].to(torch.float32)

        diff = pred - target
        loss = diff + nn.functional.softplus(-2.0 * diff)
        return loss.mean() / 10.0

    def forward(
            self,
            x_latent: torch.Tensor,
            x_true: torch.Tensor,
            mask: torch.Tensor,
            data: Data,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass of Auto-Encoder Loss.

                Args:
                    x_latent: Latent representation of the data.
                    x_true: True (scaled) values.
                    mask: Masking from the MAE. True means hidden
                        during the encoding and only those are used.
                    data: Data object, containing information of the batch.

                Returns: Tuple of loss, predictions and targets.
                """
        x_pred = self.proj(x_latent)
        return self.compute_loss(x_pred, x_true, mask, data), x_pred, x_true