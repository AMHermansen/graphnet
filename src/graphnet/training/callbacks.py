"""Callback class(es) for using during model training."""

import logging
import os
from collections.abc import Sequence
from itertools import chain
from typing import Dict, List, Optional, Any, Union
import warnings

import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from tqdm.std import Bar

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, Callback
from lightning.pytorch.utilities import rank_zero_only
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from graphnet.models.lightweight_model import LightweightModel
from graphnet.utilities.logging import Logger


class PiecewiseLinearLR(LRScheduler):
    """Interpolate learning rate linearly between milestones."""

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        factors: List[float],
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Construct `PiecewiseLinearLR`.

        For each milestone, denoting a specified number of steps, a factor
        multiplying the base learning rate is specified. For steps between two
        milestones, the learning rate is interpolated linearly between the two
        closest milestones. For steps before the first milestone, the factor
        for the first milestone is used; vice versa for steps after the last
        milestone.

        Args:
            optimizer: Wrapped optimizer.
            milestones: List of step indices. Must be increasing.
            factors: List of multiplicative factors. Must be same length as
                `milestones`.
            last_epoch: The index of the last epoch.
            verbose: If ``True``, prints a message to stdout for each update.
        """
        # Check(s)
        if milestones != sorted(milestones):
            raise ValueError("Milestones must be increasing")
        if len(milestones) != len(factors):
            raise ValueError(
                "Only multiplicative factor must be specified for each milestone."
            )

        self.milestones = milestones
        self.factors = factors
        super().__init__(optimizer, last_epoch, verbose)

    def _get_factor(self) -> np.ndarray:
        # Linearly interpolate multiplicative factor between milestones.
        return np.interp(self.last_epoch, self.milestones, self.factors)

    def get_lr(self) -> List[float]:
        """Get effective learning rate(s) for each optimizer."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        return [base_lr * self._get_factor() for base_lr in self.base_lrs]


class ProgressBar(TQDMProgressBar):
    """Custom progress bar for graphnet.

    Customises the default progress in pytorch-lightning.
    """

    def _common_config(self, bar: Bar) -> Bar:
        bar.unit = " batch(es)"
        bar.colour = "green"
        return bar

    def init_validation_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_validation_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_predict_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_predict_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_test_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_test_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_train_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_train_tqdm()
        bar = self._common_config(bar)
        return bar

    def get_metrics(self, trainer: Trainer, model: LightningModule) -> Dict:
        """Override to not show the version number in the logging."""
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_epoch_start(
        self, trainer: Trainer, model: LightningModule
    ) -> None:
        """Print the results of the previous epoch on a separate line.

        This allows the user to see the losses/metrics for previous epochs
        while the current is training. The default behaviour in pytorch-
        lightning is to overwrite the progress bar from previous epochs.
        """
        if trainer.current_epoch > 0:
            self.train_progress_bar.set_postfix(
                self.get_metrics(trainer, model)
            )
            print("")
        super().on_train_epoch_start(trainer, model)
        self.train_progress_bar.set_description(
            f"Epoch {trainer.current_epoch:2d}"
        )

    def on_train_epoch_end(
        self, trainer: Trainer, model: LightningModule
    ) -> None:
        """Log the final progress bar for the epoch to file.

        Don't duplciate to stdout.
        """
        super().on_train_epoch_end(trainer, model)

        if rank_zero_only.rank == 0:
            # Construct Logger
            logger = Logger()

            # Log only to file, not stream
            h = logger.handlers[0]
            assert isinstance(h, logging.StreamHandler)
            level = h.level
            h.setLevel(logging.ERROR)
            logger.info(str(super().train_progress_bar))
            h.setLevel(level)


class WriteValToParquet(Callback):
    """Callback to write validation predictions to parquet."""

    _file_extension = ".parquet"

    def __init__(
        self,
        output_dir: str,
        output_file_prefix: str = "validation_result",
        additional_attributes: Optional[List[str]] = None,
        prediction_labels: Optional[List[str]] = None,
        val_epoch_frequency: int = 1,
    ):
        """Construct ´WriteValToParquet´ callback to store val results.

        Args:
            output_dir: Directory of parquet files.
            output_file_prefix: Prefix in output_files.
            additional_attributes: Additional attributes to extract.
            prediction_labels: Labels to predict. If None extracts values from LightweightModel.
            val_epoch_frequency: Frequency of writing. Defaults to 1 i.e. every epoch is written.
        """
        self._output_dir = output_dir
        self._output_file_prefix = output_file_prefix
        self._additional_attributes = additional_attributes or []
        self._prediction_labels = prediction_labels
        self._val_epoch_frequency = val_epoch_frequency
        self._cache: Dict[str, List] = {}

    def on_fit_start(self, trainer: Trainer, model: LightweightModel) -> None:
        """Reset cache."""
        self._reset_cache(model)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        model: LightweightModel,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Write results to cache if relevant epoch."""
        if (
            trainer.current_epoch % self._val_epoch_frequency
            == self._val_epoch_frequency - 1
        ):
            self._write_to_cache(outputs, batch, model)

    def on_validation_epoch_end(
        self, trainer: Trainer, model: LightweightModel
    ) -> None:
        """Write cache to parquet and reset."""
        if (
            trainer.current_epoch % self._val_epoch_frequency
            == self._val_epoch_frequency - 1
        ):
            self._write_cache_to_disk(
                os.path.join(
                    self._output_dir,
                    f"{self._output_file_prefix}_epoch{trainer.current_epoch}{self._file_extension}",
                )
            )
            self._reset_cache(model)

    def _reset_cache(self, model: LightweightModel) -> None:
        if self._prediction_labels is None:
            self._prediction_labels = model.prediction_labels
        self._cache = {
            k: []
            for k in chain(
                self._additional_attributes, self._prediction_labels
            )
        }

    def _write_to_cache(
        self, outputs: Any, batch: Batch, model: LightweightModel
    ) -> None:
        for idx, pred_label in enumerate(model.prediction_labels):
            self._cache[pred_label] += outputs[:, idx].tolist()
        for idx, attribute in enumerate(self._additional_attributes):
            self._cache[attribute].extend(batch[attribute])

    def _write_cache_to_disk(self, filename: str) -> None:
        df = pd.DataFrame(self._cache)
        df.to_parquet(filename)
