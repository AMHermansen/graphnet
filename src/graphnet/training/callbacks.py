"""Callback class(es) for using during model training."""

import logging
import os
from itertools import chain
from typing import Dict, List, Optional, Any

import pandas as pd
from torch_geometric.data import Batch
from tqdm.std import Bar

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, Callback
from lightning.pytorch.utilities import rank_zero_only

from graphnet.models.lightweight_model import LightweightModel
from graphnet.utilities.logging import Logger


class GNProgressBar(TQDMProgressBar):
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

        This allows the user to see the losses/metrics for previous
        epochs while the current is training. The default behaviour in
        pytorch- lightning is to overwrite the progress bar from
        previous epochs.
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
        results_folder: str = "results",
        output_file_prefix: str = "validation_result",
        additional_attributes: Optional[List[str]] = None,
        prediction_labels: Optional[List[str]] = None,
        val_epoch_frequency: int = 1,
        overwrite_results_folder: bool = False,
    ):
        """Construct ´WriteValToParquet´ callback to store val results.

        Args:
            output_dir: Directory of parquet files.
            results_folder: Folder inside outdir to store results.
            output_file_prefix: Prefix in output_files.
            additional_attributes: Additional attributes to extract.
            prediction_labels: Labels to predict. If None extracts values from LightweightModel.
            val_epoch_frequency: Frequency of writing. Defaults to 1 i.e. every epoch is written.
            overwrite_results_folder: If true, might overwrite existing folder.
        """
        if results_folder:
            output_dir = os.path.join(output_dir, results_folder)
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=overwrite_results_folder)
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
        outputs = outputs["preds"][0]
        for idx, pred_label in enumerate(chain(model.prediction_labels)):
            # print(idx, pred_label, outputs)
            self._cache[pred_label].extend(outputs[:, idx].tolist())
        for idx, attribute in enumerate(self._additional_attributes):
            self._cache[attribute].extend(batch[attribute].tolist())

    def _write_cache_to_disk(self, filename: str) -> None:
        df = pd.DataFrame(self._cache)
        df.to_parquet(filename)
