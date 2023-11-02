"""Callbacks to write results to disk."""
import os
from abc import abstractmethod, ABC
from itertools import chain
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

import pandas as pd
from lightning import Callback, Trainer
from torch_geometric.data import Batch


if TYPE_CHECKING:
    from graphnet.models.lightweight_model import LightweightModel  # type: ignore[attr-defined]


class BaseCacheValWriter(Callback, ABC):
    """Callback to write validation predictions to internal cache."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        results_folder: str = "results",
        output_file_prefix: str = "validation_result_",
        additional_attributes: Optional[List[str]] = None,
        prediction_labels: Optional[List[str]] = None,
        val_epoch_frequency: int = 1,
        overwrite_results_folder: bool = False,
    ):
        """Construct ´BaseCacheValWriter´ callback to store val results.

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

    def on_fit_start(
        self, trainer: Trainer, model: "LightweightModel"
    ) -> None:
        """Reset cache."""
        self._reset_cache(model)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        model: "LightweightModel",
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
        self, trainer: Trainer, model: "LightweightModel"
    ) -> None:
        """Write cache to parquet and reset."""
        if (
            trainer.current_epoch % self._val_epoch_frequency
            == self._val_epoch_frequency - 1
        ):
            self._write_cache_to_disk(
                os.path.join(
                    self._output_dir,
                    f"{self._output_file_prefix}epoch{trainer.current_epoch}{self.file_extension}",
                ),
                trainer,
                model,
            )
        self._reset_cache(model)

    def _reset_cache(self, model: "LightweightModel") -> None:
        if self._prediction_labels is None:
            self._prediction_labels = model.prediction_labels
        self._cache = {
            k: []
            for k in chain(
                self._additional_attributes, self._prediction_labels
            )
        }

    def _write_to_cache(
        self, outputs: Any, batch: Batch, model: "LightweightModel"
    ) -> None:
        outputs = outputs["preds"][0]
        for idx, pred_label in enumerate(chain(model.prediction_labels)):
            self._cache[pred_label].extend(outputs[:, idx].tolist())
        for idx, attribute in enumerate(self._additional_attributes):
            self._cache[attribute].extend(batch[attribute].tolist())

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return file extension of data."""
        pass

    @abstractmethod
    def _write_cache_to_disk(
        self, filename: str, trainer: Trainer, model: "LightweightModel"
    ) -> None:
        pass


class WriteValToParquet(BaseCacheValWriter):
    """Callback to write validation predictions parquet file."""

    @property
    def file_extension(self) -> str:
        """Return file extension of data."""
        return ".parquet"

    def _write_cache_to_disk(
        self, filename: str, trainer: Trainer, model: "LightweightModel"
    ) -> None:
        pd.DataFrame(self._cache).to_parquet(filename)


class WriteValToParquetWithPlot(WriteValToParquet):
    """Callback to write to parquet files and create plots."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        results_folder: str = "results",
        output_file_prefix: str = "validation_result",
        additional_attributes: Optional[List[str]] = None,
        prediction_labels: Optional[List[str]] = None,
        val_epoch_frequency: int = 1,
        overwrite_results_folder: bool = False,
        style: Optional[Union[List[str], dict, Path]] = None,
    ):
        """Construct ´WriteValToParquetWithPlot´ callback to store val results.

        Args:
            output_dir: Directory of parquet files.
            results_folder: Folder inside outdir to store results.
            output_file_prefix: Prefix in output_files.
            additional_attributes: Additional attributes to extract.
            prediction_labels: Labels to predict. If None extracts values from LightweightModel.
            val_epoch_frequency: Frequency of writing. Defaults to 1 i.e. every epoch is written.
            overwrite_results_folder: If true, might overwrite existing folder.
            style: Style to use for plots.
                See https://matplotlib.org/stable/api/style_api.html for details.
        """
        self._style = style
        super().__init__(
            output_dir=output_dir,
            results_folder=results_folder,
            output_file_prefix=output_file_prefix,
            additional_attributes=additional_attributes,
            prediction_labels=prediction_labels,
            val_epoch_frequency=val_epoch_frequency,
            overwrite_results_folder=overwrite_results_folder,
        )

    def _write_cache_to_disk(
        self, filename: str, trainer: Trainer, model: "LightweightModel"
    ) -> None:
        super()._write_cache_to_disk(filename, trainer, model)
        for task in model.tasks:
            task.plot(
                pd.DataFrame(self._cache),
                self._output_dir,
                trainer.current_epoch,
                self._style,
                self._output_file_prefix,
            )
