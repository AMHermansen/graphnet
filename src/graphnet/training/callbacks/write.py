"""Callbacks to write results to disk."""
import os
import warnings
from abc import abstractmethod, ABC
from itertools import chain
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from icecream import ic
from lightning import Callback, Trainer, LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from torch_geometric.data import Batch, Data
import torch_geometric


if TYPE_CHECKING:
    from graphnet.models.lightweight_model import LightweightModel, Model  # type: ignore[attr-defined]


class BaseCacheWriter(Callback, ABC):
    """Callback to write validation predictions to internal cache."""

    def __init__(
        self,
        stage: Optional[str] = "validation",
        output_dir: Optional[Union[str, Path]] = None,
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
        self.stage = stage
        self._overwrite_results_folder = overwrite_results_folder
        self._output_dir = output_dir
        self._results_folder = results_folder
        self._output_file_prefix = output_file_prefix
        self._additional_attributes = additional_attributes or []
        self._prediction_labels = prediction_labels
        self._val_epoch_frequency = val_epoch_frequency
        self._cache: Dict[str, List] = {}

        if self.stage not in ["validation", "predict"]:
            raise ValueError(
                f"stage must be either 'validation' or 'predict', got {self.stage}"
            )

    def setup(self, trainer: Trainer, model: "LightweightModel", stage: str) -> None:
        if self._output_dir is None:
            self._output_dir = trainer.default_root_dir
        if self._results_folder:
            self._output_dir = os.path.join(self._output_dir, self._results_folder)
        os.makedirs(self._output_dir, exist_ok=self._overwrite_results_folder)
        self._reset_cache(model)

    def on_fit_start(
        self, trainer: Trainer, model: "LightweightModel"
    ) -> None:
        """Reset cache."""
        if self.stage == "validation":
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
        if self.stage != "validation":
            return
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
        ) and self.stage == "validation":
            self._write_cache_to_disk(
                os.path.join(
                    self._output_dir,
                    f"{self._output_file_prefix}epoch{trainer.current_epoch}{self.file_extension}",
                ),
                trainer,
                model,
            )
        self._reset_cache(model)

    def on_predict_batch_end(
        self,
        trainer: "Trainer",
        model: "LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.stage != "predict":
            return
        self._write_to_cache(outputs, batch, model)

    def on_predict_epoch_end(
        self, trainer: "Trainer", model: "LightweightModel",
    ):
        if self.stage != "predict":
            return
        self._write_cache_to_disk(
            os.path.join(
                self._output_dir,
                f"{self._output_file_prefix}"
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
        outputs = outputs["preds"][0]  # might need to only select [0] for some models
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


class WriteToParquet(BaseCacheWriter):
    """Callback to write validation predictions parquet file."""

    def __init__(
            self,
            stage: Optional[str] = "validation",
            output_dir: Optional[Union[str, Path]] = None,
            results_folder: str = "results",
            output_file_prefix: str = "validation_result_",
            additional_attributes: Optional[List[str]] = None,
            prediction_labels: Optional[List[str]] = None,
            val_epoch_frequency: int = 1,
            overwrite_results_folder: bool = False,
    ):
        super().__init__(
            stage=stage,
            output_dir=output_dir,
            results_folder=results_folder,
            output_file_prefix=output_file_prefix,
            additional_attributes=additional_attributes,
            prediction_labels=prediction_labels,
            val_epoch_frequency=val_epoch_frequency,
            overwrite_results_folder=overwrite_results_folder,
        )

    @property
    def file_extension(self) -> str:
        """Return file extension of data."""
        return ".parquet"

    def _write_cache_to_disk(
            self, filename: str, trainer: Trainer, model: "LightweightModel"
    ) -> None:
        pd.DataFrame(self._cache).to_parquet(filename)


class WriteValToParquet(WriteToParquet):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "WriteValToParquet is deprecated, use WriteToParquet instead",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


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


class MAEWriteCB(BasePredictionWriter):
    """Callback to write results from Masked AutoEncoder to disk."""

    def __init__(self, output_dir: str):
        """Initialize callback.

        Args:
            output_dir: Directory to write results to.
        """
        super().__init__("batch")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._cache: Dict[str, List] = {}

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Dict[str, Any],
        batch_indices: Any,  # Don't know what this is...
        batch: Data,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Write results to disk after each batch.

        Args:
            trainer: Trainer object.
            pl_module: Lightning Module.
            prediction: Predictions from Lightning Module, expects dictionary containing "mask", "x_pred", "cls_pred", "cls_target".
            batch_indices: No idea.
            batch: The data the model received.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.

        Returns: None.
        """
        ae_mask = prediction["mask"].detach().cpu().numpy()
        x_pred = prediction["x_pred"]
        x_t1 = prediction["x_true"]

        for k, v in x_pred.items():
            x_pred[k] = v.to(torch.float32).detach().cpu().numpy()
        for k, v in x_t1.items():
            x_t1[k] = v.to(torch.float32).detach().cpu().numpy()
        for k, v in prediction["cls_pred"].items():
            prediction["cls_pred"][k] = (
                v.to(torch.float32).detach().cpu().numpy()
            )
        for k, v in prediction["cls_target"].items():
            prediction["cls_target"][k] = (
                v.to(torch.float32).detach().cpu().numpy()
            )

        x_true, padding_mask = torch_geometric.utils.to_dense_batch(
            batch.x, batch.batch
        )
        x_true = x_true.detach().cpu().numpy()
        padding_mask = padding_mask.detach().cpu().numpy()

        os.makedirs(f"{self.output_dir}/batch_{batch_idx}", exist_ok=True)
        np.save(f"{self.output_dir}/batch_{batch_idx}/ae_mask.npy", ae_mask)
        np.save(f"{self.output_dir}/batch_{batch_idx}/x_true.npy", x_true)
        np.save(
            f"{self.output_dir}/batch_{batch_idx}/padding_mask.npy",
            padding_mask,
        )
        self._save_dict(
            prediction["cls_pred"],
            f"{self.output_dir}/batch_{batch_idx}/cls_pred",
        )
        self._save_dict(
            prediction["cls_target"],
            f"{self.output_dir}/batch_{batch_idx}/cls_target",
        )
        self._save_dict(x_pred, f"{self.output_dir}/batch_{batch_idx}/x_pred")
        self._save_dict(x_t1, f"{self.output_dir}/batch_{batch_idx}/x_true1")

    @staticmethod
    def _save_dict(dict, path):
        for k, v in dict.items():
            np.save(f"{path}_{k}.npy", v)


class WriteBatchToNumpy(BasePredictionWriter):
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize callback.

        Args:
            output_dir: Directory to write results to.
        """
        super().__init__("batch")
        self.output_dir = output_dir

    def setup(self, trainer: "pl.Trainer", model: "Model", stage: str) -> None:
        if not self.output_dir:
            self.output_dir = trainer.default_root_dir
        os.makedirs(self.output_dir, exist_ok=True)
        model.info(f"Saving outputs to {self.output_dir}/results/")

    def _write(self, result_dict: Dict[str, Union[Dict, torch.Tensor]], batch_no: int, model: "Model", path: str):
        os.makedirs(path, exist_ok=True)
        for key, value in result_dict.items():
            if not isinstance(key, str):
                model.warning_once(f"Found key: {key} in results dict. Corresponding values not saved")
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().to(torch.float32).numpy()
                np.save(f"{path}/{key}.npy", value)
            elif isinstance(value, dict):
                self._write(result_dict=value, batch_no=batch_no, model=model, path=f"{path}/key")
            else:
                # Shouldn't reach this
                model.warning_once(f"Unknown value for {key = }, {type(value) = }")

    def write_on_batch_end(
            self,
            trainer: Trainer,
            model: "Model",
            prediction: Dict[str, Any],
            batch_indices: Any,  # Don't know what this is...
            batch: Data,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        # Initial recursive call.
        path = f"{self.output_dir}/results/batch{batch_idx}"
        self._write(result_dict=prediction, batch_no=batch_idx, model=model, path=path)
        x_true, padding_mask = torch_geometric.utils.to_dense_batch(
            batch.x, batch.batch
        )
        x_true = x_true.detach().cpu().numpy()
        padding_mask = padding_mask.detach().cpu().numpy()
        np.save(f"{path}/x_true.npy", x_true)
        np.save(f"{path}/padding_mask.npy", padding_mask)
