"""Datamodule class."""
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, asdict
from itertools import chain
from pathlib import Path
from typing import Union, List, Dict, Optional, Callable, Any, TypeVar, TYPE_CHECKING

import numpy as np
import pandas as pd
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from torch.utils.data import DataLoader

from graphnet.models.graphs import GraphDefinition
from graphnet.training.utils import (
    make_dataloader, collate_fn,
)
from graphnet.utilities.logging import Logger


if TYPE_CHECKING:
    from graphnet.data.dataset.sharded import ParquetSharded


def _convert_to_dict(input_value: Any, value_type: type) -> Dict[str, Any]:
    if isinstance(input_value, value_type):
        input_value = [input_value for _ in range(3)]
    if isinstance(input_value, Sequence):
        input_value = {
            k: v for k, v in zip(["train", "val", "test"], input_value)
        }
    if input_value is None:
        input_value = {k: None for k in ["train", "val", "test"]}
    if isinstance(input_value, dict):
        return input_value
    else:
        raise ValueError(
            f"Argument {input} has an unexpected type of {type(input)}"
        )


def _get_unique_keys(*dicts: Dict[Any, Any]) -> List[Any]:
    return list(set(chain.from_iterable(sub.keys() for sub in dicts)))


def _process_selection(sel: str) -> Union[List[int], str]:
    if (file_type := sel.split(".")[-1]) == "csv":
        return pd.read_csv(sel).reset_index(drop=True)['event_no'].ravel().tolist()
    elif file_type == "parquet":
        return pd.read_parquet(sel)["event_no"].tolist()
    else:
        return sel


class SQLiteDataModule(Logger, LightningDataModule):
    """SqliteDataModule for lightning integration."""

    def __init__(
        self,
        db: Union[str, List[str], Dict[str, str]],
        pulsemaps: Union[str, List[str]],
        graph_definition: GraphDefinition,
        features: List[str],
        truth: List[str],
        *,
        batch_size: Union[int, List[int], Dict[str, int]],
        train_shuffle: bool = True,
        selection: Optional[
            Union[
                List[str],
                List[List[int]],
                Dict[str, str],
                Dict[str, List[int]],
            ]
        ] = None,
        num_workers: Union[int, List[int], Dict[str, int]] = 10,
        persistent_workers: bool = True,
        node_truth: List[str] = None,
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: List[int] = None,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        index_column: str = "event_no",
        labels: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize a new instance of a SQLiteDataModule.

        Args:
            db: The database or data source for the operation.
            pulsemaps: The pulsemaps to be used in the operation.
            graph_definition: The definition of the graph structure to be used in the operation.
            features: A list of feature names to be used in the operation.
            truth: A list of truth names to be used in the operation.
            batch_size: The batch size for data processing.
            train_shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.
            selection (optional): Selection criteria for data.
            num_workers (optional): The number of worker processes for data loading. Defaults to 10.
            persistent_workers (bool, optional): Whether to use persistent workers for data loading. Defaults to True.
            node_truth (optional): A list of node truth names. Defaults to None.
            truth_table (str, optional): The name of the truth table in the database. Defaults to "truth".
            node_truth_table (optional): The name of the node truth table in the database. Defaults to None.
            string_selection (optional): A sequence of integers for string-based selection. Defaults to None.
            loss_weight_table (optional): The name of the loss weight table in the database. Defaults to None.
            loss_weight_column (optional): The name of the loss weight column in the loss weight table. Defaults to None.
            index_column (str, optional): The name of the index column in the database. Defaults to "event_no".
            labels (optional): A dictionary of label names and their corresponding functions for label processing. Defaults to None.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        if isinstance(selection, Sequence) and isinstance(selection[0], str):
            selection = {
                k: _process_selection(v)
                for k, v in zip(["train", "val", "test"], selection)
            }
        if isinstance(selection, Dict) and isinstance(selection["train"], str):
            selection = {
                k: _process_selection(v)
                for k, v in selection.items()
            }

        self._db = _convert_to_dict(db, str)
        self._pulsemaps = pulsemaps
        self._graph_definition = graph_definition
        self._features = features
        self._truth = truth
        self._batch_size = _convert_to_dict(batch_size, int)
        self._train_shuffle = train_shuffle
        self._num_workers = _convert_to_dict(num_workers, int)
        self._persistent_workers = persistent_workers
        self._node_truth = node_truth
        self._truth_table = truth_table
        self._node_truth_table = node_truth_table
        self._string_selection = string_selection
        self._loss_weight_table = loss_weight_table
        self._loss_weight_column = loss_weight_column
        self._index_column = index_column
        self._labels = labels

        self._common_kwargs = dict(
            pulsemaps=self._pulsemaps,
            graph_definition=self._graph_definition,
            features=self._features,
            truth=self._truth,
            persistent_workers=self._persistent_workers,
            node_truth=self._node_truth,
            truth_table=self._truth_table,
            node_truth_table=self._node_truth_table,
            string_selection=self._string_selection,
            loss_weight_table=self._loss_weight_table,
            loss_weight_column=self._loss_weight_column,
            index_column=self._index_column,
            labels=self._labels,
        )

        self._selection = selection

        self._all_keys = _get_unique_keys(
            self._batch_size, self._selection, self._num_workers, self._db  # type: ignore
        )

        assert 2 <= len(self._all_keys) <= 3
        assert all(key in ["test", "val", "train"] for key in self._all_keys)
        assert "train" in self._all_keys
        assert "val" in self._all_keys

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Create train dataloader."""
        key = "train"
        return make_dataloader(
            db=self._db[key],
            batch_size=self._batch_size[key],
            selection=self._selection[key],  # type: ignore
            num_workers=self._num_workers[key],
            shuffle=self._train_shuffle,
            **self._common_kwargs,  # type: ignore
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Create val dataloader."""
        key = "val"
        return make_dataloader(
            db=self._db[key],
            batch_size=self._batch_size[key],
            selection=self._selection[key],  # type: ignore
            num_workers=self._num_workers[key],
            shuffle=False,
            **self._common_kwargs,  # type: ignore
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Create test dataloader."""
        key = "test"
        return make_dataloader(
            db=self._db[key],
            batch_size=self._batch_size[key],
            selection=self._selection[key],  # type: ignore
            num_workers=self._num_workers[key],
            shuffle=False,
            **self._common_kwargs,  # type: ignore
        )

    predict_dataloader = test_dataloader


@dataclass
class ShardedDatasetConfig:
    meta_path: Union[str, Path]
    pulsemap: str
    features: List[str]
    truth: Optional[Union[List[str], str]] = None
    node_truth: Optional[List[str]] = None
    index_column: str = "event_no"
    pulsemap_path: Optional[Union[str, Path]] = None
    selection: Optional[List[int]] = None
    node_feature_hook: Optional[
        Callable[[np.ndarray, int, "ParquetSharded"], np.ndarray]] = None,


class ShardedDataModule(Logger, LightningDataModule):
    T = TypeVar("T")

    def __init__(
            self,
            graph_definition: GraphDefinition,
            dataset_config: ShardedDatasetConfig,
            selections: Dict[str, Union[str, List[int]]],
            train_loader_config: Optional[Dict[str, Any]] = None,
            val_loader_config: Optional[Dict[str, Any]] = None,
            test_loader_config: Optional[Dict[str, Any]] = None,
            predict_loader_config: Optional[Dict[str, Any]] = None,
            common_loader_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        train_loader_config = self._default_if_none(train_loader_config, {})
        val_loader_config = self._default_if_none(val_loader_config, {})
        test_loader_config = self._default_if_none(test_loader_config, {})
        predict_loader_config = self._default_if_none(predict_loader_config, {})
        common_loader_config = self._default_if_none(common_loader_config, {})

        self._validate_compatible_selection_argument(selections)
        self.selections = selections

        self._common_dataset_config = dataset_config
        from graphnet.data.dataset.sharded.sharded import ParquetSharded
        self.base_dataset = ParquetSharded(
            graph_definition=graph_definition,
            **asdict(self._common_dataset_config)
        )

        self._train_loader_config = self._add_loader_dicts(common_loader_config, train_loader_config)
        self._val_loader_config = self._add_loader_dicts(common_loader_config, val_loader_config)
        self._test_loader_config = self._add_loader_dicts(common_loader_config, test_loader_config)
        self._predict_loader_config = self._add_loader_dicts(common_loader_config, predict_loader_config)

    def train_dataloader(self):
        return DataLoader(
            self.base_dataset.spawn_subdataset(self.selections["train"]),
            **self._train_loader_config
        )

    def val_dataloader(self):
        return DataLoader(
            self.base_dataset.spawn_subdataset(self.selections["val"]),
            **self._val_loader_config
        )

    def test_dataloader(self):
        return DataLoader(
            self.base_dataset.spawn_subdataset(self.selections["test"]),
            **self._test_loader_config
        )

    def predict_dataloader(self):
        return DataLoader(
            self.base_dataset.spawn_subdataset(self.selections["predict"]),
            **self._predict_loader_config
        )

    @staticmethod
    def _add_loader_dicts(dict1, dict2):
        d = deepcopy(dict1)
        for k, v in dict2.items():
            d[k] = v
        d["collate_fn"] = collate_fn
        return d

    @staticmethod
    def _default_if_none(value: Optional[T], default: T) -> T:
        return value if value is not None else default

    def _validate_compatible_selection_argument(self, selections):
        for key in selections:
            if key not in ["train", "val", "test", "predict"]:
                raise ValueError(f"Unexpected key in selection dictionary: {key}")
        for key, value in selections.items():
            if isinstance(value, str):
                value = _process_selection(value)
            if isinstance(value, str):
                raise ValueError(f"Unexpected string value in selection Dictionary:\n"
                                 f"{key=}: {value=}")

            assert(isinstance(value, list) and isinstance(value[0], int))
            selections[key] = value

        if "predict" not in selections and "test" in selections:
            self.info("No prediction selections found, using test selections for predictions")
            selections["predict"] = selections["test"]
