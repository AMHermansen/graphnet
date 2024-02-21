import os
from copy import deepcopy
from pathlib import Path
from typing import Union, List, Optional, Any, Tuple, TypeVar, Dict, Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data

from graphnet.models.detector import IceCube86
from graphnet.models.graphs import GraphDefinition, NodesAsPulses, KNNGraph


class ParquetSharded(Dataset):
    T = TypeVar("T")

    def _apply_selection(self, selection):
        if selection is not None:
            self._meta_info = self._meta_info.loc[selection]

    def _create_graph(
        self,
        features: np.ndarray,
        truth_dict: Dict[str, Any],
    ) -> Data:
        """Create Pytorch Data (i.e. graph) object.

        Args:
            features: List of tuples, containing event features.
            truth_dict: Dictionary containing truth information.

        Returns:
            Graph object.
        """
        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Create list of truth dicts with labels
        truth_dicts = [labels_dict, truth_dict]

        # Catch cases with no reconstructed pulses
        if len(features):
            node_features = np.asarray(features)
        else:
            node_features = np.array([]).reshape((0, len(self._features)))

        # Construct graph data object
        assert self._graph_definition is not None
        graph = self._graph_definition(
            input_features=node_features,
            input_feature_names=self._features,
            truth_dicts=truth_dicts,
            data_path=self._meta_path,
        )
        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        if "pid" in truth_dict.keys():
            abs_pid = abs(truth_dict["pid"])
            sim_type = truth_dict["sim_type"]

            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": int(abs_pid == 13),
                "muon_stopped": int(truth_dict.get("stopped_muon") == 1),
                "noise": int((abs_pid == 1) & (sim_type != "data")),
                "neutrino": int(
                    (abs_pid != 13) & (abs_pid != 1)
                ),  # @TODO: `abs_pid in [12,14,16]`?
                "v_e": int(abs_pid == 12),
                "v_u": int(abs_pid == 14),
                "v_t": int(abs_pid == 16),
                "track": int(
                    (abs_pid == 14) & (truth_dict["interaction_type"] == 1)
                ),
                "cascade": 1 ^ int(
                    (abs_pid == 14) & (truth_dict["interaction_type"] == 1)
                ),
                "dbang": self.get_dbang_label(truth_dict),
                "corsika": int(abs_pid > 20),
                "is_data": int(sim_type == "data"),
            }
        else:
            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": -1,
                "muon_stopped": -1,
                "noise": -1,
                "neutrino": -1,
                "v_e": -1,
                "v_u": -1,
                "v_t": -1,
                "track": -1,
                "cascade": -1,
                "dbang": -1,
                "corsika": -1,
            }
        return labels_dict

    def __init__(
            self,
            meta_path: Union[str, Path],
            pulsemap: str,
            truth: Optional[Union[List[str], str]],
            features: List[str],
            graph_definition: GraphDefinition,
            *,
            node_truth: Optional[List[str]] = None,
            index_column: str = "event_no",
            selection: Optional[List[int]] = None,
            pulsemap_path: Optional[Union[str, Path]] = None,
            node_feature_hook: Optional[Callable[[np.ndarray, int, "ParquetSharded"], np.ndarray]] = None,
    ):
        self._meta_path = meta_path
        self._pulsemap = pulsemap
        self._truth = ([truth] if isinstance(truth, str) else truth)
        self._features = features
        self._graph_definition = graph_definition
        self._node_feature_hook = node_feature_hook

        self._pulsemap_path = self.get_default_if_none(
            pulsemap_path,
            os.path.join(*meta_path.split("/")[:-1], pulsemap)
        )
        self._pulsemap_path = f"/{self._pulsemap_path}"

        self._node_truth = node_truth
        self._index_column = index_column
        self._selection = selection

        if (meta_path_file_type := meta_path.split(".")[-1]) == "csv":
            self._meta_info = pd.read_csv(meta_path)
        elif meta_path_file_type == "parquet":
            self._meta_info = pd.read_parquet(meta_path)
        else:
            raise ValueError(f"Unknown filetype for meta_path: {meta_path_file_type}")

        self._meta_info.set_index("event_no", drop=False, inplace=True)

        if self._truth is not None:
            self._meta_info = self._meta_info[self._truth]

        self._apply_selection(selection)

    def __len__(self):
        return len(self._meta_info)

    def __getitem__(self, idx):
        truth_dict = self.convert_df_row_to_dict(self._meta_info, idx)
        node_features = pd.read_parquet(
            os.path.join(
                self._pulsemap_path,
                f"event{(event_no := int(truth_dict[self._index_column]))}.parquet"
            )
        )[self._features].values

        node_features = self.node_hook(
            node_features,
            event_no,
        )

        graph = self._create_graph(
            node_features,
            truth_dict
        )
        return graph

    def node_hook(self, node_features: np.ndarray, event_no: int):
        if self._node_feature_hook is not None:
            return self._node_feature_hook(node_features, event_no, self)
        return node_features

    def spawn_subdataset(self, new_selection):
        new_dataset = deepcopy(self)
        new_dataset._apply_selection(new_selection)
        return new_dataset

    @staticmethod
    def get_default_if_none(value: Optional[T], default: T) -> T:
        return value if value is not None else default

    @staticmethod
    def get_dbang_label(truth_dict: Dict[str, Any]) -> int:
        """Get label for double-bang classification."""
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1

    @staticmethod
    def convert_df_row_to_dict(df, row_id):
        row = df.iloc[row_id]
        return {k: row[k] for k in df.columns}


if __name__ == "__main__":
    data = ParquetSharded(
        meta_path="/lustre/hpc/icecube/andreash/workspace/data_scripts/parquet_sharded/truth.parquet",
        pulsemap="SplitInIcePulses",
        truth=None,
        features=["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde", "pmt_area"],
        graph_definition=KNNGraph(IceCube86(), NodesAsPulses()),
    )
    print(len(data))
    train_data = data.spawn_subdataset(pd.read_csv(""))
