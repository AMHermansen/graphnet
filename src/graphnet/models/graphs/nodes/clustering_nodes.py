"""Module containing clustering node definitions."""
from typing import List, Tuple

from torch_geometric.data import Data

from .nodes import NodeDefinition
import numpy as np
import torch


class TimeClusteringNodes(NodeDefinition):
    def __init__(
            self,
            max_clusters: int,
            cluster_size: int,
    ):
        super().__init__()
        self._max_clusters = max_clusters
        self._cluster_size = cluster_size

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:
        x = x.numpy()
        l, f = x.shape
        surplus = l % self._cluster_size
        if surplus != 0:
            x = np.pad(x, ((0, self._cluster_size - surplus), (0, 0)), mode='constant', constant_values=0)
        x = x.reshape(-1, self._cluster_size, f)
        x = x[:self._max_clusters, ...]
        x = torch.from_numpy(x)
        return Data(x=x)  # , self._output_feature_names
