"""Module containing clustering node definitions."""
from typing import List, Tuple, Optional

from torch_geometric.data import Data
from torch_geometric.nn.pool import knn

from graphnet.models.graphs.nodes import NodeDefinition
import numpy as np
import torch

from graphnet.models.graphs.nodes.utils import fps


class TimeClusteringNodes(NodeDefinition):
    def __init__(
        self,
        max_clusters: int,
        cluster_size: int,
    ):
        raise NotImplementedError(
            "Implementation of this NodeDefinition is not finished."
        )
        super().__init__()
        self._max_clusters = max_clusters
        self._cluster_size = cluster_size

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names

    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id=False
    ) -> Tuple[Data, List[str]]:
        x = x.numpy()
        l, f = x.shape
        surplus = l % self._cluster_size
        if surplus != 0:
            x = np.pad(
                x,
                ((0, self._cluster_size - surplus), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        x = x.reshape(-1, self._cluster_size, f)
        x = x[: self._max_clusters, ...]
        x = torch.from_numpy(x)
        return Data(x=x)  # , self._output_feature_names


class FPS_KNN_ClusterNodes(NodeDefinition):
    def __init__(
        self,
        n_clusters: int,
        cluster_size: int,
        knn_features: Optional[List[int]] = None,
        fps_features: Optional[List[int]] = None,
        start_idx: Optional[int] = None,
    ):
        super().__init__()
        self._number_of_clusters = n_clusters
        self._cluster_size = cluster_size
        self._knn_features = knn_features
        self._fps_features = fps_features
        self._start_idx = start_idx

    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id: bool = False
    ):
        arr = x.numpy()
        l, f = arr.shape
        knn_features = (
            self._knn_features
            if self._knn_features is not None
            else list(range(f))
        )
        fps_arr = (
            arr[:, self._fps_features]
            if self._fps_features is not None
            else arr
        )

        seeds = fps(fps_arr, self._number_of_clusters, self._start_idx)
        seeds = torch.from_numpy(arr[seeds])
        seeds_knn = seeds[:, knn_features]
        x_knn = x[:, knn_features]
        group_indices = knn(x_knn, seeds_knn, self._cluster_size)[1]
        groups = x[group_indices]
        groups = groups.reshape(-1, self._cluster_size, f)
        return Data(x=groups)


class TimeOrderedFPS_KNN_ClusterNodes(NodeDefinition):
    def __init__(
        self,
        n_clusters: int,
        cluster_size: int,
        knn_features: Optional[List[int]] = None,
        fps_features: Optional[List[int]] = None,
        start_idx: Optional[int] = None,
    ):
        super().__init__()
        self._number_of_clusters = n_clusters
        self._cluster_size = cluster_size
        self._knn_features = knn_features
        self._fps_features = fps_features
        self._start_idx = start_idx

    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id: bool = False
    ):
        arr = x.numpy()
        if include_sensor_id:
            arr = arr[:, :-1]
            sensor_ids = x[:, -1]
        l, f = arr.shape
        knn_features = (
            self._knn_features
            if self._knn_features is not None
            else list(range(f))
        )
        fps_arr = (
            arr[:, self._fps_features]
            if self._fps_features is not None
            else arr
        )

        seeds = fps(fps_arr, self._number_of_clusters, self._start_idx)
        seeds = torch.from_numpy(arr[seeds])
        seeds_knn = seeds[:, knn_features]
        x_knn = x[:, knn_features]
        group_indices = knn(x_knn, seeds_knn, self._cluster_size)[1]
        groups = x[group_indices]
        groups = groups.reshape(-1, self._cluster_size, f)
        graph = Data(x=groups)
        if include_sensor_id:
            sensor_ids = sensor_ids[group_indices]
            graph.sensor_id = sensor_ids
        return graph
