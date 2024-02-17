"""Implementation of DynEdge architecture used in.

                    IceCube - Neutrinos in Deep Ice
Reconstruct the direction of neutrinos from the Universe to the South Pole
Kaggle competition.

Solution by TITO.
"""
from itertools import chain
from typing import List, Tuple, Optional, Union

import torch
from torch import Tensor, LongTensor

from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from graphnet.models.components.layers import DynTrans
from graphnet.models.gnn.gnn import RawGNN
from graphnet.models.utils import calculate_xyzt_homophily

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class DynEdgeTITO(RawGNN):
    """DynEdgeTITO (dynamical edge convolutional with Transformer) model."""

    def __init__(
        self,
        nb_inputs: int,
        features_subset: List[int] = None,
        dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        graph_mlp_layers: Optional[List[int]] = None,
    ):
        """Construct `DynEdgeTITO`.

        Args:
            nb_inputs: Number of input features on each node.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0, 1, 2, 3].
            dyntrans_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynTrans` layer.
                Defaults to [(256, 256), (256, 256), (256, 256), (256, 256)].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            use_global_features: Whether to use global features after pooling.
            use_post_processing_layers: Whether to use post-processing layers
                after the `DynTrans` layers.
        """
        # DynTrans layer sizes
        if dyntrans_layer_sizes is None:
            dyntrans_layer_sizes = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

        assert isinstance(dyntrans_layer_sizes, list)
        assert len(dyntrans_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dyntrans_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dyntrans_layer_sizes)
        assert all(
            all(size > 0 for size in sizes) for sizes in dyntrans_layer_sizes
        )

        self._dyntrans_layer_sizes = dyntrans_layer_sizes

       # Base class constructor
        super().__init__()

        # Remaining member variables()
        self._graph_mlp_layers = self.set_default_if_none(graph_mlp_layers, [])
        self._activation = torch.nn.LeakyReLU()
        self._nb_inputs = nb_inputs
        self._features_subset = features_subset or [0, 1, 2, 3]
        self._construct_layers()

    @property
    def nb_inputs(self) -> int:
        return self._nb_inputs

    @property
    def nb_outputs(self) -> int:
        return self._nb_outputs

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dyntrans_layer_sizes:
            conv_layer = DynTrans(
                [nb_latent_features] + list(sizes),
                aggr="mean",
                features_subset=self._features_subset,
                n_head=8,
            )
            self._conv_layers.append(conv_layer)
            nb_latent_features = sizes[-1]
        graph_mlp_layers = [nb_latent_features] + self._graph_mlp_layers
        self._graph_mlp = torch.nn.Sequential(
            *chain(
                *[
                    [
                        torch.nn.Linear(nb_in, nb_out),
                        torch.nn.GELU(),
                    ]
                    for nb_in, nb_out in zip(graph_mlp_layers[:-1], graph_mlp_layers[1:])
                ]
            )
        )
        self._nb_outputs = graph_mlp_layers[-1]

    def forward(self, data: Data) -> Data:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # DynEdge-convolutions
        for conv_layer in self._conv_layers:
            x = conv_layer(x, edge_index, batch)
        x = self._graph_mlp(x)
        data.encoded_x = x
        return data
