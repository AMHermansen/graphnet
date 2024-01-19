from abc import abstractmethod
from itertools import chain
from typing import List, Optional

import torch
from torch import Tensor, LongTensor, nn
from torch_geometric.data import Data
from torch_scatter import scatter_min, scatter_max, scatter_sum, scatter_mean

from graphnet.models.model import Model
from graphnet.models.utils import calculate_xyzt_homophily

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class Aggregation(Model):

    @property
    @abstractmethod
    def nb_inputs(self):
        pass

    @property
    @abstractmethod
    def nb_outputs(self):
        pass

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        pass


class StandardPooling(Aggregation):
    def __init__(
            self,
            nb_inputs: int,
            nb_original_graph_features: int,
            global_pooling_schemes: Optional[List[str]] = None,
            use_global_features: bool = True,
            readout_mlp_layers: Optional[List[int]] = None,
    ):
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        global_pooling_schemes = ["max"] if global_pooling_schemes is None else global_pooling_schemes

        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                        pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._nb_inputs = nb_inputs
        self._nb_original_features = nb_original_graph_features
        self._global_pooling_schemes = global_pooling_schemes
        self._use_global_features = use_global_features
        initial_output = nb_inputs * len(self._global_pooling_schemes) + (
            5 + self._nb_original_features if self._use_global_features else 0
        )
        self._readout_mlp_layers = [initial_output] + self.set_default_if_none(readout_mlp_layers, [])

        self._readout_mlp = nn.Sequential(
            *[
                x for x in chain(
                    *[
                        (nn.Linear(l1, l2), nn.GELU())
                        for l1, l2 in zip(self._readout_mlp_layers[:-1], self._readout_mlp_layers[1:])
                    ]
                )
            ]
        )

    @property
    def nb_inputs(self):
        return self._nb_inputs

    @property
    def nb_outputs(self):
        return self._readout_mlp_layers[-1]

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    @staticmethod
    def _calculate_global_variables(
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, data: Data) -> torch.Tensor:
        global_variables = self._calculate_global_variables(
            data.x,
            data.edge_index,
            data.batch,
            torch.log10(data.n_pulses),
        )
        x = data.encoded_x
        batch = data.batch

        x = self._global_pooling(x, batch=batch)
        if self._use_global_features:
            x = torch.cat([x, global_variables], dim=1)
        return self._readout_mlp(x)
