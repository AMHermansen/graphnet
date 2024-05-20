"""Base GNN-specific `Model` class(es)."""

from abc import abstractmethod
from typing import Optional, List, final

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from graphnet.models import Model
from graphnet.models.gnn.aggregator import Aggregator

class RawGNN(Model):
    """Base class for all core GNN models in graphnet."""

    def __init__(
        self
    ) -> None:
        """Construct `GNN`."""
        # Base class constructor
        super().__init__()

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of input features."""

    @property
    @abstractmethod
    def nb_outputs(self) -> int:
        """Return number of output features."""

    @abstractmethod
    def forward(self, data: Data) -> Data:
        """Apply learnable forward pass in model."""


class GNN(Model):
    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of input features."""

    @property
    @abstractmethod
    def nb_outputs(self) -> int:
        """Return number of output features."""

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass in model."""


@final
class StandardGNN(GNN):
    def __init__(self, gnn: RawGNN, aggregation: Aggregator):
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._gnn = gnn
        self._aggregation = aggregation

    def forward(self, data: Data) -> torch.Tensor:
        data = self._gnn(data)
        return self._aggregation(data)

    @property
    def nb_inputs(self) -> int:
        return self._gnn.nb_inputs

    @property
    def nb_outputs(self) -> int:
        return self._aggregation.nb_outputs


@final
class ValueAwareGNN(GNN):
    _initial_mlp_scale = 4

    @property
    def nb_inputs(self) -> int:
        return self._gnn.nb_inputs

    @property
    def nb_outputs(self) -> int:
        return self._aggregation.nb_outputs

    def __init__(self, gnn: RawGNN, aggregator: Aggregator, value_key):
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._gnn = gnn
        self._aggregation = aggregator
        self.value_key = value_key

        number_of_in_features = self._gnn.nb_inputs

        self._in_mlp = nn.Sequential(
            nn.Linear(number_of_in_features+1, (hidden_size := (number_of_in_features*self._initial_mlp_scale))),
            nn.GELU(),
            nn.Linear(hidden_size, number_of_in_features)
        )

    def forward(self, data: Data):
        x, mask = to_dense_batch(data.x, data.batch)
        data.orig_x = data.x

        value = data[self.value_key]
        value = value.unsqueeze(1).unsqueeze(1)
        value = value.repeat(1, x.shape[1], 1)

        x = torch.cat([x, value], dim=-1)
        data["x"] = self._in_mlp(x[mask])

        data = self._gnn(data)
        return self._aggregation(data)
