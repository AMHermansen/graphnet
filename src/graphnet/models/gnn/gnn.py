"""Base GNN-specific `Model` class(es)."""

from abc import abstractmethod
from typing import Optional, List

from torch import Tensor
from torch_geometric.data import Data

from graphnet.models import Model


class GNN(Model):
    """Base class for all core GNN models in graphnet."""

    def __init__(
        self, nb_inputs: int, readout_layer_sizes: Optional[List[int]] = None
    ) -> None:
        """Construct `GNN`."""
        # Base class constructor
        super().__init__()

        # Member variables
        self._nb_inputs = nb_inputs
        self._readout_layer_sizes = readout_layer_sizes or [256, 128]
        self._nb_outputs = self._readout_layer_sizes[-1]

    @property
    def nb_inputs(self) -> int:
        """Return number of input features."""
        return self._nb_inputs

    @property
    def nb_outputs(self) -> int:
        """Return number of output features."""
        return self._nb_outputs

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass in model."""
