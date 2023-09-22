"""Node definitions that crop the number of pulses in a pulsemap."""

from torch_geometric.data import Data
import torch
from torch import nn
from typing import Callable, Any

from .nodes import NodeDefinition
from graphnet.utilities.config import save_model_config


class PulsesCroppedValue(NodeDefinition):
    """Represent each node as pulse with an upper limit of nodes."""

    @property
    def nb_outputs(self) -> int:
        """Return number of outputs."""
        return self.nb_inputs

    def __init__(
        self,
        max_pulses: int,
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Construct `PulsesCroppedByCharge`.".

        Selects at most 'max_pulses' number of pulses, chosen as those with the smallest
         value after applying transform.

        Args:
            max_pulses: Maximal number of pulses allowed in a pulsemap.
             transform: Transform applied to the input tensor, before determining order.
            **kwargs: kwargs passed to NodeDefinitions constructor.
        """
        transform = transform or nn.Identity()
        super().__init__(**kwargs)  # noqa
        self.max_pulses = max_pulses
        self.transform = transform

    # abstract method(s)
    def _construct_nodes(self, x: torch.tensor) -> Data:
        if x.shape[0] < self.max_pulses:
            return Data(x=x)
        return Data(
            x=x[
                torch.sort(self.transform(x), dim=0).indices,
                :,
            ]
        )


class PulsesCroppedRandomly(NodeDefinition):
    """Represent each node as pulse with an upper limit of nodes."""

    @property
    def nb_outputs(self) -> int:
        """Return number of outputs."""
        return self.nb_inputs

    def __init__(self, max_pulses: int, **kwargs: Any):
        """Construct 'PulsesCroppedRandomly'.

        Selects at most 'max_pulses' number of pulses, chosen randomly.

        Args:
            max_pulses: Maximal number of pulses allowed in the pulsemap.
            **kwargs: kwargs passed to NodeDefinitions constructor.
        """
        super().__init__(**kwargs)  # noqa
        self.max_pulses = max_pulses

    def _construct_nodes(self, x: torch.tensor) -> Data:
        if x.shape[0] < self.max_pulses:
            return Data(x=x)
        return Data(x=x[torch.randperm(x.shape[0])[: self.max_pulses]])
