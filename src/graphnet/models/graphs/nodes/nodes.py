"""Class(es) for building/connecting graphs."""

from typing import List, Optional, Tuple, Union
from abc import abstractmethod

import torch
from torch_geometric.data import Data

from graphnet.utilities.decorators import final
from graphnet.models import Model
from graphnet.models.graphs.utils import (
    cluster_summarize_with_percentiles,
    identify_indices,
)
from copy import deepcopy


class NodeDefinition(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    def __init__(
        self, input_feature_names: Optional[List[str]] = None
    ) -> None:
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        if input_feature_names is not None:
            self.set_output_feature_names(
                input_feature_names=input_feature_names
            )

    @final
    def forward(
        self, x: torch.tensor, include_sensor_id: bool = False
    ) -> Tuple[Data, List[str]]:
        """Construct nodes from raw node features.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            node_feature_names: list of names for each column in ´x´.

        Returns:
            graph: a graph without edges
            new_features_name: List of new feature names.
        """
        graph = self._construct_nodes(x=x, include_sensor_id=include_sensor_id)
        try:
            self._output_feature_names
        except AttributeError as e:
            self.error(
                f"""{self.__class__.__name__} was instantiated without
                       `input_feature_names` and it was not set prior to this
                       forward call. If you are using this class outside a
                       `GraphDefinition`, please instatiate
                       with `input_feature_names`."""
            )  # noqa
            raise e
        return graph, self._output_feature_names

    @property
    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting
        classes.
        """
        return len(self._output_feature_names)

    @final
    def set_number_of_inputs(self, input_feature_names: List[str]) -> None:
        """Return number of inputs expected by node definition.

        Args:
            input_feature_names: name of each input feature column.
        """
        assert isinstance(input_feature_names, list)
        self.nb_inputs = len(input_feature_names)

    @final
    def set_output_feature_names(self, input_feature_names: List[str]) -> None:
        """Set output features names as a member variable.

        Args:
            input_feature_names: List of column names of the input to the
            node definition.
        """
        self._output_feature_names = self._define_output_feature_names(
            input_feature_names
        )

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Construct names of output columns.

        Args:
            input_feature_names: List of column names for the input data.

        Returns:
            A list of column names for each column in
            the node definition output.
        """
        return input_feature_names

    @abstractmethod
    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id: bool = False
    ) -> Tuple[Data, List[str]]:
        """Construct nodes from raw node features ´x´.

        Args:
            include_sensor_id:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            feature_names: List of names for reach column in `x`. Identical
            order of appearance. Length `d`.

        Returns:
            graph: graph without edges.
            new_node_features: A list of node features names.
        """

    def _maybe_split_sensor_id(
        self, features_input: torch.Tensor, add_sensor_id: bool
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        if add_sensor_id:
            sensor_id = features_input[:, -1]
            features_input = features_input[:, :-1]
        else:
            sensor_id = None
        return features_input, sensor_id

    def _maybe_add_sensor_id(
        self, data: Data, maybe_sensor_id: Optional[torch.Tensor] = None
    ) -> Data:
        if maybe_sensor_id is not None:
            data.sensor_id = maybe_sensor_id
        else:
            data.sensor_id = -torch.ones(data.x.shape[0])
        return data


class NodesAsPulses(NodeDefinition):
    """Represent each measured pulse of Cherenkov Radiation as a node."""

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names

    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id: bool = False
    ) -> Tuple[Data, List[str]]:
        x, maybe_sensor_id = self._maybe_split_sensor_id(x, include_sensor_id)
        data = Data(x=x)
        data = self._maybe_add_sensor_id(data, maybe_sensor_id)
        return data


class MAENodes(NodesAsPulses):
    """Choses nodes to mask for MAE training."""

    _xyz_idx = [0, 1, 2]
    _time_idx = [3]
    _charge_idx = [4]

    def __init__(
        self,
        max_length: int,
        masking_fraction: float,
        input_feature_names: Optional[List[str]] = None,
    ):
        """Construct MAENodes.

        Args:
            max_length: Maximum number of pulses to keep.
            masking_fraction: Fraction of pulses to mask.
            input_feature_names: (Optional) column names for input features.
        """
        self.max_length = max_length
        self.masking_fraction = masking_fraction
        super().__init__(input_feature_names)

    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id: bool = False
    ) -> Tuple[Data, List[str]]:
        # Crop pulses and get number of pulses to keep unmasked
        n_pulses = min(x.shape[0], self.max_length)
        n_keep = int(n_pulses * (1 - self.masking_fraction))
        if include_sensor_id:
            sensor_id = x[:, -1]
            x = x[:, :-1]

        # Construct masking indices
        noise = torch.rand(n_pulses, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=0).to(torch.int64)
        ids_restore = torch.argsort(ids_shuffle, dim=0).to(torch.int64)
        ids_keep = ids_shuffle[:n_keep]

        # construct data and add relevant data.
        data = Data(x=x[:n_pulses])
        if include_sensor_id:
            data.sensor_id = sensor_id[:n_pulses]
        data.ids_keep = ids_keep
        data.ids_restore = ids_restore
        data.n_keep = n_keep
        # self._add_charge_center(x, data)
        # self._add_time_duration(x, data)
        return data

    def _add_charge_center(self, x: torch.Tensor, data: Data) -> None:
        xyz = x[:, self._xyz_idx]
        charge = x[:, self._charge_idx]
        charge /= torch.sum(charge)
        charge_center = torch.mean(xyz * charge, dim=0)

        data.xq_mean = charge_center[0]
        data.yq_mean = charge_center[1]
        data.zq_mean = charge_center[2]

    def _add_time_duration(self, x: torch.Tensor, data: Data) -> None:
        time = x[:, self._time_idx]
        data.time_duration = torch.max(time) - torch.min(time)


class TimeShiftedNodes(NodeDefinition):
    def __init__(self, mu=0.2, sigma=0.2, input_feature_names=None):
        """Construct TimeShiftedNodes.

        Args:
            mu: Mean of the normal distribution to sample from.
            sigma: Standard deviation of the normal distribution to sample from.
            input_feature_names: (Optional) column names for input features.
        """
        self.mu = mu
        self.sigma = sigma
        super().__init__(input_feature_names)

    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id: bool = False
    ) -> Tuple[Data, List[str]]:
        # Sample from normal distribution
        perturbed = (torch.rand(1) > 0.5).to(torch.float32)
        noise = (
            torch.normal(
                self.mu, self.sigma, size=(x.shape[0], 1), device=x.device
            )
            * perturbed
        )
        # Add noise to time
        x[:, 3] += noise[:, 0]
        # Construct data
        data = Data(x=x)
        data.perturbed = perturbed
        return data


class PercentileClusters(NodeDefinition):
    """Represent nodes as clusters with percentile summary node features.

    If `cluster_on` is set to the xyz coordinates of DOMs
    e.g. `cluster_on = ['dom_x', 'dom_y', 'dom_z']`, each node will be a
    unique DOM and the pulse information (charge, time) is summarized using
    percentiles.
    """

    def __init__(
        self,
        cluster_on: List[str],
        percentiles: List[int],
        add_counts: bool = True,
        input_feature_names: Optional[List[str]] = None,
    ) -> None:
        """Construct `PercentileClusters`.

        Args:
            cluster_on: Names of features to create clusters from.
            percentiles: List of percentiles. E.g. `[10, 50, 90]`.
            add_counts: If True, number of duplicates is added to output array.
            input_feature_names: (Optional) column names for input features.
        """
        self._cluster_on = cluster_on
        self._percentiles = percentiles
        self._add_counts = add_counts
        # Base class constructor
        super().__init__(input_feature_names=input_feature_names)

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        (
            cluster_idx,
            summ_idx,
            new_feature_names,
        ) = self._get_indices_and_feature_names(
            input_feature_names, self._add_counts
        )
        self._cluster_indices = cluster_idx
        self._summarization_indices = summ_idx
        return new_feature_names

    def _get_indices_and_feature_names(
        self,
        feature_names: List[str],
        add_counts: bool,
    ) -> Tuple[List[int], List[int], List[str]]:
        cluster_idx, summ_idx, summ_names = identify_indices(
            feature_names, self._cluster_on
        )
        new_feature_names = deepcopy(self._cluster_on)
        for feature in summ_names:
            for pct in self._percentiles:
                new_feature_names.append(f"{feature}_pct{pct}")
        if add_counts:
            # add "counts" as the last feature
            new_feature_names.append("counts")
        return cluster_idx, summ_idx, new_feature_names

    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id: bool = False
    ) -> Data:
        # Cast to Numpy
        x = x.numpy()
        # Construct clusters with percentile-summarized features
        if hasattr(self, "_summarization_indices"):
            array = cluster_summarize_with_percentiles(
                x=x,
                summarization_indices=self._summarization_indices,
                cluster_indices=self._cluster_indices,
                percentiles=self._percentiles,
                add_counts=self._add_counts,
            )
        else:
            self.error(
                f"""{self.__class__.__name__} was not instatiated with
                `input_feature_names` and has not been set later.
                Please instantiate this class with `input_feature_names`
                if you're using it outside `GraphDefinition`."""
            )  # noqa
            raise AttributeError

        return Data(x=torch.tensor(array))
