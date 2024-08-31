from typing import Any, List, Optional, Dict, Callable, Union
import torch
from torch_geometric.data import Data
from graphnet.models.graphs.graph_definition import GraphDefinition
from graphnet.models.detector import Detector
from graphnet.models import Model
import numpy as np


class DenseData(Model):
    def __init__(
            self, 
            detector: Detector, 
            dtype: Optional[torch.dtype] = None,
            max_pulses: int = 128,
    ) -> None:
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        dtype = dtype if dtype is not None else torch.float32

        self.to(dtype)
        self.detector = detector
        self._detector = detector
        self.geometry = detector.geometry_table

        self._geometry_string_column_name = detector.string_index_name
        self._geometry_dom_number_column_name = "dom_number"  # hardcoded for now
        self.max_pulses = max_pulses
        self.number_of_strings = self.geometry[self._geometry_string_column_name].nunique()
        self.number_of_doms = self.geometry[self._geometry_dom_number_column_name].nunique()
    
    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        truth_dicts: Optional[List[Dict[str, Any]]] = None,
        custom_label_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
        data_path: Optional[str] = None,
    ) -> Data:
        """Transform input features into a dense data representation."""

        counts_matrix = np.zeros((self.number_of_strings, self.number_of_doms), dtype=int)

        dense_data = torch.zeros((self.max_pulses, self.number_of_strings, self.number_of_doms, len(input_feature_names) + 1), dtype=self.dtype)
        
        for row in input_features:
            string, dom_number = self.geometry.loc[tuple(row[:3])][[self._geometry_string_column_name, self._geometry_dom_number_column_name]]
            string = int(string) - 1
            dom_number = int(dom_number) - 1
            counts_matrix[string, dom_number] += 1
            if counts_matrix[string, dom_number] <= self.max_pulses:
                dense_data[string, dom_number, counts_matrix[string, dom_number] - 1, :] = self._transform_row(row, input_feature_names)
        
        max_pulses = np.max(counts_matrix)
        graph = Data(x=dense_data[:max_pulses, ...])
        graph.n_pulses = input_features.shape[0]

        if data_path is not None:
            graph["dataset_path"] = data_path

        graph = self._add_loss_weights(
            graph=graph,
            loss_weight=loss_weight,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
        )

        if truth_dicts is not None:
            graph = self._add_truth(graph=graph, truth_dicts=truth_dicts)
        
        # Attach custom truth labels
        if custom_label_functions is not None:
            graph = self._add_custom_labels(
                graph=graph, custom_label_functions=custom_label_functions
            )
        graph["graph_definition"] = self.__class__.__name__
        return graph

        
    def _add_custom_labels(
        self,
        graph: Data,
        custom_label_functions: Dict[str, Callable[..., Any]],
    ) -> Data:
        # Add custom labels to the graph
        for key, fn in custom_label_functions.items():
            graph[key] = fn(graph)
        return graph


    def _transform_row(self, row: np.ndarray, input_feature_names) -> torch.Tensor:
        """Transform a row of input features into a torch tensor."""
        transformed_input_features = torch.from_numpy(row).to(self.dtype).view(1, -1)
        transformed_input_features = self.detector._standardize(transformed_input_features, input_feature_names).view(-1)
        transformed_input_features = torch.nn.functional.pad(
            transformed_input_features,
            (0, 1),
            "constant",
            1.,
        )
        return transformed_input_features

    def _add_loss_weights(
        self,
        graph: Data,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
    ) -> Data:
        """Attempt to store a loss weight in the graph for use during training.

        I.e. `graph[loss_weight_column] = loss_weight`

        Args:
            loss_weight: The non-negative weight to be stored.
            graph: Data object representing the event.
            loss_weight_column: The name under which the weight is stored in
                                 the graph.
            loss_weight_default_value: The default value used if
                                        none was retrieved.

        Returns:
            A graph with loss weight added, if available.
        """
        # Add loss weight to graph.
        if loss_weight is not None and loss_weight_column is not None:
            # No loss weight was retrieved, i.e., it is missing for the current
            # event.
            if loss_weight < 0:
                if loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{loss_weight_column} "
                        "but loss_weight_default_value is None."
                    )
                graph[loss_weight_column] = torch.tensor(
                    self._loss_weight_default_value, dtype=self.dtype
                ).reshape(-1, 1)
            else:
                graph[loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self.dtype
                ).reshape(-1, 1)
        return graph
    
    def _add_truth(
        self, graph: Data, truth_dicts: List[Dict[str, Any]]
    ) -> Data:
        """Add truth labels from ´truth_dicts´ to ´graph´.

        I.e. ´graph[key] = truth_dict[key]´


        Args:
            graph: graph where the label will be stored
            truth_dicts: dictionary containing the labels

        Returns:
            graph with labels
        """
        # Write attributes, either target labels, truth info or original
        # features.
        for truth_dict in truth_dicts:
            for key, value in truth_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to graph."
                        )
                    )
        return graph


# Cursed code to make DenseData objects pass the isinstance check in the graphnet.models.StandardModel class, 
# without inheriting from it, because GraphDefinition is a torch module, any subclass needs to call super().__init__ in it own constructor, which forces the GraphDefinition constructor to be called.
# The data construction pipeline in GraphDefinition doesn't allow any hooks at the moment.
# Crucially NodeDefinition does not have access to detector layout, which is needed to construct a dense data representation.
GraphDefinition.register(DenseData)
