"""Class defining the GraphNeT CLI."""
from typing import Union, List

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser

from graphnet.models.gnn.aggregator import Aggregator
from graphnet.models.gnn.gnn import GNN, RawGNN, StandardGNN
from graphnet.models.graphs import GraphDefinition
from graphnet.models.task import Task


class GraphnetCLI(LightningCLI):
    """CLI class for GraphNeT."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """blah."""
        # Add components to expected args.
        parser.add_subclass_arguments(
            RawGNN,
            "gnn",
            help="The main feature-learner used for the model.",
            required=True,
        )
        parser.add_subclass_arguments(
            Aggregator,
            "aggregator",
            help="The aggregator component for the GNN",
            required=True,
        )
        parser.add_subclass_arguments(
            Task,
            "tasks",
            type=Union[List[Task], Task],
            help="Tasks",
            required=True,
        )
        parser.add_subclass_arguments(
            GraphDefinition,
            "graph_definition",
            help="GraphDefinition used for preprocessing.",
            required=True,
        )
        # Link components to correct modules.
        parser.link_arguments(
            "graph_definition", "data.graph_definition", apply_on="instantiate"
        )
        parser.link_arguments("gnn", "model.gnn", apply_on="instantiate")
        parser.link_arguments("aggregator", "model.aggregator", apply_on="instantiate")
        parser.link_arguments("tasks", "model.tasks", apply_on="instantiate")

        # Link input-output between components.
        parser.link_arguments(
            "graph_definition.nb_outputs",
            "gnn.init_args.nb_inputs",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "gnn.nb_outputs",
            "aggregator.init_args.nb_inputs",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "graph_definition.nb_outputs",
            "aggregator.init_args.nb_original_graph_features",
            apply_on="instantiate"
        )
        parser.link_arguments(
            "aggregator.nb_outputs",
            "tasks.init_args.hidden_size",
            apply_on="instantiate",
        )

