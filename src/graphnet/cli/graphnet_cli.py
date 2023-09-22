"""Class defining the GraphNeT CLI."""
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser


class GraphnetCLI(LightningCLI):
    """CLI class for GraphNeT."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """blah."""
        parser.link_arguments(
            "data.graph_definition",
            "model.graph_definition",
        )
        parser.link_arguments(
            "data.graph_definition.nb_outputs",
            "model.gnn.init_args.nb_inputs",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "model.gnn.layer",
            "model.tasks.init_args.hidden_size",
            apply_on="parse",
        )
