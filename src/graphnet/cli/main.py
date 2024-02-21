"""GraphNeT CLI Entrypoint."""
from graphnet.cli.graphnet_cli import GraphnetCLI
from graphnet.models import LightweightModel
from graphnet.data import SQLiteDataModule
from graphnet.models import Model

from graphnet.models import *  # noqa: F401
from graphnet.models.gnn import *  # noqa: F401
from graphnet.models.graphs import *  # noqa: F401
from graphnet.models.graphs.nodes import *  # noqa: F401
from graphnet.models.graphs.edges import *  # noqa: F401
from graphnet.models.task import *  # noqa: F401
from graphnet.training.loss_functions import *  # noqa: F401


def main_cli() -> None:
    """CLI entrypoint of GraphNeT."""
    cli = GraphnetCLI(  # noqa
        LightweightModel,
        seed_everything_default=2023,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__:":
    main_cli()  # type ignore
