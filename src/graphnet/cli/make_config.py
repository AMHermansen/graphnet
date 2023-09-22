"""Script to make config templates."""
from graphnet.models import Model
from lightning.pytorch.cli import LightningCLI

from graphnet.models import *  # noqa: F401
from graphnet.models.gnn import *  # noqa: F401
from graphnet.models.graphs import *  # noqa: F401
from graphnet.models.graphs.nodes import *  # noqa: F401
from graphnet.models.graphs.edges import *  # noqa: F401
from graphnet.models.task import *  # noqa: F401


def config_cli() -> None:
    """CLI entrypoint of GraphNeT."""
    cli = LightningCLI(  # noqa
        Model,
        subclass_mode_model=True,
    )


if __name__ == "__main__:":
    config_cli()  # type ignore
