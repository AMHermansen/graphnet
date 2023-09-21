"""GraphNeT CLI Entrypoint."""
from graphnet.cli.graphnet_cli import GraphnetCLI
from graphnet.models import LightweightModel
from graphnet.data import SQLiteDataModule
from graphnet.models import Model


def main_cli() -> None:
    """CLI entrypoint of GraphNeT."""
    cli = GraphnetCLI(  # noqa
        Model, SQLiteDataModule, subclass_mode_model=True
    )


if __name__ == "__main__:":
    main_cli()  # type ignore
