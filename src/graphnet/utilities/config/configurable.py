"""Bases for all configurable classes in  `graphnet`."""

from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Union

from graphnet.utilities.config.base_config import BaseConfig
from graphnet.utilities.decorators import final


class Configurable(ABC):
    """Base class for all configurable classes in graphnet."""

    def __init__(self) -> None:
        """Construct `Configurable`."""
        self._config: BaseConfig

        # Base class constructor
        super().__init__()

    @final
    @property
    def config(self) -> BaseConfig:
        """Return configuration to re-create the instance."""
        try:
            return self._config
        except AttributeError:
            raise AttributeError(
                "Config was not set. "
                "Did you wrap the class constructor with `save_config`?"
            )

    @final
    def save_config(self, path: str) -> None:
        """Save Config to `path` as YAML file."""
        self.config.dump(path)

    @classmethod
    @abstractmethod
    def from_config(cls, source: Union[BaseConfig, str]) -> Any:
        """Construct instance from `source` configuration."""

    @staticmethod
    def _unsafe_parse_string(s: str) -> Any:
        if not isinstance(s, str):
            raise ValueError("Can only parse strings")
        if s[0] == "!":
            return eval(s[1:])
