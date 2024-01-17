from typing import Union, Any, TYPE_CHECKING

from torchmetrics import Accuracy

from graphnet.utilities.config import ModelConfig
from graphnet.utilities.config.base_config import get_all_argument_values
from graphnet.utilities.config.parsing import traverse_and_apply

if TYPE_CHECKING:
    from graphnet.models import Model


def save_outside_model(cls):
    def _replace_model_instance_with_config(
            obj: Union["Model", Any]
    ) -> Union[ModelConfig, Any]:
        """Replace `Model` instances in `obj` with their `ModelConfig`."""
        from graphnet.models import Model

        if isinstance(obj, Model):
            return obj.config
        else:
            return obj
    class GN(cls):
        def __new__(cls_, *args, **kwargs):
            created_obj = super().__new__(cls_, *args, **kwargs)
            cfg = get_all_argument_values(created_obj.__init__, *args, **kwargs)
            cfg = traverse_and_apply(cfg, _replace_model_instance_with_config)

            # Store config in
            created_obj._config = ModelConfig(
                class_name=str(f"!external {cls.__module__} {cls.__name__}"),
                arguments=dict(**cfg),
            )
            return created_obj

    GN.__name__ = cls.__name__
    GN.__module__ = cls.__module__
    return GN


if __name__ == "__main__":
    from graphnet.models import Model
    AccCls = save_outside_model(Accuracy)
    a = AccCls(task="binary")
    print(a._config)
    b = Model.from_config(a._config, trust=True)
    print(b)