"""Common decorators."""
import warnings
from functools import wraps
from typing import Any, Dict


try:
    from typing import final
except ImportError:  # Python version < 3.8

    # Identity decorator
    def final(f):  # type: ignore  # noqa: D103
        return f


def deprecate_kwarg(kwarg_name_mapping: Dict[str, Any]):
    """

    Args:
        kwarg_name_mapping: Mapping old name to new name.

    Returns:

    """
    def wrapper_func(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = {}
            for key in kwargs:
                if key in kwarg_name_mapping:
                    warnings.warn(
                        f"Deprecated kwarg name received: '{key}'. "
                        f"Please use '{kwarg_name_mapping[key]}' going forward.",
                        DeprecationWarning
                    )
                    new_kwargs[kwarg_name_mapping[key]] = kwargs[key]
                else:
                    new_kwargs[key] = kwargs[key]
            return func(*args, **new_kwargs)
        return wrapper
    return wrapper_func
