"""Base physics task-specific `Model` class(es)."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING, List, Tuple, Union, Callable, Optional, Dict, \
    Type
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.nn import Linear, ModuleDict
from torch_geometric.data import Data
from torchmetrics import Metric

if TYPE_CHECKING:
    # Avoid cyclic dependency
    from graphnet.training.loss_functions import LossFunction  # type: ignore[attr-defined]

from graphnet.models import Model
from graphnet.utilities.decorators import final


class Task(Model):
    """Base class for all reconstruction and classification tasks."""

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""

    @property
    @abstractmethod
    def default_target_labels(self) -> List[str]:
        """Return default target labels."""
        return self._default_target_labels

    @property
    @abstractmethod
    def default_prediction_labels(self) -> List[str]:
        """Return default prediction labels."""
        return self._default_prediction_labels

    @property
    def default_metrics(self) -> Dict[str, Tuple[Union[Type[Metric], Callable[[], Metric]], bool]]:
        return {}

    def plot(
        self,
        cache: pd.DataFrame,
        output_dir: str,
        epoch: int,
        style: Optional[Union[List[str], dict, Path]] = None,
        output_prefix: Optional[str] = "",
    ) -> bool:
        """Create a plot of the model output and saves it to the output_dir.

        Args:
            cache: Cache of results from the model.
            output_dir: Directory to save the plot to.
            epoch: The current epoch, used for naming the plot.
            style: Path to a preferred stylesheet for the plot.
            output_prefix: Prefix to add to the plot name.

        Returns: True if the plot was successfully created, False otherwise.
        """
        self.warning(
            "plot method was called, but not implemented for this task."
        )
        return False

    def __init__(
        self,
        *,
        hidden_size: int,
        loss_function: Union[str, "LossFunction"],
        target_labels: Optional[Union[str, List[str]]] = None,
        prediction_labels: Optional[Union[str, List[str]]] = None,
        transform_prediction_and_target: Optional[Union[Callable, str]] = None,
        transform_target: Optional[Union[Callable, str]] = None,
        transform_inference: Optional[Union[Callable, str]] = None,
        transform_support: Optional[Union[Callable, str]] = None,
        loss_weight: Optional[Union[Callable, str]] = None,
        train_metrics: Optional[Dict[str, Tuple[Metric, bool]]] = None,
        val_metrics: Optional[Dict[str, Tuple[Metric, bool]]] = None,
    ):
        """Construct `Task`.

        Args:
            hidden_size: The number of nodes in the layer feeding into this
                tasks, used to construct the affine transformation to the
                predicted quantity.
            loss_function: Loss function appropriate to the task.
            target_labels: Name(s) of the quantity/-ies being predicted, used
                to extract the  target tensor(s) from the `Data` object in
                `.compute_loss(...)`.
            prediction_labels: The name(s) of each column that is predicted by
                the model during inference. If not given, the name will
                automatically be set to `target_label + _pred`.
            transform_prediction_and_target: Optional function to transform
                both the predicted and target tensor before passing them to the
                loss function. Useful e.g. for having the model predict
                quantities on a physical scale, but transforming this scale to
                O(1) for a numerically stable loss computation.
            transform_target: Optional function to transform only the target
                tensor before passing it, and the predicted tensor, to the loss
                function. Useful e.g. for having the model predict a
                transformed version of the target quantity, e.g. the log10-
                scaled energy, rather than the physical quantity itself. Used
                in conjunction with `transform_inference` to perform the
                inverse transform on the predicted quantity to recover the
                physical scale.
            transform_inference: Optional function to inverse-transform the
                model prediction to recover a physical scale. Used in
                conjunction with `transform_target`.
            transform_support: Optional tuple to specify minimum and maximum
                of the range of validity for the inverse transforms
                `transform_target` and `transform_inference` in case this is
                restricted. By default, the invertibility of `transform_target`
                is tested on the range [-1e6, 1e6].
            loss_weight: Name of the attribute in `data` containing per-event
                loss weights.
        """
        # Base class constructor
        super().__init__()
        # Check(s)
        if target_labels is None:
            target_labels = self.default_target_labels

        if train_metrics is None:
            train_metrics, train_metrics_log = self._instantiate_metrics(prefix="val_")
        else:
            train_metrics, train_metrics_log = self._split_metric_dict(train_metrics)

        if val_metrics is None:
            val_metrics, val_metrics_log = self._instantiate_metrics(prefix="val_")
        else:
            val_metrics, val_metrics_log = self._split_metric_dict(val_metrics)

        self.train_metrics = ModuleDict(train_metrics)
        self.train_metrics_log = train_metrics_log
        self.val_metrics = ModuleDict(val_metrics)
        self.val_metrics_log = val_metrics_log

        if isinstance(target_labels, str):
            target_labels = [target_labels]

        if prediction_labels is None:
            prediction_labels = self.default_prediction_labels
        if isinstance(prediction_labels, str):
            prediction_labels = [prediction_labels]

        if isinstance(loss_function, str):
            loss_function = self._unsafe_parse_string(loss_function)

        assert isinstance(target_labels, List)  # mypy
        assert isinstance(prediction_labels, List)  # mypy
        # Member variables
        self._regularisation_loss: Optional[float] = None
        self._target_labels = target_labels
        self._prediction_labels = prediction_labels
        self._loss_function = loss_function
        self._inference = False
        self._loss_weight = loss_weight

        self._transform_prediction_training: Callable[
            [Tensor], Tensor
        ] = lambda x: x
        self._transform_prediction_inference: Callable[
            [Tensor], Tensor
        ] = lambda x: x
        self._transform_target: Callable[[Tensor], Tensor] = lambda x: x

        if isinstance(transform_prediction_and_target, str):
            transform_prediction_and_target = self._unsafe_parse_string(
                transform_prediction_and_target
            )
        if isinstance(transform_target, str):
            transform_target = self._unsafe_parse_string(transform_target)
        if isinstance(transform_inference, str):
            transform_inference = self._unsafe_parse_string(
                transform_inference
            )
        if isinstance(transform_support, str):
            transform_support = self._unsafe_parse_string(transform_support)

        self._validate_and_set_transforms(
            transform_prediction_and_target,  # type: ignore
            transform_target,  # type: ignore
            transform_inference,  # type: ignore
            transform_support,  # type: ignore
        )

        # Mapping from last hidden layer to required size of input
        self._affine = Linear(hidden_size, self.nb_inputs)

    @final
    def forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Forward pass."""
        self._regularisation_loss = 0  # Reset
        x = self._affine(x)
        x = self._forward(x)
        return self._transform_prediction(x)

    @final
    def _transform_prediction(
        self, prediction: Union[Tensor, Data]
    ) -> Union[Tensor, Data]:
        if self._inference:
            return self._transform_prediction_inference(prediction)
        else:
            return self._transform_prediction_training(prediction)

    @abstractmethod
    def _forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Syntax like `.forward`, for implentation in inheriting classes."""

    def compute_loss(self, pred: Union[Tensor, Data], data: Data) -> Tensor:
        """Compute loss of `pred` wrt.

        target labels in `data`.
        """
        target = torch.stack(
            [data[label] for label in self._target_labels], dim=1
        )
        target = self._transform_target(target)

        if self._loss_weight is not None:
            weights = data[self._loss_weight]
        else:
            weights = None
        loss = (
            self._loss_function(pred, target, weights=weights)  # type: ignore
            + self._regularisation_loss
        )

        return loss

    def compute_metrics(self, pred: Union[Tensor, Data], data: Data, train=False):
        target = torch.stack(
            [data[label] for label in self._target_labels], dim=1
        )
        metrics = self.train_metrics if train else self.val_metrics

        for metric in metrics.values():
            metric(pred, target)

    @final
    def inference(self) -> None:
        """Activate inference mode."""
        self._inference = True

    @final
    def train_eval(self) -> None:
        """Deactivate inference mode."""
        self._inference = False

    @final
    def _validate_and_set_transforms(
        self,
        transform_prediction_and_target: Union[Callable, None],
        transform_target: Union[Callable, None],
        transform_inference: Union[Callable, None],
        transform_support: Union[Tuple, None],
    ) -> None:
        """Validate and set transforms.

        Assert that a valid combination of transformation arguments are
        passed and update the corresponding functions.
        """
        # Checks
        assert not (
            (transform_prediction_and_target is not None)
            and (transform_target is not None)
        ), "Please specify at most one of `transform_prediction_and_target` and `transform_target`"
        if (transform_target is not None) != (transform_inference is not None):
            self.warning(
                "Setting one of `transform_target` and `transform_inference`, but not "
                "the other."
            )

        if transform_target is not None:
            assert transform_target is not None
            assert transform_inference is not None

            if transform_support is not None:
                assert transform_support is not None

                assert (
                    len(transform_support) == 2
                ), "Please specify min and max for transformation support."
                x_test = torch.from_numpy(
                    np.linspace(transform_support[0], transform_support[1], 10)
                )
            else:
                x_test = np.logspace(-6, 6, 12 + 1)
                x_test = torch.from_numpy(
                    np.concatenate([-x_test[::-1], [0], x_test])
                )

            # Add feature dimension before inference transformation to make it
            # match the dimensions of a standard prediction. Remove it again
            # before comparison. Temporary
            try:
                t_test = torch.unsqueeze(transform_target(x_test), -1)
                t_test = torch.squeeze(transform_inference(t_test), -1)
                valid = torch.isfinite(t_test)

                assert torch.allclose(t_test[valid], x_test[valid]), (
                    "The provided transforms for targets during training and "
                    "predictions during inference are not inverse. Please "
                    "adjust transformation functions or support."
                )
                del x_test, t_test, valid

            except IndexError:
                self.warning(
                    "transform_target and/or transform_inference rely on "
                    "indexing, which we won't validate. Please make sure that "
                    "they are mutually inverse, i.e. that\n"
                    "  x = transform_inference(transform_target(x))\n"
                    "for all x that are within your target range."
                )

        # Set transforms
        if transform_prediction_and_target is not None:
            self._transform_prediction_training = (
                transform_prediction_and_target
            )
            self._transform_target = transform_prediction_and_target
        else:
            if transform_target is not None:
                self._transform_target = transform_target
            if transform_inference is not None:
                self._transform_prediction_inference = transform_inference

    @staticmethod
    def _split_metric_dict(metric_dict: Dict[str, Tuple[Metric, bool]]):
        metrics = {}
        log_metrics = {}
        for key, value in metric_dict.items():
            metrics[key] = value[0]
            log_metrics[key] = value[1]
        return metrics, log_metrics

    def _instantiate_metrics(self, prefix) -> Tuple[Dict[str, Metric], Dict[str, bool]]:
        metrics = {}
        log_metrics = {}
        for key, value in self.default_metrics.items():
            metrics[f"{prefix}_{key}"] = value[0]()
            log_metrics[f"{prefix}_{key}"] = value[1]
        return metrics, log_metrics


class IdentityTask(Task):
    """Identity, or trivial, task."""

    def __init__(
        self,
        nb_outputs: int,
        target_labels: Union[List[str], Any],
        *args: Any,
        **kwargs: Any,
    ):
        """Construct IdentityTask.

        Return the `nb_outputs` as a direct, affine transformation of
        the last hidden layer.
        """
        self._nb_inputs = nb_outputs
        self._default_target_labels = (
            target_labels
            if isinstance(target_labels, list)
            else [target_labels]
        )
        self._default_prediction_labels = [
            f"target_{i}_pred" for i in range(nb_outputs)
        ]

        super().__init__(*args, **kwargs)
        # Base class constructor

    @property
    def default_target_labels(self) -> List[str]:
        """Return default target labels."""
        return self._default_target_labels

    @property
    def default_prediction_labels(self) -> List[str]:
        """Return default prediction labels."""
        return self._default_prediction_labels

    @property
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""
        return self._nb_inputs

    def _forward(self, x: Tensor) -> Tensor:
        # Leave it as is.
        return x
