"""Class(es) for creating nodes per dom with summary variables as features."""

from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np
from torch_geometric.data import Data
import torch

from .nodes import NodeDefinition
from graphnet.utilities.config import save_model_config
from icecream import ic


class MircoSummaryNodes(NodeDefinition):
    """Create summary variables for times and charges per dom.

    Summary variables are inspired by
    https://arxiv.org/pdf/2101.11589.pdf.
    """

    _x_index = 0
    _y_index = 1
    _z_index = 2
    _time_index = 3
    _charge_index = 4
    _first_time_index = 100 / 3.0e4
    _second_time_index = 500 / 3.0e4
    _charge_scale = 2
    _charge_quantile_times = np.array([0.2, 0.5])
    _generated_features = 9 + len(_charge_quantile_times)

    @property
    def nb_outputs(self) -> int:
        """Return number of outputs."""
        return self._generated_features

    def __init__(
        self,
        charge_at_times: Optional[List[float]] = None,
        time_quantiles: Optional[List[float]] = None,
        dom_specifiers: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        """Create per DOM summary variables.

        Args:
            charge_at_times: temp
            time_quantiles: temp
            dom_specifiers: temp
            index_permutation: temp
            **kwargs:
        """
        super().__init__(**kwargs)  # noqa
        self.charge_at_times = charge_at_times or [
            self._first_time_index,
            self._second_time_index,
        ]
        if time_quantiles is None:
            self.time_quantiles = self._charge_quantile_times
        else:
            self.time_quantiles = time_quantiles
        if dom_specifiers is None:
            self.dom_specifiers = np.arange(
                3
            )  # use 3 first columns to specify dom
        else:
            self.dom_specifiers = np.array(dom_specifiers)

    def _construct_nodes(
        self, x: torch.Tensor, include_sensor_id: bool = False
    ) -> Data:
        dom_separated_pulsemap = self._split_pulsemap_to_doms(x)
        summary_data = np.empty(
            (len(dom_separated_pulsemap), self._generated_features)
        )
        for idx, (key, dom_time_series) in enumerate(
            dom_separated_pulsemap.items()
        ):
            charge_cumulative = np.cumsum(
                dom_time_series[:, self._charge_index]
            )
            charge_deposit_indices = np.searchsorted(
                charge_cumulative, self._charge_quantile_times
            )
            summary_data[idx, :] = np.array(
                [
                    dom_time_series[0, self._x_index],  # x_loc
                    dom_time_series[0, self._y_index],  # y_loc
                    dom_time_series[0, self._z_index],  # z_loc
                    t_min := np.min(
                        dom_time_series[:, self._time_index]
                    ),  # first time
                    self.max(
                        dom_time_series[:, self._time_index]
                    ),  # last time
                    np.mean(dom_time_series[:, self._time_index]),  # mean time
                    np.std(
                        dom_time_series[:, self._time_index], ddof=1
                    ),  # std time
                    np.log10(
                        1 + np.sum(dom_time_series[:, self._charge_index])
                    )
                    / self._charge_scale,  # Total charge
                    *[
                        np.log10(
                            1
                            + np.sum(
                                dom_time_series[
                                    dom_time_series[:, self._time_index]
                                    < t_min + time_index,
                                    self._charge_index,
                                ]
                            )
                        )
                        for time_index in self.charge_at_times
                    ],  # Charge at times
                    np.log10(
                        1
                        + np.sum(
                            dom_time_series[
                                dom_time_series[:, self._time_index]
                                < t_min + self._first_time_index,
                                self._charge_index,
                            ]
                        )
                    ),  # Charge in first window
                    np.log10(
                        1
                        + np.sum(
                            dom_time_series[
                                dom_time_series[:, self._time_index]
                                < t_min + self._second_time_index,
                                self._charge_index,
                            ]
                        )
                    ),  # Charge in second window
                    *[
                        dom_time_series[charge_index, self._time_index]
                        for charge_index in charge_deposit_indices
                    ],  # Time quantiles
                ]
            )
            d = Data(x=torch.from_numpy(summary_data))
            print(d)
        return d

    def _split_pulsemap_to_doms(
        self, x: torch.Tensor
    ) -> Dict[Tuple, np.ndarray]:
        x = x.numpy()
        x_sort = x
        sorted_indices = np.lexsort(x_sort)
        dom_time_series: Dict[Tuple, Union[List[np.ndarray], np.ndarray]] = {}
        for idx in sorted_indices:
            if (
                dom_key := tuple(x[idx, self.dom_specifiers])
            ) in dom_time_series:
                dom_time_series[dom_key].append(x[idx, :])
            else:
                dom_time_series[dom_key] = [x[idx, :]]
        for key in dom_time_series:
            dom_time_series[key] = np.array(dom_time_series[key])
        return dom_time_series
