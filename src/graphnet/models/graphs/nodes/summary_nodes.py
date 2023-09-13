"""Class(es) for creating nodes per dom with summary variables as features."""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from torch_geometric.data import Data
import torch

from .nodes import NodeDefinition
from graphnet.utilities.config import save_model_config


class MircoSummaryNodes(NodeDefinition):
    """Create summary variables for times and charges per dom.

    Summary variables are inspired by https://arxiv.org/pdf/2101.11589.pdf.
    """

    _x_index = 0
    _y_index = 1
    _z_index = 2
    _time_index = 3
    _charge_index = 4
    _first_time_index = 100 / 3.0e4
    _second_time_index = 500 / 3.0e4

    @save_model_config
    def __init__(
        self,
        times: Optional[List[float]] = None,
        time_quantiles: Optional[List[float]] = None,
        dom_specifiers: Optional[List[int]] = None,
        index_permutation: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        """Create per DOM summary variables.

        Args:
            times: temp
            time_quantiles: temp
            dom_specifiers: temp
            index_permutation: temp
            **kwargs:
        """
        super().__init__(**kwargs)
        self.times = times
        self.time_quantiles = time_quantiles
        if dom_specifiers is None:
            self.dom_specifiers = np.arange(
                3
            )  # use 3 first columns to specify dom
        else:
            self.dom_specifiers = np.array(dom_specifiers)

        if index_permutation is not None:
            self.index_permutation = np.empty_like(index_permutation)
            self.index_permutation[index_permutation] = np.arange(
                len(index_permutation)
            )
        else:
            self.index_permutation = index_permutation

    def _construct_nodes(self, x: torch.Tensor) -> Any:
        dom_separated_pulsemap = self._split_pulse_map_to_doms(x)
        summary_data = []
        for key, dom_time_series in dom_separated_pulsemap.items():
            summary_data.append(
                np.array(
                    [
                        dom_time_series[0, self._x_index],
                        dom_time_series[0, self._y_index],
                        dom_time_series[0, self._z_index],
                        t_min := np.min(dom_time_series[:, self._time_index]),
                        self.max(dom_time_series[:, self._time_index]),
                        np.mean(dom_time_series[:, self._time_index]),
                        np.std(dom_time_series[:, self._time_index], ddof=1),
                        np.log10(
                            1 + np.sum(dom_time_series[:, self._charge_index])
                        ),
                        np.log10(
                            1
                            + np.sum(
                                dom_time_series[
                                    dom_time_series[:, self._time_index]
                                    < t_min + self._first_time_index,
                                    self._charge_index,
                                ]
                            )
                        ),
                        np.log10(
                            1
                            + np.sum(
                                dom_time_series[
                                    dom_time_series[:, self._time_index]
                                    < t_min + self._second_time_index,
                                    self._charge_index,
                                ]
                            )
                        ),
                    ]
                )
            )
        pass

    def _split_pulse_map_to_doms(self, x: torch.Tensor) -> Any:
        x = x.numpy()
        if self.index_permutation is not None:
            x_sort = x[:, self.index_permutation]
        else:
            x_sort = x
        sorted_indices = np.lexsort(x_sort)
        dom_time_series: Dict[Tuple, torch.tensor] = {}
        for idx in sorted_indices:
            if (
                dom_key := tuple(x[idx, self.dom_specifiers])
            ) in dom_time_series:
                dom_time_series[dom_key].append(x[idx, :])
            else:
                dom_time_series[dom_key] = [
                    np.delete(x[idx, :], self.dom_specifiers)
                ]
        for key in dom_time_series:
            dom_time_series[key] = np.array(dom_time_series[key])
        return dom_time_series
