from typing import Optional, Callable

import numpy as np
import torch


def fps(arr: np.ndarray, n_sample: int, start_idx: Optional[int] = None):
    n_points, n_dim = arr.shape
    if (start_idx is None) or (start_idx < 0):
        start_idx = np.random.randint(0, n_points)
    sampled_indices = [start_idx]
    min_distances = np.full(n_points, np.inf)
    for _ in range(n_sample - 1):
        current_point = arr[sampled_indices[-1]]
        dist_to_current_point = np.linalg.norm(arr - current_point, axis=1)
        min_distances = np.minimum(min_distances, dist_to_current_point)
        farthest_point_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_point_idx)
        return np.array(sampled_indices)


class CoordinateTransform:
    def __init__(self, coordinate_idx: int, reversed_order: bool = False):
        self.coordinate_idx = coordinate_idx
        self._reversed_order_sign = 1 if reversed_order else -1

    @property
    def reverse_order(self):
        return self._reversed_order_sign == 1

    @reverse_order.setter
    def reverse_order(self, reversed_order: bool):
        self._reversed_order_sign = 1 if reversed_order else -1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.coordinate_idx] * self._reversed_order_sign
