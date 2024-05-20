from typing import Union, List

import torch
import math


class LogTransform:
    def __init__(self, base: float = 10.):
        self.base = base

    def __call__(self, x: torch.Tensor):
        return torch.log(x) / math.log(self.base)


class PowTransform:
    def __init__(self, base: float = 10.):
        self.base = base

    def __call__(self, x: torch.Tensor):
        return torch.pow(self.base, x)


class ScaleTransform:
    def __init__(self, scales: Union[float, List[float]]):
        self.scales = scales

    def __call__(self, x: torch.Tensor):
        if isinstance(self.scales, float):
            return x / self.scales
        assert x.shape[-1] == len(self.scales)
        for value_idx, scale in enumerate(self.scales):
            x[..., value_idx] /= scale
        return x


class UnScaleTransform:
    def __init__(self, scales: Union[float, List[float]]):
        self.scales = scales

    def __call__(self, x: torch.Tensor):
        if isinstance(self.scales, float):
            return x * self.scales
        assert x.shape[-1] == len(self.scales)
        for value_idx, scale in enumerate(self.scales):
            x[..., value_idx] *= scale
        return x
