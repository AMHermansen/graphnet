from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .sharded import ParquetSharded


def _remove(arr, l1_importance, number_to_remove):
    pos_to_keep = np.argsort(l1_importance)[:-number_to_remove]
    pos_to_keep = np.sort(pos_to_keep)  # restore original order
    return arr[pos_to_keep, ...]


@dataclass
class RemoverConfig:
    max_pulses: int
    fraction_to_remove: float


class NodeHook:
    pass


class NormImportanceRemover(NodeHook):
    def __init__(
            self,
            config: RemoverConfig,
            latent_folder: str,
            batch_size: int = 256,
            remove_most_important: bool = True
    ):
        self._max_pulses = config.max_pulses
        self._fraction_to_remove = config.fraction_to_remove
        self._latent_folder = latent_folder
        self._batch_size = batch_size
        self.importance_sign = 1 if remove_most_important else -1

    def __call__(self, arr: np.ndarray, idx: int, dataset: "ParquetSharded") -> np.ndarray:
        arr = arr[:self._max_pulses, ...]
        l1_importance = self._load_latent(idx) * self.importance_sign
        number_to_remove = int(self._fraction_to_remove * arr.shape[0]) + 1  # round up
        arr = _remove(arr, l1_importance, number_to_remove)
        return arr

    def _load_latent(self, idx) -> np.ndarray:
        raise NotImplementedError(f"Please implement _load_latent for {self.__class__.__name__}")


class RandomRemover(NodeHook):
    def __init__(
            self,
            config: RemoverConfig,
    ):
        self._max_pulses = config.max_pulses
        self._fraction_to_remove = config.fraction_to_remove
        self._rng = np.random.default_rng()

    def __call__(self, arr: np.ndarray, idx: int, dataset: "ParquetSharded") -> np.ndarray:
        arr = arr[:self._max_pulses]
        random_importance_score = self._rng.uniform(size=arr.shape[0])
        number_to_remove = int(self._fraction_to_remove * arr.shape[0]) + 1  # round up
        arr = _remove(arr, random_importance_score, number_to_remove)
        return arr


class L1ImportanceRemover(NormImportanceRemover):
    def _load_latent(self, idx):
        batch_number, event_number = divmod(idx, self._batch_size)
        batch_info = np.load(f"{self._latent_folder}/batch_info{batch_number}.npy")
        latent_features = np.load(f"{self._latent_folder}/node_features{batch_number}.npy")
        event_latent = latent_features[batch_info == event_number]
        return np.sum(np.abs(event_latent), axis=-1)


class RelativeL1Importance(NormImportanceRemover):
    def _load_latent(self, idx) -> np.ndarray:
        batch_number, event_number = divmod(idx, self._batch_size)
        batch_info = np.load(f"{self._latent_folder}/batch_info{batch_number}.npy")
        latent_features = np.load(f"{self._latent_folder}/node_features{batch_number}.npy")
        event_latent = latent_features[batch_info == event_number]
        return self._compute_importance(event_latent)

    @staticmethod
    def _compute_importance(event_latent: np.ndarray) -> np.ndarray:
        number_of_pulses, number_of_features = event_latent.shape
        event_mat = (
            np.stack([event_latent for _ in range(number_of_pulses)]) / (number_of_pulses - 1)
            - np.stack([np.diag(event_latent[:, i]) for i in range(number_of_features)], axis=-1)
        )
        return np.sum(np.abs(np.sum(event_mat, axis=1)), axis=-1)
