import sqlite3
from itertools import chain
from typing import List, TYPE_CHECKING

import numpy as np
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from graphnet.data.dataset import Dataset
from tqdm import tqdm


def get_n_pulses(
    database: str, features: List[str], pulsemap: str, event_no: int
):
    with sqlite3.connect(database) as con:
        query = f'select {", ".join(features)} from {pulsemap} where event_no = {event_no}'
        return len(con.execute(query).fetchall())


class SequenceBucketSampler(Sampler):
    """Sequence bucketing sampler."""

    def __init__(
        self,
        sequence_lengths: List[int],
        batch_size: int,
        shuffle: bool = False,
    ):
        super().__init__()
        self.sequence_lengths = sequence_lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._sorted_indices = np.argsort(sequence_lengths)
        self.buckets = self._create_buckets()

        self._current_epoch = 0

    def _create_buckets(self):
        buckets = [
            self._sorted_indices[i : i + self.batch_size]
            for i in range(0, len(self.sequence_lengths), self.batch_size)
        ][:-1]
        return buckets

    def __iter__(self):
        if self.shuffle:
            yield from np.random.permutation(self.buckets).tolist()
        else:
            yield from self.buckets

    def __len__(self):
        return len(self.sequence_lengths) // self.batch_size


# Way too slow to get lengths of all events
class SequenceBucketingDatasetSampler(SequenceBucketSampler):
    def __init__(
        self, dataset: "Dataset", batch_size: int, shuffle: bool = False
    ):
        event_counts = [
            get_n_pulses(
                dataset._path,
                dataset._features,
                dataset._pulsemaps[0],
                event_no,
            )
            for event_no in tqdm(dataset._indices)
        ]
        super().__init__(event_counts, batch_size, shuffle)
