import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import knn_graph

from graphnet.models.graphs import KNNGraph, NodesAsPulses
from graphnet.models.detector import IceCube86

class DummyDataGenerator:
    def __init__(self, features: int = 7, pulsemap_len_range=(5, 15)):
        self.features = features
        self.pulsemap_len_range = pulsemap_len_range

        self._gd = KNNGraph(IceCube86(), NodesAsPulses(), nb_nearest_neighbours=2)

    def get_data(self, n_samples: int = 1):
        data_list = []
        rng = np.random.default_rng()
        for _ in tqdm(range(n_samples)):
            n = rng.integers(*self.pulsemap_len_range)
            data_obj = self._gd(rng.uniform(low=0., high=1., size=(n, self.features)), ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde", "pmt_area"])
            data_list.append(data_obj)
        return Batch.from_data_list(data_list)


if __name__ == "__main__":
    dg = DummyDataGenerator()
    print(dg.get_data(2))
