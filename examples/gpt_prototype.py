from graphnet.models.graphs import GraphDefinition
from graphnet.models.graphs.nodes.clustering_nodes import TimeClusteringNodes
from graphnet.models.detector import IceCube86
import numpy as np


if __name__ == "__main__":
    g = GraphDefinition(IceCube86(), TimeClusteringNodes(200, 8))
    a = np.arange(10*7)
    a = a.reshape(10, 7)
    print(a.shape)
    print(g(a, list(IceCube86().feature_map().keys())).x.shape)
    print("Success")


