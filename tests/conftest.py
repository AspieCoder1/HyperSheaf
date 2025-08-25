import pytest
from hypersheaf.data import HeteroHypergraph
import torch


@pytest.fixture
def hypergraph() -> HeteroHypergraph:
    num_nodes = 5
    features = torch.rand(num_nodes, 16)
    he_index = torch.tensor(
        [
            [0, 1, 2, 0, 1, 3, 4, 1, 2, 4],
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
        ]
    )
    hyperedge_features = torch.rand(3, 16)
    hyperedge_types = torch.tensor([0, 1, 0])
    node_types = torch.tensor([0, 1, 0, 1, 0])
    return HeteroHypergraph(
        x=features,
        hyperedge_index=he_index,
        hyperedge_types=hyperedge_types,
        node_types=node_types,
        hyperedge_features=hyperedge_features,
    )
