import pytest
from hypersheaf.data import HeteroHypergraph
import torch


@pytest.fixture
def hypergraph() -> HeteroHypergraph:
    num_nodes = 5
    num_node_types = 2
    num_hyperedge_types = 2
    features = torch.rand(num_nodes, 16)
    he_index = torch.tensor(
        [[0, 1, 2, 0, 1, 3, 4, 1, 2, 4], [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]]
    )
    hyperedge_features = torch.rand(5, 16)
    hyperedge_types = torch.randint(0, num_hyperedge_types, (3,))
    node_types = torch.randint(0, num_node_types, (num_nodes,))
    return HeteroHypergraph(
        x=features,
        hyperedge_index=he_index,
        hyperedge_types=hyperedge_types,
        node_types=node_types,
        hyperedge_features=hyperedge_features,
    )
