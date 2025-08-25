import pytest
import torch

from hypersheaf.data import HeteroHypergraph


@pytest.fixture
def permuted_hypergraph(hypergraph: HeteroHypergraph) -> HeteroHypergraph:
    # Permuting the node features
    node_feats = hypergraph.x
    node1 = node_feats[0, :]
    node2 = node_feats[1, :]
    permuted_node_feats = hypergraph.x.clone().detach()
    permuted_node_feats[0, :] = node2
    permuted_node_feats[1, :] = node1

    permuted_he_index = torch.tensor(
        [
            [1, 0, 2, 1, 0, 3, 4, 0, 2, 4],
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
        ]
    )

    return HeteroHypergraph(
        x=permuted_node_feats,
        hyperedge_index=permuted_he_index,
        node_types=hypergraph.node_types,
        hyperedge_types=hypergraph.hyperedge_types,
        hyperedge_features=hypergraph.hyperedge_features,
    )
