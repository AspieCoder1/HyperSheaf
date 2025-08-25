from hypersheaf.data import HeteroHypergraph


def test_hetero_hypergraph(hypergraph: HeteroHypergraph):
    assert hypergraph.num_nodes == 5
    assert hypergraph.num_hyperedges == 3
    assert hypergraph.num_node_types == 2
    assert hypergraph.num_hyperedge_types == 2
    assert hypergraph.n_x == hypergraph.num_nodes
