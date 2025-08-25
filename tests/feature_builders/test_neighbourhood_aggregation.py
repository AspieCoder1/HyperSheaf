from hypersheaf.data import HeteroHypergraph
from hypersheaf.feature_builders.neighbour_aggregation import (
    NodeMeanHeFeatBuilder,
    EquivariantHeFeatBuilder,
)
from .helpers import (
    are_he_feats_permutation_invariant,
    are_node_features_permutation_invariant,
)


def test_node_mean_he_feat_builder(hypergraph: HeteroHypergraph):
    he_index = hypergraph.hyperedge_index
    feat_builder = NodeMeanHeFeatBuilder()
    assert str(feat_builder) == "NodeMeanHeFeatBuilder()"

    xs, es = feat_builder(hypergraph.x, hypergraph.hyperedge_features, he_index)
    assert xs.shape == (he_index.size(-1), 16)
    assert es.shape == (he_index.size(-1), 16)


def test_node_mean_he_feat_builder_he_feats_permutation_invariant(
    hypergraph: HeteroHypergraph, permuted_hypergraph: HeteroHypergraph
):
    feat_builder = NodeMeanHeFeatBuilder()
    assert are_he_feats_permutation_invariant(
        feat_builder, hypergraph, permuted_hypergraph
    )
    assert are_node_features_permutation_invariant(
        feat_builder, hypergraph, permuted_hypergraph
    )


def test_equivariance_he_feat_builder(hypergraph: HeteroHypergraph):
    he_index = hypergraph.hyperedge_index
    feat_builder = EquivariantHeFeatBuilder(
        num_node_feats=hypergraph.num_node_features, hidden_channels=16, out_channels=16
    )
    assert str(feat_builder) == "EquivariantHeFeatBuilder()"

    xs, es = feat_builder(hypergraph.x, hypergraph.hyperedge_features, he_index)
    assert xs.shape == (he_index.size(-1), 16)
    assert es.shape == (he_index.size(-1), 16)


def test_equivariance_he_feat_builder_he_feats_permutation_equivariant(
    hypergraph: HeteroHypergraph, permuted_hypergraph: HeteroHypergraph
):
    feat_builder = EquivariantHeFeatBuilder(
        num_node_feats=hypergraph.num_node_features, hidden_channels=16, out_channels=16
    )
    assert are_he_feats_permutation_invariant(
        feat_builder, hypergraph, permuted_hypergraph
    )
