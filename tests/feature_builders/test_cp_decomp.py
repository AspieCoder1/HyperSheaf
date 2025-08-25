from .helpers import (
    are_he_feats_permutation_invariant,
    are_node_features_permutation_invariant,
)

from hypersheaf.data import HeteroHypergraph
from hypersheaf.feature_builders.cp_decomp import CPDecompHeFeatBuilder


def test_cp_decomp_feat_builder(hypergraph: HeteroHypergraph):
    feat_builder = CPDecompHeFeatBuilder(hidden_channels=16)
    assert str(feat_builder) == "CPDecompHeFeatBuilder(16)"

    he_index = hypergraph.hyperedge_index
    xs, es = feat_builder(hypergraph.x, hypergraph.hyperedge_features, he_index)
    assert xs.shape == (he_index.size(-1), 16)
    assert es.shape == (he_index.size(-1), 16)

    xs1, es1 = feat_builder.compute_he_features(
        hypergraph.x, hypergraph.hyperedge_features, he_index
    )
    assert xs1.shape == (he_index.size(-1), 16)
    assert es1.shape == (he_index.size(-1), 16)


def test_cp_decomp_feat_builder_he_feats_permutation_invariant(
    hypergraph: HeteroHypergraph,
    permuted_hypergraph: HeteroHypergraph,
):
    feat_builder = CPDecompHeFeatBuilder(hidden_channels=16)
    assert are_he_feats_permutation_invariant(
        feat_builder, hypergraph, permuted_hypergraph
    )
    assert are_node_features_permutation_invariant(
        feat_builder, hypergraph, permuted_hypergraph
    )
