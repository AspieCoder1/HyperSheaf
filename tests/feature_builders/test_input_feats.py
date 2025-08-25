import torch

from hypersheaf.data import HeteroHypergraph
from hypersheaf.feature_builders.input_feats import InputFeatsHeFeatBuilder
from .helpers import (
    are_he_feats_permutation_invariant,
    are_node_features_permutation_invariant,
)


def test_input_feats_builder(hypergraph: HeteroHypergraph):
    he_index = hypergraph.hyperedge_index
    feat_builder = InputFeatsHeFeatBuilder()
    assert str(feat_builder) == "InputFeatsHeFeatBuilder()"

    xs, es = feat_builder(hypergraph.x, hypergraph.hyperedge_features, he_index)
    assert xs.shape == (he_index.size(-1), 16)
    assert es.shape == (he_index.size(-1), 16)
    assert torch.allclose(
        xs, torch.index_select(hypergraph.x, dim=0, index=he_index[0]), atol=1e-6
    )
    assert torch.allclose(
        es,
        torch.index_select(hypergraph.hyperedge_features, dim=0, index=he_index[1]),
        atol=1e-6,
    )

    xs1, es1 = feat_builder.compute_he_features(
        hypergraph.x, hypergraph.hyperedge_features, he_index
    )
    assert xs1.shape == (he_index.size(-1), 16)
    assert es1.shape == (he_index.size(-1), 16)
    assert torch.allclose(
        xs1, torch.index_select(hypergraph.x, dim=0, index=he_index[0]), atol=1e-6
    )
    assert torch.allclose(
        es1,
        torch.index_select(hypergraph.hyperedge_features, dim=0, index=he_index[1]),
        atol=1e-6,
    )


def test_input_feat_builder_he_feats_permutation_invariant(
    hypergraph: HeteroHypergraph,
    permuted_hypergraph: HeteroHypergraph,
):
    feat_builder = InputFeatsHeFeatBuilder(hidden_channels=16)
    assert are_he_feats_permutation_invariant(
        feat_builder, hypergraph, permuted_hypergraph
    )
    assert are_node_features_permutation_invariant(
        feat_builder, hypergraph, permuted_hypergraph
    )
