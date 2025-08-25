import torch

from hypersheaf.data import HeteroHypergraph
from hypersheaf.feature_builders.base_builder import BaseHeFeatBuilder


def are_he_feats_permutation_invariant(
    feat_builder: BaseHeFeatBuilder,
    hypergraph: HeteroHypergraph,
    permuted_hypergraph: HeteroHypergraph,
) -> bool:
    """Checks if the hyperedge features are permutation invariant"""
    _, es = feat_builder(
        hypergraph.x, hypergraph.hyperedge_features, hypergraph.hyperedge_index
    )
    _, es1 = feat_builder(
        permuted_hypergraph.x,
        permuted_hypergraph.hyperedge_features,
        permuted_hypergraph.hyperedge_index,
    )
    return torch.allclose(es, es1, atol=1e-6)


def are_node_features_permutation_invariant(
    feat_builder: BaseHeFeatBuilder,
    hypergraph: HeteroHypergraph,
    permuted_hypergraph: HeteroHypergraph,
) -> bool:
    """Checks if the node features are permutation invariant"""
    xs, _ = feat_builder(
        hypergraph.x, hypergraph.hyperedge_features, hypergraph.hyperedge_index
    )
    xs1, _ = feat_builder(
        permuted_hypergraph.x,
        permuted_hypergraph.hyperedge_features,
        permuted_hypergraph.hyperedge_index,
    )
    return torch.allclose(xs, xs1, atol=1e-6)
