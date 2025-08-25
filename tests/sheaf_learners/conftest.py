import pytest
from .helpers import SheafLearnerInput

from hypersheaf.feature_builders.input_feats import InputFeatsHeFeatBuilder
from hypersheaf.data import HeteroHypergraph


@pytest.fixture
def sheaf_learner_input(hypergraph: HeteroHypergraph) -> SheafLearnerInput:
    feat_builder = InputFeatsHeFeatBuilder()
    xs, es = feat_builder.compute_he_features(
        hypergraph.x, hypergraph.hyperedge_features, hypergraph.hyperedge_index
    )
    return SheafLearnerInput(
        node_feats=xs,
        he_feats=es,
        he_index=hypergraph.hyperedge_index,
        node_types=hypergraph.node_types,
        he_types=hypergraph.hyperedge_types,
    )
