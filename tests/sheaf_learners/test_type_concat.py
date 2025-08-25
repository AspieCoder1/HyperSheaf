import torch
import pytest
import torch.nn.functional as F
from typing import Callable

from hypersheaf.sheaf_learners.type_concat import TypeConcatSheafLearner
from hypersheaf.data import HeteroHypergraph
from helpers import SheafLearnerInput


@pytest.fixture
def sheaf_learner(hypergraph: HeteroHypergraph) -> TypeConcatSheafLearner:
    return TypeConcatSheafLearner(
        node_feats=16,
        out_channels=32,
        act_fn="relu",
        num_node_types=hypergraph.num_node_types,
        num_he_types=hypergraph.num_hyperedge_types,
    )


def test_type_concat(
    sheaf_learner: TypeConcatSheafLearner, sheaf_learner_input: SheafLearnerInput
):
    assert str(sheaf_learner) == "TypeConcatSheafLearner(16, 32)"
    assert sheaf_learner.act_fn == "relu"

    out1 = sheaf_learner(
        sheaf_learner_input.node_feats,
        sheaf_learner_input.he_feats,
        sheaf_learner_input.he_index,
        sheaf_learner_input.node_types,
        sheaf_learner_input.he_types,
    )
    assert out1.shape == (sheaf_learner_input.n_x, 32)


def test_type_concat_predict_sheaf(
    sheaf_learner: TypeConcatSheafLearner, sheaf_learner_input: SheafLearnerInput
):
    out1 = sheaf_learner(
        sheaf_learner_input.node_feats,
        sheaf_learner_input.he_feats,
        sheaf_learner_input.he_index,
        sheaf_learner_input.node_types,
        sheaf_learner_input.he_types,
    )

    assert out1.shape == (sheaf_learner_input.n_x, 32)


@pytest.mark.parametrize(
    "sheaf_act,pytorch_callable",
    [
        ("relu", F.relu),
        ("sigmoid", F.sigmoid),
        ("tanh", F.tanh),
        ("elu", F.elu),
        ("None", lambda x: x),
    ],
)
def test_type_concat_act_fn(
    sheaf_learner: TypeConcatSheafLearner,
    sheaf_learner_input: SheafLearnerInput,
    sheaf_act: str,
    pytorch_callable: Callable[[torch.Tensor], torch.Tensor],
):
    sheaf_learner.act_fn = sheaf_act

    out1 = sheaf_learner(
        sheaf_learner_input.node_feats,
        sheaf_learner_input.he_feats,
        sheaf_learner_input.he_index,
        sheaf_learner_input.node_types,
        sheaf_learner_input.he_types,
    )
    out2 = sheaf_learner.predict_sheaf(
        sheaf_learner_input.node_feats,
        sheaf_learner_input.he_feats,
        sheaf_learner_input.he_index,
        sheaf_learner_input.node_types,
        sheaf_learner_input.he_types,
    )

    assert out1.shape == (sheaf_learner_input.n_x, 32)
    assert out2.shape == (sheaf_learner_input.n_x, 32)
    assert torch.allclose(out1, pytorch_callable(out2), atol=1e-6)
