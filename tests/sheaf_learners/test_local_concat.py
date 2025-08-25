import torch
import pytest
import torch.nn.functional as F
from typing import Callable

from hypersheaf.sheaf_learners.local_concat import LocalConcatSheafLearner
from .helpers import SheafLearnerInput


@pytest.fixture
def sheaf_learner() -> LocalConcatSheafLearner:
    return LocalConcatSheafLearner(node_feats=16, out_channels=32)


def test_local_concat(
    sheaf_learner: LocalConcatSheafLearner, sheaf_learner_input: SheafLearnerInput
):
    assert str(sheaf_learner) == "LocalConcatSheafLearner(16, 32)"
    assert sheaf_learner.act_fn == "relu"

    out1 = sheaf_learner(
        sheaf_learner_input.node_feats,
        sheaf_learner_input.he_feats,
        sheaf_learner_input.he_index,
        sheaf_learner_input.node_types,
        sheaf_learner_input.he_types,
    )
    assert out1.shape == (sheaf_learner_input.n_x, 32)


def test_local_concat_predict_sheaf(
    sheaf_learner: LocalConcatSheafLearner, sheaf_learner_input: SheafLearnerInput
):
    out1 = sheaf_learner.predict_sheaf(
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
def test_local_concat_act_fn(
    sheaf_learner: LocalConcatSheafLearner,
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
