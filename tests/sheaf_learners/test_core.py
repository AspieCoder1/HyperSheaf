import pytest
import torch
import torch.nn.functional as F
from typing import Callable

from hypersheaf.sheaf_learners.core import HeteroSheafLearner


@pytest.mark.parametrize(
    "sheaf_act,pytorch_callable",
    [
        ("relu", F.relu),
        ("sigmoid", F.sigmoid),
        ("tanh", F.tanh),
        ("elu", F.elu),
        ("None", None),
    ],
)
def test_sheaf_act(
    sheaf_act: str, pytorch_callable: Callable[[torch.Tensor], torch.Tensor]
):
    sheaf_learner = HeteroSheafLearner(act_fn=sheaf_act)
    x = torch.randn(4, 2)
    out = sheaf_learner.sheaf_act(x)

    expected = x
    if pytorch_callable is not None:
        expected = pytorch_callable(x)

    assert torch.allclose(out, expected)
    assert out.shape == (4, 2)
