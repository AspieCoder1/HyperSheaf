import pytest
import torch

from hypersheaf.utils.mlp import MLP


@pytest.mark.parametrize("normalisation", ["bn", "ln", "None"])
@pytest.mark.parametrize("input_norm", [True, False])
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_mlp(normalisation: str, input_norm: bool, num_layers: int):
    mlp = MLP(
        in_channels=2,
        hidden_channels=1,
        out_channels=6,
        num_layers=num_layers,
        normalisation=normalisation,
        input_norm=input_norm,
    )
    mlp.reset_parameters()

    x = torch.randn(4, 2)
    assert mlp(x).shape == (4, 6)


def test_mlp_error():
    with pytest.raises(ValueError, match="normalisation must be one of bn, ln or None"):
        MLP(
            in_channels=1,
            hidden_channels=1,
            out_channels=1,
            num_layers=1,
            normalisation="random_normalisation_strategy",
        )
