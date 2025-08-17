import pytest
import torch

from hypersheaf.utils.orthogonal import Orthogonal


def all_orthogonal(tensor: torch.Tensor) -> bool:
    """Helper function to check if a batch of tensors are orthogonal."""
    products = torch.bmm(tensor, tensor.mT)
    identity_batch = torch.eye(tensor.shape[-1]).repeat(tensor.shape[0], 1, 1)
    return torch.allclose(products, identity_batch, atol=1e-5)


@pytest.mark.parametrize(
    "orthogonal_map", ["matrix_exp", "cayley", "householder", "euler"]
)
def test_valid_transformation_type(orthogonal_map: str):
    orth_model = Orthogonal(d=5, orthogonal_map=orthogonal_map)

    assert orth_model.d == 5
    assert orth_model.orthogonal_map == orthogonal_map


def test_invalid_transformation_type():
    with pytest.raises(
        ValueError, match="Unsupported transformations random_orthogonal_map"
    ):
        Orthogonal(d=5, orthogonal_map="random_orthogonal_map")

def test_orthogonal_matrix_exp():
    orth_model = Orthogonal(d=5, orthogonal_map="matrix_exp")

    x = torch.randn((4, orth_model.d * (orth_model.d + 1) // 2))
    out = orth_model(x)

    assert all_orthogonal(out)

def test_orthogonal_cayley():
    orth_model = Orthogonal(d=5, orthogonal_map="cayley")

    x = torch.randn((4, orth_model.d * (orth_model.d + 1) // 2))
    out = orth_model(x)

    assert all_orthogonal(out)

def test_orthogonal_householder():
    orth_model = Orthogonal(d=5, orthogonal_map="householder")

    x = torch.randn((4, 10))
    out = orth_model(x)
    assert all_orthogonal(out)

def test_orthogonal_euler_2d():
    orth_model = Orthogonal(d=2, orthogonal_map="euler")

    x = torch.randn((4, 1))
    out = orth_model(x)

    assert all_orthogonal(out)

def test_orthogonal_euler_3d():
    orth_model = Orthogonal(d=3, orthogonal_map="euler")

    x = torch.randn((4, 3)).clip(-1, 1)
    out = orth_model(x)

    assert all_orthogonal(out)