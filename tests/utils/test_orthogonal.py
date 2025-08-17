import pytest

from hypersheaf.utils.orthogonal import Orthogonal


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
