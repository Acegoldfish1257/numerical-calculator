import pytest
from methods.linear_algebra.conditioning import matrix_conditioning


def test_diagonal():
    assert matrix_conditioning([[1, 0], [0, 100]])["result"] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        matrix_conditioning([[1, 2, 3]])
