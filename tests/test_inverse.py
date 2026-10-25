import numpy as np
import pytest
from methods.linear_algebra.inverse import matrix_inverse


def test_inverse():
    A = np.array([[4.0, 7], [2, 6]])
    assert np.allclose(matrix_inverse(A)["result"], np.linalg.inv(A), atol=1e-12)
    with pytest.raises(ValueError):
        matrix_inverse([[1, 2], [2, 4]])
