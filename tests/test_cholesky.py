import numpy as np
import pytest
from methods.linear_algebra.cholesky import cholesky_factorization


def test_spd():
    A = np.array([[4.0, 2], [2, 3]])
    assert np.allclose(
        cholesky_factorization(A)["result"], np.linalg.cholesky(A), atol=1e-12
    )
    with pytest.raises(ValueError):
        cholesky_factorization([[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        cholesky_factorization([[1, 2], [0, 1]])
