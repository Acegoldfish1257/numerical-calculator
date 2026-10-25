import numpy as np
import pytest
from methods.linear_algebra.lu import lu_factorization


def test_reconstruction():
    A = np.array([[0.0, 2, 3], [4, 5, 6], [7, 8, 10]])
    r = lu_factorization(A)["result"]
    assert np.allclose(r["P"] @ A, r["L"] @ r["U"], atol=1e-12)
    assert np.allclose(np.diag(r["L"]), 1)
    with pytest.raises(ValueError):
        lu_factorization([[1, 1], [1, 1]])
