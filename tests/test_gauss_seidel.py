import numpy as np
import pytest
from methods.linear_algebra.gauss_seidel import gauss_seidel


def test_solve_and_diverge():
    A = np.array([[4.0, 1], [1, 3]])
    b = np.array([1.0, 2.0])
    r = gauss_seidel(A, b)
    assert r["converged"] and np.allclose(r["result"], np.linalg.solve(A, b), atol=1e-8)
    assert not gauss_seidel([[1, 3], [3, 1]], [1, 2], max_iter=10)["converged"]
    with pytest.raises(ValueError):
        gauss_seidel([[0, 1], [1, 1]], [1, 2])
