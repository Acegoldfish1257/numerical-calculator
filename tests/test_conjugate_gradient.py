import numpy as np
import pytest
from methods.optimization.conjugate_gradient import conjugate_gradient


def test_spd():
    A = np.array([[4.0, 1], [1, 3]])
    b = np.array([1.0, 2])
    r = conjugate_gradient(A, b)
    assert r["converged"] and np.allclose(
        r["result"], np.linalg.solve(A, b), atol=1e-12
    )
    with pytest.raises(ValueError):
        conjugate_gradient([[1, 2], [2, 1]], [1, 2])
