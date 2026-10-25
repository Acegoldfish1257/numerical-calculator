import numpy as np
import pytest
from methods.eigenvalues.inverse_iteration import inverse_iteration


def test_nearest():
    A = np.array([[2.0, 1], [1, 3]])
    r = inverse_iteration(A, shift=1.0)
    assert r["converged"] and r["result"]["eigenvalue"] == pytest.approx(
        np.linalg.eigvalsh(A)[0], abs=1e-10
    )
    with pytest.raises(ValueError):
        inverse_iteration([[1, 0], [0, 2]], shift=1)
