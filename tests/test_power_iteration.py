import numpy as np
import pytest
from methods.eigenvalues.power_iteration import power_iteration


def test_dominant():
    A = np.array([[2.0, 1], [1, 3]])
    r = power_iteration(A)
    assert r["converged"] and r["result"]["eigenvalue"] == pytest.approx(
        np.linalg.eigvalsh(A)[-1], abs=1e-10
    )
    with pytest.raises(ValueError):
        power_iteration(A, x0=[0, 0])
