import numpy as np
import pytest
from methods.best_approximation.least_squares import least_squares


@pytest.mark.parametrize("solver", ["qr", "svd"])
def test_fit(solver):
    A = np.array([[1.0, 0], [1, 1], [1, 2], [1, 3]])
    b = np.array([1.0, 2, 2, 4])
    assert np.allclose(
        least_squares(A, b, solver=solver)["result"],
        np.linalg.lstsq(A, b, rcond=None)[0],
        atol=1e-12,
    )


def test_rank():
    with pytest.raises(ValueError):
        least_squares([[1, 1], [2, 2]], [1, 2])
    assert np.allclose(
        least_squares([[1, 1], [2, 2]], [1, 2], solver="svd")["result"], [0.5, 0.5]
    )
