import numpy as np
import pytest
from methods.linear_algebra.gaussian_elimination import gaussian_elimination


@pytest.mark.parametrize("pivoting", ["partial", "scaled"])
def test_swap(pivoting):
    A = np.array([[0.0, 2.0, 1.0], [1.0, -2.0, -3.0], [3.0, -1.0, 2.0]])
    b = A @ np.array([1.0, 2.0, 3.0])
    assert np.allclose(
        gaussian_elimination(A, b, pivoting=pivoting)["result"], [1, 2, 3], atol=1e-12
    )


def test_invalid():
    with pytest.raises(ValueError):
        gaussian_elimination([[1, 2], [2, 4]], [1, 2])
    with pytest.raises(ValueError):
        gaussian_elimination([[0, 1], [1, 0]], [1, 2], pivoting="none")
