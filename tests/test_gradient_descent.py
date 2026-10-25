import numpy as np
import pytest
from methods.optimization.gradient_descent import gradient_descent


def test_quadratic():
    r = gradient_descent(
        lambda x: (x[0] - 2) ** 2 + 2 * (x[1] + 1) ** 2,
        lambda x: np.array([2 * (x[0] - 2), 4 * (x[1] + 1)]),
        [0, 0],
    )
    assert r["converged"] and np.allclose(r["result"], [2, -1], atol=1e-8)
    with pytest.raises(ValueError):
        gradient_descent(lambda x: 0, lambda x: [1, 2], [1])
