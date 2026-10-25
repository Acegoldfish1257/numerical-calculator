import numpy as np
import pytest
from methods.interpolation.lagrange import lagrange


def test_polynomial():
    t = np.linspace(-1, 3, 9)
    assert np.allclose(
        lagrange([0, 1, 2], [1, 2, 5], t)["result"], t * t + 1, atol=1e-12
    )


def test_invalid():
    with pytest.raises(ValueError):
        lagrange([0, 0], [1, 2], [1])
