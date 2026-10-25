import numpy as np
import pytest
from methods.best_approximation.gram_schmidt import gram_schmidt


@pytest.mark.parametrize("variant", ["modified", "classical"])
def test_orthogonal(variant):
    A = np.array([[1.0, 1], [1, 0], [0, 1]])
    r = gram_schmidt(A, variant=variant)["result"]
    assert np.allclose(r["Q"].T @ r["Q"], np.eye(2), atol=1e-12)
    assert np.allclose(r["Q"] @ r["R"], A, atol=1e-12)
    with pytest.raises(ValueError):
        gram_schmidt([[1, 1], [1, 1]])
