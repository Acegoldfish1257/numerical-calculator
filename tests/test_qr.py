import numpy as np
import pytest
from methods.best_approximation.qr import qr_factorization


def test_qr():
    A = np.array([[1.0, 2], [3, 4], [5, 6]])
    r = qr_factorization(A)["result"]
    assert np.allclose(r["Q"] @ r["R"], A, atol=1e-12)
    assert np.allclose(r["Q"].T @ r["Q"], np.eye(2), atol=1e-12)
    with pytest.raises(ValueError):
        qr_factorization([[1, 2]])
