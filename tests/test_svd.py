import numpy as np
from scipy.linalg import svdvals
from methods.best_approximation.svd import svd_factorization


def test_rectangular():
    A = np.array([[1.0, 2, 3], [4, 5, 6]])
    r = svd_factorization(A)["result"]
    assert np.allclose((r["U"] * r["s"]) @ r["Vt"], A, atol=1e-12)
    assert np.allclose(r["s"], svdvals(A), atol=1e-12)
