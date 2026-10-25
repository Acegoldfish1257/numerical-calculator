import numpy as np
from core.validation import checked, matrix
from core.results import result


@checked
def svd_factorization(A):
    """Reduced SVD A=U diag(s) V^T using NumPy/LAPACK.

    Input: finite rectangular real A. Output: U,s,Vt, singular components and
    reconstruction residual. Rank uses eps*max(shape)*largest singular value.
    Singular vectors have arbitrary signs; repeated singular values permit
    different bases. Library backend is intentional, not a manual SVD solver.
    """
    A = matrix(A)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    total = float(np.sum(s * s))
    rows = []
    for i, v in enumerate(s):
        rows.append(
            dict(
                component=i + 1,
                singular_value=v,
                energy_fraction=0.0 if total == 0 else float(v * v / total),
            )
        )
    rank = int(np.sum(s > np.finfo(float).eps * max(A.shape) * s[0]))
    return result(
        dict(U=U, s=s, Vt=Vt),
        rows,
        float(np.linalg.norm(A - (U * s) @ Vt, np.inf)),
        error_kind="Reconstruction residual",
        rank=rank,
        backend="numpy.linalg.svd (LAPACK)",
    )
