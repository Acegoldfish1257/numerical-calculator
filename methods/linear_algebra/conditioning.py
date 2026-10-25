import numpy as np
from core.validation import checked, matrix
from core.results import result


@checked
def matrix_conditioning(A, *, norm="2"):
    """Compute kappa_p(A)=||A||_p ||A^-1||_p; kappa_2=sigma_max/sigma_min.

    Inputs: square finite A and norm 1,2,inf. Output: condition number and
    singular-value table (NumPy backend). Singular matrices have infinite
    condition number; a large number signals sensitivity, not solver error.
    """
    A = matrix(A, square=True)
    norms = {"1": 1, "2": 2, "inf": np.inf}
    if norm not in norms:
        raise ValueError("Norm must be 1, 2, or inf.")
    singular = np.linalg.svd(A, compute_uv=False)
    value = float(np.linalg.cond(A, p=norms[norm]))
    return result(
        value,
        [dict(index=i + 1, singular_value=v) for i, v in enumerate(singular)],
        norm=norm,
        backend="numpy.linalg.cond",
        note="Sensitivity measure, not an error estimate",
    )
