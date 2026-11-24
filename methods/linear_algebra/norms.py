"""Vector L1, L2 and L-infinity norm calculator."""

import numpy as np

from core.results import result
from core.validation import array, checked


@checked
def vector_norms(values):
    """Compute ||v||_1, ||v||_2 and ||v||_infinity for a finite real vector.

    Input is a nonempty one-dimensional vector. Output is a mapping of all
    three norms plus absolute component contributions. Norms are nonnegative;
    a zero vector has all three equal to zero.
    """
    v = array(values, "Vector")
    absolute = np.abs(v)
    norms = {"L1": float(np.sum(absolute)), "L2": float(np.linalg.norm(v, 2)), "L∞": float(np.max(absolute))}
    return result(
        norms,
        [dict(component=i + 1, value=float(value), absolute_value=float(abs(value))) for i, value in enumerate(v)],
        None,
        True,
        "Vector norms computed.",
        formulas={"L1": "Σ|vᵢ|", "L2": "√Σ(vᵢ²)", "L∞": "max|vᵢ|"},
        error_kind="Error estimate",
    )
