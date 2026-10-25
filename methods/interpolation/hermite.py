import numpy as np
from core.validation import checked, points, array
from core.results import result


@checked
def hermite(x, y, dy, x_eval):
    """Confluent Newton interpolation with repeated nodes and f' values.

    Inputs: n distinct x, n y and n dy, evaluation points. Output: polynomial
    values of degree <=2n-1, divided-difference steps and coefficients.
    Smooth data assumed; high degrees and close nodes are ill-conditioned.
    """
    x, y, order = points(x, y)
    dy = array(dy, "Derivatives")
    if len(dy) != len(x) or len(x) > 15:
        raise ValueError("Supply one derivative per node, with at most 15 nodes.")
    dy = dy[order]
    t = array(x_eval, "Evaluation points")
    z = np.repeat(x, 2)
    Q = np.zeros((len(z), len(z)))
    Q[:, 0] = np.repeat(y, 2)
    for i in range(len(x)):
        Q[2 * i + 1, 1] = dy[i]
        if i:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])
    for j in range(2, len(z)):
        for i in range(j, len(z)):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])
    coefficients = np.diag(Q)
    values = np.full_like(t, coefficients[-1])
    for i in range(len(z) - 2, -1, -1):
        values = values * (t - z[i]) + coefficients[i]
    history = [
        dict(order=i, node=z[i], coefficient=coefficients[i]) for i in range(len(z))
    ]
    return result(values, history, coefficients=coefficients, evaluation_points=t)
