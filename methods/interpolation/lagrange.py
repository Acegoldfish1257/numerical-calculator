import numpy as np
from core.validation import checked, points, array
from core.results import result


@checked
def lagrange(x, y, x_eval):
    """Evaluate P(t)=sum_i y_i product_{j!=i}(t-x_j)/(x_i-x_j).

    Inputs: distinct nodes x, matching y, evaluation array. Returns values,
    per-node basis contributions and no unprovided error bound. High degree,
    nearly repeated nodes and extrapolation can amplify roundoff/oscillation.
    """
    x, y, _ = points(x, y)
    t = array(x_eval, "Evaluation points")
    if len(x) > 30:
        raise ValueError("Use at most 30 interpolation nodes.")
    values = np.zeros_like(t)
    history = []
    for i in range(len(x)):
        basis = np.ones_like(t)
        for j in range(len(x)):
            if i != j:
                basis *= (t - x[j]) / (x[i] - x[j])
        contribution = y[i] * basis
        values += contribution
        history.append(
            dict(
                node=i,
                x=x[i],
                y=y[i],
                first_basis=basis[0],
                first_contribution=contribution[0],
            )
        )
    return result(values, history, formula="Lagrange basis", evaluation_points=t)
