import numpy as np
from core.validation import checked, points, array, scalar
from core.results import result


@checked
def cubic_spline(x, y, x_eval, *, boundary="natural", endpoint_slopes=None):
    """Construct C2 piecewise cubics by solving for node second derivatives M.

    Interior: h[i-1]M[i-1]+2(h[i-1]+h[i])M[i]+h[i]M[i+1]
    =6(slope[i]-slope[i-1]). Inputs: nodes, values, in-range evaluation points,
    natural or clamped boundary and optional endpoint slopes. Output: values,
    node M table, segment coefficients. Repeated nodes/extrapolation rejected.
    """
    x, y, _ = points(x, y)
    if len(x) > 200:
        raise ValueError("Use at most 200 spline nodes.")
    t = array(x_eval)
    if np.any(t < x[0]) or np.any(t > x[-1]):
        raise ValueError("Spline evaluation points must be inside the data interval.")
    n = len(x)
    h = np.diff(x)
    slopes = np.diff(y) / h
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        A[i, i - 1 : i + 2] = [h[i - 1], 2 * (h[i - 1] + h[i]), h[i]]
        rhs[i] = 6 * (slopes[i] - slopes[i - 1])
    if boundary == "natural":
        A[0, 0] = A[-1, -1] = 1
    elif boundary == "clamped":
        if endpoint_slopes is None or len(endpoint_slopes) != 2:
            raise ValueError("Clamped splines need two endpoint slopes.")
        left, right = map(scalar, endpoint_slopes)
        A[0, :2] = [2 * h[0], h[0]]
        A[-1, -2:] = [h[-1], 2 * h[-1]]
        rhs[0] = 6 * (slopes[0] - left)
        rhs[-1] = 6 * (right - slopes[-1])
    else:
        raise ValueError("Boundary must be natural or clamped.")
    M = np.linalg.solve(A, rhs)
    coeff = np.column_stack(
        (
            y[:-1],
            slopes - h * (2 * M[:-1] + M[1:]) / 6,
            M[:-1] / 2,
            np.diff(M) / (6 * h),
        )
    )
    indices = np.clip(np.searchsorted(x, t, side="right") - 1, 0, n - 2)
    dx = t - x[indices]
    c = coeff[indices]
    values = c[:, 0] + dx * (c[:, 1] + dx * (c[:, 2] + dx * c[:, 3]))
    return result(
        values,
        [dict(x=xi, y=yi, second_derivative=mi) for xi, yi, mi in zip(x, y, M)],
        coefficients=coeff,
        boundary=boundary,
    )
