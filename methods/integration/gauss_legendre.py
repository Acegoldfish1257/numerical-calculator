import numpy as np
from core.validation import checked, interval, integer, evaluate
from core.results import result


@checked
def gauss_legendre(f, a, b, *, nodes=4):
    """Q=(b-a)/2 sum w_i*f((a+b)/2+(b-a)*t_i/2).

    t_i are roots of Legendre P_n, w_i=2/((1-t_i^2)*P_n'(t_i)^2).
    Inputs: f, ordered bounds, 1..64 nodes. Output: integral, Legendre and
    transformed nodes/weights, n versus n+1 difference indicator. Polynomial
    degree <=2n-1 is exact up to rounding; endpoint singularities need care.
    Nodes/weights are computed by NumPy's Legendre routine.
    """
    a, b = interval(a, b)
    nodes = integer(nodes, "Nodes", 1, 64)
    z, w = np.polynomial.legendre.leggauss(nodes)
    x = a + (z + 1) * (b - a) / 2
    weights = w * (b - a) / 2
    y = np.array([evaluate(f, v) for v in x])
    q = float(weights @ y)
    zn, wn = np.polynomial.legendre.leggauss(nodes + 1)
    fine = float(
        (b - a)
        / 2
        * sum(wi * evaluate(f, a + (zi + 1) * (b - a) / 2) for zi, wi in zip(zn, wn))
    )
    return result(
        q,
        [
            dict(
                node=i + 1,
                legendre_node=zi,
                x=xi,
                weight=wi,
                value=yi,
                contribution=wi * yi,
            )
            for i, (zi, xi, wi, yi) in enumerate(zip(z, x, weights, y))
        ],
        abs(fine - q),
        error_kind="n vs n+1 node difference",
        legendre_degree=nodes,
        legendre_power_coefficients=np.polynomial.legendre.leg2poly([0] * nodes + [1]),
    )
