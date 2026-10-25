"""Data-oriented calculator pages and SciPy cross-checks."""

import numpy as np
from scipy.interpolate import BarycentricInterpolator, KroghInterpolator, CubicSpline
from services.common import Field, Method, parse_vector
from core.expressions import parse_function
from methods.interpolation.lagrange import lagrange
from methods.interpolation.hermite import hermite
from methods.interpolation.error_bound import interpolation_error_bound
from methods.curve_fitting.least_squares_fit import least_squares_fit
from methods.curve_fitting.polynomial_approximation import polynomial_approximation
from methods.curve_fitting.cubic_spline import cubic_spline
from methods.curve_fitting.model_fit import fit_model

DATA_FIELDS = [
    Field("x", "Data x values", "0, 1, 2"),
    Field("y", "Data y values", "1, 2, 5"),
    Field("t", "Evaluation points", "0, 0.25, 0.5, 1, 1.5, 2"),
]


def data_run(p, kind):
    x, y = parse_vector(p["x"]), parse_vector(p["y"])
    t = parse_vector(p["t"]) if "t" in p else np.linspace(min(x), max(x), 100)
    grid = np.linspace(min(x), max(x), 300)
    if kind == "lagrange":
        r = lagrange(x, y, t)
        curve = lagrange(x, y, grid)["result"]
        reference = BarycentricInterpolator(x, y)(t)
    elif kind == "hermite":
        dy = parse_vector(p["dy"])
        r = hermite(x, y, dy, t)
        curve = hermite(x, y, dy, grid)["result"]
        order = np.argsort(x)
        reference = KroghInterpolator(
            np.repeat(x[order], 2), np.column_stack((y[order], dy[order])).ravel()
        )(t)
    elif kind == "spline":
        r = cubic_spline(x, y, t)
        curve = cubic_spline(x, y, grid)["result"]
        order = np.argsort(x)
        reference = CubicSpline(x[order], y[order], bc_type="natural")(t)
    elif kind == "model":
        r=fit_model(x,y,p["model"]); curve=fit_model(x,y,p["model"],grid)["result"]; reference=None
    else:
        basis = [parse_function(text.strip()) for text in p["basis"].split(";")]
        r = least_squares_fit(x, y, basis=basis, x_eval=t)
        curve = least_squares_fit(x, y, basis=basis, x_eval=grid)["result"]
        reference = None
    r["diagnostics"]["evaluations"] = [
        dict(x=float(ti), value=float(vi)) for ti, vi in zip(t, r["result"])
    ]
    if reference is not None:
        r["diagnostics"]["scipy_max_difference"] = float(
            np.max(np.abs(r["result"] - reference))
        )
    r["plot"] = dict(
        x=grid,
        series={"Interpolant / fit": curve},
        points=(x, y),
        xlabel="x",
        ylabel="y",
    )
    return r


def bound_run(p):
    x, t = parse_vector(p["x"]), parse_vector(p["t"])
    r = interpolation_error_bound(x, t, derivative_bound=p["M"], kind=p["kind"])
    r["plot"] = dict(x=t, series={"Conditional bound": r["result"]})
    return r


def polynomial_run(p):
    f = parse_function(p["f"])
    t = np.linspace(p["a"], p["b"], 200)
    r = polynomial_approximation(
        f, p["a"], p["b"], p["degree"], sample_count=p["samples"], x_eval=t
    )
    r["plot"] = dict(
        x=t, series={"f(x)": [f(v) for v in t], "Approximation": r["result"]}
    )
    return r


DATA_METHODS = [
    Method(
        "Interpolation",
        "Lagrange",
        "P(x) is the weighted sum of Lagrange basis polynomials. Distinct nodes required; high-degree and extrapolated values can be unreliable.",
        DATA_FIELDS,
        lambda p: data_run(p, "lagrange"),
    ),
    Method(
        "Interpolation",
        "Hermite",
        "Confluent divided differences match both function values and first derivatives at each node.",
        [*DATA_FIELDS, Field("dy", "Derivative values at data nodes", "0, 2, 4")],
        lambda p: data_run(p, "hermite"),
    ),
    Method(
        "Interpolation",
        "Interpolation error bound",
        "Enter M bounding the n-th derivative (Lagrange) or 2n-th derivative (Hermite), over all nodes and evaluation points.",
        [
            DATA_FIELDS[0],
            DATA_FIELDS[2],
            Field("M", "Derivative magnitude bound M", 2.0, "number"),
            Field("kind", "Interpolant", "lagrange", "choice", ("lagrange", "hermite")),
        ],
        bound_run,
    ),
    Method(
        "Interpolation",
        "Least-squares curve fit",
        "Fits a linear combination of your basis functions. The displayed error is training RMSE, not a prediction-error bound.",
        [
            *DATA_FIELDS,
            Field("basis", "Basis functions, separated by semicolons", "1; x; x^2"),
        ],
        lambda p: data_run(p, "fit"),
    ),
    Method("Approximation and fitting", "Non-polynomial model fit", "Fit y=a*b^x or y=a*x^b by logarithmic linearization, then display the converted coefficients and equation.", [*DATA_FIELDS[:2], Field("model", "Model", "exponential", "choice", ("exponential", "power"))], lambda p: data_run(p, "model")),
    Method(
        "Approximation and fitting",
        "Cubic spline",
        "Natural cubic spline: second derivative is zero at both endpoints. Evaluation must stay inside the data range.",
        DATA_FIELDS,
        lambda p: data_run(p, "spline"),
    ),
    Method(
        "Approximation and fitting",
        "Polynomial approximation",
        "Discrete least-squares approximation from equally spaced samples; the power basis uses a normalized coordinate in [-1,1].",
        [
            Field("f", "f(x)", "exp(x)"),
            Field("a", "Lower bound", 0.0, "number"),
            Field("b", "Upper bound", 1.0, "number"),
            Field("degree", "Polynomial degree", 3, "integer"),
            Field("samples", "Sample count", 50, "integer"),
        ],
        polynomial_run,
    ),
]
