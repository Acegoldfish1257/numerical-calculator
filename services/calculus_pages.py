"""Quadrature, differentiation and ODE pages."""

import warnings
import numpy as np
from scipy.integrate import quad, solve_ivp, IntegrationWarning
from core.expressions import parse_function
from services.common import Field, Method, function_plot
from methods.integration.midpoint import midpoint
from methods.integration.trapezoidal import trapezoidal
from methods.integration.simpson import simpson
from methods.integration.simpson_three_eighth import simpson_three_eighth
from methods.integration.newton_cotes import newton_cotes
from methods.integration.gauss_legendre import gauss_legendre
from methods.differentiation.finite_difference import finite_difference
from methods.ode_solvers.rk4 import rk4

INTEGRAL_FIELDS = [
    Field("f", "f(x)", "exp(x)"),
    Field("a", "Lower limit", 0.0, "number"),
    Field("b", "Upper limit", 1.0, "number"),
]


def integration_run(p, fn):
    f = parse_function(p["f"])
    if fn is newton_cotes:
        r = fn(f, p["a"], p["b"], order=p["order"], panels=p["panels"])
    elif fn is gauss_legendre:
        r = fn(f, p["a"], p["b"], nodes=p["nodes"])
    else:
        r = fn(f, p["a"], p["b"], p["n"])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", IntegrationWarning)
        reference, err = quad(f, p["a"], p["b"], epsabs=1e-11, epsrel=1e-11)
    r["diagnostics"].update(
        scipy_quad=reference,
        scipy_error_estimate=err,
        reference_difference=abs(r["result"] - reference),
        reference_warnings=[str(w.message) for w in recorded],
    )
    r["plot"] = function_plot(f, p["a"], p["b"])
    r["plot"]["points"] = (
        [row["x"] for row in r["history"]],
        [row["value"] for row in r["history"]],
    )
    return r


def differentiation_run(p):
    f = parse_function(p["f"])
    r = finite_difference(f, p["x"], p["h"], formula=p["formula"])
    reference = f.derivative(order=r["diagnostics"]["derivative_order"])(p["x"])
    r["diagnostics"].update(
        symbolic_derivative=reference,
        reference_difference=abs(r["result"] - reference),
        stencil=r["history"],
    )
    rows = []
    for h in np.logspace(-1, -9, 17):
        try:
            q = finite_difference(f, p["x"], float(h), formula=p["formula"])["result"]
            rows.append(dict(h=h, approximation=q, absolute_error=abs(q - reference)))
        except ValueError:
            continue
    if rows:
        r["history"] = rows
        r["plot"] = dict(
            x=[row["h"] for row in rows],
            series={
                "Absolute error": [max(row["absolute_error"], 1e-18) for row in rows]
            },
            xlabel="Step size h",
            ylabel="Absolute error (plot floor 1e-18)",
            logx=True,
            logy=True,
        )
    return r


def ode_run(p):
    f = parse_function(p["f"], ("t", "y"))
    r = rk4(f, p["t0"], p["y0"], p["end"], p["h"])
    t, y = r["diagnostics"]["t"], r["diagnostics"]["y"]
    reference = solve_ivp(
        lambda t, y: [f(t, float(y[0]))],
        (p["t0"], p["end"]),
        [p["y0"]],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )
    if not reference.success or reference.y.shape[1] != len(t):
        r["diagnostics"]["reference_unavailable"] = reference.message
        r["plot"] = dict(x=t, series={"RK4": y}, xlabel="t", ylabel="y")
    else:
        r["diagnostics"]["max_difference_from_solve_ivp"] = float(
            np.max(np.abs(y - reference.y[0]))
        )
        r["plot"] = dict(
            x=t,
            series={"RK4": y, "SciPy solve_ivp": reference.y[0]},
            xlabel="t",
            ylabel="y",
        )
    return r


CALCULUS_METHODS = []
for name, fn, n, description in [
    (
        "Midpoint",
        midpoint,
        100,
        "Composite midpoint: h times the sum of midpoint function values. Second-order accuracy for smooth functions.",
    ),
    (
        "Trapezoidal",
        trapezoidal,
        100,
        "Composite trapezoids: h times the weighted node sum, with endpoint weights 1/2.",
    ),
    (
        "Simpson 1/3",
        simpson,
        20,
        "Composite Simpson 1/3: weights 1,4,2,...,4,1 times h/3. Requires an even subinterval count.",
    ),
]:
    CALCULUS_METHODS.append(
        Method(
            "Integration",
            name,
            description,
            [*INTEGRAL_FIELDS, Field("n", "Subinterval count", n, "integer")],
            lambda p, fn=fn: integration_run(p, fn),
            group="Simpson" if fn is simpson else "",
        )
    )
CALCULUS_METHODS.extend(
    [
        Method(
            "Integration",
            "Newton–Cotes",
            "Closed rules with 2 to 7 equally spaced nodes per panel. Displays computed weights and each contribution.",
            [
                *INTEGRAL_FIELDS,
                Field("order", "Order (nodes per panel minus one)", 4, "integer"),
                Field("panels", "Panel count", 4, "integer"),
            ],
            lambda p: integration_run(p, newton_cotes),
        ),
        Method(
            "Integration",
            "Gauss–Legendre",
            "Nodes are roots of the Legendre polynomial P_n. An n-node rule is exact through polynomial degree 2n-1, up to rounding.",
            [*INTEGRAL_FIELDS, Field("nodes", "Number of nodes", 4, "integer")],
            lambda p: integration_run(p, gauss_legendre),
        ),
        Method(
            "Differentiation and ODEs",
            "Finite differences",
            "Forward/backward first differences are O(h); central first and second differences are O(h²). The plot shows truncation error versus roundoff as h changes.",
            [
                Field("f", "f(x)", "sin(x)"),
                Field("x", "Evaluation point", 0.5, "number"),
                Field("h", "Step size", 0.001, "number"),
                Field(
                    "formula",
                    "Difference formula",
                    "central_first",
                    "choice",
                    (
                        "central_first",
                        "forward_first",
                        "backward_first",
                        "central_second",
                    ),
                ),
            ],
            differentiation_run,
        ),
        Method(
            "Differentiation and ODEs",
            "Classical RK4",
            "Solves a scalar initial-value problem y'=f(t,y). Four slopes per step; final step is shortened to reach the endpoint. Explicit RK4 may be unsuitable for stiff problems.",
            [
                Field("f", "f(t,y)", "y"),
                Field("t0", "Initial time", 0.0, "number"),
                Field("y0", "Initial value y(t0)", 1.0, "number"),
                Field("end", "Final time", 1.0, "number"),
                Field("h", "Step size", 0.1, "number"),
            ],
            ode_run,
        ),
    ]
)
CALCULUS_METHODS.append(Method("Integration", "Simpson 3/8", "Composite Simpson 3/8; n must be a multiple of 3.", [*INTEGRAL_FIELDS, Field("n", "Subinterval count", 12, "integer")], lambda p: integration_run(p, simpson_three_eighth), group="Simpson"))
