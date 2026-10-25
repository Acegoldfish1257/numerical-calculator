"""Method descriptors keep the Streamlit UI independent of algorithms."""

from services.common import Field, Method, function_plot
import numpy as np
from scipy import optimize
from core.expressions import parse_function
from methods.root_finding.bisection import bisection
from methods.root_finding.fixed_point import fixed_point
from methods.root_finding.newton_raphson import newton_raphson
from methods.root_finding.secant import secant
from methods.theory.calculus_theorems import theorem_check, taylor_remainder_bound
from methods.interpolation.finite_difference_formulas import difference_formula
from core.results import result


def root_run(p, algorithm=bisection):
    f = parse_function(p["f"])
    options = dict(xtol=p["tol"], ftol=p["tol"], max_iter=p["max_iter"])
    if algorithm is fixed_point:
        g=parse_function(p["g"]); gp=g.derivative()
        if abs(gp(p["x0"]))>=1: raise ValueError("Fixed-point iteration rejected: |g'(x0)| must be below 1 for local convergence.")
        r = fixed_point(g, p["x0"], f=f, **options)
    elif algorithm is newton_raphson:
        r = newton_raphson(f, f.derivative(), p.get("x0",(p["a"]+p["b"])/2), **options)
    elif algorithm is secant:
        r = secant(f, p["x0"], p["x1"], **options)
    else:
        if f(p["a"])*f(p["b"])>0: raise ValueError("Bisection requires f(a) and f(b) to have opposite signs.")
        r = bisection(f, p["a"], p["b"], **options)
    try:
        if algorithm is bisection:
            reference = optimize.brentq(f, p["a"], p["b"])
            reference_method = "scipy.optimize.brentq"
        else:
            reference = optimize.newton(f, p["x0"], fprime=f.derivative(), maxiter=200)
            reference_method = "scipy.optimize.newton (independent reference, may reach a different root)"
        r["diagnostics"].update(
            reference=reference,
            reference_difference=abs(r["result"] - reference),
            reference_method=reference_method,
        )
    except (ValueError, RuntimeError) as exc:
        r["diagnostics"]["reference_unavailable"] = str(exc)
    a, b = (p["a"], p["b"]) if "a" in p else (p["x0"] - 2, p["x0"] + 2)
    r["plot"] = function_plot(f, a, b)
    return r


ROOT_FIELDS = [
    Field(
        "f",
        "f(x)",
        "x^3-x-2",
        help="Use * for multiplication, ^ or ** for powers; sin, cos, tan, exp, log, sqrt, pi and e are supported.",
    ),
    Field("a", "Lower bound a", 1.0, "number"),
    Field("b", "Upper bound b", 2.0, "number"),
    Field("tol", "Absolute tolerance", 1e-8, "number"),
    Field("max_iter", "Maximum iterations", 100, "integer"),
]

METHODS = [
    Method(
        "Root finding",
        "Bisection",
        "c = (a + b) / 2. Requires a continuous function and opposite endpoint signs. Convergence requires both a small bracket and a small residual.",
        ROOT_FIELDS,
        root_run,
    )
]

ITER_FIELDS = [
    ROOT_FIELDS[0],
    Field("x0", "Initial guess x0", 1.5, "number"),
    *ROOT_FIELDS[3:],
]
G_FIELD = Field(
    "g",
    "Iteration mapping g(x)",
    "exp(log(x+2)/3)",
    help="Choose a rearrangement x=g(x); local convergence usually needs |g'(x)|<1.",
)
METHODS.extend(
    [
        Method(
            "Root finding",
            "Fixed-point iteration",
            "x[k+1] = g(x[k]). The original f(x) residual is checked as well as the iterate difference.",
            [ITER_FIELDS[0], G_FIELD, *ITER_FIELDS[1:]],
            lambda p: root_run(p, fixed_point),
        ),
        Method(
            "Root finding",
            "Newton–Raphson",
            "x[k+1] = x[k] - f(x[k])/f'(x[k]). Derivative is obtained symbolically. Zero derivatives and poor guesses can cause failure.",
            ITER_FIELDS,
            lambda p: root_run(p, newton_raphson),
        ),
        Method(
            "Root finding",
            "Secant",
            "Uses the slope through two previous iterates. Equal function values and guesses outside the domain can cause failure.",
            [
                *ITER_FIELDS[:2],
                Field("x1", "Second guess x1", 2.0, "number"),
                *ITER_FIELDS[2:],
            ],
            lambda p: root_run(p, secant),
        ),
    ]
)


def compare_roots(p):
    parse_function(p["f"])  # A malformed shared function is a page-level input error.
    rows = []
    histories = {}
    for name, algorithm in (
        ("Bisection", bisection),
        ("Fixed point", fixed_point),
        ("Newton", newton_raphson),
        ("Secant", secant),
    ):
        try:
            r = root_run(p, algorithm)
            histories[name] = r["history"]
            rows.append(
                dict(
                    method=name,
                    root=r["result"],
                    iterations=r["history"][-1]["iteration"],
                    residual=r["diagnostics"].get("residual"),
                    scipy_difference=r["diagnostics"].get("reference_difference"),
                    converged=r["converged"],
                    status=r["message"],
                )
            )
        except ValueError as exc:
            rows.append(
                dict(
                    method=name,
                    root=None,
                    iterations=None,
                    residual=None,
                    scipy_difference=None,
                    converged=False,
                    status=str(exc),
                )
            )
    r = result(
        "Four-method comparison",
        rows,
        converged=all(row["converged"] for row in rows),
        message="Comparison complete; inspect each method's status",
        method_histories=histories,
        note="Residuals are |f(x)|. A small residual is not a root-error bound. Methods may converge to different roots.",
    )
    if histories:
        length = max(len(h) for h in histories.values())
        r["plot"] = dict(
            x=list(range(1, length + 1)),
            series={
                name: [abs(row.get("f(x)", row.get("residual", 0))) for row in h]
                + [np.nan] * (length - len(h))
                for name, h in histories.items()
            },
            xlabel="Recorded iteration",
            ylabel="Absolute residual",
        )
    return r


METHODS.append(
    Method(
        "Root finding",
        "Compare all four",
        "Runs the same f(x) through all four solvers. Supply a valid g(x), bracket, and initial guesses; failures are reported separately.",
        [
            *ROOT_FIELDS,
            G_FIELD,
            Field("x0", "Initial guess x0", 1.5, "number"),
            Field("x1", "Second guess x1", 2.0, "number"),
        ],
        compare_roots,
    )
)

from services.data_pages import DATA_METHODS

METHODS.extend(DATA_METHODS)
from services.calculus_pages import CALCULUS_METHODS

METHODS.extend(CALCULUS_METHODS)
from services.advanced_pages import ADVANCED_METHODS

METHODS.extend(ADVANCED_METHODS)
METHODS.extend([
    Method("Calculus theory", "Continuity and IVT/Bolzano", "Checks sampled continuity and the Intermediate Value/Bolzano sign-change condition; samples cannot prove continuity.", [ROOT_FIELDS[0],Field("a","Lower bound",0.,"number"),Field("b","Upper bound",1.,"number"),Field("theorem","Concept","continuity","choice",("continuity","bolzano","ivt"))], lambda p: theorem_check(parse_function(p["f"]),p["a"],p["b"],p["theorem"])),
    Method("Calculus theory", "Rolle and MVT", "Checks endpoint equality for Rolle or derivative/secant agreement for the Mean Value Theorem.", [ROOT_FIELDS[0],Field("df","Derivative df(x)","1"),Field("a","Lower bound",0.,"number"),Field("b","Upper bound",1.,"number"),Field("theorem","Concept","rolle","choice",("rolle","mvt"))], lambda p: theorem_check(parse_function(p["f"]),p["a"],p["b"],p["theorem"],parse_function(p["df"]))),
    Method("Calculus theory", "Taylor remainder bound", "Conditional bound M|x-c|^(n+1)/(n+1)! for Taylor's remainder.", [Field("M","Derivative bound M",2.,"number"),Field("degree","Taylor degree",2,"integer"),Field("center","Expansion center",0.,"number"),Field("x","Evaluation point",1.,"number")], lambda p: taylor_remainder_bound(p["M"],p["degree"],p["center"],p["x"])),
    Method("Interpolation", "Newton difference formulas", "Equally spaced Newton forward, backward, and central Gauss-forward difference tables.", [Field("x","Nodes","0,1,2,3"),Field("y","Values","0,1,4,9"),Field("point","Evaluation point",1.5,"number"),Field("formula","Formula","forward","choice",("forward","backward","gauss_forward"))], lambda p: difference_formula(__import__('services.common',fromlist=['parse_vector']).parse_vector(p["x"]),__import__('services.common',fromlist=['parse_vector']).parse_vector(p["y"]),p["point"],p["formula"]))
])
