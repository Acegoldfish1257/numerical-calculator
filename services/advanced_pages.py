"""Arithmetic, matrix, approximation and optimization page orchestration."""

import numpy as np
from scipy import linalg, optimize
from services.common import Field, Method, parse_vector, parse_matrix
from core.expressions import parse_function
from methods.computer_arithmetic.floating_point import floating_point_representation
from methods.computer_arithmetic.arithmetic import floating_point_arithmetic
from methods.computer_arithmetic.error_analysis import numerical_error
from methods.computer_arithmetic.significant_digits import significant_digits
from methods.curve_fitting.taylor import taylor_approximation
from methods.linear_algebra.gaussian_elimination import gaussian_elimination
from methods.linear_algebra.lu import lu_factorization
from methods.linear_algebra.cholesky import cholesky_factorization
from methods.linear_algebra.conditioning import matrix_conditioning
from methods.linear_algebra.inverse import matrix_inverse
from methods.linear_algebra.jacobi import jacobi
from methods.linear_algebra.gauss_seidel import gauss_seidel
from methods.best_approximation.gram_schmidt import gram_schmidt
from methods.best_approximation.qr import qr_factorization
from methods.best_approximation.svd import svd_factorization
from methods.best_approximation.least_squares import least_squares
from methods.optimization.gradient_descent import gradient_descent
from methods.optimization.newton import newton_optimization
from methods.optimization.conjugate_gradient import conjugate_gradient
from methods.eigenvalues.power_iteration import power_iteration
from methods.eigenvalues.inverse_iteration import inverse_iteration


def arithmetic_run(p, kind):
    if kind == "representation":
        r = floating_point_representation(p["value"], precision=p["precision"])
    elif kind == "arithmetic":
        r = floating_point_arithmetic(
            p["a"], p["b"], operation=p["operation"], precision=p["precision"]
        )
    elif kind == "digits":
        r = significant_digits(p["value"], p["digits"], mode=p["mode"])
    else:
        r = numerical_error(p["approximate"], p["reference"])
    r["plot"] = dict(
        kind="bar",
        x=["Absolute error"],
        series={"Error": [r["error_estimate"]]},
        ylabel="Absolute error",
    )
    return r


def taylor_run(p):
    f = parse_function(p["f"])
    t = parse_vector(p["t"])
    r = taylor_approximation(f, p["center"], p["degree"], t)
    order = np.argsort(t)
    r["plot"] = dict(
        x=t[order],
        series={
            "Taylor polynomial": r["result"][order],
            "f(x)": r["diagnostics"]["actual_values"][order],
        },
    )
    return r


def matrix_run(p, fn):
    A = parse_matrix(p["A"])
    if fn is gaussian_elimination:
        r = fn(A, parse_vector(p["b"]), pivoting=p["pivoting"])
    elif fn in (jacobi, gauss_seidel, conjugate_gradient):
        r = fn(
            A,
            parse_vector(p["b"]),
            x0=parse_vector(p["x0"]),
            tol=p["tol"],
            max_iter=p["limit"],
        )
    elif fn is least_squares:
        r = fn(A, parse_vector(p["b"]), solver=p["solver"])
    elif fn in (power_iteration, inverse_iteration):
        options = dict(x0=parse_vector(p["x0"]), tol=p["tol"], max_iter=p["limit"])
        if fn is inverse_iteration:
            options["shift"] = p["shift"]
        r = fn(A, **options)
    elif fn is gram_schmidt:
        r = fn(A, variant=p["variant"])
    elif fn is matrix_conditioning:
        r = fn(A, norm=p["norm"])
    else:
        r = fn(A)
    value = r["result"]
    if fn in (
        gaussian_elimination,
        jacobi,
        gauss_seidel,
        conjugate_gradient,
        least_squares,
    ):
        b = parse_vector(p["b"])
        reference = linalg.lstsq(A, b)[0] if fn is least_squares else linalg.solve(A, b)
        r["diagnostics"]["scipy_solution_difference"] = float(
            np.linalg.norm(value - reference)
        )
        if fn in (jacobi, gauss_seidel, conjugate_gradient):
            r["plot"] = dict(
                x=[row["iteration"] for row in r["history"]],
                series={"Residual": [row["residual"] for row in r["history"]]},
                xlabel="Iteration",
                ylabel="Residual norm",
            )
        else:
            r["plot"] = dict(
                kind="bar",
                x=[str(i + 1) for i in range(len(value))],
                series={"Solution components": value},
                xlabel="Component",
            )
    elif fn in (power_iteration, inverse_iteration):
        eigenvalues = linalg.eigvalsh(A)
        nearest = float(
            eigenvalues[np.argmin(np.abs(eigenvalues - value["eigenvalue"]))]
        )
        r["diagnostics"].update(
            nearest_scipy_eigenvalue=nearest,
            reference_difference=abs(nearest - value["eigenvalue"]),
        )
        r["plot"] = dict(
            x=[row["iteration"] for row in r["history"]],
            series={"Eigenpair residual": [row["residual"] for row in r["history"]]},
            xlabel="Iteration",
            ylabel="Residual norm",
        )
    elif fn is matrix_conditioning:
        r["plot"] = dict(
            kind="bar",
            x=[str(row["index"]) for row in r["history"]],
            series={"Singular values": [row["singular_value"] for row in r["history"]]},
        )
    elif fn is svd_factorization:
        r["diagnostics"]["scipy_singular_value_difference"] = float(
            np.max(np.abs(value["s"] - linalg.svdvals(A)))
        )
        r["plot"] = dict(
            kind="bar",
            x=[str(i + 1) for i in range(len(value["s"]))],
            series={"Singular values": value["s"]},
        )
    else:
        if fn is matrix_inverse:
            r["diagnostics"]["scipy_inverse_difference"] = float(
                np.linalg.norm(value - linalg.inv(A))
            )
        if fn is cholesky_factorization:
            r["diagnostics"]["scipy_factor_difference"] = float(
                np.linalg.norm(value - linalg.cholesky(A, lower=True))
            )
        shown = (
            value
            if isinstance(value, np.ndarray)
            else value.get("Q", value.get("U", A))
        )
        r["plot"] = dict(kind="matrix", matrix=shown, title="Computed matrix / factor")
    return r


def optimization_run(p, fn):
    parsed = parse_function(p["f"], ("x", "y"))
    start = parse_vector(p["x0"])
    if len(start) != 2:
        raise ValueError("This objective page needs two initial values: x and y.")
    first = [parsed.derivative(variable=i) for i in range(2)]
    f = lambda x: parsed(*x)
    grad = lambda x: np.array([d(*x) for d in first])
    if fn is newton_optimization:
        second = [[first[i].derivative(variable=j) for j in range(2)] for i in range(2)]
        hessian = lambda x: np.array([[d(*x) for d in row] for row in second])
        r = fn(f, grad, hessian, start, tol=p["tol"], max_iter=p["limit"])
    else:
        r = fn(
            f, grad, start, initial_step=p["step"], tol=p["tol"], max_iter=p["limit"]
        )
    reference = optimize.minimize(f, start, jac=grad, method="BFGS", tol=1e-10)
    r["diagnostics"].update(
        scipy_objective=float(reference.fun),
        scipy_status=str(reference.message),
        scipy_success=bool(reference.success),
        point_difference=float(np.linalg.norm(r["result"] - reference.x)),
    )
    r["plot"] = dict(
        x=[row["iteration"] for row in r["history"]],
        series={"Objective": [row["objective"] for row in r["history"]]},
        xlabel="Iteration",
        ylabel="Objective value",
    )
    return r


PRECISION = Field(
    "precision", "Binary precision", "float64", "choice", ("float64", "float32")
)
MATRIX = Field("A", "Matrix A (one row per line)", "4, 1\n1, 3", "area")
RHS = Field("b", "Vector b", "1, 2")
ITER = [
    Field("x0", "Initial vector", "0, 0"),
    Field("tol", "Residual tolerance", 1e-8, "number"),
    Field("limit", "Maximum iterations", 500, "integer"),
]
ADVANCED_METHODS = [
    Method(
        "Computer arithmetic",
        "Floating-point representation",
        "Inspect IEEE-754 bits, machine epsilon and representation error. Subnormal values use a different leading-bit convention.",
        [Field("value", "Decimal input", "0.1"), PRECISION],
        lambda p: arithmetic_run(p, "representation"),
    ),
    Method(
        "Computer arithmetic",
        "Floating-point arithmetic",
        "Compare selected-precision binary arithmetic with a 100-digit decimal reference.",
        [
            Field("a", "Operand a", "0.1"),
            Field("b", "Operand b", "0.2"),
            Field(
                "operation",
                "Operation",
                "add",
                "choice",
                ("add", "subtract", "multiply", "divide"),
            ),
            PRECISION,
        ],
        lambda p: arithmetic_run(p, "arithmetic"),
    ),
    Method(
        "Computer arithmetic",
        "Rounding and truncation",
        "Decimal significant digits: ties-to-even rounding or truncation toward zero. Shows normalized scientific notation.",
        [
            Field("value", "Decimal input", "123.456"),
            Field("digits", "Significant digits", 4, "integer"),
            Field("mode", "Mode", "round", "choice", ("round", "truncate")),
        ],
        lambda p: arithmetic_run(p, "digits"),
    ),
    Method(
        "Computer arithmetic",
        "Numerical errors",
        "Absolute, relative and percentage errors against a supplied reference. Relative error is undefined when the reference is zero.",
        [
            Field("approximate", "Approximation", "3.14"),
            Field("reference", "Reference value", "3.141592653589793"),
        ],
        lambda p: arithmetic_run(p, "error"),
    ),
    Method(
        "Approximation and fitting",
        "Taylor approximation",
        "P_n(x)=sum f^(k)(c)(x-c)^k/k!. Displays coefficients and sampled errors against the original function.",
        [
            Field("f", "f(x)", "exp(x)"),
            Field("center", "Expansion center", 0.0, "number"),
            Field("degree", "Degree", 4, "integer"),
            Field("t", "Evaluation points", "0, 0.25, 0.5, 0.75, 1"),
        ],
        taylor_run,
    ),
    Method(
        "Linear algebra",
        "Gaussian elimination",
        "Elimination and back substitution, with optional partial or scaled partial pivoting. A residual is not a forward-error bound.",
        [
            MATRIX,
            RHS,
            Field(
                "pivoting",
                "Pivoting strategy",
                "partial",
                "choice",
                ("partial", "scaled", "none"),
            ),
        ],
        lambda p: matrix_run(p, gaussian_elimination),
    ),
    Method(
        "Linear algebra",
        "LU factorization",
        "Partial-pivoted Doolittle factorization P A = L U. Displays factors and row-swap history.",
        [MATRIX],
        lambda p: matrix_run(p, lu_factorization),
    ),
    Method(
        "Linear algebra",
        "Cholesky factorization",
        "A = L Lᵀ for symmetric positive-definite matrices. Displays each computed entry.",
        [MATRIX],
        lambda p: matrix_run(p, cholesky_factorization),
    ),
    Method(
        "Linear algebra",
        "Matrix inverse",
        "Solves A x_j = e_j for each inverse column using partial-pivoted Gaussian elimination.",
        [MATRIX],
        lambda p: matrix_run(p, matrix_inverse),
    ),
    Method(
        "Linear algebra",
        "Matrix conditioning",
        "κ(A)=||A|| ||A⁻¹||. For the 2-norm this is the ratio of largest to smallest singular values. NumPy backend.",
        [MATRIX, Field("norm", "Matrix norm", "2", "choice", ("2", "1", "inf"))],
        lambda p: matrix_run(p, matrix_conditioning),
    ),
    Method(
        "Linear algebra",
        "Jacobi",
        "Each new component uses only the previous iterate. Strict diagonal dominance is sufficient for convergence.",
        [MATRIX, RHS, *ITER],
        lambda p: matrix_run(p, jacobi),
    ),
    Method(
        "Linear algebra",
        "Gauss–Seidel",
        "Components use newly updated values as soon as they are available. Convergence is not guaranteed for arbitrary matrices.",
        [MATRIX, RHS, *ITER],
        lambda p: matrix_run(p, gauss_seidel),
    ),
    Method(
        "Best approximation",
        "Linear least squares",
        "Minimizes ||Ax-b||₂ with Householder QR, or NumPy SVD for a minimum-norm solution.",
        [MATRIX, RHS, Field("solver", "Solver", "qr", "choice", ("qr", "svd"))],
        lambda p: matrix_run(p, least_squares),
    ),
    Method(
        "Best approximation",
        "Gram–Schmidt",
        "Constructs orthonormal columns; modified Gram–Schmidt usually preserves orthogonality better than the classical form.",
        [
            MATRIX,
            Field(
                "variant", "Variant", "modified", "choice", ("modified", "classical")
            ),
        ],
        lambda p: matrix_run(p, gram_schmidt),
    ),
    Method(
        "Best approximation",
        "QR factorization",
        "Householder reflections construct a reduced QR factorization. Requires rows ≥ columns.",
        [MATRIX],
        lambda p: matrix_run(p, qr_factorization),
    ),
    Method(
        "Best approximation",
        "SVD",
        "Reduced singular value decomposition A = U diag(s) Vᵀ. This page explicitly uses NumPy/LAPACK and verifies against SciPy.",
        [MATRIX],
        lambda p: matrix_run(p, svd_factorization),
    ),
    Method(
        "Optimization",
        "Conjugate gradient",
        "Minimizes ½xᵀAx-bᵀx for a symmetric positive-definite matrix. This is linear CG for a quadratic objective.",
        [MATRIX, RHS, *ITER],
        lambda p: matrix_run(p, conjugate_gradient),
    ),
]
OPT_FIELDS = [
    Field("f", "Objective f(x,y)", "(x-2)^2 + 2*(y+1)^2"),
    Field("x0", "Initial x, y", "0, 0"),
    Field("tol", "Gradient tolerance", 1e-8, "number"),
    Field("limit", "Maximum iterations", 500, "integer"),
]
ADVANCED_METHODS.extend(
    [
        Method(
            "Optimization",
            "Gradient descent",
            "Unconstrained two-variable objective with symbolic gradient and Armijo backtracking. A stationary point need not be a global minimum.",
            [*OPT_FIELDS, Field("step", "Initial trial step", 1.0, "number")],
            lambda p: optimization_run(p, gradient_descent),
        ),
        Method(
            "Optimization",
            "Newton optimization",
            "Uses the symbolic Hessian and an Armijo line search. This implementation requires a positive-definite Hessian at every iterate.",
            OPT_FIELDS,
            lambda p: optimization_run(p, newton_optimization),
        ),
        Method(
            "Eigenvalues",
            "Power iteration",
            "Real symmetric matrices. Targets a largest-magnitude eigenpair if the initial vector has the required component; convergence checks the eigenpair residual.",
            [MATRIX, Field("x0", "Initial vector", "1, 1"), *ITER[1:]],
            lambda p: matrix_run(p, power_iteration),
        ),
        Method(
            "Eigenvalues",
            "Inverse iteration",
            "Real symmetric matrices. Targets an eigenvalue near the fixed shift; the shifted matrix must be nonsingular.",
            [
                MATRIX,
                Field("x0", "Initial vector", "1, 1"),
                Field("shift", "Shift", 0.0, "number"),
                *ITER[1:],
            ],
            lambda p: matrix_run(p, inverse_iteration),
        ),
    ]
)
