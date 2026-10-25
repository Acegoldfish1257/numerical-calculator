"""Explicit expected values and tolerances; shared by pytest and table export."""

from dataclasses import dataclass
import math
import numpy as np
from core.expressions import parse_function
from methods.root_finding.bisection import bisection
from methods.root_finding.fixed_point import fixed_point
from methods.root_finding.newton_raphson import newton_raphson
from methods.root_finding.secant import secant
from methods.interpolation.lagrange import lagrange
from methods.interpolation.hermite import hermite
from methods.interpolation.error_bound import interpolation_error_bound
from methods.curve_fitting.polynomial_approximation import polynomial_approximation
from methods.curve_fitting.least_squares_fit import least_squares_fit
from methods.curve_fitting.cubic_spline import cubic_spline
from methods.curve_fitting.taylor import taylor_approximation
from methods.integration.midpoint import midpoint
from methods.integration.trapezoidal import trapezoidal
from methods.integration.simpson import simpson
from methods.integration.newton_cotes import newton_cotes
from methods.integration.gauss_legendre import gauss_legendre
from methods.differentiation.finite_difference import finite_difference
from methods.ode_solvers.rk4 import rk4
from methods.computer_arithmetic.floating_point import floating_point_representation
from methods.computer_arithmetic.arithmetic import floating_point_arithmetic
from methods.computer_arithmetic.error_analysis import numerical_error
from methods.computer_arithmetic.significant_digits import significant_digits
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


@dataclass
class Case:
    method: str
    problem: str
    expected: object
    tolerance: float
    compute: object


def cases():
    f = lambda x: x**3 - x - 2
    root = 1.5213797068045676
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    solution = np.array([1 / 11, 7 / 11])
    objective = lambda x: (x[0] - 2) ** 2 + 2 * (x[1] + 1) ** 2
    gradient = lambda x: np.array([2 * (x[0] - 2), 4 * (x[1] + 1)])
    return [
        Case(
            "Bisection",
            "x^3-x-2 on [1,2]",
            root,
            1e-8,
            lambda: bisection(f, 1, 2)["result"],
        ),
        Case(
            "Fixed point",
            "g(x)=(x+2)^(1/3), x0=1.5",
            root,
            1e-8,
            lambda: fixed_point(lambda x: (x + 2) ** (1 / 3), 1.5, f=f)["result"],
        ),
        Case(
            "Newton-Raphson",
            "x^3-x-2, x0=1.5",
            root,
            1e-10,
            lambda: newton_raphson(f, lambda x: 3 * x * x - 1, 1.5)["result"],
        ),
        Case(
            "Secant",
            "x^3-x-2, x0=1,x1=2",
            root,
            1e-10,
            lambda: secant(f, 1, 2)["result"],
        ),
        Case(
            "Lagrange",
            "x^2+1 nodes [0,1,2]; query 0.5",
            [1.25],
            1e-12,
            lambda: lagrange([0, 1, 2], [1, 2, 5], [0.5])["result"],
        ),
        Case(
            "Hermite",
            "x^3 with derivative, nodes [0,2]; query 0.5",
            [0.125],
            1e-12,
            lambda: hermite([0, 2], [0, 8], [0, 12], [0.5])["result"],
        ),
        Case(
            "Interpolation bound",
            "nodes [0,1], query .5, M=2",
            [0.25],
            1e-12,
            lambda: interpolation_error_bound([0, 1], [0.5], derivative_bound=2)[
                "result"
            ],
        ),
        Case(
            "Polynomial approximation",
            "x^2+2, degree 2, query 0.5",
            [2.25],
            1e-12,
            lambda: polynomial_approximation(
                lambda x: x * x + 2, 0, 1, 2, x_eval=[0.5]
            )["result"],
        ),
        Case(
            "Curve fit",
            "y=1+2x, x=[0,1,2]; fitted coefficients",
            [1, 2],
            1e-12,
            lambda: least_squares_fit(
                [0, 1, 2], [1, 3, 5], basis=[lambda x: 1, lambda x: x]
            )["diagnostics"]["coefficients"],
        ),
        Case(
            "Cubic spline",
            "linear data [0,1,2]; query 0.5",
            [0.5],
            1e-12,
            lambda: cubic_spline([0, 1, 2], [0, 1, 2], [0.5])["result"],
        ),
        Case(
            "Taylor",
            "x^3, center 1, degree 3; queries 0,1,2",
            [0, 1, 8],
            1e-12,
            lambda: taylor_approximation(parse_function("x^3"), 1, 3, [0, 1, 2])[
                "result"
            ],
        ),
        Case(
            "Midpoint",
            "integral exp(x), [0,1], n=100",
            math.expm1(1),
            8e-6,
            lambda: midpoint(math.exp, 0, 1, 100)["result"],
        ),
        Case(
            "Trapezoidal",
            "integral exp(x), [0,1], n=100",
            math.expm1(1),
            1.5e-5,
            lambda: trapezoidal(math.exp, 0, 1, 100)["result"],
        ),
        Case(
            "Simpson",
            "integral exp(x), [0,1], n=20",
            math.expm1(1),
            1e-7,
            lambda: simpson(math.exp, 0, 1, 20)["result"],
        ),
        Case(
            "Newton-Cotes",
            "integral exp(x), [0,1], order=4, panels=4",
            math.expm1(1),
            1e-9,
            lambda: newton_cotes(math.exp, 0, 1, order=4, panels=4)["result"],
        ),
        Case(
            "Gauss-Legendre",
            "integral exp(x), [0,1], nodes=4",
            math.expm1(1),
            1e-9,
            lambda: gauss_legendre(math.exp, 0, 1, nodes=4)["result"],
        ),
        Case(
            "Central difference",
            "d exp(x)/dx at x=1,h=1e-4",
            math.e,
            1e-7,
            lambda: finite_difference(math.exp, 1, 1e-4)["result"],
        ),
        Case(
            "RK4",
            "y'=y,y(0)=1,t_end=1,h=.1",
            math.e,
            3e-6,
            lambda: rk4(lambda t, y: y, 0, 1, 1, 0.1)["result"],
        ),
        Case(
            "Floating point",
            "1.5 in float32",
            1.5,
            0.0,
            lambda: floating_point_representation("1.5", precision="float32")["result"],
        ),
        Case(
            "Binary arithmetic",
            "0.1+0.2, float64",
            0.3,
            1e-15,
            lambda: floating_point_arithmetic("0.1", "0.2")["result"],
        ),
        Case(
            "Error analysis",
            "absolute error of 3.1 versus 3",
            0.1,
            1e-15,
            lambda: numerical_error("3.1", "3")["result"],
        ),
        Case(
            "Significant digits",
            "123.456 rounded to 4 significant digits",
            123.5,
            0.0,
            lambda: float(significant_digits("123.456", 4)["result"]),
        ),
        Case(
            "Gaussian elimination",
            "A=[[4,1],[1,3]],b=[1,2]",
            solution,
            1e-12,
            lambda: gaussian_elimination(A, b)["result"],
        ),
        Case(
            "LU",
            "A=[[4,1],[1,3]]; reconstruction residual",
            0.0,
            1e-12,
            lambda: lu_factorization(A)["error_estimate"],
        ),
        Case(
            "Cholesky",
            "A=[[4,1],[1,3]]; reconstruction residual",
            0.0,
            1e-12,
            lambda: cholesky_factorization(A)["error_estimate"],
        ),
        Case(
            "Conditioning",
            "diag(1,100), 2-norm",
            100.0,
            1e-12,
            lambda: matrix_conditioning([[1, 0], [0, 100]])["result"],
        ),
        Case(
            "Inverse",
            "A=[[4,1],[1,3]]; inverse residual",
            0.0,
            1e-12,
            lambda: matrix_inverse(A)["error_estimate"],
        ),
        Case(
            "Jacobi",
            "A=[[4,1],[1,3]],b=[1,2]",
            solution,
            1e-8,
            lambda: jacobi(A, b)["result"],
        ),
        Case(
            "Gauss-Seidel",
            "A=[[4,1],[1,3]],b=[1,2]",
            solution,
            1e-8,
            lambda: gauss_seidel(A, b)["result"],
        ),
        Case(
            "Gram-Schmidt",
            "A=[[4,1],[1,3]]; orthogonality residual",
            0.0,
            1e-12,
            lambda: gram_schmidt(A)["diagnostics"]["orthogonality_error"],
        ),
        Case(
            "Householder QR",
            "A=[[4,1],[1,3]]; reconstruction residual",
            0.0,
            1e-12,
            lambda: qr_factorization(A)["error_estimate"],
        ),
        Case(
            "SVD",
            "singular values of diag(3,2)",
            [3, 2],
            1e-12,
            lambda: svd_factorization([[3, 0], [0, 2]])["result"]["s"],
        ),
        Case(
            "Linear least squares",
            "A=[[1,0],[1,1],[1,2]],b=[1,3,5]",
            [1, 2],
            1e-12,
            lambda: least_squares([[1, 0], [1, 1], [1, 2]], [1, 3, 5])["result"],
        ),
        Case(
            "Gradient descent",
            "min (x-2)^2+2(y+1)^2, start (0,0)",
            [2, -1],
            1e-8,
            lambda: gradient_descent(objective, gradient, [0, 0])["result"],
        ),
        Case(
            "Newton optimization",
            "same quadratic, start (0,0)",
            [2, -1],
            1e-12,
            lambda: newton_optimization(
                objective, gradient, lambda x: np.diag([2.0, 4.0]), [0, 0]
            )["result"],
        ),
        Case(
            "Conjugate gradient",
            "min .5*x^T*A*x-b^T*x; A=[[4,1],[1,3]],b=[1,2]",
            solution,
            1e-12,
            lambda: conjugate_gradient(A, b)["result"],
        ),
        Case(
            "Power iteration",
            "dominant eigenvalue of [[4,1],[1,3]]",
            (7 + math.sqrt(5)) / 2,
            1e-10,
            lambda: power_iteration(A)["result"]["eigenvalue"],
        ),
        Case(
            "Inverse iteration",
            "smallest eigenvalue of [[4,1],[1,3]],shift=0",
            (7 - math.sqrt(5)) / 2,
            1e-10,
            lambda: inverse_iteration(A)["result"]["eigenvalue"],
        ),
    ]


def evaluate_case(case):
    actual = np.asarray(case.compute(), dtype=float)
    expected = np.asarray(case.expected, dtype=float)
    if actual.shape != expected.shape:
        raise AssertionError(f"{case.method}: unexpected output shape")
    error = float(np.max(np.abs(actual - expected)))
    return actual, error, bool(np.all(np.isfinite(actual)) and error <= case.tolerance)
