# Implementation scope mapping

| Source requirement | Implementation | Verification |
| --- | --- | --- |
| Streamlit browser GUI | `app.py`, `ui/application.py`, method catalog | Default and bad-input AppTest on every page; live browser check |
| One file per numerical method | 38 files under `methods/` | Imports and method-specific tests |
| Polynomial/trigonometric/exponential/logarithmic inputs | `core/expressions.py` | Valid functions, symbolic derivative, injection/domain rejection tests |
| Arithmetic and numerical errors | `methods/computer_arithmetic/` | IEEE bit pattern, decimal comparisons, zero-reference behavior |
| Four root methods and comparison | `methods/root_finding/`, comparison service | Cubic root, invalid brackets, cycles, zero derivatives, SciPy |
| Lagrange, Hermite, interpolation error | `methods/interpolation/` | Polynomial reproduction and known bounds |
| Polynomial approximation, fitting, spline | `methods/curve_fitting/` | Known fits, SciPy natural/clamped spline comparison |
| Taylor polynomial and error (outline) | `methods/curve_fitting/taylor.py` | Exact cubic and exponential remainder bound |
| Five quadrature families and Legendre nodes | `methods/integration/` | exp integral, polynomial exactness, SciPy quad |
| Finite differences and step-size effect | `methods/differentiation/` | Known derivative, order-of-accuracy, cancellation limits |
| Runge-Kutta and SciPy ODE reference | `methods/ode_solvers/rk4.py` | y'=y, solve_ivp, final partial step, size limits |
| Linear systems, pivoting, factorization, conditioning | `methods/linear_algebra/` | Known systems, pivot swaps, singular/non-SPD inputs, reconstruction |
| Matrix inversion (outline) | `methods/linear_algebra/inverse.py` | Inverse identity residual and NumPy comparison |
| Iterative linear methods | Jacobi and Gauss-Seidel | Known systems and divergent examples |
| Least squares, Gram-Schmidt, QR, SVD | `methods/best_approximation/` | Projection, orthogonality, reconstruction, singular values |
| Gradient, CG, Newton optimization (outline) | `methods/optimization/` | Quadratic minima, SPD checks, Hessian validation |
| Eigenvalue/eigenvector approximation | `methods/eigenvalues/` | Symmetric eigenpairs, residuals, singular shift rejection |
| Pinned dependencies and run instructions | Requirements files, README | Installed environment and actual execution |
| Expected/actual test evidence | `docs/test_results.md`, `.csv` | Generated from 38 executable examples |

Choices for unspecified scope: classical RK4, Jacobi/Gauss-Seidel, real symmetric eigenpairs, and linear CG for SPD quadratics. These are explicit implementation choices that can be revised when the instructor specifies a narrower or different calculator test.

The course outline also contains conceptual background, Python fundamentals, NumPy usage, vector spaces and norms. These support the implementations; they do not require standalone calculator pages. Additional Runge-Kutta variants, nonsymmetric/complex eigenproblems, nonlinear CG and further classroom algorithms remain possible course-driven extensions.

Report, handwritten work and slides are separate submission components in the brief; no report or slides were authored in this project, per the user's updated scope.
