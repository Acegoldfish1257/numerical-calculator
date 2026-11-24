# Implementation scope mapping

| Source requirement | Implementation | Verification |
| --- | --- | --- |
| Streamlit browser GUI | `app.py`, `ui/application.py`, method catalog | Default and bad-input AppTest on every page; live browser check |
| One file per active numerical method | `methods/` | Imports and method-specific tests |
| Polynomial/trigonometric/exponential/logarithmic inputs | `core/expressions.py` | Valid functions, symbolic derivative, injection/domain rejection tests |
| Arithmetic and numerical errors | `methods/computer_arithmetic/` | IEEE bit pattern, decimal comparisons, zero-reference behavior |
| Four root methods and comparison | `methods/root_finding/`, comparison service | Cubic root, invalid brackets, cycles, zero derivatives, SciPy |
| Lagrange, interpolation error, cubic spline | `methods/interpolation/` and spline implementation | Polynomial reproduction and known bounds |
| Polynomial approximation and fitting | `methods/curve_fitting/` | Known fits |
| Taylor polynomial and error (outline) | `methods/curve_fitting/taylor.py` | Exact cubic and exponential remainder bound |
| Composite quadrature and Legendre nodes | `methods/integration/` | exp integral, polynomial exactness, SciPy quad |
| Midpoint RK2 and SciPy ODE reference | `methods/ode_solvers/rk2.py` | y'=y, solve_ivp, final partial step, size limits |
| Linear systems, pivoting, factorization, conditioning | `methods/linear_algebra/` | Known systems, pivot swaps, singular/non-SPD inputs, reconstruction |
| Matrix inversion (outline) | `methods/linear_algebra/inverse.py` | Inverse identity residual and NumPy comparison |
| Iterative linear methods | Jacobi and Gauss-Seidel | Known systems and divergent examples |
| Least squares, Gram-Schmidt and QR | `methods/best_approximation/` | Projection, orthogonality and reconstruction |
| Gradient descent | `methods/optimization/` | Quadratic minimum |
| Eigenvalue/eigenvector approximation | `methods/eigenvalues/` | Symmetric eigenpairs, residuals, singular shift rejection |
| Pinned dependencies and run instructions | Requirements files, README | Installed environment and actual execution |
| Expected/actual test evidence | `docs/test_results.md`, `.csv` | Generated from executable examples |

The course outline also contains conceptual background, Python fundamentals, NumPy usage and vector spaces. L1, L2 and L-infinity norms have a dedicated calculator page; other background topics support the implementations.

Report, handwritten work and slides are separate submission components in the brief; no report or slides were authored in this project, per the user's updated scope.
