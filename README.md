# CS-2008 Numerical Calculator

A local browser calculator built with Streamlit, Python, NumPy, SciPy, SymPy and Matplotlib. It implements 38 numerical method modules and a four-method root comparison page. Each page provides relevant inputs, a Run button, a result, a step table, a Matplotlib plot and numerical diagnostics.

## Install and run

Tested on Windows with Python 3.12. Use Python 3.12 to reproduce the verified environment.

From PowerShell in this project folder, for a fresh installation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The project virtual environment on this computer already has the dependencies installed. Start it using the last command above. Open the local URL printed by Streamlit, normally http://localhost:8501. Stop the server with Ctrl+C in its terminal.

Alternatively, activate the environment and run the exact requested command:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

If PowerShell blocks activation, use the direct `.venv\Scripts\python.exe` commands; no execution-policy change is needed. On Linux/macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

`requirements.txt` pins the direct dependencies that were tested. `requirements-lock.txt` pins the installed runtime dependencies for closer reproduction. `requirements-dev.txt` adds the optional source formatter.

## Use the calculator

1. Select a topic and method in the sidebar. On a narrow browser, expand the sidebar with the arrow at the top left.
2. Edit the example inputs. Use explicit multiplication (`2*x`), `^` or `**` for powers, and radians for trigonometry.
3. Press **Run computation**. Results stay visible during table downloads until you run again or select a different method.
4. Read the convergence status and diagnostics, not just the headline answer. Download the step table as CSV when useful.

Functions support `sin`, `cos`, `tan`, `exp`, `log` (natural logarithm), `sqrt`, `sinh`, `cosh`, `tanh`, `pi`, `e` and `E`. Root/integration/differentiation functions use `x`; ODEs use `t,y`; optimization objectives use `x,y`. Numeric exponents and simple fractions are supported, for example `(x+2)^(1/3)`. Variable exponentials can be written `exp(x)` or `exp(x*log(2))`.

Use comma- or space-separated values for vectors, one row per line for matrices, and semicolon-separated functions for a curve-fitting basis (`1; x; x^2`). Mathematical expressions are not accepted inside numeric data fields.

## Coverage

| Topic | Implemented methods |
| --- | --- |
| Computer arithmetic | IEEE-754 float32/64 representation, machine epsilon, decimal rounding/truncation and normalized notation, binary arithmetic, absolute/relative/percentage error |
| Root finding | Bisection, fixed point, Newton-Raphson, secant, four-method comparison |
| Interpolation | Lagrange, Hermite with first derivatives, conditional error bounds |
| Approximation and fitting | Sampled polynomial least squares, automatic exponential/power model fitting with logarithmic linearization, natural/clamped cubic spline API, Taylor approximation |
| Integration | Composite midpoint, trapezoidal, Simpson 1/3, Simpson 3/8, general closed Newton-Cotes orders 1-6, Gauss-Legendre |
| Differentiation and ODEs | Forward/backward/central first differences, central second difference, step-size error study, scalar classical RK4 |
| Linear algebra | Gaussian elimination with no/partial/scaled pivoting, LU, Cholesky, inverse, condition numbers, Jacobi, Gauss-Seidel |
| Best approximation | Linear least squares (QR/SVD), classical/modified Gram-Schmidt, Householder QR, SVD |
| Optimization | Two-variable gradient descent, Newton optimization, linear conjugate gradient for SPD quadratics |
| Eigenvalues | Power and shifted inverse iteration for real symmetric matrices |

The supplied official brief determines the project scope. The course outline adds context, including Taylor approximation, inversion, conjugate gradient and Newton optimization. General Python/programming lessons and mathematical background are not separate calculator pages. See `docs/requirements_matrix.md` for the mapping. Report and slide creation are outside this implementation's scope, as requested.

## Architecture

```text
app.py                       Streamlit entry point; no numerical logic
core/                        Validated expression parsing, numeric validation, result contract
methods/<topic>/<method>.py   One implementation module per numerical method
services/                    Page descriptors, input conversion, comparisons and references
ui/                          Streamlit controls and Matplotlib rendering
tests/                       Method, failure-case, parser and UI checks
examples/                    Reproducible verification problems
scripts/                     Verification-table generator
docs/                        Requirements mapping, test evidence, development checkpoints
```

Algorithms do not import Streamlit. Each numerical function documents its formula, arguments, output and assumptions, and returns:

```python
{
    "result": ...,           # scalar, array, string, or named factors
    "history": [...],        # list of table rows
    "error_estimate": ...,   # None when no justified estimate is available
    "converged": ...,        # True/False for iterations, otherwise None
    "message": "...",
    "diagnostics": {...},
}
```

Invalid input raises `ValueError`; the UI renders it with `st.error`. Iteration exhaustion returns `converged=False`. A final UI exception boundary prevents unexpected computation/rendering errors from exposing an unhandled traceback in the page. This does not make an ordinary Python process immune to operating-system failures or resource exhaustion.

## Numerical interpretation and limits

- Residual, iterate difference, reference difference, RMSE, conditioning and error bounds are different quantities; the UI labels each explicitly.
- Bisection assumes continuity. A sign change alone does not prove a root across a discontinuity. The implementation requires both a narrow bracket and a small residual, unless an exact floating-point zero is found.
- Fixed point needs a user-supplied `g(x)`. A small original-equation residual is also required. Local methods and SciPy can find different roots.
- Interpolation bounds require a valid higher-derivative bound over the whole relevant interval. Data alone cannot supply that bound.
- Composite quadrature error estimates use refinement and smooth asymptotic error assumptions. Newton-Cotes and Gaussian differences are indicators, not rigorous bounds. Integration currently requires finite ordered bounds; improper integrals are outside scope.
- Non-polynomial model fitting supports y=a*b^x and y=a*x^b. The app derives the logarithmic linear form, estimates coefficients, converts them back and displays the fitted equation.
- Cubic-spline UI uses natural boundary conditions. The Python API additionally supports clamped endpoint slopes. Spline extrapolation is rejected.
- Polynomial approximation is discrete sampled least squares, not minimax or continuous orthogonal projection. Curve fitting is linear in its specified coefficients.
- RK4 handles one real scalar ODE and forward time. It is fixed-step, with a shortened final step, and is not a general stiff-system solver.
- Gradient descent/Newton pages handle two real variables. Newton optimization requires a positive-definite Hessian at every step. Linear CG minimizes an SPD quadratic; nonlinear CG is not implemented.
- Eigenpair pages accept real symmetric matrices. The starting vector and spectral gaps affect which eigenpair is found. Small eigenpair residual alone does not establish that it is the dominant/nearest eigenpair.
- Most core algorithms are explicit educational implementations. SVD, condition numbers, curve-fit least squares and Gaussian node generation deliberately use NumPy routines; these backends are labeled. SciPy supplies independent comparisons where appropriate.
- Expressions are limited to 300 characters, an allowlisted AST, bounded powers/depth and supported names before `sympy.parse_expr` and `lambdify`. No application code calls Python `eval`. SymPy internally uses evaluation/code generation; this is a local educational tool, not a public untrusted-code service. Further process isolation would be needed for public multi-user hosting.
- Input sizes are capped. Examples: matrices 50x50, RK4 at most 10000 steps, Lagrange 30 nodes, Hermite 15 nodes, splines 200 nodes. Errors explain these limits.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/generate_verification_table.py
```

The generated `docs/test_results.md` and `docs/test_results.csv` contain expected values, actual results, absolute errors and stated tolerances for 38 reproducible examples. Pytest also checks adversarial expression syntax, invalid inputs on every page, nonconvergence, factor reconstruction and independent SciPy references. Streamlit `AppTest` tests the Run buttons, tables and error messages; a local browser check verifies actual page rendering.

Typical examples are the root of `x^3-x-2`, the integral of `exp(x)` on `[0,1]`, and `y'=y, y(0)=1` through `t=1`.

## References and attribution

- Project scope: user-provided `CS2008 - Project Details.pdf` and `CS2008 Numerical Computing Outline.docx`.
- [SymPy parsing documentation](https://docs.sympy.org/latest/modules/parsing.html): parsing behavior and untrusted-input warning.
- [Streamlit AppTest documentation](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest): application test interface.
- [NumPy documentation](https://numpy.org/doc/) and [SciPy documentation](https://docs.scipy.org/doc/scipy/): reference numerical libraries.

This code was developed with AI assistance. Review and understand the algorithms, record any course-specific changes, and acknowledge external assistance according to your instructor's requirements.
