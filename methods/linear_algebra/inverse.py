import numpy as np
from core.validation import checked, matrix
from core.results import result
from methods.linear_algebra.gaussian_elimination import gaussian_elimination


@checked
def matrix_inverse(A):
    """Compute inverse columns by solving A*x_j=e_j with partial pivoting.

    Input: nonsingular square A. Output: inverse, column-solve residual table,
    ||A*A^-1-I||_inf. Singular/numerically singular pivots rejected. Inversion
    is educational; solve Ax=b directly when only a solution is needed.
    """
    A = matrix(A, square=True)
    identity = np.eye(len(A))
    columns = []
    history = []
    for j in range(len(A)):
        r = gaussian_elimination(A, identity[:, j])
        columns.append(r["result"])
        history.append(dict(column=j + 1, solve_residual=r["error_estimate"]))
    inverse = np.column_stack(columns)
    return result(
        inverse,
        history,
        float(np.linalg.norm(A @ inverse - identity, np.inf)),
        error_kind="Inverse residual",
    )
