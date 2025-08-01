import numpy as np

def check_square_psd_and_eigenvalues(matrix):
    """
    Checks if the input matrix is square and positive semidefinite (PSD).
    If so, returns its eigenvalues. Otherwise, returns None.

    Parameters:
        matrix (np.ndarray): Input matrix.

    Returns:
        np.ndarray or None: Eigenvalues if matrix is square and PSD, else None.
    """
    matrix = np.array(matrix)
    # Check if square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        print("Matrix is not square.")
        return None
    # Check if symmetric
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        print("Matrix is not symmetric, so cannot be PSD.")
        return None
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)
    # Check if all eigenvalues are >= 0 (PSD)
    if np.all(eigenvalues >= -1e-8):  # Allow for small numerical errors
        print("Matrix is square and positive semidefinite.")
        return eigenvalues
    else:
        print("Matrix is not positive semidefinite.")
        return None