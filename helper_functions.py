import numpy as np
from scipy.linalg import sqrtm, eigh, svd
import qutip


def trace_out(rho, trace_out_index):
    dim = rho.shape[0]
    n = int(np.log2(dim))
    if any(i < 0 or i >= n for i in trace_out_index):
        raise ValueError(
            f"Invalid trace_out_index: Indices must be in range [0, n-1], n = {n}, trace_out_index = {trace_out_index}"
        )

    rho_qutip = qutip.Qobj(rho, dims=[[2] * n, [2] * n])
    sel = [i for i in range(n) if i not in trace_out_index]
    return rho_qutip.ptrace(sel).full()


## the Uhlmann is sqyared, the paper uses this
def trace_norm_rho_rho_delta(rho1, rho2):
    """
    Computes the trace norm || sqrt(rho1) sqrt(rho2) ||_1
    for two density matrices rho_sigma, rho_(sigma + delta).
    """
    # 1) Compute the principal square root of rho1 and rho2

    # print("rho.dtype =", rho1.dtype)
    # print("rho_delta.dtype =", rho2.dtype)

    sqrt_rho1 = sqrtm(rho1).astype(np.complex128)
    sqrt_rho2 = sqrtm(rho2).astype(np.complex128)

    # print("rho.dtype =", sqrt_rho1.dtype)
    # print("rho_delta.dtype =", sqrt_rho2.dtype)

    # 2) Form the product sqrt(rho1)*sqrt(rho2)
    product = sqrt_rho1 @ sqrt_rho2

    # 3) Compute all singular values (σ_i)
    singular_vals = np.linalg.svd(product, compute_uv=False)

    # 4) The trace norm is the sum of the singular values
    return np.sum(singular_vals)


def compute_eigen_decomposition(rho):
    """
    Compute the eigenvalues and eigenvectors of a density matrix.

    Args:
        rho (np.ndarray): Density matrix.

    Returns:
        tuple: (eigvals, eigvecs) where:
            eigvals (np.ndarray): Eigenvalues in descending order.
            eigvecs (np.ndarray): Corresponding eigenvectors.
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argsort(eigvals)[::-1]  # Sort in descending order
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    return eigvals, eigvecs


def truncate_density_matrix(rho, m, DEBUG=False):
    """
    Truncate a density matrix to its m-largest eigenvalues/eigenvectors.

    Args:
        rho (np.ndarray): Density matrix.
        m (int): Number of largest eigenvalues to keep.
        DEBUG (bool): If True, print truncated eigenvalues.

    Returns:
        np.ndarray: Truncated density matrix.
    """
    eigvals, eigvecs = compute_eigen_decomposition(rho)

    # Keep only the m-largest eigenvalues and eigenvectors
    eigvals_trunc = eigvals[:m]
    eigvecs_trunc = eigvecs[:, :m]

    if DEBUG:
        print("Truncated eigenvalues:", eigvals_trunc)

    # Reconstruct the truncated density matrix
    rho_trunc = sum(
        eigvals_trunc[i] * np.outer(eigvecs_trunc[:, i], eigvecs_trunc[:, i].conj())
        for i in range(m)
    )

    return rho_trunc


def get_truncated_eigen_decomposition(rho, m):
    """
    Get the truncated eigenvalues and eigenvectors.

    Args:
        rho (np.ndarray): Density matrix.
        m (int): Number of largest eigenvalues to keep.

    Returns:
        tuple: (eigvals_trunc, eigvecs_trunc) where:
            eigvals_trunc (np.ndarray): Truncated eigenvalues.
            eigvecs_trunc (np.ndarray): Truncated eigenvectors.
    """
    eigvals, eigvecs = compute_eigen_decomposition(rho)
    eigvals_trunc = eigvals[:m]
    eigvecs_trunc = eigvecs[:, :m]
    return eigvals_trunc, eigvecs_trunc


def is_density_matrix(mat):
    """Check if a matrix is a valid density matrix."""
    return (
        np.allclose(mat, mat.conj().T)
        and np.all(np.linalg.eigvals(mat) >= -1e-10)
        and np.isclose(np.trace(mat), 1)
    )


def fidelity_pure_states(rho_1, rho_2):
    # Extract the state vectors from the density matrices (simce it's pure, the eigenvector is the state  that generated it)
    psi_1 = np.linalg.eigh(rho_1)[1][:, -1]  # Last eigenvector (they are sorted)
    psi_2 = np.linalg.eigh(rho_2)[1][:, -1]

    # fidelity as squared inner product
    fidelity = np.abs(np.vdot(psi_1, psi_2)) ** 2

    return fidelity


def fidelity(rho, rho_delta, root=False, DEBUG=False):
    if not (is_density_matrix(rho) and is_density_matrix(rho_delta)):
        raise ValueError("Inputs must be valid density matrices.")

    sqrt_rho = sqrtm(rho)

    X = sqrtm(sqrt_rho @ rho_delta @ sqrt_rho)

    # F = np.min(np.array([(np.real(np.trace(X))), 1]))

    if root == True:
        F = np.trace(X)
    else:  # standard formula
        F = (np.trace(X)) ** 2

    if DEBUG:
        # print("rho squared trace:", np.real(np.trace(rho @ rho)))
        # print("rho + delta squared trace:", np.real(np.trace(rho_delta @ rho_delta)))
        print(f"Fidelity F = {F}")
        # print("Difference between matrices:\n", np.abs(rho - rho_delta))
        # print("Maximum difference:", np.max(np.abs(rho - rho_delta)))
    # Enforce valid range [0, 1] by clipping
    return np.clip(np.real(F), 0, 1)


def fidelity_robust(rho, sigma):
    """
    Compute the quantum fidelity between two density matrices rho and sigma.
    """
    sqrt_rho = sqrtm(rho)
    X = sqrtm(sqrt_rho @ sigma @ sqrt_rho)

    # Ensure X is real and positive semi-definite
    X = (X + X.conj().T) / 2  # force Hermiticity
    X = np.real(X)  # Remove imaginary part
    X[X < 0] = 0  # Clip negative values from numerical errors
    #  Fidelity
    F = np.trace(X)
    # Enforce valid range [0, 1] by clipping
    return np.clip(F**2, 0, 1)


def random_haar_ket(n):
    """
    Generate a random ket of size n, distributed according to the Haar measure.

    Args:
        n (int): Dimension of the ket.

    Returns:
        numpy.ndarray: A normalized complex vector (ket) of dimension n.
    """
    # Step 1: Generate random complex vector with normally distributed components
    z = np.random.randn(n) + 1j * np.random.randn(n)

    # Step 2: Normalize the vector to get a ket
    ket = z / np.linalg.norm(z)

    return ket
