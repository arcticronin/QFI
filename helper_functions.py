import numpy as np
from scipy.linalg import sqrtm, eigh, svd


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


def truncate_density_matrix(rho, m, DEBUG=False):
    """
    Truncate a density matrix to its m-largest eigenvalues/eigenvectors.
    """
    # rho truncated
    eigvals, eigvecs = np.linalg.eigh(rho)

    idx = np.argsort(eigvals)[::-1]  # Descending order (largest first)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Step 2: Truncate rho to m-largest eigenvalues/eigenvectors
    eigvals_trunc = eigvals[:m]
    eigvecs_trunc = eigvecs[:, :m]

    # if DEBUG:
    #    print(eigvals_trunc)

    # Construct the truncated density matrix
    rho_trunc = sum(
        eigvals_trunc[i] * np.outer(eigvecs_trunc[:, i], eigvecs_trunc[:, i].conj())
        for i in range(m)
    )

    return rho_trunc


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
