import numpy as np
from scipy.linalg import sqrtm


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


def fidelity(rho, rho_delta, DEBUG=False):
    if not (is_density_matrix(rho) and is_density_matrix(rho_delta)):
        raise ValueError("Inputs must be valid density matrices.")

    sqrt_rho = sqrtm(rho)

    X = sqrtm(sqrt_rho @ rho_delta @ sqrt_rho)

    # F = np.min(np.array([(np.real(np.trace(X))), 1]))
    F = np.trace(X)

    if DEBUG:
        # print("rho squared trace:", np.real(np.trace(rho @ rho)))
        # print("rho + delta squared trace:", np.real(np.trace(rho_delta @ rho_delta)))
        print(f"Fidelity F = {F}")
        # print("Difference between matrices:\n", np.abs(rho - rho_delta))
        # print("Maximum difference:", np.max(np.abs(rho - rho_delta)))

    return np.real(F)


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
