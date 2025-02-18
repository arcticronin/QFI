import numpy as np
from qiskit.quantum_info import DensityMatrix

# from density_matrix_from_exp import generate_rho_rho_delta
# from ising_density_2D import (
#    generate_density_matrices_with_perturbation as generate_rho_rho_delta,
# )

from density_generator import IsingQuantumState

# debug
DEBUG = 1


# --------------------------------------------------
# Truncated Quantum Fisher Information (TQFI) bounds
# --------------------------------------------------


def compute_tqfi_bounds(rho, rho_delta, m, delta):
    """
    Compute the Truncated Quantum Fisher Information (TQFI) bounds.

    This method follows the approach from the theoretical framework:
    1. Perform eigenvalue decomposition of rho.
    2. Truncate to the m-largest eigenvalues (principal components).
    3. Compute truncated and generalized fidelities (L R).
    4. Estimate the lower and upper bounds of TQFI.

    Parameters:
    - rho: Density matrix at parameter theta.
    - rho_delta: Density matrix at parameter theta + delta.
    - m: Truncation parameter for principal components.
    - delta: Small shift in parameter for derivative approximation.

    Returns:
    - lower_tqfi: Lower bound of TQFI.
    - upper_tqfi: Upper bound of TQFI.
    """
    # Step 1: Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(rho)

    idx = np.argsort(eigvals)[::-1]  # Descending order
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Step 2: Truncate to m-largest eigenvalues/eigenvectors
    eigvals_trunc = eigvals[:m]
    eigvecs_trunc = eigvecs[:, :m]

    # Construct the truncated density matrix
    rho_trunc = sum(
        eigvals_trunc[i] * np.outer(eigvecs_trunc[:, i], eigvecs_trunc[:, i].conj())
        for i in range(m)
    )

    # Step 3: Compute truncated fidelity (Fidelity between truncated rho and rho_delta)
    fidelity_truncated = np.real(np.trace(np.dot(rho_trunc, rho_delta)))

    # Generalized fidelity incorporates truncation errors
    fidelity_generalized = fidelity_truncated + np.sqrt(
        max(0, (1 - np.trace(rho_trunc)) * (1 - np.trace(rho_delta)))
    )

    # Step 4: Compute TQFI bounds using fidelity definitions
    lower_tqfi = 8 * (1 - fidelity_generalized) / (delta**2)
    upper_tqfi = 8 * (1 - fidelity_truncated) / (delta**2)

    # Intermediate results for debugging
    print(f"Fidelity (Truncated): {fidelity_truncated}")
    print(f"Fidelity (Generalized): {fidelity_generalized}")

    return lower_tqfi, upper_tqfi


# ---------------------------------------------------------
# Sub/Super Quantum Fisher Information (SSQFI) bounds
# ---------------------------------------------------------


def compute_ssqfi_bounds(rho, rho_delta, delta):
    """
    Compute the Sub and Super Quantum Fisher Information (SSQFI) bounds.

    This method is based on subfidelity and superfidelity formulas derived from the paper:
    1. Compute subfidelity and superfidelity.
    2. Derive SSQFI bounds using fidelity bounds.

    Parameters:
    - rho: Density matrix at parameter theta.
    - rho_delta: Density matrix at parameter theta + delta.
    - delta: Small shift in parameter for derivative approximation.

    Returns:
    - lower_ssqfi: Lower bound of SSQFI.
    - upper_ssqfi: Upper bound of SSQFI.
    """
    # Step 1: Basic trace calculations
    trace_rho_sigma = np.trace(np.dot(rho, rho_delta))
    trace_rho2 = np.trace(np.dot(rho, rho))
    trace_sigma2 = np.trace(np.dot(rho_delta, rho_delta))

    # Step 2: Compute Subfidelity (with numerical stability)
    sub_fidelity = trace_rho_sigma + np.sqrt(
        max(
            0,
            2
            * (
                trace_rho_sigma**2
                - np.trace(np.dot(np.dot(rho, rho_delta), np.dot(rho, rho_delta)))
            ),
        )
    )

    # Step 3: Compute Superfidelity
    super_fidelity = trace_rho_sigma + np.sqrt(
        max(0, (1 - trace_rho2) * (1 - trace_sigma2))
    )

    # Step 4: Compute SSQFI bounds
    lower_ssqfi = 8 * (1 - super_fidelity) / (delta**2)
    upper_ssqfi = 8 * (1 - sub_fidelity) / (delta**2)

    # Intermediate results for debugging
    print(f"Subfidelity: {sub_fidelity}")
    print(f"Superfidelity: {super_fidelity}")

    return lower_ssqfi, upper_ssqfi


# ---------------------------
# Main Simulation Parameters
# ---------------------------
n = 2  # Number of qubits
m = 2  # Truncation parameter
delta = 0.001  # Small error for derivative approximation

# Generate density matrix (simulated state)
a_x = 1.0  # Coefficient for s_x s_x
h_z = 0.5  # Coefficient for s_z

# Generating the quantum states rho(theta) and rho(theta + delta)
# rho_theta, rho_theta_delta = generate_rho_rho_delta(a_x, h_z, delta)

qs = IsingQuantumState(n=3, a_x=a_x, h_z=h_z)

rho_theta, rho_theta_delta = qs.generate_density_matrices_with_perturbation()

# Compute TQFI bounds
lower_tqfi, upper_tqfi = compute_tqfi_bounds(rho_theta, rho_theta_delta, m, delta)

# Compute SSQFI bounds
lower_ssqfi, upper_ssqfi = compute_ssqfi_bounds(rho_theta, rho_theta_delta, delta)

# ---------------------------
# Output the Final Results
# ---------------------------
print("TQFI Bounds:")
print(f"Lower TQFI Bound: {lower_tqfi}")
print(f"Upper TQFI Bound: {upper_tqfi}")

print("\nSSQFI Bounds:")
print(f"Lower SSQFI Bound: {lower_ssqfi}")
print(f"Upper SSQFI Bound: {upper_ssqfi}")
