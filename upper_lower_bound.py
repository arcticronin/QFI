import numpy as np
from qiskit.quantum_info import DensityMatrix, state_fidelity

from density_matrix_from_exp import generate_rho_rho_delta

## VQFIE variational quantum fischer information estimator

# problem: no algorithm to compute QFI for mixed state
# goal: approximate it


def compute_fidelity(rho, sigma):
    # Exact fidelity computation using Qiskit:
    # Fidelity = (Trace[√(√ρσ√ρ)])^2
    # squared overlap of those 2 states
    return state_fidelity(DensityMatrix(rho), DensityMatrix(sigma))


def compute_superfidelity(rho, sigma):
    # Superfidelity approximation
    trace_rho_sigma = np.trace(np.dot(rho, sigma))
    trace_rho2 = np.trace(np.dot(rho, rho))
    trace_sigma2 = np.trace(np.dot(sigma, sigma))
    return trace_rho_sigma + np.sqrt((1 - trace_rho2) * (1 - trace_sigma2))


def compute_subfidelity(rho, sigma):
    # Subfidelity approximation
    trace_rho_sigma = np.trace(np.dot(rho, sigma))
    trace_rho_sigma2 = np.trace(np.dot(np.dot(rho, sigma), np.dot(rho, sigma)))
    return trace_rho_sigma - np.sqrt(2 * (trace_rho_sigma**2 - trace_rho_sigma2))


def compute_truncated_fidelity(rho, sigma, m):
    # Truncated fidelity using the m-largest eigenvalues/vectors
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argsort(eigvals)[::-1]  # Sort eigenvalues in descending order
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep only the m-largest eigenvalues/vectors
    trunc_eigvals = eigvals[:m]
    trunc_eigvecs = eigvecs[:, :m]

    # Construct truncated state
    rho_trunc = sum(
        trunc_eigvals[i] * np.outer(trunc_eigvecs[:, i], trunc_eigvecs[:, i])
        for i in range(m)
    )

    # Compute fidelity for truncated states
    fidelity = np.trace(np.dot(rho_trunc, sigma))
    return fidelity


def compute_upper_bound(fidelity, delta):
    # Upper bound formula
    return 8 * (1 - fidelity) / (delta**2)


def compute_lower_bound(fidelity, delta):
    # Lower bound formula
    return 8 * (1 - fidelity) / (delta**2)


# Parameters
n = 2  # Number of qubits
m = 2  # Truncation parameter
delta = 0.1  # error << 1

# Generate density matrix
a_x = 1.0  # Coefficient for s_x s_x
h_x = 0.5  # Coefficient for s_z

rho_theta, rho_theta_delta = generate_rho_rho_delta(a_x, h_x, delta)

# Compute fidelities
superfidelity = compute_superfidelity(rho_theta, rho_theta_delta)
subfidelity = compute_subfidelity(rho_theta, rho_theta_delta)
truncated_fidelity = compute_truncated_fidelity(rho_theta, rho_theta_delta, m)

# Compute bounds
upper_bound = compute_upper_bound(superfidelity, delta)
lower_bound_truncated = compute_lower_bound(truncated_fidelity, delta)
lower_bound_subfidelity = compute_lower_bound(subfidelity, delta)

# Output results
print(f"Superfidelity: {superfidelity}")
print(f"Upper Bound: {upper_bound}")
print(f"Lower Bound (Truncated): {lower_bound_truncated}")
print(f"Lower Bound (Subfidelity): {lower_bound_subfidelity}")
