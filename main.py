import numpy as np
from qiskit.quantum_info import DensityMatrix

from density_matrix_from_exp import generate_rho_rho_delta
from deprecated.density_matrix_gen import prova


# Truncated Quantum Fisher Information (TQFI) bounds
def compute_tqfi_bounds(rho, rho_delta, m):
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argsort(eigvals)[::-1]  # Descending order
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Truncate to m-largest eigenvalues/eigenvectors
    eigvals_trunc = eigvals[:m]
    eigvecs_trunc = eigvecs[:, :m]

    rho_trunc = sum(
        eigvals_trunc[i] * np.outer(eigvecs_trunc[:, i], eigvecs_trunc[:, i])
        for i in range(m)
    )

    # # Compute truncated fidelity
    # fidelity_truncated = np.trace(np.dot(rho_trunc, rho_delta))

    # # Compute generalized fidelity
    # fidelity_generalized = fidelity_truncated + np.sqrt(
    #     (1 - np.trace(rho_trunc)) * (1 - np.trace(rho_delta))
    # )

    ##### FIX for the case where i get imaginary values

    # Compute truncated fidelity
    fidelity_truncated = np.real(np.trace(np.dot(rho_trunc, rho_delta)))

    # Compute generalized fidelity
    fidelity_generalized = fidelity_truncated + np.sqrt(
        max(0, (1 - np.trace(rho_trunc)) * (1 - np.trace(rho_delta)))
    )
    #####
    print(f"fidelity generalized: {fidelity_generalized}")

    # Compute TQFI bounds
    lower_tqfi = 8 * (1 - fidelity_generalized) / (delta**2)
    upper_tqfi = 8 * (1 - fidelity_truncated) / (delta**2)
    return lower_tqfi, upper_tqfi


# Sub/Super Quantum Fisher Information (SSQFI) bounds
def compute_ssqfi_bounds(rho, rho_delta):
    trace_rho_sigma = np.trace(np.dot(rho, rho_delta))
    trace_rho2 = np.trace(np.dot(rho, rho))
    trace_sigma2 = np.trace(np.dot(rho_delta, rho_delta))

    sub_fidelity = trace_rho_sigma - np.sqrt(
        2
        * (
            trace_rho_sigma**2
            - np.trace(np.dot(np.dot(rho, rho_delta), np.dot(rho, rho_delta)))
        )
    )
    super_fidelity = trace_rho_sigma + np.sqrt((1 - trace_rho2) * (1 - trace_sigma2))

    # Compute SSQFI bounds
    lower_ssqfi = 8 * (1 - super_fidelity) / (delta**2)
    upper_ssqfi = 8 * (1 - sub_fidelity) / (delta**2)
    return lower_ssqfi, upper_ssqfi


# Parameters
n = 2  # Number of qubits
m = 2  # Truncation parameter
delta = 0.001  # error << 1

# Generate density matrix
a_x = 1.0  # Coefficient for s_x s_x
h_x = 0.5  # Coefficient for s_z

rho_theta, rho_theta_delta = generate_rho_rho_delta(a_x, h_x, delta)
# rho_theta, rho_theta_delta = prova()

# Compute TQFI bounds
lower_tqfi, upper_tqfi = compute_tqfi_bounds(rho_theta, rho_theta_delta, m)

# Compute SSQFI bounds
lower_ssqfi, upper_ssqfi = compute_ssqfi_bounds(rho_theta, rho_theta_delta)

# Output results
print("TQFI Bounds:")
print(f"Lower TQFI Bound: {lower_tqfi}")
print(f"Upper TQFI Bound: {upper_tqfi}")

print("\nSSQFI Bounds:")
print(f"Lower SSQFI Bound: {lower_ssqfi}")
print(f"Upper SSQFI Bound: {upper_ssqfi}")
"""
## part 2, variational approach 
# from appendices of paper

from deprecated_deprecated_subroutines import variational_state_eigensolver, generalized_swap_test

print("VQSE and Generalized SWAP Test, part 2 subroutines")

# random density matrices
rho = np.random.rand(4, 4)
rho = rho @ rho.T  # posdef
rho /= np.trace(rho)  # norm

sigma = np.random.rand(4, 4)
sigma = sigma @ sigma.T
sigma /= np.trace(sigma)

# U(parameters) = Exp[ - i (a_x XX *  a_y YY *  a_z ZZ) ]
# U|00> = |psi>
# a_x a_y a_z in 0 2pi, parameter to be estimated
# XX sigma X tensore sigma X primo qubit secondo qubit
# delta can be used to increment a little parameters

import sys

sys.exit()

## variational i need to use VQSE file, other is deprecated

# VQSE
eigvals, eigvecs, params = variational_state_eigensolver(
    rho, m=2, num_qubits=2, num_params=10
)

# Generalized SWAP Test
fidelity = generalized_swap_test(rho, sigma, num_qubits=2)

print("Largest eigenvalues:", eigvals)
print("Fidelity (from SWAP test):", fidelity)
"""
