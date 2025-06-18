import numpy as np
from importlib import reload
import helper_functions

reload(helper_functions)

"""
TODO eigenvalue crossing can be a problem at some points, at a drop of the fidelity between 2 matrices
"""

# Globals
# Set to True to use the alternative version of truncation
# This version uses the same eigenspace of the first density matrix to project the second one (theoretically sound)
Eigenvalue_crossing_protection = True


# Eq. (6) — Induced bound on QFI from a fidelity-like quantity f(ρ_θ, ρ_θ+δ)
def I_induced_bound(f_theta_theta_delta, delta):
    return 8 * (1 - f_theta_theta_delta) / (delta**2)


# Eq. (13) — Subfidelity E(ρ, σ)
def E_subfidelity(rho, sigma):
    trace_rho_sigma = np.trace(rho @ sigma)
    term = trace_rho_sigma**2 - np.trace(rho @ sigma @ rho @ sigma)
    return trace_rho_sigma + np.sqrt(2 * term)


# Eq. (14) — Superfidelity R(ρ, σ)
def R_superfidelity(rho, sigma):
    tr_rho_sigma = np.trace(rho @ sigma)
    tr_rho2 = np.trace(rho @ rho)
    tr_sigma2 = np.trace(sigma @ sigma)

    root_term = (1 - tr_rho2) * (1 - tr_sigma2)
    root_term = max(root_term, 0)  # Ensure non-negative for numerical safety

    return np.real(tr_rho_sigma + np.sqrt(root_term))


# clean version, but try to usse just real to avoid numerical issues
def R_superfidelity_clean(rho, sigma):
    trace_rho_sigma = np.trace(rho @ sigma)
    trace_rho_squared = np.trace(rho @ rho)
    trace_sigma_squared = np.trace(sigma @ sigma)
    return trace_rho_sigma + np.sqrt(
        (1 - trace_rho_squared) * (1 - trace_sigma_squared)
    )


# √E(ρ, σ) — used in Eq. (15) as the lower bound of fidelity
def sqrt_E_subfidelity(rho, sigma):
    return np.sqrt(E_subfidelity(rho, sigma))


# √R(ρ, σ) — used in Eq. (15) as the upper bound of fidelity
def sqrt_R_superfidelity(rho, sigma):
    return np.sqrt(R_superfidelity(rho, sigma))


def compute_tqfi_bounds(rho, rho_delta, m, delta, DEBUG=False):
    """
    Compute the Truncated Quantum Fisher Information (TQFI) and SSQFI bounds.

    Parameters:
    - rho: Density matrix at parameter theta (the probe state).
    - rho_delta: Density matrix at parameter theta + delta (the error state).
    - m: Number of principal components to keep.
    - delta: Small shift in the parameter theta.
    - DEBUG: Whether to print debug information.

    Returns:
    - Dictionary with:
        - fidelity_truncated
        - fidelity_truncated_generalized
        - lower_tqfi, upper_tqfi
        - fidelity_true, qfi_fidelity
        - sub_qfi_bound (√R)
        - super_qfi_bound (√E)
        - H_delta (max of two lower bounds)
        - J_delta (min of two upper bounds)
    """
    # Step 1: Truncate density matrices (parallely using eigenvalues, works only on low rank high purity states)

    if Eigenvalue_crossing_protection == False:
        rho_trunc = helper_functions.truncate_density_matrix(rho, m)
        rho_delta_trunc = helper_functions.truncate_density_matrix(rho_delta, m)

    # Step 1 alternative version, use same eigenspace of the first one to project the second
    else:
        rho_trunc, rho_delta_trunc = (
            helper_functions.truncate_rho_and_project_rho_delta(
                rho, rho_delta, m, DEBUG=False
            )
        )

    # Step 2: Compute truncated and generalized fidelities
    fidelity_truncated = helper_functions.uhlmann_fidelity_root(
        rho_trunc, rho_delta_trunc
    )

    trace_rho = np.trace(rho_trunc)
    trace_rho_delta = np.trace(rho_delta_trunc)
    correction_term = np.sqrt(np.clip((1 - trace_rho) * (1 - trace_rho_delta), 0, 1))
    fidelity_truncated_generalized = fidelity_truncated + correction_term

    correction_term = np.sqrt(
        max(0, (1 - np.trace(rho_trunc)) * (1 - np.trace(rho_delta_trunc)))
    )

    fidelity_truncated_generalized = fidelity_truncated + correction_term

    # Step 3: Compute true fidelity (for comparison)
    fidelity_true = helper_functions.fidelity(rho, rho_delta, root=True, DEBUG=DEBUG)
    uhlmann_fidelity = helper_functions.uhlmann_fidelity_root(rho, rho_delta)
    fidelity_pennylane = helper_functions.fidelity_pennylane(rho, rho_delta)

    # Step 4: TQFI bounds from Eq. (6)
    lower_tqfi = I_induced_bound(fidelity_truncated_generalized, delta)
    upper_tqfi = I_induced_bound(fidelity_truncated, delta)
    qfi_fidelity = I_induced_bound(fidelity_true, delta)

    # Step 5: SSQFI bounds from Eq. (15)
    sqrt_E = sqrt_E_subfidelity(rho_trunc, rho_delta_trunc)
    sqrt_R = sqrt_R_superfidelity(rho_trunc, rho_delta_trunc)

    sub_qfi_bound = I_induced_bound(sqrt_R, delta)  # Lower bound
    super_qfi_bound = I_induced_bound(sqrt_E, delta)  # Upper bound

    # Step 6: Dynamics-agnostic bounds from Eq. (18)
    H_delta = max(lower_tqfi, sub_qfi_bound)
    J_delta = min(upper_tqfi, super_qfi_bound)

    return {
        "fidelity_truncated": fidelity_truncated,
        "fidelity_truncated_generalized": fidelity_truncated_generalized,
        "correction_term": correction_term,
        "fidelity_true": fidelity_true,
        "fidelity_pennylane": fidelity_pennylane,
        "lower_tqfi": lower_tqfi,
        "upper_tqfi": upper_tqfi,
        "qfi_fidelity": qfi_fidelity,
        "sub_qfi_bound": sub_qfi_bound,
        "super_qfi_bound": super_qfi_bound,
        "H_delta": H_delta,
        "J_delta": J_delta,
        "uhlmann_fidelity": uhlmann_fidelity,
    }
