import numpy as np
from importlib import reload
import helper_functions

reload(helper_functions)


def I_induced_bound(f_theta_theta_delta, delta):
    return (8 * (1 - f_theta_theta_delta)) / (delta**2)


def E_subfidelity(rho, sigma):
    trace_rho_theta = np.trace(rho @ sigma)
    E = trace_rho_theta + np.sqrt(
        2 * (((trace_rho_theta) ** 2) - np.trace(rho @ sigma @ rho @ sigma))
    )
    return E


def R_superfidelity(rho, sigma):
    R = np.trace(rho @ sigma) + np.sqrt(
        (1 - np.trace(rho @ rho)) * (1 - np.trace(sigma @ sigma))
    )
    return R


def compute_tqfi_bounds(rho, rho_delta, m, delta, DEBUG=False):
    """
    Compute the Truncated Quantum Fisher Information (TQFI) bounds.

    This method follows the approach from the theoretical framework:
    1. Perform eigenvalue decomposition of rho.
    2. Truncate to the m-largest eigenvalues (principal components).
    3. Compute truncated and generalized fidelities (L R).
    4. Estimate the lower and upper bounds of TQFI.

    Parameters:
    - rho: Density matrix at parameter theta. ("probe" state)
    - rho_delta: Density matrix at parameter theta + delta. ("error" state)
    - m: Truncation parameter for principal components.
    - delta: Small shift in parameter for derivative approximation.

    Returns:
    - results dictionary containing
        - lower_tqfi: Lower bound of TQFI.
        - upper_tqfi: Upper bound of TQFI.
        - fidelity_truncated
        - fidelity_generalized
        - true_fidelity
        - true_qfi


    """
    # Step 1: Eigenvalue decomposition (linalg.eigh returns them in ascending order)

    # rho truncated
    eigvals, eigvecs = np.linalg.eigh(rho)

    idx = (np.argsort(eigvals))[::-1]  # Descending order
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

    # repeat the above for rho_delta
    # Step 1: rho delta trunc
    eigvals_delta, eigvecs_delta = np.linalg.eigh(rho_delta)

    idx = (np.argsort(eigvals_delta))[::-1]  # Descending order
    eigvals_delta, eigvecs_delta = eigvals_delta[idx], eigvecs_delta[:, idx]
    # Step 2:
    eigvals_trunc_delta = eigvals_delta[:m]
    eigvecs_trunc_delta = eigvecs_delta[:, :m]

    # if DEBUG:
    #    print(eigvals_trunc_delta)

    rho_delta_trunc = sum(
        eigvals_trunc_delta[i]
        * np.outer(eigvecs_trunc_delta[:, i], eigvecs_trunc_delta[:, i].conj())
        for i in range(m)
    )

    # Step 3: Compute truncated fidelity

    # (Fidelity between truncated rho and truncated rho_delta)
    fidelity_truncated = np.real(np.trace(np.dot(rho_trunc, rho_delta_trunc)))

    # Generalized fidelity incorporates truncation errors
    fidelity_generalized = fidelity_truncated + np.sqrt(
        max(0, (1 - np.trace(rho_trunc)) * (1 - np.trace(rho_delta_trunc)))
    )

    # Step 3.2: (optional) compute true fidelity
    true_fidelity = helper_functions.fidelity(rho, rho_delta, DEBUG=True)

    # Step 4: Compute TQFI bounds using fidelity definitions
    lower_tqfi = 8 * (1 - fidelity_truncated) / (delta**2)
    upper_tqfi = 8 * (1 - fidelity_generalized) / (delta**2)

    ## limit d-> 0 of true_qfi = -4 * (delta**2) * true_fidelity
    ## appriximated by
    true_qfi = 8 * (1 - true_fidelity) / (delta**2)

    # if DEBUG:
    # Intermediate results for debugging
    # print(f"Fidelity (Truncated): {fidelity_truncated}")
    # print(f"Fidelity (Generalized): {fidelity_generalized}")

    # step 5: Subfidelity and superbounds (B-2 on theoretical framework on paper)

    E = np.real(E_subfidelity(rho_trunc, rho_delta_trunc))
    R = np.real(R_superfidelity(rho_trunc, rho_delta_trunc))

    sub_qfi_bound = I_induced_bound(f_theta_theta_delta=np.sqrt(E), delta=delta)

    super_qfi_bound = I_induced_bound(f_theta_theta_delta=np.sqrt(R), delta=delta)

    # create a result dictionary
    results = {
        "fidelity_truncated": fidelity_truncated,
        "fidelity_generalized": fidelity_generalized,
        "lower_tqfi": lower_tqfi,
        "upper_tqfi": upper_tqfi,
        "true_fidelity": true_fidelity,
        "true_qfi": true_qfi,
        "sub_qfi_bound": sub_qfi_bound,
        "super_qfi_bound": super_qfi_bound,
    }

    return results
