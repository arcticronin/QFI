import numpy as np
from importlib import reload
import helper_functions

reload(helper_functions)


# eq (6)
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
    rho_trunc = helper_functions.truncate_density_matrix(rho, m)
    #
    rho_delta_trunc = helper_functions.truncate_density_matrix(rho_delta, m)

    # Step 2: Compute truncated fidelities

    # F - Fidelity between truncated rho and truncated rho_delta
    fidelity_truncated = helper_functions.trace_norm_rho_rho_delta(
        rho_trunc, rho_delta_trunc
    )

    # F* - Generalized fidelity incorporates truncation errors
    fidelity_truncated_generalized = fidelity_truncated + np.sqrt(
        max(0, (1 - np.trace(rho_trunc)) * (1 - np.trace(rho_delta_trunc)))
    )

    # Step 3: (optional) compute true fidelity
    fidelity_true = helper_functions.fidelity(rho, rho_delta, root=True, DEBUG=True)

    # Step 4: Compute TQFI bounds using fidelity definitions (they are INVERTED!!)
    lower_tqfi = 8 * (1 - fidelity_truncated_generalized) / (delta**2)
    upper_tqfi = 8 * (1 - fidelity_truncated) / (delta**2)

    ## This is an appriximation
    qfi_fidelity = 8 * (1 - fidelity_true) / (delta**2)

    # if DEBUG:
    # Intermediate results for debugging
    # print(f"Fidelity (Truncated): {fidelity_truncated}")
    # print(f"Fidelity (Generalized): {fidelity_generalized}")

    # step 5: Subfidelity and superbounds (B-2 on theoretical framework on paper)

    E = np.real(E_subfidelity(rho_trunc, rho_delta_trunc))
    R = np.real(R_superfidelity(rho_trunc, rho_delta_trunc))

    # from the paper it uses the square root of E and R in place of the fidelity ()hard for mixed states)
    sub_qfi_bound = I_induced_bound(f_theta_theta_delta=np.sqrt(R), delta=delta)

    super_qfi_bound = I_induced_bound(f_theta_theta_delta=np.sqrt(E), delta=delta)

    # create a result dictionary
    results = {
        "fidelity_truncated": fidelity_truncated,
        "fidelity_truncated_generalized": fidelity_truncated_generalized,
        "lower_tqfi": lower_tqfi,
        "upper_tqfi": upper_tqfi,
        "fidelity_true": fidelity_true,
        "qfi_fidelity": qfi_fidelity,
        "sub_qfi_bound": sub_qfi_bound,
        "super_qfi_bound": super_qfi_bound,
    }

    return results
