import numpy as np


def compute_tqfi_bounds(rho, rho_delta, m, delta, DEBUG=False):
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
    # Step 1: Eigenvalue decomposition (linalg.eigh returns them in ascending order)
    eigvals, eigvecs = np.linalg.eigh(rho)

    idx = (np.argsort(eigvals))[::-1]  # Descending order
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Step 2: Truncate to m-largest eigenvalues/eigenvectors
    eigvals_trunc = eigvals[:m]
    eigvecs_trunc = eigvecs[:, :m]

    if DEBUG:
        print(eigvals_trunc)

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
    lower_tqfi = (
        8 * (1 - fidelity_generalized) / (delta**2)
    )  # TODO check this delta on the paper
    upper_tqfi = 8 * (1 - fidelity_truncated) / (delta**2)

    if DEBUG:
        # Intermediate results for debugging
        print(f"Fidelity (Truncated): {fidelity_truncated}")
        print(f"Fidelity (Generalized): {fidelity_generalized}")

    return lower_tqfi, upper_tqfi
