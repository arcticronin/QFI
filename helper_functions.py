import numpy as np
from scipy.linalg import sqrtm, eigh, svd
import qutip
import pennylane as qml


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


def uhlmann_fidelity_root(rho, sigma, tol=1e-10, strict=True):
    """
    Compute the Uhlmann fidelity (non-squared) between two density matrices.

        F(ρ, σ) = || sqrt(ρ) sqrt(σ) ||_1

    Parameters:
        rho (ndarray): First density matrix
        sigma (ndarray): Second density matrix
        tol (float): Tolerance for treating imaginary parts as numerical noise
        strict (bool): If True, raises an error when imaginary part is too large;
                       otherwise returns the real part if imaginary part is negligible

    Returns:
        float: Uhlmann fidelity in [0, 1]
    """
    # Step 1: Compute matrix square roots
    sqrt_rho = sqrtm(rho)
    sqrt_sigma = sqrtm(sigma)

    # Step 2: Product of square roots
    product = sqrt_rho @ sqrt_sigma

    # Step 3: Singular values → trace norm
    singular_vals = svd(product, compute_uv=False)
    fidelity = np.sum(singular_vals)

    # Step 4: Handle complex numerical artifacts
    if np.iscomplexobj(fidelity):
        imag_part = np.abs(np.imag(fidelity))
        if imag_part < tol * np.abs(fidelity):
            return np.real(fidelity)
        elif strict:
            raise ValueError(f"Fidelity has non-negligible imaginary part: {fidelity}")
        else:
            print(
                "Warning: fallback, in uhlmann fidelity root, try setting strict = True or ignore"
            )  # warn
            return np.real(fidelity)  # fallback
    else:
        return fidelity


def uhlmann_fidelity_root_unsafe(rho1, rho2):
    """
    Computes the Uhlmann fidelity (non-squared) between two density matrices:
        F(ρ, σ) = || sqrt(ρ) sqrt(σ) ||_1

    This is the trace norm of the product of the square roots of ρ and σ,
    which is the quantity used in fidelity-based QFI bounds (e.g. Eq. 6, 12 in the paper).

    Parameters:
        rho1: First density matrix (e.g. ρ_θ)
        rho2: Second density matrix (e.g. ρ_{θ+δ})

    Returns:
        Uhlmann fidelity (not squared): a real number in [0, 1]
    """
    sqrt_rho1 = sqrtm(rho1).astype(np.complex128)
    sqrt_rho2 = sqrtm(rho2).astype(np.complex128)
    product = sqrt_rho1 @ sqrt_rho2

    singular_vals = np.linalg.svd(product, compute_uv=False)
    return np.real(np.sum(singular_vals))


def fidelity_pennylane(rho_1, rho_2):
    return qml.math.fidelity(rho_1, rho_2)


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
        rho (np.ndarray): Density matrix (Hermitian, positive semi-definite).
        m (int): Number of largest eigenvalues to keep.
        DEBUG (bool): If True, print truncated eigenvalues.

    Returns:
        np.ndarray: Truncated density matrix (Hermitian, trace ≤ 1).
    """
    eigvals, eigvecs = compute_eigen_decomposition(rho)  # Already sorted

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


def truncate_rho_and_project_rho_delta(rho, rho_delta, m, DEBUG=False):
    """
    Truncate `rho` to its top-m eigencomponents and project `rho_delta`
    onto the same eigenvectors.

    Args:
        rho (np.ndarray): Density matrix to truncate.
        rho_delta (np.ndarray): Matrix to project on `rho`'s eigenbasis.
        m (int): Number of top eigenvectors/eigenvalues to retain.
        DEBUG (bool): If True, print debug info.

    Returns:
        tuple:
            - rho_trunc (np.ndarray): Truncated version of `rho`.
            - rho_delta_proj (np.ndarray): `rho_delta` projected in same truncated basis.
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argsort(eigvals)[::-1]  # Descending
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_trunc = eigvals[:m]
    eigvecs_trunc = eigvecs[:, :m]

    if DEBUG:
        print("Top-m eigenvalues of rho:", eigvals_trunc)

    # Truncate rho using its eigenvalues and eigenvectors
    rho_trunc = eigvecs_trunc @ np.diag(eigvals_trunc) @ eigvecs_trunc.conj().T

    # Project rho_delta onto the same eigenvectors (in full space)
    rho_delta_proj = (
        eigvecs_trunc
        @ (eigvecs_trunc.conj().T @ rho_delta @ eigvecs_trunc)
        @ eigvecs_trunc.conj().T
    )

    return rho_trunc, rho_delta_proj


def get_truncated_eigen_decomposition(rho, m):
    """
    Just used as debugging info in pipeline
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
    A Haar ket is a pure quantum state (a normalized complex vector) that is uniformly distributed according to the
    Haar measure over the complex unit sphere in a Hilbert space.

    Idea: take both real and im part from a normal distribution, the joint probability distribution of the components becomes rotationally invariant under unitary transformations due to the properties of the normal distribution.

    The normalization ensures that the ket lies on the unit sphere, while the rotational invariance ensures the sampling is uniform according to the Haar measure.

    Args:
        n (int): Dimension of the ket.

    Returns:
        numpy.ndarray: A normalized complex vector (ket) of dimension n.
    """

    # random complex vector with normally distributed components
    z = np.random.randn(n) + 1j * np.random.randn(n)
    # morlmalize
    ket = z / np.linalg.norm(z)

    return ket


def random_mixed_density_matrix(N, n):
    print("Generating random mixed density matrix")
    # Generate random initial state
    ket = random_haar_ket(2**N)

    rho = np.outer(ket, np.conj(ket))

    trace_out_index = np.random.choice(range(N), size=N - n, replace=False)

    return trace_out(rho=rho, trace_out_index=trace_out_index)
