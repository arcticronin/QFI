import numpy as np
from scipy.linalg import expm

DEBUG = True


def pauli_matrices():
    """Returns the Pauli matrices."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def generate_rho(a_x, h_z):
    """
    Generates the density matrix rho using the unitary evolution defined by
    U = exp(-i (a_x s_x s_x + h_x s_z)) applied to the initial state |00>.

    Parameters:
    - a_x: Coefficient for the s_x ⊗ s_x term in the Hamiltonian.
    - h_z: Coefficient for the s_z term in the Hamiltonian.

    Returns:
    - rho: The 2-qubit density matrix as numpy array).
    """
    # Get Pauli matrices
    paulis = pauli_matrices()
    I, X, Z = paulis["I"], paulis["X"], paulis["Z"]

    # Define the Hamiltonian H = a_x * (s_x ⊗ s_x) + h_z * (s_z ⊗ I)
    # it defines system dynamics
    # H = a_x * np.kron(X, X) + h_x * np.kron(Z, I)  ## check
    H = a_x * np.kron(X, X) + h_z * np.kron(Z, I) + h_z * np.kron(I, Z)  ## check

    if DEBUG:
        print(f"Hamiltonian H = {H}")
        print("terms:")
        print(a_x * np.kron(X, X))
        print(h_z * np.kron(Z, I))
        print(h_z * np.kron(I, Z))

    # Compute the unitary operator U = exp(-iH)
    U = expm(-1j * H)

    # Define the initial state |00> in the computational basis
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)

    # Apply U to |00>
    psi = U @ ket_00

    # Construct the density matrix rho = |psi><psi|
    rho = np.outer(psi, np.conj(psi))

    return rho


def generate_rho_rho_delta(a_x, h_x, delta=0.01):
    """
    Generate rho and rho+delta using exponential

            U = exp(-i (a_x s_x s_x + h_x s_z))

    applied to the initial state |00>.
    """
    return (generate_rho(a_x, h_x), generate_rho(a_x, h_x + delta))


# State preparation: Random density matrix for demonstration
def generate_random_density_matrix(n):
    mat = np.random.rand(2**n, 2**n)
    mat = mat @ mat.T  # Ensure positive semi-definite
    return mat / np.trace(mat)  # Normalize


## when i will need a micxed state i could use a mixture of two states
# rho_mixed = p * rho1 + (1 - p) * rho2  # 0 ≤ p ≤ 1
