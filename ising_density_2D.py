import numpy as np
from scipy.linalg import expm
import pandas as pd

# Define Pauli matrices


def pauli_matrices():
    """
    Returns a dictionary of Pauli matrices.
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def generate_density_matrix(a_x, h_z):
    """
    Generates the density matrix rho for a two-qubit system evolving under an Ising-like Hamiltonian.

    The Hamiltonian is defined as:
        H = a_x * (X ⊗ X) + h_z * (Z ⊗ I) + h_z * (I ⊗ Z)

    Parameters:
    - a_x: Coupling coefficient for the interaction term (X ⊗ X).
    - h_z: Coefficient for the external field in the Z direction.

    Returns:
    - rho: The 2-qubit density matrix as a numpy array.
    """
    # Retrieve Pauli matrices
    paulis = pauli_matrices()
    I, X, Z = paulis["I"], paulis["X"], paulis["Z"]

    # Define the Hamiltonian: Ising interaction + external field
    H = a_x * np.kron(X, X) + h_z * np.kron(Z, I) + h_z * np.kron(I, Z)

    print(H)

    # Compute the unitary evolution operator U = exp(-iH)
    U = expm(-1j * H)

    print("la U e': ")
    print(pd.DataFrame(U))

    # Define the initial state |00> in the computational basis
    ket_00 = np.array([1, 0, 0, 0], dtype=complex)

    # Apply U to |00> to obtain the evolved state
    psi = U @ ket_00

    # Construct the density matrix rho = |psi⟩⟨psi|
    rho = np.outer(psi, np.conj(psi))

    return rho


def generate_density_matrices_with_perturbation(a_x, h_z, delta=0.01):
    """
    Generates two density matrices: one with (a_x, h_z) and another with a perturbed field (h_z + delta).

    Parameters:
    - a_x: Coupling coefficient for the interaction term (X ⊗ X).
    - h_z: Coefficient for the external field in the Z direction.
    - delta: Small perturbation to the external field h_z.

    Returns:
    - (rho, rho_perturbed): Tuple containing the original and perturbed density matrices.
    """
    return generate_density_matrix(a_x, h_z), generate_density_matrix(a_x, h_z + delta)


def generate_random_positive_density_matrix(n):
    """
    Generates a random valid density matrix for an n-qubit system.

    Parameters:
    - n: Number of qubits.

    Returns:
    - rho: A random valid density matrix (Hermitian, PSD, trace = 1).
    """
    dim = 2**n  # Dimension of the Hilbert space
    mat = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    mat = mat @ mat.conj().T  # Ensure Hermitian & positive semi-definite
    return mat / np.trace(mat)  # Normalize to ensure trace = 1
