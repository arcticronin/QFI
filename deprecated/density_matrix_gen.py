import numpy as np
import pandas as pd


def pauli_matrices():
    """Returns the Pauli matrices"""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def density_matrix_from_pauli_basis(coefficients, delta=0.0):
    """
    Constructs a density matrix for a 2-qubit system using Pauli basis.
    For this task i just need the XX, YY, ZZ coefficients, but i will keep the general case.

    Parameters:
    coefficients (dict): Keys are strings like 'II', 'IX', etc., and values are coefficients.

    Returns:
    numpy.ndarray: The 2-qubit density matrix.
    """
    # Get Pauli matrices
    paulis = pauli_matrices()

    # Initialize density matrix
    dim = 4  # 2-qubit system
    rho = np.zeros((dim, dim), dtype=complex)

    # Build the density matrix
    for key, coeff in coefficients.items():
        # Parse Pauli operators
        sigma1, sigma2 = paulis[key[0]], paulis[key[1]]
        rho += coeff * np.kron(sigma1, sigma2)

    rho_delta = np.zeros((dim, dim), dtype=complex)
    for key, coeff in coefficients.items():
        # Parse Pauli operators
        sigma1, sigma2 = paulis[key[0]], paulis[key[1]]
        rho_delta += (coeff + delta) * np.kron(sigma1, sigma2)

    rho /= np.trace(rho)  ## normalize
    rho_delta /= np.trace(rho_delta)  ## normalize

    return rho, rho_delta


# # Example usage for the general case
# coefficients = {
#     "II": 1.0,
#     "IX": 0.0,
#     "IY": 0.0,
#     "IZ": 0.0,
#     "XI": 0.0,
#     "XX": 0.5,
#     "XY": 0.0,
#     "XZ": 0.0,
#     "YI": 0.0,
#     "YX": 0.0,
#     "YY": -0.5,
#     "YZ": 0.0,
#     "ZI": 0.0,
#     "ZX": 0.0,
#     "ZY": 0.0,
#     "ZZ": 0.5,
# }

# case for our project
coefficients = {
    "II": 1.0,  ## ensures non zero trace (so i can ALWAYs normalize it later on)
    "XX": 0.5,
    "YY": 0.2,
    "ZZ": 0.5,
}

rho, rho_delta = density_matrix_from_pauli_basis(coefficients, delta=0.01)


def prova():
    return rho, rho_delta


print("\nDensity Matrix rho:\n", pd.DataFrame(rho))
# print("Trace of Density Matrix rho:", np.trace(rho))

print("\nDensity Matrix rho+delta:\n", pd.DataFrame(rho_delta))
# print("Trace of Density Matrix rho+delta:", np.trace(rho_delta))

diff = rho - rho_delta
print("\ndifference:\n", pd.DataFrame(diff))
