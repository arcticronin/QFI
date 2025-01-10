import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
import sys


def prepare_vqse_circuit(num_qubits):
    """
    Creates a parameterized two-local circuit for the VQSE algorithm.

    Args:
        num_qubits (int): Number of qubits.

    Returns:
        TwoLocal: The parameterized two-local circuit.
    """
    ansatz = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        entanglement="full",
        reps=3,
        skip_final_rotation_layer=False,
        parameter_prefix="\u03b8",
    )
    return ansatz


def vqse(num_qubits, density_matrix):
    """
    Variational Quantum State Eigensolver (VQSE) using StatevectorEstimator.

    Args:
        num_qubits (int): Number of qubits.
        density_matrix (np.ndarray): The density matrix of the quantum state.

    Returns:
        eigenvalue (float): Approximated eigenvalue.
        optimal_parameters (np.ndarray): Optimal variational parameters.
    """
    # Create a parameterized ansatz circuit
    ansatz = prepare_vqse_circuit(num_qubits)

    # Convert the density matrix into a SparsePauliOp
    operator = SparsePauliOp.from_operator(Operator(density_matrix))

    # Instantiate the StatevectorEstimator
    estimator = StatevectorEstimator()

    # Define a cost function for optimization
    def cost_function(parameters):
        # Bind parameters to the ansatz
        bound_circuit = ansatz.assign_parameters(parameters)
        # Prepare the pub as (circuit, observable)
        pub = [(bound_circuit, operator)]
        # Compute the expectation value using the estimator
        job = estimator.run(pub)
        result = job.result()
        return np.real(result.values[0])  # Expectation value

    # Initialize random parameters for the ansatz
    initial_parameters = np.random.rand(ansatz.num_parameters)

    # Minimize the cost function using SciPy's optimizer
    result = minimize(cost_function, initial_parameters, method="COBYLA")

    # Return the minimized eigenvalue and optimal parameters
    return result.fun, result.x


def generalized_swap_test(num_qubits):
    """
    Creates a quantum circuit implementing the generalized SWAP test.

    Args:
        num_qubits (int): Number of qubits for the quantum states.

    Returns:
        QuantumCircuit: The SWAP test circuit.
    """
    qc = QuantumCircuit(num_qubits + 1)  # Include ancilla
    ancilla = 0
    data_qubits = list(range(1, num_qubits + 1))

    # Apply Hadamard on ancilla
    qc.h(ancilla)

    # Apply controlled SWAP gates
    for i in range(len(data_qubits) - 1):
        qc.cswap(ancilla, data_qubits[i], data_qubits[i + 1])

    # Final Hadamard on ancilla
    qc.h(ancilla)
    return qc


def compute_truncated_fidelity(eigenvalues, eigenvectors, rho):
    """
    Compute the fidelity for the truncated state.

    Args:
        eigenvalues (list): List of eigenvalues.
        eigenvectors (list): List of eigenvectors as NumPy arrays.
        rho (np.ndarray): Density matrix of the quantum state.

    Returns:
        float: Fidelity value.
    """
    fidelity = sum(
        np.sqrt(eigenvalues[i] * eigenvalues[j])
        * np.abs(eigenvectors[i].conj().T @ rho @ eigenvectors[j])
        for i in range(len(eigenvalues))
        for j in range(len(eigenvalues))
    )
    return fidelity


# Define a 2-qubit density matrix (example)
num_qubits = 2
density_matrix = np.array(
    [
        [0.5, 0.1, 0.1, 0.2],
        [0.1, 0.3, 0.0, 0.1],
        [0.1, 0.0, 0.3, 0.1],
        [0.2, 0.1, 0.1, 0.4],
    ]
)

# Run VQSE
eigenvalue, optimal_parameters = vqse(num_qubits, density_matrix)

print("Eigenvalue:", eigenvalue)
print("Optimal Parameters:", optimal_parameters)

# Prepare SWAP test circuit
swap_test = generalized_swap_test(num_qubits)

# Compute Truncated Fidelity
eigenvalues = [0.7, 0.3]
eigenvectors = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])]
truncated_fidelity = compute_truncated_fidelity(
    eigenvalues, eigenvectors, density_matrix
)

print("Truncated Fidelity:", truncated_fidelity)
