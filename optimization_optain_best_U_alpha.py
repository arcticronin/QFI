import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer


# Define helper functions (same as before)
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

    # Compute truncated fidelity
    fidelity_truncated = np.trace(np.dot(rho_trunc, rho_delta))

    # Compute generalized fidelity
    fidelity_generalized = fidelity_truncated + np.sqrt(
        (1 - np.trace(rho_trunc)) * (1 - np.trace(rho_delta))
    )

    # Compute TQFI bounds
    lower_tqfi = 8 * (1 - fidelity_generalized) / (delta**2)
    upper_tqfi = 8 * (1 - fidelity_truncated) / (delta**2)
    return lower_tqfi, upper_tqfi


def generate_random_density_matrix(n):
    mat = np.random.rand(2**n, 2**n)
    mat = mat @ mat.T  # Ensure positive semi-definite
    return mat / np.trace(mat)  # Normalize


# Variational unitary definition
def apply_variational_unitary(rho_in, params, n):
    """Applies a parameterized unitary to rho_in."""
    # Build the quantum circuit
    qc = QuantumCircuit(n)
    idx = 0
    for _ in range(2):  # Two layers for simplicity
        for qubit in range(n):
            qc.ry(params[idx], qubit)  # Parameterized Y-rotation
            idx += 1
        for qubit in range(n - 1):
            qc.cx(qubit, qubit + 1)  # Entangling CNOT gate

    # Convert circuit to unitary matrix
    backend = Aer.get_backend("unitary_simulator")
    qc = transpile(qc, backend)
    unitary = Operator(qc).data  # Get the unitary matrix

    # Apply the unitary transformation
    rho_out = unitary @ rho_in @ unitary.conj().T
    return rho_out


# Optimization loop
def cost_function(params, rho_in, rho_theta_delta, n, m, delta):
    """Cost function to minimize: negative lower TQFI bound."""
    rho_alpha = apply_variational_unitary(rho_in, params, n)
    lower_tqfi, _ = compute_tqfi_bounds(rho_alpha, rho_theta_delta, m)
    return -lower_tqfi  # Negative because we maximize QFI


# Parameters
n = 2  # Number of qubits
m = 2  # Truncation parameter
delta = 0.001  # Small parameter shift
num_params = n * 2  # Two layers of parameterized gates

# Generate input states
rho_in = generate_random_density_matrix(n)
rho_theta_delta = generate_random_density_matrix(n)

# Initialize parameters for optimization
initial_params = np.random.rand(num_params)

# Optimize the parameters
result = minimize(
    cost_function,
    initial_params,
    args=(rho_in, rho_theta_delta, n, m, delta),
    method="COBYLA",  # Constrained optimization
    options={"maxiter": 200},
)

# Get optimized parameters and results
optimized_params = result.x
optimized_rho = apply_variational_unitary(rho_in, optimized_params, n)
final_lower_tqfi, final_upper_tqfi = compute_tqfi_bounds(
    optimized_rho, rho_theta_delta, m
)

# Output results
print("Optimized Parameters:", optimized_params)
print(f"Final Lower TQFI Bound: {final_lower_tqfi}")
print(f"Final Upper TQFI Bound: {final_upper_tqfi}")
