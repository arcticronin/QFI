import pennylane as qml
import numpy as np
from math import sqrt
from scipy.optimize import minimize

# Device for mixed-state simulation (using a small number of qubits for illustration)
n_qubits = 2
dev = qml.device("default.mixed", wires=n_qubits)


# Example state preparation for ρ_θ (noisy entangled state on 2 qubits)
def prepare_rho(theta):
    # Prepare a Bell-like state with parameter theta
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    # Apply some depolarizing noise on each qubit (to simulate a mixed state)
    qml.DepolarizingChannel(0.05, wires=0)
    qml.DepolarizingChannel(0.05, wires=1)


# Parameterized ansatz for the diagonalizing unitary U (with L layers of single-qubit rotations + CNOTs)
def ansatz_U(params, wires):
    # params shape: (L, n_qubits, 3) for RZ-RY-RZ rotations on each qubit per layer
    layers = params.shape[0]
    for L_idx in range(layers):
        # Single-qubit rotations (Euler angles) on each qubit
        for q in range(n_qubits):
            qml.RZ(params[L_idx, q, 0], wires=wires[q])
            qml.RY(params[L_idx, q, 1], wires=wires[q])
            qml.RZ(params[L_idx, q, 2], wires=wires[q])
        # Entangling layer (CNOT chain)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[wires[q], wires[q + 1]])


# QNode to apply U to ρ_θ and return the density matrix (for computing the cost)
@qml.qnode(dev)
def rotated_state(params, theta):
    prepare_rho(theta)
    ansatz_U(params, wires=range(n_qubits))
    return qml.state()  # returns the density matrix of U ρ_θ U†


# Cost function: negative sum of squared diagonal entries of U ρ_θ U†
# (Maximizing this is equivalent to minimizing off-diagonals:contentReference[oaicite:5]{index=5})
def cost(params, theta):
    rho_matrix = rotated_state(params, theta)
    # Get diagonal probabilities of the rotated state
    probs = np.real(np.diag(rho_matrix))
    return -np.sum(probs**2)


# Initialize ansatz parameters (e.g., 2 layers of rotations for a 2-qubit system)
L = 2
init_params = 0.1 * np.random.randn(L, n_qubits, 3)

# Optimize the ansatz parameters using a gradient-free optimizer (COBYLA) for stability
theta = 0.3  # example parameter value for ρ_θ
result = minimize(
    lambda p: cost(p, theta),
    init_params,
    method="COBYLA",
    options={"maxiter": 200, "disp": False},
)
opt_params = result.x.reshape(L, n_qubits, 3)

# After optimization, obtain the diagonalized state ρ'_θ = U ρ_θ U†
rho_diag = rotated_state(opt_params, theta)
diag_elements = np.real(np.diag(rho_diag))
print("Optimized diagonal of ρ':", np.round(diag_elements, 4))
print("Off-diagonal norm^2:", np.linalg.norm(rho_diag - np.diag(diag_elements)) ** 2)


# STEP 2


# Determine the top-m eigenvalues and their indices
m = 1  # for example, take the largest eigenvalue
eigvals = np.real(np.diag(rho_diag))
sorted_indices = np.argsort(eigvals)[::-1]  # indices sorted by eigenvalue, descending
top_indices = sorted_indices[:m]
top_eigenvalues = eigvals[top_indices]
# Construct truncated ρ_θ^(m) as a diagonal matrix of top eigenvalues (in the same basis U diagonalizes ρ)
rho_m = np.zeros_like(rho_diag)
for idx in top_indices:
    rho_m[idx, idx] = eigvals[idx]
print(f"Top-{m} eigenvalues of ρ_θ:", np.round(top_eigenvalues, 4))
print(f"Tr(ρ_θ^(m)) =", np.round(np.trace(rho_m), 4))


# STEP 3


# Prepare the slightly perturbed state ρ_{θ+δ} and express it in ρ_θ's eigenbasis using U
delta = 0.05  # small perturbation


@qml.qnode(dev)
def rotated_sigma(params, theta_val, delta_val):
    # Prepare ρ_{θ+δ} on the device
    prepare_rho(theta_val + delta_val)
    ansatz_U(
        params, wires=range(n_qubits)
    )  # apply the same U to rotate into ρ_θ's eigenbasis
    return qml.state()


sigma_rot = rotated_sigma(opt_params, theta, delta)
# σ_rot is U ρ_{θ+δ} U† (matrix of size 2^n_qubits x 2^n_qubits)
# Extract the top-m x top-m sub-block corresponding to ρ_θ^(m) subspace
sigma_subblock = sigma_rot[np.ix_(top_indices, top_indices)]
print(f"Projected σ in top-{m} subspace:\n", np.round(sigma_subblock, 4))
print("Tr(σ_proj) =", np.round(np.trace(sigma_subblock), 4))


# STEP 4

# Build the T matrix: T_ij = sqrt(r_i r_j) * sigma_subblock_ij  (r_i are top eigenvalues of ρ)
r_top = top_eigenvalues  # eigenvalues of ρ_θ in the truncated subspace
# Form matrix of sqrt(r_i * r_j) for i,j in top subspace
sqrt_r_ir_j = np.sqrt(np.outer(r_top, r_top))
T = sqrt_r_ir_j * np.real(sigma_subblock)  # elementwise multiplication
# Ensure T is Hermitian (numerical symmetrization)
T = 0.5 * (T + T.T)
# Eigenvalues of T (should be non-negative)
eigvals_T = np.linalg.eigvals(T)
eigvals_T = np.clip(np.real(eigvals_T), 0, None)
# Truncated fidelity = sum of square roots of eigenvalues of T:contentReference[oaicite:23]{index=23}
F_trunc = np.sum(np.sqrt(eigvals_T))
# Generalized fidelity = F_trunc + sqrt((1 - Tr(rho_m))*(1 - Tr(sigma_subblock))):contentReference[oaicite:24]{index=24}
trace_rho_m = np.sum(r_top)
trace_sigma_m = np.real(np.trace(sigma_subblock))
F_star = F_trunc + sqrt((1 - trace_rho_m) * (1 - trace_sigma_m))

print("Truncated fidelity F =", np.round(F_trunc, 6))
print("Generalized fidelity F* =", np.round(F_star, 6))
