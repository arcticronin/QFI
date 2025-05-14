import pennylane as qml
import numpy as np
from math import sqrt
from scipy.optimize import minimize


import pennylane as qml
import numpy as np
from math import sqrt
from scipy.optimize import minimize


def vqfe_from_density_matrices(rho, rho_delta, L=2, m=1, maxiter=200):
    """Compute fidelity bounds between rho and rho_delta using top-m eigenvectors of rho.

    Args:
        rho (np.ndarray): Input density matrix ρ_θ (2^n x 2^n).
        rho_delta (np.ndarray): Perturbed density matrix ρ_{θ+δ}.
        L (int): Number of ansatz layers for the diagonalizing unitary.
        m (int): Number of eigenvectors to keep in truncated state.
        maxiter (int): Max steps for variational optimizer.

    Returns:
        dict: Fidelity bounds and intermediate results.
    """
    dim = rho.shape[0]
    n_qubits = int(np.log2(dim))
    dev = qml.device("default.mixed", wires=n_qubits)

    def ansatz_U(params, wires):
        for L_idx in range(params.shape[0]):
            for q in range(n_qubits):
                qml.RZ(params[L_idx, q, 0], wires=wires[q])
                qml.RY(params[L_idx, q, 1], wires=wires[q])
                qml.RZ(params[L_idx, q, 2], wires=wires[q])
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[wires[q], wires[q + 1]])

    def apply_unitary(params, matrix):
        """Apply U to matrix: U ρ U†."""

        @qml.qnode(dev)
        def rotated():
            qml.QubitDensityMatrix(matrix, wires=range(n_qubits))
            ansatz_U(params, wires=range(n_qubits))
            return qml.state()

        return rotated()

    def cost(params):
        """Cost: maximize squared diagonal elements (i.e., minimize off-diagonals)."""
        rotated_rho = apply_unitary(params, rho)
        probs = np.real(np.diag(rotated_rho))
        return -np.sum(probs**2)

    # Initialize and optimize ansatz
    init_params = 0.1 * np.random.randn(L, n_qubits, 3)
    result = minimize(
        lambda p: cost(p.reshape(L, n_qubits, 3)),
        init_params.flatten(),
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False},
    )
    opt_params = result.x.reshape(L, n_qubits, 3)

    # Rotate both rho and rho_delta into the learned eigenbasis
    rho_rot = apply_unitary(opt_params, rho)
    sigma_rot = apply_unitary(opt_params, rho_delta)

    # Get top-m eigencomponents from rotated ρ
    eigvals = np.real(np.diag(rho_rot))
    sorted_indices = np.argsort(eigvals)[::-1]
    top_indices = sorted_indices[:m]
    r_top = eigvals[top_indices]

    # Project rho_rot and sigma_rot into m-dimensional subspace
    rho_m = np.zeros_like(rho_rot)
    for idx in top_indices:
        rho_m[idx, idx] = rho_rot[idx, idx]
    sigma_subblock = sigma_rot[np.ix_(top_indices, top_indices)]

    # Compute T matrix for fidelity bounds
    sqrt_r_ir_j = np.sqrt(np.outer(r_top, r_top))
    T = sqrt_r_ir_j * np.real(sigma_subblock)
    T = 0.5 * (T + T.T)  # ensure Hermitian
    eigvals_T = np.linalg.eigvalsh(T)
    eigvals_T = np.clip(eigvals_T, 0, None)

    # Fidelity bounds
    F_trunc = np.sum(np.sqrt(eigvals_T))
    trace_rho_m = np.sum(r_top)
    trace_sigma_m = np.real(np.trace(sigma_subblock))
    F_star = F_trunc + sqrt(max(0, (1 - trace_rho_m) * (1 - trace_sigma_m)))

    return {
        "F_trunc": F_trunc,
        "F_star": F_star,
        "top_eigenvalues": r_top,
        "trace_rho_m": trace_rho_m,
        "trace_sigma_m": trace_sigma_m,
        "opt_params": opt_params,
        "rho_rotated": rho_rot,
        "sigma_rotated": sigma_rot,
    }


## automatically creates rho and rho delta using random kraus matrices (not my use but good for testing)
def vqfe_noisy_channels(
    n_qubits=2, theta=0.3, delta=0.05, L=2, m=1, noise_level=0.05, maxiter=200
):
    """Run the VQFE subroutine for estimating truncated and generalized fidelity.

    Args:
        n_qubits (int): Number of qubits.
        theta (float): Base parameter θ.
        delta (float): Small perturbation δ for θ + δ.
        L (int): Number of ansatz layers.
        m (int): Number of principal components (truncation rank).
        noise_level (float): Depolarizing noise strength.
        maxiter (int): Maximum number of optimization steps.

    Returns:
        dict: Fidelity results and diagnostic info.
    """
    dev = qml.device("default.mixed", wires=n_qubits)

    def prepare_rho(theta_val):
        qml.RY(theta_val, wires=0)
        qml.CNOT(wires=[0, 1])
        for i in range(n_qubits):
            qml.DepolarizingChannel(noise_level, wires=i)

    def ansatz_U(params, wires):
        layers = params.shape[0]
        for L_idx in range(layers):
            for q in range(n_qubits):
                qml.RZ(params[L_idx, q, 0], wires=wires[q])
                qml.RY(params[L_idx, q, 1], wires=wires[q])
                qml.RZ(params[L_idx, q, 2], wires=wires[q])
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[wires[q], wires[q + 1]])

    @qml.qnode(dev)
    def rotated_state(params, theta_val):
        prepare_rho(theta_val)
        ansatz_U(params, wires=range(n_qubits))
        return qml.state()

    def cost(params, theta_val):
        rho_matrix = rotated_state(params, theta_val)
        probs = np.real(np.diag(rho_matrix))
        return -np.sum(probs**2)

    def cost_flat(flat_params, theta_val):
        params = flat_params.reshape(L, n_qubits, 3)
        return cost(params, theta_val)

    init_params = 0.1 * np.random.randn(L, n_qubits, 3)
    init_params_flat = init_params.flatten()

    result = minimize(
        lambda p: cost_flat(p, theta),
        init_params_flat,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False},
    )

    opt_params = result.x.reshape(L, n_qubits, 3)
    rho_diag = rotated_state(opt_params, theta)
    diag_elements = np.real(np.diag(rho_diag))

    eigvals = diag_elements
    sorted_indices = np.argsort(eigvals)[::-1]
    top_indices = sorted_indices[:m]
    top_eigenvalues = eigvals[top_indices]

    rho_m = np.zeros_like(rho_diag)
    for idx in top_indices:
        rho_m[idx, idx] = eigvals[idx]

    @qml.qnode(dev)
    def rotated_sigma(params, theta_val, delta_val):
        prepare_rho(theta_val + delta_val)
        ansatz_U(params, wires=range(n_qubits))
        return qml.state()

    sigma_rot = rotated_sigma(opt_params, theta, delta)
    sigma_subblock = sigma_rot[np.ix_(top_indices, top_indices)]

    r_top = top_eigenvalues
    sqrt_r_ir_j = np.sqrt(np.outer(r_top, r_top))
    T = sqrt_r_ir_j * np.real(sigma_subblock)
    T = 0.5 * (T + T.T)

    eigvals_T = np.linalg.eigvals(T)
    eigvals_T = np.clip(np.real(eigvals_T), 0, None)

    F_trunc = np.sum(np.sqrt(eigvals_T))
    trace_rho_m = np.sum(r_top)
    trace_sigma_m = np.real(np.trace(sigma_subblock))
    F_star = F_trunc + sqrt((1 - trace_rho_m) * (1 - trace_sigma_m))

    return {
        "F_trunc": F_trunc,
        "F_star": F_star,
        "top_eigenvalues": top_eigenvalues,
        "trace_rho_m": trace_rho_m,
        "trace_sigma_m": trace_sigma_m,
        "opt_params": opt_params,
        "rho_diag": rho_diag,
        "sigma_rotated": sigma_rot,
    }
