import pennylane as qml
import numpy as np
from math import sqrt
from scipy.optimize import minimize
import pennylane as qml

import pennylane as qml
from pennylane import (
    numpy as np,
)  # Use PennyLane's numpy for potential autograd compatibility
from scipy.optimize import minimize
from math import sqrt


def ansatz_U(params, wires_to_act_on, L):
    """
    Hardware-efficient Ansatz for variational diagonalization.
    params shape: (L, n_subsystem_qubits, 3)
    """
    if not wires_to_act_on:
        return
    n_subsystem_qubits = len(wires_to_act_on)
    for l_idx in range(L):  # L layers
        for i, q_idx in enumerate(wires_to_act_on):  # iterate over actual wire indices
            qml.RZ(params[l_idx, i, 0], wires=q_idx)
            qml.RY(params[l_idx, i, 1], wires=q_idx)
            qml.RZ(params[l_idx, i, 2], wires=q_idx)
        # Apply CNOTs only if there's more than one qubit in the subsystem
        if n_subsystem_qubits > 1:
            for i in range(n_subsystem_qubits - 1):
                qml.CNOT(wires=[wires_to_act_on[i], wires_to_act_on[i + 1]])


def vqse_subroutine(
    circuit_fn,
    total_num_qubits,
    target_wires,
    L=2,
    maxiter=200,
):
    """
    Variational Quantum State Eigensolver (VQSE) subroutine
    to variationally diagonalize a quantum state prepared on target_wires.

    Args:
        circuit_fn (callable): Function that prepares a 'total_num_qubits'-qubit state.
                               It's assumed this prepares the target state on 'target_wires'.
        total_num_qubits (int): Total number of qubits the circuit_fn operates on and the device uses.
        target_wires (list[int]): List of qubit indices for the state to be diagonalized.
        L (int): Number of ansatz layers for variational diagonalization.
        maxiter (int): Optimization steps for variational diagonalization.

    Returns:
        tuple: (optimized_params, rotated_density_matrix)
    """
    n_subsystem_qubits = len(target_wires)
    dev = qml.device("default.mixed", wires=total_num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit_target_rotated(params_vqse):
        circuit_fn()  # Prepare the initial state
        ansatz_U(params_vqse, wires_to_act_on=target_wires, L=L)
        return qml.density_matrix(wires=target_wires)

    def cost_vqse(params_flat):
        # Params are flattened by scipy.optimize.minimize, reshape them.
        params_vqse = params_flat.reshape(L, n_subsystem_qubits, 3)
        rho_rot = circuit_target_rotated(params_vqse)
        # Cost function aims to maximize the sum of squares of diagonal elements of rho_rot
        # This encourages rho_rot to be diagonal in the computational basis[cite: 307].
        diag_elements = np.real(np.diag(rho_rot))
        return -np.sum(diag_elements**2)

    init_params_shape = (L, n_subsystem_qubits, 3)
    init_params_flat = (0.1 * np.random.random_sample(size=init_params_shape)).flatten()

    result = minimize(
        cost_vqse,
        init_params_flat,
        method="COBYLA",  # COBYLA is a gradient-free method
        options={"maxiter": maxiter, "disp": False},
    )
    opt_params_flat = result.x
    opt_params_vqse = opt_params_flat.reshape(L, n_subsystem_qubits, 3)

    rotated_density_matrix = circuit_target_rotated(opt_params_vqse)

    return opt_params_vqse, rotated_density_matrix


def vqfe_main(
    circuit_fn,
    total_num_qubits,
    active_rho_wires,
    active_rho_delta_wires,
    L=2,
    m=1,
    maxiter=200,
):
    """
    Compute truncated and generalized fidelity between two quantum states ρ and ρ_delta,
    defined on disjoint sets of qubits 'active_rho_wires' and 'active_rho_delta_wires' respectively,
    both prepared within a larger 'total_num_qubits'-qubit circuit, by variationally diagonalizing ρ.

    Args:
        circuit_fn (callable): Function that prepares a 'total_num_qubits'-qubit state.
                               It's assumed that this function prepares the desired states
                               on the qubits that will be specified by active_rho_wires
                               and active_rho_delta_wires.
        total_num_qubits (int): Total number of qubits the circuit_fn operates on and the device uses.
        active_rho_wires (list[int]): List of qubit indices for the first state (ρ).
        active_rho_delta_wires (list[int]): List of qubit indices for the second state (ρ_delta).
                                             Must be disjoint from active_rho_wires and have the same length.
        L (int): Number of ansatz layers for variational diagonalization.
        m (int): Truncation rank (top-m eigenvectors).
        maxiter (int): Optimization steps for variational diagonalization.

    Returns:
        dict: {F_trunc, F_star, top_eigenvalues, opt_params, rotated matrices, etc.}
    """
    if not isinstance(active_rho_wires, (list, tuple)) or not isinstance(
        active_rho_delta_wires, (list, tuple)
    ):
        raise ValueError(
            "active_rho_wires and active_rho_delta_wires must be lists or tuples."
        )

    if len(active_rho_wires) != len(active_rho_delta_wires):
        raise ValueError(
            "active_rho_wires and active_rho_delta_wires must have the same length."
        )
    if not active_rho_wires:  # Handles case of empty list
        raise ValueError("active_rho_wires cannot be empty.")

    # Check for disjointness
    if set(active_rho_wires) & set(active_rho_delta_wires):
        raise ValueError(
            "active_rho_wires and active_rho_delta_wires must be disjoint."
        )

    n_subsystem_qubits = len(active_rho_wires)

    # Ensure all active wires are within the total_num_qubits range
    all_active_wires = set(active_rho_wires) | set(active_rho_delta_wires)
    if max(all_active_wires, default=-1) >= total_num_qubits:
        raise ValueError("All active wire indices must be less than total_num_qubits.")
    if min(all_active_wires, default=0) < 0:
        raise ValueError("Wire indices cannot be negative.")

    dev = qml.device("default.mixed", wires=total_num_qubits)

    # Step 1: Variational Diagonalization of ρ using VQSE subroutine [cite: 303]
    opt_params_rho, rho_rot_final = vqse_subroutine(
        circuit_fn, total_num_qubits, active_rho_wires, L, maxiter
    )

    # Step 2: Apply the SAME optimized parameters to ρ_delta [cite: 319]
    @qml.qnode(dev, interface="autograd")
    def circuit_rho_delta_rotated_after_vqse():
        circuit_fn()  # Prepare the initial state on all total_num_qubits
        ansatz_U(opt_params_rho, wires_to_act_on=active_rho_delta_wires, L=L)
        return qml.density_matrix(wires=active_rho_delta_wires)

    sigma_rot_final = circuit_rho_delta_rotated_after_vqse()

    # Step 3: Extract eigenvalues from the diagonalized rho [cite: 309]
    eigvals_rho = np.real(np.diag(rho_rot_final))
    sorted_indices = np.argsort(eigvals_rho)[::-1]  # Sort in descending order
    top_indices = sorted_indices[
        :m
    ]  # Select top-m eigenvalues/eigenvectors [cite: 315]

    r_top = eigvals_rho[top_indices]  # Top-m eigenvalues of rotated rho

    # Step 4: Construct the submatrix of rotated sigma corresponding to the top-m eigenvectors of rho [cite: 315]
    sigma_sub = sigma_rot_final[np.ix_(top_indices, top_indices)]

    # Step 5: Compute the T matrix for truncated fidelity calculation [cite: 316]
    r_top_clipped = np.clip(r_top, 0, None)  # Ensure non-negative for sqrt
    sqrt_r_ir_j = np.sqrt(np.outer(r_top_clipped, r_top_clipped))

    T = sqrt_r_ir_j * np.real(sigma_sub)
    T = 0.5 * (T + T.T)  # Symmetrize T

    # Step 6: Compute truncated fidelity (F_trunc) [cite: 316, 322]
    eigvals_T = np.clip(np.linalg.eigvalsh(T), 0, None)  # Ensure non-negative for sqrt
    F_trunc = np.sum(np.sqrt(eigvals_T))

    # Step 7: Compute generalized fidelity (F_star) [cite: 316]
    trace_rho_m = np.sum(r_top)  # Trace of the truncated (rotated) rho
    trace_sigma_m = np.real(
        np.trace(sigma_sub)
    )  # Trace of the submatrix of (rotated) sigma

    term_under_sqrt = (1 - trace_rho_m) * (1 - trace_sigma_m)
    F_star = F_trunc + sqrt(max(0, term_under_sqrt))

    return {
        "F_trunc": F_trunc,
        "F_star": F_star,
        "top_eigenvalues_rho": r_top,
        "trace_rho_m": trace_rho_m,
        "trace_sigma_m": trace_sigma_m,
        "opt_params_vqse": opt_params_rho,
        "rho_rotated_final": rho_rot_final,
        "sigma_rotated_final": sigma_rot_final,
        "T_matrix_eigvals": eigvals_T,
    }


#### old implementations


def vqfe_from_circuit_disjoint(
    circuit_fn,
    total_num_qubits,  # Total wires the circuit_fn acts on
    active_rho_wires,  # List/tuple of wires for the first subsystem
    active_rho_delta_wires,  # List/tuple of wires for the second subsystem
    L=2,
    m=1,
    maxiter=200,
):
    """
    Compute truncated and generalized fidelity between two quantum states ρ and ρ_delta,
    defined on disjoint sets of qubits 'active_rho_wires' and 'active_rho_delta_wires' respectively,
    both prepared within a larger 'total_num_qubits'-qubit circuit, by variationally diagonalizing ρ.

    Args:
        circuit_fn (callable): Function that prepares a 'total_num_qubits'-qubit state.
                               It's assumed that this function prepares the desired states
                               on the qubits that will be specified by active_rho_wires
                               and active_rho_delta_wires.
        total_num_qubits (int): Total number of qubits the circuit_fn operates on and the device uses.
        active_rho_wires (list[int]): List of qubit indices for the first state (ρ).
        active_rho_delta_wires (list[int]): List of qubit indices for the second state (ρ_delta).
                                           Must be disjoint from active_rho_wires and have the same length.
        L (int): Number of ansatz layers for variational diagonalization.
        m (int): Truncation rank (top-m eigenvectors).
        maxiter (int): Optimization steps for variational diagonalization.

    Returns:
        dict: {F_trunc, F_star, top_eigenvalues, opt_params, rotated matrices, etc.}
    """
    if not isinstance(active_rho_wires, (list, tuple)) or not isinstance(
        active_rho_delta_wires, (list, tuple)
    ):
        raise ValueError(
            "active_rho_wires and active_rho_delta_wires must be lists or tuples."
        )

    if len(active_rho_wires) != len(active_rho_delta_wires):
        raise ValueError(
            "active_rho_wires and active_rho_delta_wires must have the same length."
        )
    if not active_rho_wires:  # Handles case of empty list
        raise ValueError("active_rho_wires cannot be empty.")

    # Check for disjointness
    if set(active_rho_wires) & set(active_rho_delta_wires):
        raise ValueError(
            "active_rho_wires and active_rho_delta_wires must be disjoint."
        )

    n_subsystem_qubits = len(active_rho_wires)

    # Ensure all active wires are within the total_num_qubits range
    all_active_wires = set(active_rho_wires) | set(active_rho_delta_wires)
    if max(all_active_wires, default=-1) >= total_num_qubits:
        raise ValueError("All active wire indices must be less than total_num_qubits.")
    if min(all_active_wires, default=0) < 0:
        raise ValueError("Wire indices cannot be negative.")

    dev = qml.device("default.mixed", wires=total_num_qubits)

    def ansatz_U(params, wires_to_act_on):  # Renamed for clarity
        # params shape: (L, n_subsystem_qubits, 3)
        for l_idx in range(params.shape[0]):  # L layers
            for i, q_idx in enumerate(
                wires_to_act_on
            ):  # iterate over actual wire indices
                qml.RZ(params[l_idx, i, 0], wires=q_idx)
                qml.RY(params[l_idx, i, 1], wires=q_idx)
                qml.RZ(params[l_idx, i, 2], wires=q_idx)
            # Apply CNOTs only if there's more than one qubit in the subsystem
            if len(wires_to_act_on) > 1:
                for i in range(len(wires_to_act_on) - 1):
                    qml.CNOT(wires=[wires_to_act_on[i], wires_to_act_on[i + 1]])

    # QNode to apply ansatz U to the rho subsystem and get its density matrix
    @qml.qnode(
        dev, interface="autograd"
    )  # Use autograd if using scipy minimize with pennylane numpy
    def circuit_rho_rotated(params_rho):
        circuit_fn()  # Prepare the initial state on all total_num_qubits
        ansatz_U(params_rho, wires_to_act_on=active_rho_wires)
        return qml.density_matrix(wires=active_rho_wires)

    # QNode to apply the SAME ansatz U (optimized for rho) to the rho_delta subsystem
    # and get its density matrix
    @qml.qnode(dev, interface="autograd")
    def circuit_rho_delta_rotated(params_rho):  # Takes params optimized for rho
        circuit_fn()  # Prepare the initial state on all total_num_qubits
        # IMPORTANT: The variational parameters 'params_rho' were optimized to diagonalize rho.
        # We apply these same parameters to the rho_delta subsystem.
        ansatz_U(params_rho, wires_to_act_on=active_rho_delta_wires)
        return qml.density_matrix(wires=active_rho_delta_wires)

    def cost(params_flat):
        # Params are flattened by scipy.optimize.minimize, reshape them.
        params_rho = params_flat.reshape(L, n_subsystem_qubits, 3)
        rho_rot = circuit_rho_rotated(params_rho)
        # Cost function aims to maximize the sum of squares of diagonal elements of rho_rot
        # This encourages rho_rot to be diagonal in the computational basis.
        diag_elements = np.real(np.diag(rho_rot))
        return -np.sum(diag_elements**2)

    # Initialize parameters for the ansatz U
    init_params_shape = (L, n_subsystem_qubits, 3)
    init_params_flat = (0.1 * np.random.random_sample(size=init_params_shape)).flatten()
    # Scipy optimizer expects a 1D array for parameters
    # print(f"Initial parameters shape for optimizer: {init_params_flat.shape}")

    # Variational optimization to find parameters that diagonalize rho
    result = minimize(
        cost,
        init_params_flat,
        method="COBYLA",  # COBYLA is a gradient-free method
        options={"maxiter": maxiter, "disp": False},
    )
    opt_params_flat = result.x
    opt_params_rho = opt_params_flat.reshape(L, n_subsystem_qubits, 3)

    # Get the rotated density matrices using the optimized parameters
    rho_rot_final = circuit_rho_rotated(opt_params_rho)
    # For sigma_rot, we apply the U optimized for rho to the rho_delta subsystem
    sigma_rot_final = circuit_rho_delta_rotated(opt_params_rho)

    # Extract eigenvalues (diagonal elements of rho_rot_final)
    eigvals_rho = np.real(np.diag(rho_rot_final))
    sorted_indices = np.argsort(eigvals_rho)[::-1]  # Sort in descending order
    top_indices = sorted_indices[:m]  # Select top-m eigenvalues/eigenvectors

    r_top = eigvals_rho[top_indices]  # Top-m eigenvalues of rotated rho

    # Construct the submatrix of rotated sigma corresponding to the top-m eigenvectors of rho
    # sigma_sub = sigma_rot_final[np.ix_(top_indices, top_indices)]
    # Ensure top_indices are valid for indexing if m > 2**n_subsystem_qubits
    # (though typically m <= 2**n_subsystem_qubits)
    # For Pennylane's numpy, direct slicing should work if it behaves like NumPy
    # If sigma_rot_final is a standard NumPy array, np.ix_ is fine.
    # Let's try direct slicing first, which is more common with qml.math
    # However, np.ix_ is the correct way for this kind of advanced indexing.
    # Since we used `interface="autograd"` and `pennylane.numpy`, np.ix_ should be fine.
    sigma_sub = sigma_rot_final[np.ix_(top_indices, top_indices)]

    # Compute the T matrix for truncated fidelity calculation
    # sqrt_r_ir_j = np.sqrt(np.outer(r_top, r_top))
    # Ensure r_top elements are non-negative for sqrt
    r_top_clipped = np.clip(r_top, 0, None)
    sqrt_r_ir_j = np.sqrt(np.outer(r_top_clipped, r_top_clipped))

    T = sqrt_r_ir_j * np.real(sigma_sub)
    T = 0.5 * (T + T.T)  # Symmetrize T

    # Eigenvalues of T must be non-negative for sqrt
    eigvals_T = np.clip(np.linalg.eigvalsh(T), 0, None)
    F_trunc = np.sum(np.sqrt(eigvals_T))

    trace_rho_m = np.sum(r_top)  # Trace of the truncated (rotated) rho
    trace_sigma_m = np.real(
        np.trace(sigma_sub)
    )  # Trace of the submatrix of (rotated) sigma

    # Generalized fidelity F_star
    term_under_sqrt = (1 - trace_rho_m) * (1 - trace_sigma_m)
    F_star = F_trunc + sqrt(max(0, term_under_sqrt))  # Using global sqrt

    return {
        "F_trunc": F_trunc,
        "F_star": F_star,
        "top_eigenvalues_rho": r_top,  # Eigenvalues of rotated rho (should be ~diagonal)
        "trace_rho_m": trace_rho_m,
        "trace_sigma_m": trace_sigma_m,
        "opt_params": opt_params_rho,
        "rho_rotated_final": rho_rot_final,
        "sigma_rotated_final": sigma_rot_final,  # This is U_rho_opt^{\dagger} rho_delta U_rho_opt
        "T_matrix_eigvals": eigvals_T,
        "cost_value": result.fun,
    }


def vqfe_from_circuit(circuit_fn, active_rho, active_rho_delta, L=2, m=1, maxiter=200):
    """
    Variational Quantum Fidelity Estimation between two reduced states ρ and ρ_delta.

    Args:
        circuit_fn (callable): Prepares a 2n-qubit state.
        active_rho (list[int]): Wires for the reduced state ρ.
        active_rho_delta (list[int]): Wires for the reduced state ρ_delta.
        L (int): Ansatz depth.
        m (int): Truncation rank.
        maxiter (int): Optimization steps.

    Returns:
        dict: Fidelity data and rotated matrices.
    """
    # Ensure active_rho and active_rho_delta same lenght
    if len(active_rho) != len(active_rho_delta):
        raise ValueError("active_rho and active_rho_delta must have the same length.")

    # try to fix but i dont think it works
    # dev = qml.device("default.mixed", wires=max(active_rho + active_rho_delta) + 1)

    all_wires = list(set(active_rho + active_rho_delta))
    dev = qml.device("default.mixed", wires=all_wires)

    n_active = len(active_rho)

    def ansatz_U(params, wires):
        for l in range(params.shape[0]):
            for i, q in enumerate(wires):
                qml.RZ(params[l, i, 0], wires=q)
                qml.RY(params[l, i, 1], wires=q)
                qml.RZ(params[l, i, 2], wires=q)
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

    def apply(params, wires):
        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            ansatz_U(params, wires)
            return qml.density_matrix(wires=wires)

        return circuit()

    def cost(params):
        rho_rot = apply(params, active_rho)
        diag = np.real(np.diag(rho_rot))
        return -np.sum(diag**2)

    init_params = 0.1 * np.random.randn(L, n_active, 3)
    result = minimize(
        lambda p: cost(p.reshape(L, n_active, 3)),
        init_params.flatten(),
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False},
    )
    opt_params = result.x.reshape(L, n_active, 3)

    rho_rot = apply(opt_params, active_rho)
    sigma_rot = apply(opt_params, active_rho_delta)

    eigvals = np.real(np.diag(rho_rot))
    sorted_indices = np.argsort(eigvals)[::-1]
    top_indices = sorted_indices[:m]
    r_top = eigvals[top_indices]
    sigma_sub = sigma_rot[np.ix_(top_indices, top_indices)]

    sqrt_r_ir_j = np.sqrt(np.outer(r_top, r_top))
    T = sqrt_r_ir_j * np.real(sigma_sub)
    T = 0.5 * (T + T.T)

    eigvals_T = np.clip(np.linalg.eigvalsh(T), 0, None)
    F_trunc = np.sum(np.sqrt(eigvals_T))
    trace_rho_m = np.sum(r_top)
    trace_sigma_m = np.real(np.trace(sigma_sub))
    F_star = F_trunc + np.sqrt(max(0, (1 - trace_rho_m) * (1 - trace_sigma_m)))

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


def vqfe_from_circuit_total(circuit_fn, n_qubits, L=2, m=1, maxiter=200):
    """
    Compute truncated and generalized fidelity between two quantum states ρ and ρ_delta,
    both prepared in a 2n-qubit circuit, by variationally diagonalizing ρ.

    Args:
        circuit_fn (callable): Function that prepares a 2n-qubit state such that
                               qubits [0..n-1] encode ρ and [n..2n-1] encode ρ_delta.
        n_qubits (int): Number of qubits in each state.
        L (int): Number of ansatz layers.
        m (int): Truncation rank (top-m eigenvectors).
        maxiter (int): Optimization steps for variational diagonalization.

    Returns:
        dict: {F_trunc, F_star, top_eigenvalues, opt_params, rotated matrices, etc.}
    """
    dev = qml.device("default.mixed", wires=2 * n_qubits)

    def ansatz_U(params, wires):
        for l in range(params.shape[0]):
            for i, q in enumerate(wires):
                qml.RZ(params[l, i, 0], wires=q)
                qml.RY(params[l, i, 1], wires=q)
                qml.RZ(params[l, i, 2], wires=q)
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

    def apply_to_rho(params):
        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            ansatz_U(params, wires=range(n_qubits))
            return qml.density_matrix(wires=range(n_qubits))

        return circuit()

    def apply_to_rho_delta(params):
        @qml.qnode(dev)
        def circuit():
            circuit_fn()
            ansatz_U(params, wires=range(n_qubits, 2 * n_qubits))
            return qml.density_matrix(wires=range(n_qubits, 2 * n_qubits))

        return circuit()

    def cost(params):
        rho_rot = apply_to_rho(params)
        diag = np.real(np.diag(rho_rot))
        return -np.sum(diag**2)

    init_params = 0.1 * np.random.randn(L, n_qubits, 3)
    result = minimize(
        lambda p: cost(p.reshape(L, n_qubits, 3)),
        init_params.flatten(),
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False},
    )
    opt_params = result.x.reshape(L, n_qubits, 3)

    rho_rot = apply_to_rho(opt_params)
    sigma_rot = apply_to_rho_delta(opt_params)

    eigvals = np.real(np.diag(rho_rot))
    sorted_indices = np.argsort(eigvals)[::-1]
    top_indices = sorted_indices[:m]
    r_top = eigvals[top_indices]
    sigma_sub = sigma_rot[np.ix_(top_indices, top_indices)]

    sqrt_r_ir_j = np.sqrt(np.outer(r_top, r_top))
    T = sqrt_r_ir_j * np.real(sigma_sub)
    T = 0.5 * (T + T.T)

    eigvals_T = np.clip(np.linalg.eigvalsh(T), 0, None)
    F_trunc = np.sum(np.sqrt(eigvals_T))
    trace_rho_m = np.sum(r_top)
    trace_sigma_m = np.real(np.trace(sigma_sub))
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


def vqfe_from_density_matrices_total(rho, rho_delta, L=2, m=1, maxiter=200):
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
