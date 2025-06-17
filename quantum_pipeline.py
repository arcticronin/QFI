import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
from math import sqrt

import circuit_generator
from vqfe_subroutine import vqfe_main


def main(
    N=3,
    n=2,
    time_t=1.0,
    J=1,
    delta_h_x=0.5,  # This delta is used for the perturbation in the Hamiltonian AND the QFI formula
    h_x=0.5,
    m=1,
    DEBUG=False,
    trotter_steps_K=10,
    trotter_order=2,
    trace_out_indices=None,
):
    """
    Main function to compare quantum states using VQFE and compute QFI bounds.

    Parameters:
    - N (int): Initial total number of qubits in the full system (before tracing out).
    - n (int): Final number of qubits in the subsystem after tracing out.
    - time_t (float): Time for evolution of the quantum state.
    - J (float): Coupling constant for the Ising model.
    - delta_h_x (float): This parameter serves two roles:
                         1. It's the small perturbation added to h_x for the second copy of the system
                            in the quantum circuit (H_2 = H_1(h_x + delta_h_x)).
                         2. It's the 'delta' (δ) in the finite difference approximation formula
                            for Quantum Fisher Information bounds: 8 * (1 - F) / (delta^2).
    - h_x (float): Transverse field strength.
    - m (int): Number of measurements for VQFE.
    - DEBUG (bool): Debug mode flag.
    - trotter_steps_K (int): Number of Trotter steps for Hamiltonian simulation.
    - trotter_order (int): Order of Trotter decomposition.
    - trace_out_indices (list, optional): List of qubit indices to trace out. If None, randomly chosen.

    Returns:
    - dict: A dictionary containing computed QFI bounds and fidelities.
    """

    # --- Parameter Validation and Setup ---
    if trace_out_indices is None:
        print("Randomly choosing indices to trace out.")
        if N - n < 0:
            raise ValueError("n must be less than or equal to N to trace out qubits.")
        if N - n > 0:  # Only choose if there are qubits to trace out
            trace_out_indices = np.random.choice(
                range(N), size=N - n, replace=False
            ).tolist()
        else:  # No qubits to trace out if N == n
            trace_out_indices = []

    if len(trace_out_indices) != N - n:
        raise ValueError(
            f"trace_out_indices must have length N - n ({N-n}). Found {len(trace_out_indices)}."
        )
    if n > N:
        raise ValueError("n must be less than or equal to N.")

    print("\n--- Simulation Parameters ---")
    print(f"N (initial total qubits): {N}")
    print(f"n (final subsystem qubits): {n}")
    print(f"trace_out_indices: {trace_out_indices}")
    print(f"J: {J}")
    print(f"h_x: {h_x}")
    print(f"delta_h_x (physical perturbation AND QFI formula delta): {delta_h_x}")
    print(f"m (number of measurements): {m}")
    print(f"trotter_steps_K: {trotter_steps_K}")
    print(f"trotter_order: {trotter_order}")
    print("---------------------------\n")

    # Define active wires for the subsystem after tracing out
    active_rho_wires = [x for x in list(range(N)) if x not in trace_out_indices]
    active_rho_delta_wires = [x + N for x in active_rho_wires]

    # --- Quantum Circuit Generation ---
    # delta_h_x is used here to define the second Hamiltonian H(hx + delta_hx)
    circuit_fn = circuit_generator.make_tfim_circuits_trotter_decomposition(
        J_param=J,
        hx_param=h_x,
        delta_hx=delta_h_x,  # This is the perturbation for the *second* system in the circuit
        n_qubits=N,
        time_evolution=time_t,
        trotter_steps=trotter_steps_K,
        order=trotter_order,
    )

    # --- Execute VQFE ---
    vqfe_results = vqfe_main(
        circuit_fn=circuit_fn,
        total_num_qubits=2 * N,  # Total qubits in the PennyLane device
        active_rho_wires=active_rho_wires,
        active_rho_delta_wires=active_rho_delta_wires,
        L=2,  # Example value, consider making this a parameter if it varies
        m=m,
        maxiter=256,  # Example value, consider making this a parameter
    )

    # --- Extract Results and Compute QFI Bounds ---
    F_trunc = vqfe_results["F_trunc"]  # Truncated fidelity F(ρ_θ^(m), ρ_θ+δ^(m))
    F_star = vqfe_results["F_star"]  # Generalized fidelity F_*(ρ_θ^(m), ρ_θ+δ^(m))

    # Calculate QFI Bounds based on the VQFE fidelities
    # The 'delta' in the QFI formula is precisely the delta_h_x that was used
    # to perturb the Hamiltonian and generate the second state.
    qfi_lower_bound = 8 * (1 - F_star) / (delta_h_x**2)
    qfi_upper_bound = 8 * (1 - F_trunc) / (delta_h_x**2)

    print(f"\n--- Computed Results ---")
    print(f"Truncated Fidelity (F_trunc): {F_trunc}")
    print(f"Generalized Fidelity (F_star): {F_star}")
    print(f"QFI Lower Bound (from F_star): {qfi_lower_bound}")
    print(f"QFI Upper Bound (from F_trunc): {qfi_upper_bound}")
    print("------------------------\n")

    # --- Prepare Results Dictionary ---
    results_dict = {
        "fidelity_truncated": F_trunc.item(),
        "fidelity_truncated_generalized": F_star.item(),  # Renamed for clarity as it's F_star
        "lower_tqfi": qfi_lower_bound.item(),  # Using 'lower_tqfi' as in your target dictionary
        "upper_tqfi": qfi_upper_bound.item(),  # Using 'upper_tqfi' as in your target dictionary
    }

    return results_dict


# just to test it
if __name__ == "__main__":
    main(
        N=3,
        n=2,
        time_t=1.0,
        J=1,
        delta_h_x=0.1,
        h_x=0.5,
        m=1,
        DEBUG=False,
        trotter_steps_K=10,
        trotter_order=2,
        trace_out_indices=None,
    )
