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
    delta=0.5,
    delta_h_x=0.1,
    h_x=0.5,
    m=1,
    DEBUG=False,
    derivative_delta=1e-3,
    trotter_steps_K=10,
    trotter_order=2,
    trace_out_indices=None,
):
    """
    Main function to compare quantum states using VQFE.
    Parameters:
    - N: Initial number of qubits (before tracing out).
    - n: Final number of qubits after tracing out.
    - time_t: Time for evolution.
    - J: Coupling constant.
    - delta: Parameter shift for QFI calculation.
    - delta_h_x: Perturbation in the transverse field.
    - h_x: Transverse field strength.
    - m: Number of measurements.
    - DEBUG: Debug mode flag.
    - derivative_delta: Small value for derivative approximation.
    - trotter_steps_K: Number of Trotter steps.
    - trotter_order: Order of Trotter decomposition.
    """

    # compare states
    ## parameters
    if trace_out_indices is None:
        print("randomly choosing indices to trace out")
        trace_out_indices = np.random.choice(range(N), size=N - n, replace=False)
    if len(trace_out_indices) != N - n:
        raise ValueError("trace_out_indices must have length N - n")
    if n > N:
        raise ValueError("n must be less than or equal to N")
    print("Using parameters:")
    print("N =", N, " starting number of qubits, before tracing out")
    print("n =", n, " final number of qubits")
    print("trace_out_indices =", trace_out_indices)
    print("J =", J)
    # print("h_z =",h_z)
    print("h_x =", h_x)
    print("delta_h_x =", delta_h_x)
    print("delta =", delta)
    print("m =", m)
    # print("DEBUG is set to: ",DEBUG)
    print("trotter_steps_K =", trotter_steps_K)
    print("trotter_order =", trotter_order)

    # indices for quantum part
    # trace out for the whole system composed by sys 1 and sys 2
    active_rho = [x for x in list(range(N)) if x not in trace_out_indices]
    active_rho_delta = [x + N for x in active_rho]

    measured_wires = active_rho + active_rho_delta
    discarded_wires = [x for x in range(2 * N) if x not in measured_wires]
    h_z = 0.1

    # Your existing circuit setup
    # Assuming circuit_generator, J, h_x, delta_h_x, N, time_t, trotter_steps_K, trotter_order are defined
    # and active_rho, active_rho_delta are also defined.

    circuit_fn = circuit_generator.make_tfim_circuits_trotter_decomposition(
        J,
        h_x,
        delta_h_x,
        N,
        time_evolution=time_t,
        trotter_steps=trotter_steps_K,
        order=trotter_order,
    )

    # Set the parameter shift delta (δ) for QFI calculation
    # This delta is distinct from delta_h_x used in circuit_fn for creating rho_theta+delta
    # The paper suggests a small value for delta for accurate approximation [cite: 55]
    # For the magnetometry example in the paper, they use delta=0.1 for analysis of barren plateaus [cite: 225]
    param_shift_delta = 0.1  # This is the 'δ' from Eq. 3 in the paper [cite: 55]

    # Call the vqfe_main function
    vqfe_results = vqfe_main(
        circuit_fn=circuit_fn,
        total_num_qubits=2
        * N,  # Assuming 2*N qubits are used for rho and rho_delta combined
        active_rho_wires=active_rho,
        active_rho_delta_wires=active_rho_delta,
        L=2,  # TODO set those as hyperparameter somewere
        m=m,
        maxiter=256,  # TODO
    )

    # Extract the calculated fidelities
    F_trunc = vqfe_results["F_trunc"]  # F(ρ_θ^(m), ρ_θ+δ^(m)) from Eq. 9 [cite: 72]
    F_star = vqfe_results["F_star"]  # F_*(ρ_θ^(m), ρ_θ+δ^(m)) from Eq. 9 [cite: 72]

    # Compute the QFI bounds using the formulas from the paper
    # The paper states: I_δ(f_2; ρ_θ) <= I_δ(θ; ρ_θ) <= I_δ(f_1; ρ_θ) (Eq. 5) [cite: 67]
    # where I_δ(f; ρ_θ) = 8 * (1 - f(ρ_θ, ρ_θ+δ)) / δ^2 (Eq. 6) [cite: 67]
    # And for TQFI bounds: I_δ(F_*; ρ_θ^(m)) <= I_δ(θ; ρ_θ) <= I_δ(F; ρ_θ^(m)) (Eq. 11) [cite: 76]

    # Lower Bound for QFI: I_δ(F_*; ρ_θ^(m))
    # Note: F_star is the upper bound on fidelity in Eq. 9[cite: 72], so it leads to the lower bound on QFI
    # I_δ(f_2) is the lower bound for QFI as per Eq. 5. In TQFI context, f_2 is F_star.
    QFI_lower_bound = 8 * (1 - F_star) / (param_shift_delta**2)

    # Upper Bound for QFI: I_δ(F; ρ_θ^(m))
    # Note: F_trunc is the lower bound on fidelity in Eq. 9[cite: 72], so it leads to the upper bound on QFI
    # I_δ(f_1) is the upper bound for QFI as per Eq. 5. In TQFI context, f_1 is F_trunc.
    QFI_upper_bound = 8 * (1 - F_trunc) / (param_shift_delta**2)

    print(f"Truncated Fidelity (F_trunc): {F_trunc}")
    print(f"Generalized Fidelity (F_star): {F_star}")
    print(f"QFI Lower Bound (I_delta(F_star)): {QFI_lower_bound}")
    print(f"QFI Upper Bound (I_delta(F_trunc)): {QFI_upper_bound}")

    # You can also access other results from vqfe_results if needed
    # print(f"Top eigenvalues of rotated rho: {vqfe_results['top_eigenvalues_rho']}")
    # print(f"Optimized VQSE parameters: {vqfe_results['opt_params_vqse']}")


if __name__ == "__main__":
    main(
        N=3,
        n=2,
        time_t=1.0,
        J=1,
        delta=0.5,
        delta_h_x=0.1,
        h_x=0.5,
        m=1,
        DEBUG=False,
        derivative_delta=1e-3,
        trotter_steps_K=10,
        trotter_order=2,
        trace_out_indices=None,
    )
