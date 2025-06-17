import pennylane as qml
import numpy as np

DEBUG = True


# use the version with trotter decomposition
def make_tfim_circuits(J_param, hx_param, delta_hx, n_qubits, time_evolution=1.0):
    """
    Returns a function that prepares a 2n-qubit quantum state representing:
    - First n qubits: evolved under TFIM (with -J * ZZ and -hx * X)
    - Second n qubits: evolved under TFIM (with -J * ZZ and -(hx + delta_hx) * X)

    This circuit implements the time evolution U(t) = exp(-iHt) for the
    1D Transverse-Field Ising Model Hamiltonian, which is commonly defined as:

    H = -J * sum_{i} (Z_i Z_{i+1}) - h_x * sum_{i} (X_i)

    Where:
    - J: Coupling strength for the nearest-neighbor Ising (ZZ) interaction.
         (Corresponds to 'J_param' in the function arguments).
    - h_x: Strength of the magnetic field in the transverse (X) direction.
           (Corresponds to 'hx_param' in the function arguments).

    The time evolution operator is U(t) = exp(-iHt). Substituting H:
    U(t) = exp(-i * (-J * sum_{i} Z_i Z_{i+1} - h_x * sum_{i} X_i) * t)
    U(t) = exp(i * (J * sum_{i} Z_i Z_{i+1} + h_x * sum_{i} X_i) * t)

    For a single Trotter step (or exact evolution if terms commute, which they don't here,
    but for the purpose of mapping to gates, we apply each term's evolution):
    U(t) approx Product_i [exp(i * J * t * Z_i Z_{i+1})] * Product_j [exp(i * h_x * t * X_j)]

    Args:
        J_param (float): Coupling strength for the nearest-neighbor ZZ interaction.
                         (Corresponds to 'J' in the H = -J*ZZ - hx*X convention).
        hx_param (float): Strength of the transverse X field.
                          (Corresponds to 'hx' in the H = -J*ZZ - hx*X convention).
        delta_hx (float): Small perturbation added to hx_param for the second copy.
        n_qubits (int): Number of qubits per system (2n total).
        time_evolution (float): The total time 't' for the evolution.
                                The gate parameters will be scaled by this time.
    """
    total_qubits = 2 * n_qubits

    def circuit():
        """
        The quantum circuit that prepares the two evolved TFIM states.
        """
        # Initialize all qubits in the |0⟩ state (computational basis).
        # This is the standard initial state for many quantum simulations.
        qml.BasisState(np.zeros(total_qubits, dtype=int), wires=range(total_qubits))

        # --- System 1 (qubits 0 to n-1): Evolved under H_1 = -J*ZZ - hx*X ---
        # The evolution operator for this system is U_1(t) = exp(i * (J * sum Z_i Z_{i+1} + hx * sum X_i) * t)

        # Apply ZZ interactions: exp(i * J * t * Z_i Z_{i+1})
        # PennyLane's qml.IsingZZ(phi, wires=[q1, q2]) applies exp(-i * phi * Z_q1 Z_q2).
        # To match exp(i * J * t * Z_i Z_{i+1}), we need -phi = J * t, so phi = -J * t.
        for i in range(n_qubits - 1):
            # Apply IsingZZ gate for nearest-neighbor interaction (qubits i and i+1)
            qml.IsingZZ(-J_param * time_evolution, wires=[i, i + 1])

        # Barrier for visualization: helps separate distinct parts of the circuit diagram.
        qml.Barrier(wires=range(n_qubits))

        # Apply transverse X fields: exp(i * hx * t * X_i)
        # PennyLane's qml.RX(phi, wires=q) applies exp(-i * phi/2 * X_q).
        # To match exp(i * hx * t * X_i), we need -phi/2 = hx * t, so phi = -2 * hx * t.
        # This is the crucial sign correction to match the Hamiltonian convention.
        for i in range(n_qubits):
            # Apply RX gate for transverse field on each qubit i
            qml.RX(-2 * hx_param * time_evolution, wires=i)

        qml.Barrier(wires=range(n_qubits))  # Separator for visualization

        # --- System 2 (qubits n to 2n-1): Evolved under H_2 = -J*ZZ - (hx+delta_hx)*X ---
        # This system uses the same J_param but a perturbed transverse field.
        # The evolution operator for this system is U_2(t) = exp(i * (J * sum Z_i Z_{i+1} + (hx+delta_hx) * sum X_i) * t)
        offset = n_qubits  # Offset to address qubits in the second system
        perturbed_hx = hx_param + delta_hx  # Calculate the perturbed h_x value

        # Apply ZZ interactions (same J_param as first system)
        # Angle remains -J * t
        for i in range(n_qubits - 1):
            qml.IsingZZ(-J_param * time_evolution, wires=[offset + i, offset + i + 1])

        qml.Barrier(wires=range(n_qubits, total_qubits))  # Separator for visualization

        # Apply perturbed transverse X fields
        # Angle becomes -2 * perturbed_hx * t
        for i in range(n_qubits):
            qml.RX(-2 * perturbed_hx * time_evolution, wires=offset + i)

        qml.Barrier(wires=range(n_qubits, total_qubits))  # Separator for visualization

        # The circuit implicitly returns the quantum state when executed on a device
        # that supports state vector simulation (e.g., 'default.qubit').
        # For measurement-based tasks, you would add measurement operations here.
        # Example: return qml.state() or qml.expval(qml.PauliZ(0))

    return circuit


def make_tfim_circuits_trotter_decomposition(
    J_param,
    hx_param,
    delta_hx,
    n_qubits,
    time_evolution=1.0,
    trotter_steps=1,
    order=1,
    draw_circuit=False,
):
    """
    Returns a function that prepares a 2n-qubit quantum state representing:
    - First n qubits: evolved under TFIM (with -J * ZZ and -hx * X)
    - Second n qubits: evolved under TFIM (with -J * ZZ and -(hx + delta_hx) * X)

    This circuit implements the time evolution U(t) = exp(-iHt) for the
    1D Transverse-Field Ising Model Hamiltonian, which is commonly defined as:

    H = -J * sum_{i} (Z_i Z_{i+1}) - h_x * sum_{i} (X_i)

    The time evolution operator is U(t) = exp(-iHt).

    Note on sign convention:
    PennyLane's `qml.TrotterProduct(H_PL, t, ...)` computes exp(-i * H_PL * t).
    However, when `H_PL` is constructed from basic Pauli operators with positive coefficients,
    the convention might effectively lead to an overall phase flip relative to the standard
    QM definition of H leading to exp(-iHt).
    To align with H = -J*ZZ - hx*X and get exp(-iHt), we must pass H_PL = -H to TrotterProduct.
    So, H_PL should be: J*sum(Z_i Z_{i+1}) + hx*sum(X_i).

    Args:
        J_param (float): Coupling strength for the nearest-neighbor ZZ interaction.
                         (Corresponds to 'J' in the H = -J*ZZ - hx*X convention).
        hx_param (float): Strength of the magnetic field in the transverse (X) direction.
                          (Corresponds to 'hx' in the H = -J*ZZ - hx*X convention).
        delta_hx (float): Small perturbation added to hx_param for the second copy.
        n_qubits (int): Number of qubits per system (2n total).
        time_evolution (float): The total time 't' for the evolution.
                                The gate parameters will be scaled by this time.
        trotter_steps (int): The number of Trotter steps (K) to use for the approximation.
                             Higher values lead to better precision but more gates.
        order (int): The order of the Trotter-Suzuki decomposition (1 or 2).
        draw_circuit (bool): If True, the circuit will be drawn and printed to the console.
    """
    total_qubits = 2 * n_qubits

    # --- Define Hamiltonian for System 1 ---
    # We want to simulate H_1 = -J * sum(Z_i Z_{i+1}) - h_x * sum(X_i)
    # So, we pass H_PL_1 = J * sum(Z_i Z_{i+1}) + h_x * sum(X_i) to qml.TrotterProduct
    coeffs1 = []
    ops1 = []

    # ZZ terms: J * Z_i Z_{i+1}
    for i in range(n_qubits - 1):
        coeffs1.append(J_param)  # Corrected sign from -J_param
        ops1.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

    # X terms: hx * X_i
    for i in range(n_qubits):
        coeffs1.append(hx_param)  # Corrected sign from -hx_param
        ops1.append(qml.PauliX(i))

    H1 = qml.Hamiltonian(coeffs1, ops1)

    # --- Define Hamiltonian for System 2 ---
    # We want to simulate H_2 = -J * sum(Z_i Z_{i+1}) - (hx + delta_hx) * sum(X_i)
    # So, we pass H_PL_2 = J * sum(Z_i Z_{i+1}) + (hx + delta_hx) * sum(X_i) to qml.TrotterProduct
    coeffs2 = []
    ops2 = []
    offset = n_qubits
    perturbed_hx = hx_param + delta_hx

    # ZZ terms: J * Z_i Z_{i+1}
    for i in range(n_qubits - 1):
        coeffs2.append(J_param)  # Corrected sign from -J_param
        ops2.append(qml.PauliZ(offset + i) @ qml.PauliZ(offset + i + 1))

    # X terms: (hx + delta_hx) * X_i
    for i in range(n_qubits):
        coeffs2.append(perturbed_hx)  # Corrected sign from -perturbed_hx
        ops2.append(qml.PauliX(offset + i))

    H2 = qml.Hamiltonian(coeffs2, ops2)

    def circuit():
        """
        The quantum circuit that prepares the two evolved TFIM states using Trotterization.
        """
        # Initialize all qubits in the |0⟩ state (computational basis).
        qml.BasisState(np.zeros(total_qubits, dtype=int), wires=range(total_qubits))

        # --- Apply Trotterization for System 1 ---
        # qml.TrotterProduct receives H_PL_1 = -H_1, and calculates exp(-i * H_PL_1 * t)
        # = exp(-i * (-H_1) * t) = exp(i * H_1 * t).
        # This matches the desired U(t) = exp(-iHt) for H = -J*ZZ - hx*X.
        qml.TrotterProduct(H1, time_evolution, trotter_steps, order=order)

        # Barrier for visualization.
        qml.Barrier(wires=range(n_qubits))

        # --- Apply Trotterization for System 2 ---
        qml.TrotterProduct(H2, time_evolution, trotter_steps, order=order)

        qml.Barrier(wires=range(n_qubits, total_qubits))

    # Add drawing functionality if requested
    if draw_circuit:
        dev_for_drawing = qml.device("default.qubit", wires=total_qubits)
        qnode_for_drawing = qml.QNode(circuit, dev_for_drawing)
        print("\n--- Drawn TFIM Circuit ---")
        print(qml.draw(qnode_for_drawing)())
        print("--------------------------\n")

    return circuit


def make_xx_plus_z_circuits(j_param, hz_param, delta, n_qubits):
    """
    Returns a function that prepares a 2n-qubit quantum state:
    - First n qubits: evolved under XX + Z field (with hz_param)
    - Second n qubits: evolved under XX + Z field (with hz_param + delta)



    Args:
        j_param (float): XX interaction strength.
        hz_param (float): Base longitudinal Z-field parameter.
        delta (float): Small perturbation added to hz_param for the second copy.
        n_qubits (int): Number of qubits per system (2n total).
    """
    total_qubits = 2 * n_qubits

    def circuit():
        # Start in |0⟩^⊗2n
        qml.BasisState(np.zeros(total_qubits, dtype=int), wires=range(total_qubits))

        # --- First system (qubits 0 to n-1) ---
        for i in range(n_qubits - 1):
            qml.Hadamard(wires=i)
            qml.Hadamard(wires=i + 1)
            qml.IsingZZ(
                2 * j_param, wires=[i, i + 1]
            )  # Note: factor 2 for exp(-i * H t), t=1
            qml.Hadamard(wires=i)
            qml.Hadamard(wires=i + 1)

        for i in range(n_qubits):
            qml.RZ(2 * hz_param, wires=i)

        qml.Barrier()

        # --- Second system (qubits n to 2n-1) ---
        offset = n_qubits
        for i in range(n_qubits - 1):
            qml.Hadamard(wires=offset + i)
            qml.Hadamard(wires=offset + i + 1)
            qml.IsingZZ(2 * j_param, wires=[offset + i, offset + i + 1])
            qml.Hadamard(wires=offset + i)
            qml.Hadamard(wires=offset + i + 1)

        for i in range(n_qubits):
            qml.RZ(2 * (hz_param + delta), wires=offset + i)

        qml.Barrier()

    return circuit


def get_density_matrix_circuit(circuit_factory, active_qubits):
    """
    Returns a new function that, when executed on a device, outputs the
    density matrix of the specified active qubits after the original circuit
    has been executed.

    Args:
        circuit_factory (callable): A function that returns a PennyLane circuit.
        active_qubits (Sequence[int]): A list or tuple of qubit indices for
                                       which the density matrix should be computed.

    Returns:
        callable: A new function representing the circuit that outputs the
                  density matrix of the active qubits.
    """

    def density_matrix_circuit():
        circuit_factory()  # Execute the operations from the original circuit (using the correct name)
        return qml.density_matrix(wires=active_qubits)

    return density_matrix_circuit


def make_ising_theta_and_theta_plus_delta_circuits(j_param, hz_param, delta, n_qubits):
    """
    Returns a function that prepares a 2n-qubit quantum state:
    - First n qubits: evolved under IsingZZ + RZ(hz_param)
    - Second n qubits: evolved under IsingZZ + RZ(hz_param + delta)

    Args:
        j_param (float): ZZ interaction strength.
        hz_param (float): Base longitudinal field parameter.
        delta (float): Small perturbation added to hz_param for the second copy.
        n_qubits (int): Number of qubits per state.
        kept_wires (list[int] or None): Wires to measure (optional, unused here).

    Returns:
        function: A circuit function usable by vqfe_from_circuit.
    """
    total_qubits = 2 * n_qubits

    def circuit():
        # Start in |0⟩^⊗2n
        qml.BasisState(np.zeros(total_qubits, dtype=int), wires=range(total_qubits))

        # First system (qubits 0 to n-1) coupling
        for i in range(n_qubits - 1):
            qml.IsingZZ(j_param, wires=[i, i + 1])

        qml.Barrier()
        # apply RZ to first

        for i in range(n_qubits):
            qml.RZ(hz_param, wires=i)

        qml.Barrier()

        # Second system (qubits n to 2n-1) coupling
        offset = n_qubits
        for i in range(n_qubits - 1):
            qml.IsingZZ(j_param, wires=[offset + i, offset + i + 1])

        qml.Barrier()

        # apply RZ + delta to second
        for i in range(n_qubits):
            qml.RZ(hz_param + delta, wires=offset + i)

        qml.Barrier()

    # def circuit():
    #     # Start in |0⟩^⊗2n
    #     qml.BasisState(np.zeros(total_qubits, dtype=int), wires=range(total_qubits))

    #     # First system (qubits 0 to n-1) with hz
    #     for i in range(n_qubits - 1):
    #         qml.IsingZZ(j_param, wires=[i, i + 1])

    #     for i in range(n_qubits):
    #         qml.RZ(hz_param, wires=i)

    #     qml.Barrier()

    #     # Second system (qubits n to 2n-1) with hz + delta
    #     offset = n_qubits
    #     for i in range(n_qubits - 1):
    #         qml.IsingZZ(j_param, wires=[offset + i, offset + i + 1])
    #     for i in range(n_qubits):
    #         qml.RZ(hz_param + delta, wires=offset + i)

    #     qml.Barrier()

    return circuit
