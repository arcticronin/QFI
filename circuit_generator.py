import pennylane as qml
import numpy as np


def make_tfim_circuits(J_param, hx_param, delta_hx, n_qubits, time_evolution=1.0):
    """
    Returns a function that prepares a 2n-qubit quantum state representing:
    - First n qubits: evolved under TFIM (with -J * ZZ and -hx * X)
    - Second n qubits: evolved under TFIM (with -J * ZZ and -(hx + delta_hx) * X)

    This circuit directly implements the standard Transverse-Field Ising Model Hamiltonian:
    H = -J * ∑ Z_i Z_{i+1} - h_x * ∑ X_i

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
        # Start in |0⟩^⊗2n (computational basis initial state)
        qml.BasisState(np.zeros(total_qubits, dtype=int), wires=range(total_qubits))

        # --- First system (qubits 0 to n-1): Evolved under H = -J*ZZ - hx*X ---

        # Apply ZZ interactions: exp(-i * (-J * t) * Z_i Z_{i+1}) = exp(i * J * t * Z_i Z_{i+1})
        # The PennyLane IsingZZ gate is exp(-i * phi * Z_i Z_j).
        # So phi = -J * time_evolution
        for i in range(n_qubits - 1):
            qml.IsingZZ(-J_param * time_evolution, wires=[i, i + 1])

        qml.Barrier()  # Separator for visualization

        # Apply transverse X fields: exp(-i * (-hx * t) * X_i) = exp(i * hx * t * X_i)
        # The PennyLane RX gate is exp(-i * phi * X_i / 2).
        # So phi/2 = hx * time_evolution => phi = 2 * hx * time_evolution
        for i in range(n_qubits):
            qml.RX(2 * hx_param * time_evolution, wires=i)

        qml.Barrier()  # Separator for visualization

        # --- Second system (qubits n to 2n-1): Evolved under H = -J*ZZ - (hx+delta_hx)*X ---
        offset = n_qubits
        perturbed_hx = hx_param + delta_hx

        # Apply ZZ interactions (same J_param as first system)
        for i in range(n_qubits - 1):
            qml.IsingZZ(-J_param * time_evolution, wires=[offset + i, offset + i + 1])

        qml.Barrier()  # Separator for visualization

        # Apply perturbed transverse X fields
        for i in range(n_qubits):
            qml.RX(2 * perturbed_hx * time_evolution, wires=offset + i)

        qml.Barrier()  # Separator for visualization

        # Optional: You might return the state or measurements depending on your use case
        # E.g., return qml.state() or qml.expval(qml.PauliZ(0))

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
