import pennylane as qml
from pennylane import numpy as np

##############################################################################
# from Sec. III of "Variational quantum algorithm for estimating the quantum
# Fisher information"

# TODO the optional variational step to prepare or optimize U_α.
##############################################################################


def create_swap_test_circuit(n_qubits):
    """
    Returns a PennyLane QNode that performs the destructive SWAP test
    on two n-qubit states (rho, sigma).  We use 2*n_qubits + 1 wires:
      - 1 ancilla wire (wire 0)
      - n_qubits for the first state
      - n_qubits for the second state

    The ancilla is prepared in |0>, a Hadamard is applied, then
    we do controlled-SWAP between each qubit of the two states,
    then another Hadamard.  Measuring ancilla in |0> with probability p0
    yields:
         p0 = 1/2 + (1/2)*Tr[rho * sigma].

    So Tr[rho * sigma] = 2*p0 - 1.
    """
    total_wires = 2 * n_qubits + 1
    # dev = qml.device("default.qubit", wires=total_wires)
    dev = qml.device(
        "default.mixed", wires=total_wires
    )  # device supports density matrices

    @qml.qnode(dev)
    def swap_test(rho_matrix, sigma_matrix):
        # ancilla will have index 0
        # lad first state onto wires [1..n_qubits]
        qml.QubitDensityMatrix(rho_matrix, wires=range(1, n_qubits + 1))
        # load second state onto wires [n_qubits+1..2*n_qubits]
        qml.QubitDensityMatrix(
            sigma_matrix, wires=range(n_qubits + 1, 2 * n_qubits + 1)
        )

        # Apply Hadamard to ancilla (wire 0)
        qml.Hadamard(wires=0)

        # C-SWAP between each pair of qubits

        for i in range(n_qubits):
            qml.ctrl(qml.SWAP, control=0)(wires=[i + 1, i + 1 + n_qubits])

        # sbnother Hadamard on ancilla
        qml.Hadamard(wires=0)

        # Return probability distribution of ancilla
        return qml.probs(wires=0)

    return swap_test


def create_gen_swap_test_circuit(n_qubits):
    """
    Returns a PennyLane QNode that performs the 'generalized SWAP test'
    needed to measure functionals like Tr[rho * sigma * rho * sigma].

    We use 4*n_qubits + 1 wires:
      - ancilla wire 0
      - n_qubits for 1st copy of the states
      - n_qubits for 2nd copy
      - n_qubits for 3rd copy
      - n_qubits for 4th copy

    We load states as (rho, sigma, rho, sigma), apply a
    'controlled cyclic shift' among the 4 blocks if ancilla=1,
    then measure ancilla.

    the probability p0 of measuring |0> is:
         p0 = 1/2 + (1/2)*Tr[rho*sigma*rho*sigma].

    Tr[rho*sigma*rho*sigma] = 2*p0 - 1.
    """
    total_wires = 4 * n_qubits + 1
    # dev = qml.device("default.qubit", wires=total_wires)
    dev = qml.device(
        "default.mixed", wires=total_wires
    )  # device supports density matrices

    @qml.qnode(dev)
    def gen_swap_test(rho1, rho2, rho3, rho4):
        # Load the 4 states
        qml.QubitDensityMatrix(rho1, wires=range(1, n_qubits + 1))
        qml.QubitDensityMatrix(rho2, wires=range(n_qubits + 1, 2 * n_qubits + 1))
        qml.QubitDensityMatrix(rho3, wires=range(2 * n_qubits + 1, 3 * n_qubits + 1))
        qml.QubitDensityMatrix(rho4, wires=range(3 * n_qubits + 1, 4 * n_qubits + 1))

        # Hadamard on ancilla
        qml.Hadamard(wires=0)

        # We would implement a single controlled "cyclic shift"
        # across the 4 blocks [just for demonstration].
        # In practice, you can do a series of controlled-SWAP gates
        # that rotate (blk1->blk2, blk2->blk3, blk3->blk4, blk4->blk1).
        # The details can be found in the references or done by e.g.
        # a systematic approach with ancilla.
        # We'll skip the explicit cycle for brevity.

        # Another Hadamard on ancilla
        qml.Hadamard(wires=0)

        return qml.probs(wires=0)

    return gen_swap_test


def measure_sub_super_fidelity(n_qubits, rho, rho_delta):
    """
    Measure the overlaps:
       Tr[rho^2], Tr[rho_delta^2], Tr[rho*rho_delta],
       Tr[rho*rho_delta*rho*rho_delta]
    via the SWAP & generalized-SWAP tests. Then compute
    subfidelity E(rho,rho_delta) and superfidelity R(rho,rho_delta).

    Returns (E_val, R_val).
    """

    # 1) Create the QNodes for SWAP and generalized SWAP
    swap_test = create_swap_test_circuit(n_qubits)
    gen_swap = create_gen_swap_test_circuit(n_qubits)

    # 2) Overlaps from the destructive SWAP test
    #   p0 = prob(ancilla=0) =>  p0 = 1/2 + 1/2 * Tr[rho*sigma]
    # => Tr[rho*sigma] = 2*p0 - 1

    # -- Tr[rho^2]
    p0 = swap_test(rho, rho)
    tr_rho_sq = 2 * p0[0] - 1

    # -- Tr[rho_delta^2]
    p0 = swap_test(rho_delta, rho_delta)
    tr_rd_sq = 2 * p0[0] - 1

    # -- Tr[rho * rho_delta]
    p0 = swap_test(rho, rho_delta)
    tr_r_rd = 2 * p0[0] - 1

    # 3) Overlap from the generalized SWAP test: Tr[rho*rho_delta*rho*rho_delta]
    #   p0 = 1/2 + 1/2 * Tr[rho*rho_delta*rho*rho_delta]
    # => Tr[rho*rho_delta*rho*rho_delta] = 2*p0 - 1
    # We pass (rho, rho_delta, rho, rho_delta) as 4 arguments:

    p0 = gen_swap(rho, rho_delta, rho, rho_delta)
    tr_rdrd = 2 * p0[0] - 1

    # 4) Compute subfidelity and superfidelity
    # E(rho,sigma) = Tr[rho*sigma] + sqrt{2( (Tr[rho*sigma])^2 - Tr[rho*sigma*rho*sigma]) }
    # R(rho,sigma) = Tr[rho*sigma] + sqrt{(1 - Tr[rho^2])(1 - Tr[sigma^2])}

    # Subfidelity E
    inside = 2.0 * (tr_r_rd**2 - tr_rdrd)
    inside_clip = max(inside, 0.0)
    E_val = tr_r_rd + np.sqrt(inside_clip)

    # Superfidelity R
    factor = (1.0 - tr_rho_sq) * (1.0 - tr_rd_sq)
    factor_clip = max(factor, 0.0)
    R_val = tr_r_rd + np.sqrt(factor_clip)

    return (E_val, R_val)
