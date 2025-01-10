import numpy as np
from scipy.optimize import minimize

## stolen functions


def generalized_swap_test(rho, sigma, num_qubits):
    """
    Generalized SWAP test to estimate Tr[ρσρσ] using a cyclic permutation operator.
    """
    # Number of dimensions
    dim = 2**num_qubits

    # Step 1: Prepare ancilla and state copies
    ancilla_dim = 2  # Dimension of the ancilla qubit
    total_dim = ancilla_dim * dim**4  # Ancilla + 4 copies of n-qubit states

    # Initialize ancilla in |0⟩ state and tensor product with states
    ancilla = np.array([[1, 0], [0, 0]])  # |0⟩ state
    rho_extended = np.kron(rho, np.kron(rho, sigma))  # ρ ⊗ ρ ⊗ σ
    initial_state = np.kron(ancilla, rho_extended)

    # Step 2: Apply Hadamard gate to the ancilla
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard gate
    H_full = np.kron(H, np.eye(total_dim // ancilla_dim))  # Extend to total dimension
    state_after_hadamard = H_full @ initial_state

    # Step 3: Define the cyclic permutation operator
    Sn = np.zeros((total_dim, total_dim))  # Cyclic permutation matrix
    for i in range(total_dim):
        Sn[i, (i + dim**2) % total_dim] = 1  # Cycle indices of ρ ⊗ σ ⊗ ρ ⊗ σ

    # Step 4: Apply controlled cyclic permutation gate
    controlled_Sn = np.block(
        [
            [np.eye(total_dim // 2), np.zeros((total_dim // 2, total_dim // 2))],
            [np.zeros((total_dim // 2, total_dim // 2)), Sn],
        ]
    )
    state_after_swap = controlled_Sn @ state_after_hadamard

    # Step 5: Apply another Hadamard gate to the ancilla
    state_after_final_hadamard = H_full @ state_after_swap

    # Step 6: Measure ancilla in |0⟩ state to compute probability
    prob_0 = np.abs(state_after_final_hadamard[0]) ** 2  # Probability of ancilla in |0⟩
    fidelity = 2 * prob_0 - 1  # Fidelity-like measure from probability

    return fidelity


def variational_state_eigensolver(rho, m, num_qubits, num_params, max_iter=100):
    """
    Variational Quantum State Eigensolver (VQSE) to find the largest m eigenvalues
    and eigenvectors of the density matrix rho.
    """
    dim = 2**num_qubits  # Full dimension of the system

    def cost_function(params, rho):
        """Cost function to minimize: -Tr[H * V(rho)V†]."""
        # Eigen-decompose the density matrix
        eigvals, eigvecs = np.linalg.eigh(rho)
        idx = np.argsort(eigvals)[::-1]  # Sort eigenvalues in descending order
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Take the largest m eigenvalues and eigenvectors
        eigvals_trunc = eigvals[:m]
        eigvecs_trunc = eigvecs[:, :m]

        # Construct the truncated density matrix
        projected_rho = np.zeros((dim, dim))  # Zero-padded to match full dimension
        for i in range(m):
            projected_rho += eigvals_trunc[i] * np.outer(
                eigvecs_trunc[:, i], eigvecs_trunc[:, i]
            )

        # Minimize the negative overlap
        return -np.trace(projected_rho @ rho)  # Use full-dimension matrices here

    # Initialize random parameters for variational Ansatz
    initial_params = np.random.rand(num_params)

    # Optimization using COBYLA
    result = minimize(
        cost_function,
        initial_params,
        args=(rho,),
        method="COBYLA",
        options={"maxiter": max_iter},
    )
    params = result.x

    # Return eigenvalues and eigenvectors after optimization
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:m]
    eigvecs = eigvecs[:, idx][:, :m]

    return eigvals, eigvecs, params
