import numpy as np
from scipy.linalg import expm
import helper_functions
from scipy.linalg import expm  # For matrix exponential


class TransverseFieldIsingModel:
    """
    Simulates a 1D Transverse-Field Ising Model (TFIM) with nearest-neighbor
    ZZ interactions and a transverse X field.

    Hamiltonian (most common form):
        H = -J * ∑ Z_i Z_{i+1} - h_x * ∑ X_i

    Where:
    - J: Coupling strength for the nearest-neighbor Ising (ZZ) interaction.
          J > 0: Ferromagnetic interaction (spins prefer to align).
          J < 0: Antiferromagnetic interaction (spins prefer to anti-align).
    - h_x: Strength of the magnetic field in the transverse (X) direction.
           This term drives quantum fluctuations and can lead to quantum phase transitions.

    Open boundary conditions are used.
    """

    def __init__(self, n, J, h_x, initial_state="0", DEBUG=False):
        """
        Initializes the TransverseFieldIsingModel.

        Parameters:
        - n (int): Number of qubits (spin sites).
        - J (float): Coupling strength for the Z_i Z_{i+1} interaction term.
                     (Note: The minus sign is included in the Hamiltonian construction).
        - h_x (float): Strength of the transverse X field.
                       (Note: The minus sign is included in the Hamiltonian construction).
        - initial_state (str or np.ndarray): Initial state of the system.
                                             "0": All spins in the |0> state (eigenstate of Z with eigenvalue +1).
                                             "H": Hadamard state (equal superposition of all computational basis states).
                                             ndarray: A custom state vector (1D array) or density matrix (2D array).
        - DEBUG (bool): If True, print diagnostic information during initialization.
        """
        self.n = n
        self.J = J
        self.h_x = h_x
        self.paulis = self._pauli_matrices()
        self.dim = 2**n  # Total Hilbert space dimension

        # Prepare initial state (stored as a density matrix for generality)
        if isinstance(initial_state, np.ndarray):
            if (
                initial_state.ndim == 1 and initial_state.shape[0] == self.dim
            ):  # If given a state vector
                self.initial_rho = np.outer(initial_state, initial_state.conj())
            elif initial_state.ndim == 2 and initial_state.shape == (
                self.dim,
                self.dim,
            ):  # If given a density matrix
                self.initial_rho = initial_state
            else:
                raise ValueError(
                    f"Invalid shape for initial_state. Expected ({self.dim},) or ({self.dim}, {self.dim})."
                )

        elif initial_state == "0":
            # Initial state |00...0> (all spins in Z-up direction in computational basis)
            state = np.zeros(self.dim, dtype=complex)
            state[0] = 1.0
            self.initial_rho = np.outer(state, state.conj())

        elif initial_state == "H":
            # Initial state |++++...> (equal superposition, often called the "Hadamard state")
            state = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
            self.initial_rho = np.outer(state, state.conj())
        else:
            raise ValueError("initial_state must be '0', 'H', or a valid numpy array.")

        # Basic validation for the initial density matrix
        if not np.allclose(self.initial_rho, self.initial_rho.conj().T):
            raise ValueError("Initial density matrix is not Hermitian.")
        if not np.isclose(np.trace(self.initial_rho), 1):
            raise ValueError("Initial density matrix trace is not 1.")
        if DEBUG:
            print(f"Initial ρ:\n{self.initial_rho}")

    def _pauli_matrices(self):
        """Returns a dictionary of standard Pauli matrices."""
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return {"I": I, "X": X, "Y": Y, "Z": Z}

    def _single_site_operator(self, op_matrix, site_idx):
        """
        Constructs a multi-qubit operator by applying 'op_matrix' to a
        single qubit at 'site_idx' and identity to all other qubits.
        """
        ops = [self.paulis["I"]] * self.n
        ops[site_idx] = op_matrix

        # Kronecker product construction
        full_op = ops[0]
        for i in range(1, self.n):
            full_op = np.kron(full_op, ops[i])
        return full_op

    def _two_site_operator(self, op_matrix_i, op_matrix_j, site_i, site_j):
        """
        Constructs a multi-qubit operator by applying 'op_matrix_i' to 'site_i',
        'op_matrix_j' to 'site_j', and identity to all other qubits.
        Assumes site_i and site_j are distinct.
        """
        ops = [self.paulis["I"]] * self.n
        ops[site_i] = op_matrix_i
        ops[site_j] = op_matrix_j

        # Kronecker product construction
        full_op = ops[0]
        for i in range(1, self.n):
            full_op = np.kron(full_op, ops[i])
        return full_op

    def construct_hamiltonian(self):
        """
        Builds the Hamiltonian for the 1D Transverse-Field Ising Model:
        H = -J * ∑ Z_i Z_{i+1} - h_x * ∑ X_i

        The sums are over nearest neighbors for the ZZ term and all sites for the X term.
        Open boundary conditions are assumed.
        """
        X = self.paulis["X"]
        Z = self.paulis["Z"]

        # Initialize Hamiltonian as a zero matrix of the correct dimension
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # Add the ZZ nearest-neighbor interaction terms
        for i in range(self.n - 1):  # Iterate up to n-2 for pairs (i, i+1)
            # Term: -J * Z_i Z_{i+1}
            H += -self.J * self._two_site_operator(Z, Z, i, i + 1)

        # Add the transverse X field terms
        for i in range(self.n):  # Iterate over all sites
            # Term: -h_x * X_i
            H += -self.h_x * self._single_site_operator(X, i)

        return H

    def evolve_density_matrix(self, time):
        """
        Evolves the initial density matrix under the Hamiltonian for a given time `t`
        using unitary evolution: ρ(t) = U(t) ρ(0) U(t)†, where U(t) = exp(-iHt).

        Parameters:
        - time (float): The total evolution time `t`.

        Returns:
        - rho_evolved (np.ndarray): The evolved 2^n-dimensional density matrix.
        """
        H = self.construct_hamiltonian()
        U = expm(-1j * H * time)  # Include time in the exponential
        rho_evolved = U @ self.initial_rho @ U.conj().T
        return rho_evolved

    def generate_perturbed_density_matrices(self, delta_h_x, time):
        """
        Generates two density matrices: one evolved with the original (J, h_x) parameters
        and another with a perturbed transverse field (h_x + delta_h_x).

        Parameters:
        - delta_h_x (float): Small perturbation to the transverse field h_x.
        - time (float): The evolution time `t` for both density matrices.

        Returns:
        - (rho_original, rho_perturbed): Tuple containing the original and perturbed density matrices.
        """
        # Evolve the original density matrix with the current h_x
        rho_original = self.evolve_density_matrix(time)

        # Temporarily modify the transverse field for the perturbed calculation
        original_h_x = self.h_x  # Store original h_x value
        self.h_x += delta_h_x  # Apply perturbation

        # Evolve with the perturbed field
        rho_perturbed = self.evolve_density_matrix(time)

        # Restore original h_x for subsequent operations
        self.h_x = original_h_x

        return rho_original, rho_perturbed

    def generate_mixed_states_with_perturbation(
        self, trace_out_indices, delta_h_x, time
    ):
        """
        Generates two mixed states by tracing out specified qubits from the evolved pure states:
        one from the original parameters and one from the perturbed transverse field.

        This function assumes the existence of a 'trace_out' helper function that takes
        (rho, trace_out_index, total_qubits).

        Parameters:
        - trace_out_indices (list or int): Index or list of indices of qubits to trace out.
        - delta_h_x (float): Small perturbation to the transverse field h_x.
        - time (float): The evolution time `t` for the initial pure states.

        Returns:
        - (rho_mixed_original, rho_mixed_perturbed): Tuple containing the original and perturbed mixed density matrices.
        """
        # Generate the full density matrices (pure states) with and without perturbation
        rho_pure, rho_delta_pure = self.generate_perturbed_density_matrices(
            delta_h_x=delta_h_x, time=time
        )

        # Reduce both matrices by tracing out specified qubits
        rho_mixed_original = helper_functions.trace_out(
            rho=rho_pure,
            trace_out_index=trace_out_indices,
        )
        rho_mixed_perturbed = helper_functions.trace_out(
            rho=rho_delta_pure,
            trace_out_index=trace_out_indices,
        )
        return rho_mixed_original, rho_mixed_perturbed

    def compute_drho_numerical(self, param_value, delta_deriv=1e-4, time=0):
        """
        Computes the numerical derivative of the evolved density matrix with respect
        to the transverse field h_x, using a central difference approximation.

        Parameters:
        - param_value (float): The specific value of h_x at which to compute the derivative.
                                (The class's h_x parameter will be temporarily set to this value).
        - delta_deriv (float, optional): Small step size for numerical differentiation.
                                        The total step used is 2 * delta_deriv.
        - time (float): The evolution time `t` for the density matrices used in the derivative.

        Returns:
        - drho (np.ndarray): Numerical derivative of rho with respect to h_x.
        """
        # Store original h_x to restore it later
        original_h_x = self.h_x

        # Calculate rho at (param_value + delta_deriv)
        self.h_x = param_value + delta_deriv
        rho_plus_delta = self.evolve_density_matrix(time)

        # Calculate rho at (param_value - delta_deriv)
        self.h_x = param_value - delta_deriv
        rho_minus_delta = self.evolve_density_matrix(time)

        # Restore original h_x to maintain the object's state
        self.h_x = original_h_x

        # Central difference formula: drho/d(param) = (rho(param+delta) - rho(param-delta)) / (2*delta)
        drho = (rho_plus_delta - rho_minus_delta) / (2 * delta_deriv)
        return drho

    def compute_qfi_with_sld(self, h_x_val, time=0, delta_h_x_num_deriv=1e-4):
        """
        Compute the Quantum Fisher Information (QFI) for a mixed quantum state
        with respect to the transverse field (h_x), using the symmetric logarithmic derivative (SLD) method.

        This function calculates QFI for the density matrix evolved to a specific time.

        Parameters:
        - h_x_val (float): The specific value of the transverse field h_x at which to compute QFI.
                            (The class's h_x parameter will be temporarily set to this value during calculation).
        - time (float): The evolution time `t` for the density matrix.
        - delta_h_x_num_deriv (float, optional): Step size for numerical differentiation of drho.

        Returns:
        - F_Q (float): Quantum Fisher Information (QFI) for the given state and its derivative.
        """
        # Store original h_x, then set to h_x_val for QFI calculation
        original_h_x = self.h_x
        self.h_x = h_x_val

        # Get the density matrix at the specified h_x_val and time
        rho = self.evolve_density_matrix(time)

        # Compute the numerical derivative of the density matrix with respect to h_x
        # Corrected: Changed delta_param to delta_deriv to match the function signature
        drho = self.compute_drho_numerical(
            param_value=h_x_val, delta_deriv=delta_h_x_num_deriv, time=time
        )

        # Compute the eigenvalues and eigenvectors of the density matrix (rho must be Hermitian)
        # np.linalg.eigh is used for Hermitian matrices.
        eigenvalues, eigenvectors = np.linalg.eigh(rho)

        # Initialize QFI sum
        F_Q = 0.0

        # Compute matrix elements of SLD in the eigenbasis of rho
        # The QFI for a mixed state rho with eigenvalues p_k and eigenvectors |k>
        # with respect to a parameter lambda is given by:
        # F_Q = sum_{i,j} (2 * |<i| d(rho)/d(lambda) |j>|^2) / (p_i + p_j)
        # where d(rho)/d(lambda) is the derivative of rho with respect to lambda.
        for i in range(len(eigenvalues)):
            for j in range(len(eigenvalues)):
                # Avoid division by zero for nearly zero or zero eigenvalues
                # If the sum of eigenvalues is effectively zero, the term is skipped.
                if np.isclose(eigenvalues[i] + eigenvalues[j], 0.0, atol=1e-10):
                    continue

                # Project drho into the eigenbasis of rho: <v_i | drho | v_j>
                drho_ij = np.dot(
                    eigenvectors[:, i].conj().T, np.dot(drho, eigenvectors[:, j])
                )

                # Add term to Quantum Fisher Information (QFI) using the corrected formula
                # F_Q = sum_{i,j} (2 * |<i| drho |j>|^2) / (eigenvalues[i] + eigenvalues[j])
                F_Q += (2 * np.abs(drho_ij) ** 2) / (eigenvalues[i] + eigenvalues[j])

        # Restore original h_x value after computation
        self.h_x = original_h_x

        return F_Q
