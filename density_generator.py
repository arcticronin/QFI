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

    def generate_perturbed_density_matrices(self, delta_h_x, time=0):
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
        self, trace_out_indices, delta_h_x, time=0
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

    def compute_drho_numerical(self, param_value, delta_param=1e-4, time=0):
        """
        Computes the numerical derivative of the evolved density matrix with respect
        to the transverse field h_x, using a central difference approximation.

        Parameters:
        - param_value (float): The specific value of h_x at which to compute the derivative.
                               (The class's h_x parameter will be temporarily set to this value).
        - delta_param (float, optional): Small step size for numerical differentiation.
                                        The total step used is 2 * delta_param.
        - time (float): The evolution time `t` for the density matrices used in the derivative.

        Returns:
        - drho (np.ndarray): Numerical derivative of rho with respect to h_x.
        """
        # Store original h_x to restore it later
        original_h_x = self.h_x

        # Calculate rho at (param_value + delta_param)
        self.h_x = param_value + delta_param
        rho_plus_delta = self.evolve_density_matrix(time)

        # Calculate rho at (param_value - delta_param)
        self.h_x = param_value - delta_param
        rho_minus_delta = self.evolve_density_matrix(time)

        # Restore original h_x to maintain the object's state
        self.h_x = original_h_x

        # Central difference formula: drho/d(param) = (rho(param+delta) - rho(param-delta)) / (2*delta)
        drho = (rho_plus_delta - rho_minus_delta) / (2 * delta_param)
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
        drho = self.compute_drho_numerical(
            param_value=h_x_val, delta_param=delta_h_x_num_deriv, time=time
        )

        # Compute the eigenvalues and eigenvectors of the density matrix (rho must be Hermitian)
        eigenvalues, eigenvectors = np.linalg.eigh(rho)

        # Initialize QFI sum
        F_Q = 0.0

        # Compute matrix elements of SLD in the eigenbasis of rho
        for i in range(len(eigenvalues)):
            for j in range(len(eigenvalues)):
                # Avoid division by zero for nearly zero or zero eigenvalues
                # Check if the sum of eigenvalues is effectively zero
                if np.isclose(eigenvalues[i] + eigenvalues[j], 0.0, atol=1e-10):
                    continue  # Skip term if sum of eigenvalues is too small to avoid numerical instability

                # Project drho into the eigenbasis of rho: <v_i | drho | v_j>
                drho_ij = np.dot(
                    eigenvectors[:, i].conj().T, np.dot(drho, eigenvectors[:, j])
                )

                # Calculate the element of the Symmetric Logarithmic Derivative (SLD)
                L_ij = 2 * drho_ij / (eigenvalues[i] + eigenvalues[j])

                # Add term to Quantum Fisher Information (QFI)
                F_Q += (
                    np.abs(L_ij) ** 2 * eigenvalues[j]
                )  # This is the standard formula for QFI of mixed states

        # Restore original h_x value after computation
        self.h_x = original_h_x

        return F_Q


class SpinChainXXZ:
    """
    Simulates a 1D spin chain with nearest-neighbor XX interactions and local Z fields.

    Hamiltonian:
        H = J * ∑ X_i X_{i+1} + h_z * ∑ Z_i

    Open boundary conditions are used.
    """

    def __init__(self, n, J, h_z, initial_state="0", DEBUG=False):
        """
        Parameters:
        - n (int): Number of qubits.
        - J (float): Coupling strength for the X_i X_{i+1} interaction term.
        - h_z (float): Strength of the longitudinal Z field.
        - initial_state (str or np.ndarray): Initial state ("0", "H", or ndarray)
        - DEBUG (bool): Print diagnostic information.
        """
        self.n = n
        self.J = J
        self.h_z = h_z
        self.paulis = self._pauli_matrices()
        dim = 2**n

        # Prepare initial state
        if isinstance(initial_state, np.ndarray):
            if initial_state.shape == (dim,):
                self.initial_rho = np.outer(initial_state, initial_state.conj())
            elif initial_state.shape == (dim, dim):
                self.initial_rho = initial_state
            else:
                raise ValueError("Invalid shape for initial_state.")

        elif initial_state == "0":
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
            self.initial_rho = np.outer(state, state.conj())

        elif initial_state == "H":
            state = np.ones(dim, dtype=complex) / np.sqrt(dim)
            self.initial_rho = np.outer(state, state.conj())
        else:
            raise ValueError("initial_state must be '0', 'H', or a valid ndarray.")

        if DEBUG:
            print(f"Initial ρ:\n{self.initial_rho}")

        if not np.allclose(self.initial_rho, self.initial_rho.conj().T):
            raise ValueError("Initial matrix is not Hermitian.")
        if not np.isclose(np.trace(self.initial_rho), 1):
            raise ValueError("Initial matrix trace is not 1.")

    def _pauli_matrices(self):
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return {"I": I, "X": X, "Z": Z}

    def _single_site_operator(self, op, site):
        """Builds a tensor product operator with `op` at position `site`."""
        ops = [self.paulis["I"]] * self.n
        ops[site] = op
        out = ops[0]
        for i in range(1, self.n):
            out = np.kron(out, ops[i])
        return out

    def _two_site_interaction(self, op, site1, site2):
        """Builds an operator applying `op` to qubits `site1` and `site2`."""
        ops = [self.paulis["I"]] * self.n
        ops[site1] = op
        ops[site2] = op
        out = ops[0]
        for i in range(1, self.n):
            out = np.kron(out, ops[i])
        return out

    def _construct_hamiltonian(self):
        """
        Builds:
        H = J * ∑ X_i X_{i+1} + h_z * ∑ Z_i
        """
        X = self.paulis["X"]
        Z = self.paulis["Z"]
        H = np.zeros((2**self.n, 2**self.n), dtype=complex)

        # XX interactions
        for i in range(self.n - 1):
            H += self.J * self._two_site_interaction(X, i, i + 1)

        # Z fields
        for i in range(self.n):
            H += self.h_z * self._single_site_operator(Z, i)

        return H

    def generate_density_matrix(self):
        """
        Evolve initial state under H: ρ → U ρ U†

        Returns:
        - np.ndarray: Final density matrix after unitary evolution.
        """
        H = self._construct_hamiltonian()
        U = expm(-1j * H)
        return U @ self.initial_rho @ U.conj().T

    def generate_density_matrix(self):
        """
        Evolves the initial density matrix under the Hamiltonian using unitary evolution.

        Returns:
        - rho: The evolved 2^n-dimensional density matrix as a numpy array.
        """
        # Construct the Hamiltonian
        H = self._construct_hamiltonian()

        # Unitary evolution operator
        U = expm(-1j * H)

        # Evolve the density matrix:
        # ρ(t) = U ρ(0) U†
        rho = U @ self.initial_rho @ U.conj().T

        return rho

    def generate_density_matrices_with_perturbation(self, delta=0.01):
        """
        Generates two density matrices: one with (J_xx, h_z) and another with a perturbed field (h_z + delta).

        Parameters:
        - delta: Small perturbation to the external field h_z.

        Returns:
        - (rho, rho_perturbed): Tuple containing the original and perturbed density matrices.
        """
        rho = self.generate_density_matrix()

        # Temporarily modify the field (h_z + theta)
        original_hz = self.h_z
        self.h_z += delta
        rho_perturbed = self.generate_density_matrix()

        # Restore original h_z
        self.h_z = original_hz

        return rho, rho_perturbed

    def generate_mixed_density_matrices_with_perturbation(
        self, trace_out_indices, delta=0.01
    ):
        """
        Generates a mixed state by tracing out one or more qubits.
        """

        rho_pure, rho_delta_pure = self.generate_density_matrices_with_perturbation(
            delta=delta
        )

        rho_mixed = helper_functions.trace_out(
            rho=rho_pure, trace_out_index=trace_out_indices
        )
        rho_delta_mixed = helper_functions.trace_out(
            rho=rho_delta_pure, trace_out_index=trace_out_indices
        )

        return rho_mixed, rho_delta_mixed

    # @staticmethod
    # def generate_random_positive_density_matrix(n):
    #     """
    #     Generates a random valid density matrix for an n-qubit system.

    #     Parameters:
    #     - n: Number of qubits.

    #     Returns:
    #     - rho: A random valid density matrix (Hermitian, PSD, trace = 1).
    #     """
    #     dim = 2**n  # Dimension of the Hilbert space
    #     mat = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    #     mat = mat @ mat.conj().T  # Ensure Hermitian & positive semi-definite
    #     return mat / np.trace(mat)  # Normalize to ensure trace = 1

    # def generate_mixed(self, delta=0.01):
    #     """
    #     Generates a mixed state by tracing out one or more qubits.

    #     Parameters:
    #         delta (float): Perturbation size for the density matrix.

    #     Returns:
    #         rho (numpy.ndarray): Reduced density matrix after tracing out.
    #         rho_delta (numpy.ndarray): Perturbed reduced density matrix.
    #     """

    #     # Generate perturbed density matrices
    #     rho_0, rho_delta_0 = self.generate_density_matrices_with_perturbation(
    #         delta=delta
    #     )

    #     # Reduce both matrices
    #     rho_reduced = helper_functions.trace_out(rho_0, self.trace_out_index)
    #     rho_delta_reduced = helper_functions.trace_out(
    #         rho_delta_0, self.trace_out_index
    #     )

    #     return rho_reduced, rho_delta_reduced

    def compute_qfi_with_sld(
        self, delta, d=1e-5
    ):  # this is the delta for the numerical derivative
        """
        Compute the Quantum Fisher Information (QFI) for a mixed quantum state,
        using the symmetric logarithmic derivative (SLD).

        Parameters:
        rho  : ndarray
            Density matrix of the quantum state (NxN, Hermitian, positive semi-definite, trace = 1)
        drho : ndarray
            Derivative of the density matrix with respect to the parameter (NxN)

        Returns:
        F_Q : float
            Quantum Fisher Information (QFI) for the given state and its derivative
        """

        rho = self.generate_density_matrix()

        drho = self.compute_drho(
            delta=delta, d=d
        )  # Compute the derivative of the density matrix
        # d rho(theta)
        # d theta

        # Compute the eigenvalues and eigenvectors of the density matrix
        eigenvalues, eigenvectors = np.linalg.eigh(rho)

        # Initialize QFI sum
        F_Q = 0

        # Compute matrix elements of SLD in the eigenbasis of rho
        for i in range(len(eigenvalues)):
            for j in range(len(eigenvalues)):
                if eigenvalues[i] + eigenvalues[j] > 1e-10:  # Avoid division by zero
                    L_ij = (
                        2
                        * np.dot(
                            eigenvectors[:, i].conj().T,
                            np.dot(drho, eigenvectors[:, j]),
                        )
                        / (eigenvalues[i] + eigenvalues[j])
                    )
                    F_Q += np.abs(L_ij) ** 2 * eigenvalues[j]

        return F_Q

    def compute_drho(self, delta, d=1e-4):
        """
        Compute the numerical derivative of the density matrix with respect to theta.

        Parameters:
        rho_func : function
            Function that returns the density matrix as a function of theta.
        theta : float
            Parameter value at which to compute the derivative.
        delta : float, optional
            Small step size for numerical differentiation.

        Returns:
        drho : ndarray
            Numerical derivative of rho with respect to theta.
        """
        # Temporarily modify the field (h_z + theta)
        original_hz = self.h_z

        self.h_z -= d

        rho_m_delta, rho_delta = self.generate_density_matrices_with_perturbation(2 * d)
        # Restore original h_z
        self.h_z = original_hz

        return (rho_delta - rho_m_delta) / (2 * d)


# --- Example Usage ---
if __name__ == "__main__":
    # --- Dummy helper_functions for demonstration ---
    # In a real application, you would import a proper 'trace_out' function
    # from a quantum library like Qiskit or QuTiP, or implement it yourself.
    class DummyHelperFunctions:
        @staticmethod
        def trace_out(rho, trace_out_index, total_qubits):
            """
            A placeholder 'trace_out' function for demonstration purposes.
            This implementation is NOT a correct general trace-out operation.
            You MUST replace this with a proper implementation or use a library function.
            It simply returns a normalized identity matrix of the reduced dimension.
            """
            print(
                f" (Note: Using dummy trace_out for index(es) {trace_out_index}. "
                "You need to replace this with your actual trace_out implementation!)"
            )

            # For a proper trace_out, you'd need to handle single/multiple indices,
            # and permute dimensions if tracing out non-adjacent qubits.
            # This is a complex operation not generically implemented here.

            # Simple placeholder: return a normalized identity matrix of the reduced size
            # Calculate the dimension after tracing out:
            if isinstance(trace_out_index, int):
                num_traced_out = 1
            elif isinstance(trace_out_index, (list, tuple)):
                num_traced_out = len(trace_out_index)
            else:
                raise ValueError(
                    "trace_out_index must be an int or a list/tuple of ints."
                )

            remaining_qubits = total_qubits - num_traced_out
            if remaining_qubits < 0:
                raise ValueError("Cannot trace out more qubits than available.")

            if remaining_qubits == 0:
                return np.array([[1.0]])  # Return 1x1 matrix for empty system

            reduced_dim = 2**remaining_qubits
            return np.eye(reduced_dim) / reduced_dim

    # Assign the dummy for testing. REMOVE THIS AND IMPORT YOUR ACTUAL helper_functions
    # if you have a proper 'trace_out' function.
    helper_functions = DummyHelperFunctions()
    # --- End Dummy helper_functions ---

    n_qubits = 2  # Number of spins
    J_coupling = 1.0  # Ferromagnetic coupling
    h_transverse = 0.5  # Transverse field strength
    evolution_time = 1.0  # Time for evolution

    # Initialize the TFIM model
    tfim_model = TransverseFieldIsingModel(
        n=n_qubits, J=J_coupling, h_x=h_transverse, initial_state="0", DEBUG=True
    )

    # Construct and print the Hamiltonian
    hamiltonian = tfim_model.construct_hamiltonian()
    print(f"\n--- Hamiltonian for N={n_qubits}, J={J_coupling}, h_x={h_transverse} ---")
    print(f"H:\n{np.round(hamiltonian, 2)}")

    # Evolve the initial density matrix
    evolved_rho = tfim_model.evolve_density_matrix(time=evolution_time)
    print(f"\n--- Evolved Density Matrix at t={evolution_time} ---")
    print(f"ρ(t):\n{np.round(evolved_rho, 4)}")

    # Generate perturbed density matrices
    delta_hx_perturb = 0.1
    rho_orig, rho_pert = tfim_model.generate_perturbed_density_matrices(
        delta_h_x=delta_hx_perturb, time=evolution_time
    )
    print(f"\n--- Perturbed Density Matrices (h_x changed by {delta_hx_perturb}) ---")
    print(f"ρ(original h_x):\n{np.round(rho_orig, 4)}")
    print(f"ρ(perturbed h_x):\n{np.round(rho_pert, 4)}")

    # Generate mixed density matrices with perturbation
    trace_indices = 0  # Trace out the first qubit (index 0)
    print(f"\n--- Mixed Density Matrices (tracing out qubit {trace_indices}) ---")
    rho_mixed_orig, rho_mixed_pert = tfim_model.generate_mixed_states_with_perturbation(
        trace_out_indices=trace_indices, delta_h_x=delta_hx_perturb, time=evolution_time
    )
    print(f"ρ_mixed(original h_x):\n{np.round(rho_mixed_orig, 4)}")
    print(f"ρ_mixed(perturbed h_x):\n{np.round(rho_mixed_pert, 4)}")

    # Compute Quantum Fisher Information
    h_x_for_qfi = 0.5  # Compute QFI at h_x = 0.5
    time_for_qfi = 1.0  # Compute QFI for state evolved for time=1.0
    qfi_value = tfim_model.compute_qfi_with_sld(h_x_val=h_x_for_qfi, time=time_for_qfi)
    print(f"\n--- Quantum Fisher Information ---")
    print(
        f"QFI with respect to h_x at h_x={h_x_for_qfi}, t={time_for_qfi}: {qfi_value:.6f}"
    )

    # Example of computing QFI at a different h_x value
    qfi_value_2 = tfim_model.compute_qfi_with_sld(h_x_val=0.1, time=0.0)
    print(f"QFI with respect to h_x at h_x={0.1}, t={0.0}: {qfi_value_2:.6f}")

    # Test initial state validation with an invalid shape
    print("\n--- Testing initial state validation (expected error) ---")
    try:
        TransverseFieldIsingModel(n=2, J=1, h_x=1, initial_state=np.zeros(3))
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test initial state validation with a non-Hermitian matrix
    print("\n--- Testing initial state validation (expected error - non-Hermitian) ---")
    try:
        non_hermitian_rho = np.array(
            [[0.5, 0.5 + 0.1j], [0.5 - 0.1j, 0.5 + 0.2j]]
        )  # Not Hermitian
        tfim_model_error = TransverseFieldIsingModel(
            n=1, J=1, h_x=1, initial_state=non_hermitian_rho
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")
