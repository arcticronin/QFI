import numpy as np
from scipy.linalg import expm
import pandas as pd
import helper_functions


class IsingQuantumState:
    """
    A class to represent the quantum state evolution under an n-qubit Ising-like Hamiltonian.
    """

    def __init__(
        self,
        n,
        a_x,
        h_z,
        # trace_out_index,
        initial_state="0",
        DEBUG=False,
    ):
        """
        Initialize the system with given interaction and external field parameters.

        Parameters:
        - n: Number of qubits.
        - a_x: Coupling coefficient for the Ising interaction term.
        - h_z: Coefficient for the external field in the Z direction.
        # - trace_out_index: Indices of the qubits to trace out.
        - initial_state: Initial state or density matrix.
            - "0" -> Computational basis state |0...0>
            - "H" -> Uniform Hadamard state
            - ndarray -> Custom state vector or density matrix
        """
        self.n = n
        self.a_x = a_x
        self.h_z = h_z
        self.paulis = self._pauli_matrices()

        dim = 2**n

        if DEBUG:
            print(f"Dimension of the Hilbert space: {dim}")
            print(
                f"Initial state type: {type(initial_state)} | initial_state: {initial_state}"
            )

        # ---------- Handle state vector or density matrix ----------
        if isinstance(initial_state, np.ndarray):
            if initial_state.shape == (dim,):
                # State vector → Convert to density matrix
                self.initial_rho = np.outer(initial_state, initial_state.conj())
                if DEBUG:
                    print(
                        "Initial state provided as state vector -> Converted to density matrix."
                    )
            elif initial_state.shape == (dim, dim):
                self.initial_rho = initial_state
                if DEBUG:
                    print("Initial state provided as density matrix.")
            else:
                raise ValueError(f"Invalid initial state shape: {initial_state.shape}")

        elif initial_state == "0":
            # Computational basis state |0...0>
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
            self.initial_rho = np.outer(state, state.conj())
            if DEBUG:
                print("Initial state: |0...0> (converted to density matrix)")

        elif initial_state == "H":
            # Hadamard state (uniform superposition)
            state = np.ones(dim, dtype=complex) / np.sqrt(dim)
            self.initial_rho = np.outer(state, state.conj())
            if DEBUG:
                print("Initial state: Hadamard state (converted to density matrix)")

        else:
            raise ValueError(
                "Invalid initial state. Choose '0', 'H', or provide a valid custom state or matrix."
            )

        # ---------- Validate density matrix ----------
        if self.initial_rho.shape != (dim, dim):
            raise ValueError(
                f"Invalid initial matrix shape: {self.initial_rho.shape}, expected ({dim}, {dim})"
            )
        if not np.allclose(self.initial_rho, self.initial_rho.conj().T):
            raise ValueError("Initial matrix is not Hermitian.")
        # if not np.all(np.linalg.eigvals(self.initial_rho) >= 0):
        #    raise ValueError("Initial matrix is not positive semidefinite.")
        if not np.isclose(np.trace(self.initial_rho), 1):
            raise ValueError("Initial matrix trace is not 1.")

        # # ---------- Handle trace out indices ----------
        # if trace_out_index == -1:
        #     self.trace_out_index = [n - 1]
        #     if DEBUG:
        #         print("Tracing out the last qubit.")
        # else:
        #     if any(not (0 <= i < n) for i in trace_out_index):
        #         raise ValueError("Invalid trace_out_index: Index out of bounds.")
        #     self.trace_out_index = trace_out_index

    def _pauli_matrices(self):
        """Define Pauli matrices."""
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return {"I": I, "X": X, "Y": Y, "Z": Z}

    def _single_site_operator(self, op, site):
        """
        Builds an n-qubit operator that acts as 'op' on a specific qubit 'site'
        (0-indexed) and acts like the identity on the others.
        """
        # For n qubits, we do a tensor product of I on all sites except 'site'
        # where we put 'op'
        I = self.paulis["I"]
        operators = []
        for i in range(self.n):  # build the list of the operators
            operators.append(op if i == site else I)

        # Take the tensor product in order
        out = operators[0]
        for i in range(1, self.n):
            out = np.kron(out, operators[i])
        return out

    def _two_site_interaction(self, op, site1, site2):
        """
        Constructs an n-qubit operator applying 'op' on site1 and site2.
        """
        I = self.paulis["I"]
        operators = [I] * self.n  # Initialize all sites as identity
        operators[site1] = op  # modify the 2 we want to apply the operator
        operators[site2] = op

        # Compute the tensor product
        out = operators[0]
        for i in range(1, self.n):
            out = np.kron(out, operators[i])
        return out

    def _construct_hamiltonian(self):
        """
        Constructs the n-qubit Ising-like Hamiltonian:

        For a 1D chain of n qubits with OPEN boundary conditions:
            H = a_x * sum_{i=0 to n-2} (X_i X_{i+1})
                + h_z * sum_{i=0 to n-1} (Z_i)

        change boundary conditions or add more terms ???
        """
        X = self.paulis["X"]
        Z = self.paulis["Z"]

        # Build the Hamiltonian term by term.
        # Interaction terms: a_x * (X_i X_{i+1})
        H = np.zeros((2**self.n, 2**self.n), dtype=complex)

        # Sum over pairs (i, i+1) for the X-X interaction (anti_diagonal?)
        for i in range(self.n - 1):
            H += self.a_x * self._two_site_interaction(X, i, i + 1)

        # print("H at step 1")
        # print(pd.DataFrame(H))

        # Local field terms: h_z * Z_i
        for i in range(self.n):
            ## Caso i = 0, i = n-1? boundary contitions? OPEN BOUNDARY
            H += self.h_z * self._single_site_operator(Z, i)
        # print("H at step 2")
        # print(pd.DataFrame(H))

        return H

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
        Generates two density matrices: one with (a_x, h_z) and another with a perturbed field (h_z + delta).

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

    def compute_drho(self, delta, d=1e-5):
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

        self.h_z -= delta

        rho_m_delta, rho_delta = self.generate_density_matrices_with_perturbation(
            2 * delta
        )
        # Restore original h_z
        self.h_z = original_hz

        return (rho_delta - rho_m_delta) / (2 * d)
