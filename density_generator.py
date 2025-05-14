import numpy as np
from scipy.linalg import expm
import helper_functions

# import pandas as pd


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
