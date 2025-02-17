import numpy as np
from scipy.linalg import expm


class IsingQuantumState:
    """
    A class to represent the quantum state evolution under an n-qubit Ising-like Hamiltonian.
    """

    def __init__(self, n, a_x, h_z):
        """
        Initialize the system with given interaction and external field parameters.

        Parameters:
        - n: Number of qubits.
        - a_x: Coupling coefficient for the Ising interaction term.
        - h_z: Coefficient for the external field in the Z direction.
        """
        self.n = n
        self.a_x = a_x
        self.h_z = h_z
        self.paulis = self._pauli_matrices()

    @staticmethod
    def _pauli_matrices():
        """
        Returns a dictionary of single-qubit Pauli matrices.
        """
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
        for i in range(self.n):
            operators.append(op if i == site else I)
        # Take the tensor product in order
        out = operators[0]
        for i in range(1, self.n):
            out = np.kron(out, operators[i])
        return out

    def _two_site_interaction(self, op1, site1, op2, site2):
        """
        Builds an n-qubit operator that applies 'op1' on 'site1' and 'op2' on 'site2',
        acting like identity on other sites.
        """
        # We'll build single-site operators and multiply them
        # I am reusing `_single_site_operator`, but at some point if needed Ican implement a loop approach
        if site1 == site2:
            raise ValueError("site1 and site2 must be different.")

        # For clarity, let's build separately, then multiply
        # Sincle site alresdy has the identity on the other sites
        op_on_site1 = self._single_site_operator(op1, site1)
        op_on_site2 = self._single_site_operator(op2, site2)
        return op_on_site1 @ op_on_site2

    def _construct_hamiltonian(self):
        """
        Constructs the n-qubit Ising-like Hamiltonian:

        For a 1D chain of n qubits with open boundary conditions:
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
            H += self.a_x * self._two_site_interaction(X, i, X, i + 1)

        # Local field terms: h_z * Z_i
        for i in range(self.n):
            ## Caso i = 0, i = n-1? boundary contitions?
            H += self.h_z * self._single_site_operator(Z, i)
        return H

    def generate_density_matrix(self):
        """
        Generates the density matrix rho for the system evolved under the Hamiltonian
        from the initial state |0...0> (n zeros).

        Returns:
        - rho: The 2^n-dimensional density matrix as a numpy array.
        """
        # Construct the Hamiltonian
        H = self._construct_hamiltonian()
        # Unitary evolution operator
        U = expm(-1j * H)

        # Define initial state |0...0> in computational basis
        # For n qubits, |0...0> is dimension 2^n with a 1 in the first component.
        dim = 2**self.n
        ket_0n = np.zeros(dim, dtype=complex)
        ket_0n[0] = 1.0  # this is |0...0>

        # Apply U to |0...0>
        psi = U @ ket_0n

        # Construct the density matrix
        rho = np.outer(psi, np.conj(psi))
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

        # Temporarily modify the field
        original_hz = self.h_z
        self.h_z += delta
        rho_perturbed = self.generate_density_matrix()

        # Restore original h_z
        self.h_z = original_hz

        return rho, rho_perturbed

    @staticmethod
    def generate_random_positive_density_matrix(n):
        """
        Generates a random valid density matrix for an n-qubit system.

        Parameters:
        - n: Number of qubits.

        Returns:
        - rho: A random valid density matrix (Hermitian, PSD, trace = 1).
        """
        dim = 2**n  # Dimension of the Hilbert space
        mat = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        mat = mat @ mat.conj().T  # Ensure Hermitian & positive semi-definite
        return mat / np.trace(mat)  # Normalize to ensure trace = 1
