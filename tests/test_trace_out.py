import unittest
import numpy as np
import qutip

from helper_functions import trace_out


class TestTraceOut(unittest.TestCase):

    def setUp(self):
        # Create the Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
        bell_state = (qutip.basis(4, 0) + qutip.basis(4, 3)).unit()
        self.rho = (bell_state * bell_state.dag()).full()

        # Expected reduced state (maximally mixed state)
        self.expected_rho_reduced = 0.5 * np.eye(2)

    def test_trace_out_bell_state(self):
        # Convert to Qobj before tracing out (for printing)
        rho_qutip = qutip.Qobj(self.rho, dims=[[2, 2], [2, 2]])
        reduced_rho = trace_out(self.rho, trace_out_index=[0])
        reduced_rho_qutip = qutip.Qobj(reduced_rho, dims=[[2], [2]])

        # Pretty print matrices
        print("\n--- Original Density Matrix (NumPy) ---")
        print(np.array2string(self.rho, precision=4, separator=", "))

        print("\n--- Original Density Matrix (Qutip) ---")
        print(rho_qutip)

        print("\n--- Reduced Density Matrix (NumPy) ---")
        print(np.array2string(reduced_rho, precision=4, separator=", "))

        print("\n--- Reduced Density Matrix (Qutip) ---")
        print(reduced_rho_qutip)

        print("\n--- Expected Reduced Density Matrix ---")
        print(np.array2string(self.expected_rho_reduced, precision=4, separator=", "))

        # Compare with expected reduced state
        np.testing.assert_allclose(reduced_rho, self.expected_rho_reduced, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
