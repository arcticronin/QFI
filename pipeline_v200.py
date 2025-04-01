import density_generator

# from qiskit.visualization import plot_state_city
from importlib import reload
import pandas as pd
import numpy as np
import classicalQFI
import qutip
import helper_functions

reload(helper_functions)
reload(density_generator)
reload(classicalQFI)

# pd.options.display.float_format = "{:,.3f}".format


def simulation(
    N=10,
    n=7,
    a_x=1,
    h_z=0.1,
    delta=0.01,
    m=1,
    initial_state=None,
    DEBUG=False,
    derivative_delta=1e-3,
) -> np.array:

    if initial_state is None:
        if DEBUG:
            print("Generating random initial state")
        # Generate random initial state
        ket = helper_functions.random_haar_ket(2**N)

        rho = np.outer(ket, np.conj(ket))

        trace_out_index = np.random.choice(range(N), size=N - n, replace=False)

        v = helper_functions.trace_out(rho=rho, trace_out_index=trace_out_index)
    else:
        v = initial_state

    if DEBUG:
        purity = np.trace(v @ v)
        print(f"purity of initial state = {purity} ")

    # Generate new density matrix using Ising model
    model = density_generator.IsingQuantumState(
        n=n, a_x=a_x, h_z=h_z, initial_state=v, DEBUG=DEBUG
    )
    rho = model.generate_density_matrix()

    if DEBUG:
        # Calculate purity and participation ratio
        purity = np.trace(rho @ rho)
        participation_ratio = 1 / purity
        print(f"Purity of evolved state: {purity}")
        print(f"Participation ratio: {participation_ratio}")

    ####### END) generation of the matrices

    rho, rho_delta = model.generate_density_matrices_with_perturbation(delta=delta)

    ## Getting results from classical QFI
    results = classicalQFI.compute_tqfi_bounds(
        rho=rho, rho_delta=rho_delta, m=m, delta=delta
    )

    if derivative_delta is None:
        derivative_delta = (delta / 100,)

    # this method is not parameter agnostic, so it is not into classicalQFI
    qfi_from_SLD = model.compute_qfi_with_sld(delta=delta, d=derivative_delta)

    # append the true QFI to the results, taken using the pure states,
    # before the partial trace (this idea was not good)

    ## Optional overlap with |00...00>
    # ket_0n = np.zeros(n-1, dtype=complex)
    # ket_0n[0] = 1.0
    # ket_0n = ket_0n.transpose()
    # overlap = ket_0n @

    if DEBUG:  ## check trace of rho
        # print(f"trace of rho: {np.trace(rho)}")
        # print(f"trace of rho + delta: {np.trace(rho_delta)}")
        ## check also purity
        # print(f"purity of rho: {np.trace(rho @ rho)}")
        # print(f"purity of rho + delta: {np.trace(rho_delta @ rho_delta)}")
        ## check also rank
        # print(f"rank of rho: {np.linalg.matrix_rank(rho)}")
        # print(f"rank of rho + delta: {np.linalg.matrix_rank(rho_delta)}")
        # put all of the above in a dictionary
        results["trace_rho"] = np.trace(rho)
        results["trace_rho_delta"] = np.trace(rho_delta)
        results["purity_rho"] = np.trace(rho @ rho)
        results["purity_rho_delta"] = np.trace(rho_delta @ rho_delta)
        results["rank_rho"] = np.linalg.matrix_rank(rho)
        results["rank_rho_delta"] = np.linalg.matrix_rank(rho_delta)

        results["truncated_eigenvalues"] = (
            helper_functions.get_truncated_eigen_decomposition(rho=rho, m=m)[0]
        )
        results["eigenvalues"] = helper_functions.compute_eigen_decomposition(rho=rho)[
            0
        ]

    results["QFI_from_SLD"] = qfi_from_SLD

    return results
