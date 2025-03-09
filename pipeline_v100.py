import density_generator

# from qiskit.visualization import plot_state_city
from importlib import reload
import pandas as pd
import numpy as np
import classicalQFI
import qutip

reload(density_generator)
reload(classicalQFI)

# pd.options.display.float_format = "{:,.3f}".format


def simulation(
    n=5,
    a_x=1,
    h_z=0.1,
    delta=0.01,
    m=1,
    DEBUG=False,
    trace_out_index=-1,
    derivative_delta=None,
) -> np.array:

    model = density_generator.IsingQuantumState(
        n=n, a_x=a_x, h_z=h_z, trace_out_index=trace_out_index
    )
    # rho_0, rho_delta_0 = model.generate_density_matrices_with_perturbation(delta=delta)

    # plot_state_city(rho)
    # print(pd.DataFrame(rho))

    # create a qutip object, inserting the density matrix
    # specify dims to make sure qutip knows the dimension of the system
    # (if not he thinks it is more than 2 state systemsand cannot take partial trace)
    # d = [[2 for i in range(n)], [2 for i in range(n)]]

    # rho_qutip = qutip.Qobj(rho_0, dims=d)
    # rho_delta_qutip = qutip.Qobj(rho_delta_0, dims=d)

    # trace out all qubits except the last one
    # rho = rho_qutip.ptrace(sel=list(range(n - 1))).full()
    # rho_delta = rho_delta_qutip.ptrace(sel=list(range(n - 1))).full()

    rho, rho_delta = model.generate_mixed(delta=delta)

    ## Getting results from classical QFI
    results = classicalQFI.compute_tqfi_bounds(
        rho=rho, rho_delta=rho_delta, m=m, delta=delta
    )

    if derivative_delta is None:
        derivative_delta = (delta / 1000,)

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
        print()
    results["trace_rho"] = np.trace(rho)
    results["trace_rho_delta"] = np.trace(rho_delta)
    results["purity_rho"] = np.trace(rho @ rho)
    results["purity_rho_delta"] = np.trace(rho_delta @ rho_delta)
    results["rank_rho"] = np.linalg.matrix_rank(rho)
    results["rank_rho_delta"] = np.linalg.matrix_rank(rho_delta)
    results["QFI_from_SLD"] = qfi_from_SLD

    return results
