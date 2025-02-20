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


def simulation(n=5, a_x=1, h_z=0.1) -> np.array:

    model = density_generator.IsingQuantumState(n=n, a_x=a_x, h_z=h_z)
    rho_0, rho_delta_0 = model.generate_density_matrices_with_perturbation(delta=0.9)

    # plot_state_city(rho)
    # print(pd.DataFrame(rho))

    # create a qutip object, inserting the density matrix
    # specify dims to make sure qutip knows the dimension of the system
    # (if not he thinks it is more than 2 state systemsand cannot take partial trace)
    d = [[2 for i in range(n)], [2 for i in range(n)]]

    rho_qutip = qutip.Qobj(rho_0, dims=d)
    rho_delta_qutip = qutip.Qobj(rho_delta_0, dims=d)

    # trace out all qubits except the last one
    rho = rho_qutip.ptrace(sel=list(range(n - 1))).full()
    rho_delta = rho_delta_qutip.ptrace(sel=list(range(n - 1))).full()

    lower_tqfi, upper_tqfi = classicalQFI.compute_tqfi_bounds(
        rho=rho, rho_delta=rho_delta, m=1, delta=0.01
    )

    return lower_tqfi, upper_tqfi
