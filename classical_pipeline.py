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


def main(
    N=3,
    n=2,
    time_t=1.0,
    J=1,
    delta=0.5,
    delta_h_x=0.1,
    h_x=0.5,
    m=1,
    DEBUG=False,
    derivative_delta=1e-3,
    trace_out_indices=None,
    SLD=False,
):

    model = density_generator.TransverseFieldIsingModel(
        n=N, J=J, h_x=h_x, initial_state="0", DEBUG=DEBUG
    )

    rho_mixed, rho_delta_mixed = model.generate_mixed_states_with_perturbation(
        delta_h_x=delta_h_x,
        trace_out_indices=trace_out_indices,
        time=time_t,
    )

    results = classicalQFI.compute_tqfi_bounds(
        rho_mixed, rho_delta_mixed, m, delta, DEBUG=False
    )

    if SLD == True:
        qfi_from_SLD = model.compute_qfi_with_sld(
            h_x_val=h_x
        )  # todo check fderivatives
        results["qfi_from_SLD"] = qfi_from_SLD

    return results
