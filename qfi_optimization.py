import numpy as np
import pipeline_v100
from scipy.optimize import minimize_scalar


def get_lower_bound(n, a_x, delta, m, trace_out_index, derivative_delta):
    def lower_bound_for_h_z(h_z):
        return np.real(
            pipeline_v100.simulation(
                n=n,
                a_x=a_x,
                h_z=h_z,
                delta=delta,
                trace_out_index=trace_out_index,
                derivative_delta=derivative_delta,
                m=m,
                DEBUG=True,
            )["lower_tqfi"]
        )

    return lower_bound_for_h_z


def get_best_lower_bound(
    n, a_x, delta, m, trace_out_index, derivative_delta, h_z_bounds=(0, 2.5)
):
    # Generate the theta function dependent on h_z
    lower_bound_for_h_z = get_lower_bound(
        n, a_x, delta, m, trace_out_index, derivative_delta
    )

    res = minimize_scalar(
        lambda h_z: -lower_bound_for_h_z(h_z),  # Maximize by minimizing the negative
        bounds=h_z_bounds,
        method="bounded",
    )

    res.fun = -res.fun  # Restore the sign

    return res
