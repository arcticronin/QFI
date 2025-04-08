import numpy as np
import pipeline_v200
from scipy.optimize import minimize_scalar


## returns a function that computes the lower bound for a given h_z
def get_lower_bound(N, n, a_x, delta, m, initial_state, DEBUG, derivative_delta):
    """
    Returns a function that computes the lower bound for a given h_z.

    Args:
        n (int): Number of qubits.
        a_x (float): Coupling coefficient for the Ising interaction term.
        delta (float): Perturbation strength.
        m (int): Number of largest eigenvalues to keep.
        trace_out_index [int]: list of indices of the qubit to trace out.
        derivative_delta (float): Step size for the derivative.

    """

    def lower_bound_for_h_z(h_z):
        """returns the lower bound for a given h_z.

        Args:
            h_z (float): _description_

        Returns:
            float : value for the lower bound at h_z
        """
        return np.real(
            pipeline_v200.simulation(
                N=N,
                n=n,
                a_x=a_x,
                h_z=h_z,
                delta=delta,
                m=m,
                initial_state=initial_state,
                DEBUG=DEBUG,
                derivative_delta=derivative_delta,
            )[
                "lower_tqfi"
                # "sub_qfi_bound"
            ]  ## forse l'altro bound??
        )

    return lower_bound_for_h_z


def get_best_lower_bound(
    N, n, a_x, delta, m, initial_state, DEBUG, derivative_delta, h_z_bounds=(0, 2.5)
):
    """
    Returns the best lower bound for the given parameters.

    Args:
        n (int): Number of qubits.
        a_x (float): Coupling coefficient for the Ising interaction term.
        delta (float): Perturbation strength.
        m (int): Number of largest eigenvalues to keep.
        trace_out_index [int]: list of indices of the qubit to trace out.
        derivative_delta (float): Step size for the derivative.
        h_z_bounds (int, int): Bounds for h_z.

    """
    lower_bound_for_h_z = get_lower_bound(
        N=N,
        n=n,
        a_x=a_x,
        delta=delta,
        m=m,
        initial_state=initial_state,
        DEBUG=DEBUG,
        derivative_delta=derivative_delta,
    )

    # Step 1: Optimize within bounds
    res = minimize_scalar(
        lambda h_z: -lower_bound_for_h_z(h_z),
        bounds=h_z_bounds,
        method="bounded",
    )

    # Restore the sign for the optimizer's result
    res.fun = -res.fun

    # Step 2: Compute boundary values
    h_z_min = h_z_bounds[0]
    h_z_max = h_z_bounds[1]

    res.value_at_min = lower_bound_for_h_z(h_z_min)
    res.value_at_max = lower_bound_for_h_z(h_z_max)

    return res


## this keeps all the info in the dicitonary, useful for debugging purpose


def get_results_for_m_hz(N, n, a_x, delta, m, initial_state, DEBUG, derivative_delta):
    def get_results_for_h_z(h_z):
        return pipeline_v200.simulation(
            N=N,
            n=n,
            a_x=a_x,
            h_z=h_z,
            delta=delta,
            m=m,
            initial_state=initial_state,
            DEBUG=DEBUG,
            derivative_delta=derivative_delta,
        )

    return get_results_for_h_z


def extract_result_given_h_z(
    N, n, a_x, delta, m, initial_state, DEBUG, derivative_delta, res_key
):
    def res_for_h_z(h_z):
        """Returns the value associated with `res_key` for a given h_z.

        Args:
            h_z (float): Value of h_z parameter.

        Returns:
            float: The value for the lower bound at h_z.

        Raises:
            KeyError, IndexError, TypeError: If res_key is not valid for the result.
        """
        res = pipeline_v200.simulation(
            N=N,
            n=n,
            a_x=a_x,
            h_z=h_z,
            delta=delta,
            m=m,
            initial_state=initial_state,
            DEBUG=DEBUG,
            derivative_delta=derivative_delta,
        )
        try:
            return np.real(res[res_key])
        except (KeyError, IndexError, TypeError) as e:
            print(
                f"res = {res}\n"
                f"[ERROR] Failed to access res[{res_key}] — {type(e).__name__}: {e}\n"
                f"Type of res: {type(res)}\n"
                f"Available keys or indices: "
                f"{list(res.keys()) if hasattr(res, 'keys') else f'Not a dict (shape: {res.shape if hasattr(res, 'shape') else 'N/A'})'}"
            )
            raise

    return res_for_h_z
