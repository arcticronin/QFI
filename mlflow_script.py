import mlflow

import sys
import os
import numpy as np

# Add the parent folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import classical_pipeline
import quantum_pipeline
import pennylane as qml
from scipy.linalg import eigh


def main():
    # --- Global Simulation Configuration ---
    SIMULATION_TYPE = "quantum"  #  "classical" or "quantum"
    N = 5
    n = 4
    J = 1
    time_t = 1.0
    delta_h_x = 0.1  # [0.01..0.1] This delta is used for the perturbation in the Hamiltonian AND the QFI formula
    m = 2  # most meaningful eigenvectors
    DEBUG = True  # true in classical also tracks purity

    # Parameters for classical pipeline (if applicable)
    derivative_delta = 1e-2

    # Parameters for quantum pipeline (if applicable)
    trotter_steps_K = 4
    trotter_order = 1

    # --- h_x Sweep Configuration ---
    h_x_start = 0.0
    h_x_end = 2.5
    h_x_steps = 20
    h_x_values = np.linspace(h_x_start, h_x_end, h_x_steps).tolist()

    # --- Pre-calculate trace_out_indices once if they are fixed for all runs ---
    if N - n < 0:
        raise ValueError("n must be less than or equal to N to trace out qubits.")
    if N - n > 0:
        # np.random.seed(42)  # For reproducibility of random choice
        global_trace_out_indices = np.random.choice(
            range(N), size=N - n, replace=False
        ).tolist()
    else:
        global_trace_out_indices = []

    print("--- Global Simulation Configuration ---")
    print(f"SIMULATION_TYPE: {SIMULATION_TYPE}")
    print(f"N: {N} (initial total qubits, before tracing out)")
    print(f"n: {n} (final number of qubits)")
    print(f"J: {J}")
    print(f"time_t: {time_t}")
    print(f"delta_h_x (physical perturbation AND QFI formula delta): {delta_h_x}")
    print(f"Eigenvector basis size (m): {m}")
    print(f"DEBUG mode: {DEBUG}")
    print(f"h_x sweep range: {h_x_start} to {h_x_end} in {h_x_steps} steps.")
    print(f"Common trace_out_indices: {global_trace_out_indices}")
    if SIMULATION_TYPE == "classical":
        print(f"Classical-specific derivative_delta: {derivative_delta}")
    elif SIMULATION_TYPE == "quantum":
        print(f"Quantum-specific Trotter steps (K): {trotter_steps_K}")
        print(f"Quantum-specific Trotter order: {trotter_order}")
    print("---------------------------------------\n")

    # --- Start a single MLflow run for the entire sweep ---
    with mlflow.start_run(
        run_name=f"h_x_Sweep_{SIMULATION_TYPE}_N{N}_n{n}_delta_h_x{delta_h_x}"
    ):
        # Log common parameters for this consolidated run (not varying in the sweep)
        mlflow.log_param("SIMULATION_TYPE", SIMULATION_TYPE)
        mlflow.log_param("N", N)
        mlflow.log_param("n", n)
        mlflow.log_param("J", J)
        mlflow.log_param("time_t", time_t)
        mlflow.log_param("delta_h_x", delta_h_x)
        mlflow.log_param("m", m)
        mlflow.log_param("h_x_start", h_x_start)
        mlflow.log_param("h_x_end", h_x_end)
        mlflow.log_param("trace_out_indices", str(global_trace_out_indices))
        mlflow.log_param("DEBUG", DEBUG)

        if SIMULATION_TYPE == "classical":
            mlflow.log_param("derivative_delta", derivative_delta)
        elif SIMULATION_TYPE == "quantum":
            mlflow.log_param("trotter_steps_K", trotter_steps_K)
            mlflow.log_param("trotter_order", trotter_order)

        print("\n--- Starting h_x sweep within a single MLflow run ---")
        # --- Iterate through h_x values and log step-by-step ---
        for step, current_h_x in enumerate(h_x_values):
            print(f"Processing h_x = {current_h_x:.3f} (Step {step+1}/{h_x_steps})")

            results_dict = {}  # Initialize results_dict for this step

            if SIMULATION_TYPE == "classical":
                results_dict = classical_pipeline.main(
                    N=N,
                    n=n,
                    J=J,
                    delta=delta_h_x,
                    h_x=current_h_x,
                    m=m,
                    DEBUG=DEBUG,
                    derivative_delta=derivative_delta,
                    trace_out_indices=global_trace_out_indices,
                    SLD=True,
                )
            elif SIMULATION_TYPE == "quantum":
                results_dict = quantum_pipeline.main(
                    N=N,
                    n=n,
                    time_t=time_t,
                    J=J,
                    delta_h_x=delta_h_x,
                    h_x=current_h_x,
                    m=m,
                    DEBUG=DEBUG,
                    trotter_steps_K=trotter_steps_K,
                    trotter_order=trotter_order,
                    trace_out_indices=global_trace_out_indices,
                )

            # Log results as metrics for THIS STEP
            # The 'step' parameter is crucial for plotting.
            mlflow.log_metric(
                "h_x_value", current_h_x, step=step
            )  # Log h_x as a metric
            for key, value in results_dict.items():
                # if isinstance(value, complex):
                #     mlflow.log_metric(key + "_real", value.real, step=step)
                #     if not np.isclose(value.imag, 0.0, atol=1e-9):
                #         mlflow.log_metric(key + "_imag", value.imag, step=step)
                if isinstance(value, complex):
                    mlflow.log_metric(key, value.real, step=step)
                else:
                    mlflow.log_metric(key, value, step=step)

            # Optional: Add a printout for each step's logged metrics if needed for verbosity
            # print(f"  Logged metrics for this step: {results_dict}")

        print("\n--- h_x sweep completed for this MLflow run ---")


if __name__ == "__main__":
    main()
