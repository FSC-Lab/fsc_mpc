# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import time
import numpy as np
import tqdm

from quadrotor_mpc import acados_wrapper, quadrotor_model, trajectory_generator


def run_simulation(
    model: quadrotor_model.QuadrotorModel,
    solver: acados_wrapper.AcadosWrapper,
    trajectory: trajectory_generator.trajectories.Trajectory,
    reference_over_sampling: int,
):
    simulation_dt = 5e-4
    # Set quad initial state equal to the initial reference trajectory state
    quad_current_state = trajectory.states[0, :]
    model.state = quad_current_state

    u_setpoint = trajectory.inputs[0, :]

    # Sliding reference trajectory initial index
    current_idx = 0

    # Measure the MPC optimization time
    mean_opt_time = 0.0

    # Measure total simulation time
    total_sim_time = 0.0

    print("\nRunning simulation...")
    tout = [time.time()]
    yout = {"states": [], "inputs": []}
    for current_idx in tqdm.tqdm(range(len(trajectory))):

        quad_current_state = model.state

        yout["states"].append(np.array(quad_current_state))

        # ##### Optimization runtime (outer loop) ##### #
        # Get the chunk of trajectory required for the current optimization
        x_reference, u_reference = trajectory.get_reference_chunk(
            current_idx,
            solver.n_nodes,
            reference_over_sampling,
        )

        # Set the reference for the OCP
        try:
            solver.set_reference_trajectory(x_reference, u_reference)
        except acados_wrapper.AcadosWrapperException as exc:
            raise acados_wrapper.AcadosWrapperException(
                f"Failed to set reference on trajectory index {current_idx}"
            ) from exc

        # Optimize control input to reach pre-set target
        u_optimized, _ = solver.optimize(quad_current_state)
        tout.append(time.time())
        # MPC applies only first optimized input to the plant
        u_setpoint = u_optimized[0, :]
        yout["inputs"].append(u_setpoint)

        simulation_time = 0.0

        # ##### Simulation runtime (inner loop) ##### #
        control_period = trajectory.time_interval[min(current_idx, len(trajectory) - 2)]
        while simulation_time < control_period:
            simulation_time += simulation_dt
            total_sim_time += simulation_dt
            model.model_update(u_setpoint, simulation_dt)

    tout = np.asarray(tout)
    yout = {k: np.asarray(v) for k, v in yout.items()}

    # Average elapsed time per optimization
    mean_opt_time = mean_opt_time / current_idx * 1000
    return tout, yout
