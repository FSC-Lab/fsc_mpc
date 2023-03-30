# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm

from quadrotor_mpc import acados_wrapper, quadrotor_model, trajectory_generator


def run_simulation(
    model: quadrotor_model.QuadrotorModel,
    solver: acados_wrapper.AcadosWrapper,
    trajectory: trajectory_generator.trajectories.Trajectory,
    reference_over_sampling: int,
    profile_data=False,
):
    simulation_dt = 5e-4
    # Set quad initial state equal to the initial reference trajectory state
    quad_current_state = trajectory.states[0, :]
    model.state = quad_current_state

    u_setpoint = trajectory.inputs[0, :]

    # Sliding reference trajectory initial index
    current_idx = 0

    # Measure total simulation time
    total_sim_time = 0.0

    print("\nRunning simulation...")
    tout = []
    yout = {"states": [], "inputs": []}
    if profile_data:
        yout["solve_time"] = [0.0]
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
        t1 = time.time()
        u_optimized, _ = solver.optimize(quad_current_state)
        if profile_data:
            yout["solve_time"].append(time.time() - t1)
        tout.append(t1)
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
    yout = {k: np.asarray(v, dtype=np.float64) for k, v in yout.items()}

    return tout, yout


def visualize_tracking_results(
    tout,
    yout,
    trajectory: trajectory_generator.trajectories.Trajectory,
    handles=None,
    autoshow=False,
):
    create_new_handles = handles is None
    if create_new_handles:
        handles = []
        f1, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        handles.append({"fig": f1, "ax": ax})
    else:
        f1 = handles[0]["fig"]
        ax = handles[0]["ax"]

    expect_states = trajectory.states
    result_states = yout["states"]
    expect_position = expect_states[:, 0:3]
    result_position = result_states[:, 0:3]

    expect_attitude = expect_states[:, 3:7]
    result_attitude = result_states[:, 3:7]

    expect_velocity = expect_states[:, 7:10]
    result_velocity = result_states[:, 7:10]

    ax.plot(
        expect_position[:, 0],
        expect_position[:, 1],
        expect_position[:, 2],
        linewidth=2,
        label="Trajectory Reference",
    )
    ax.plot(
        result_position[:, 0],
        result_position[:, 1],
        result_position[:, 2],
        "--",
        linewidth=2,
        label="Simulated Trajectory",
    )
    ax.legend()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore
    mean_z = expect_position[:, 2].mean()
    ax.set_zlim(mean_z - 1, mean_z + 1)

    position_error = np.abs(expect_position - result_position)
    attitude_error = np.abs(
        (
            Rotation.from_quat(expect_attitude)
            * Rotation.from_quat(result_attitude).inv()
        ).as_rotvec()
    )

    velocity_error = np.abs(expect_velocity - result_velocity)
    if create_new_handles:
        f2, ax = plt.subplots(1, 3, figsize=(12, 4))
        handles.append({"fig": f2, "ax": ax})
    else:
        f2 = handles[1]["fig"]
        ax = handles[1]["ax"]

    for idx, it in enumerate("XYZ"):
        ax[0].plot(
            tout,
            position_error[:, idx],
            alpha=0.5,
            label=f"{it}-axis absolute position error",
        )
        ax[0].axhline(
            position_error[:, idx].mean(),
            color=f"C{idx}",
            label=f"{it}-axis position MAE",
        )
        ax[1].plot(
            tout,
            attitude_error[:, idx],
            alpha=0.5,
            label=f"{it}-axis absolute angular error",
        )
        ax[1].axhline(
            attitude_error[:, idx].mean(),
            color=f"C{idx}",
            label=f"{it}-axis attitude MAE",
        )
        ax[2].plot(
            tout,
            velocity_error[:, idx],
            alpha=0.5,
            label=f"{it}-axis absolute velocity error",
        )
        ax[2].axhline(
            velocity_error[:, idx].mean(),
            color=f"C{idx}",
            label=f"{it}-axis velocity MAE",
        )

    for it in ax:
        it.set_xlabel("Time (s)")
        it.legend()

    if autoshow:
        plt.show()
    return handles
