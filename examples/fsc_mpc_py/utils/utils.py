"""
Copyright © 2023 FSC Lab

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from fsc_mpc_py.trajectory_generator import MultirotorTrajectory
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from .. import mpc_interface


def run_simulation(
    sim,
    solver,
    trajectory,
    reference_over_sampling,
    profile_data=False,
    show_progress=True,
):
    # Set quad initial state equal to the initial reference trajectory state

    u_setpoint = trajectory.inputs[:, 0]

    # Sliding reference trajectory initial index
    current_idx = 0

    # Measure total simulation time
    total_sim_time = 0.0

    print("\nRunning simulation...")
    tout = []
    yout = {"states": [], "inputs": []}
    solve_time = [0.0]
    for current_idx in (
        tqdm.trange(len(trajectory)) if show_progress else range(len(trajectory))
    ):
        yout["states"].append(np.array(sim.state))

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
        except mpc_interface.MPCInterfaceException as exc:
            raise mpc_interface.MPCInterfaceException(
                f"Failed to set reference on trajectory index {current_idx}"
            ) from exc

        # Optimize control input to reach pre-set target
        t1 = time.time()
        u_optimized, _ = solver.optimize(sim.state)
        if profile_data:
            solve_time.append(time.time() - t1)
        tout.append(sim.time)
        # MPC applies only first optimized input to the plant
        u_setpoint = u_optimized[:, 0]
        yout["inputs"].append(u_setpoint)

        simulation_time = 0.0

        # ##### Simulation runtime (inner loop) ##### #
        control_period = trajectory.time_interval[min(current_idx, len(trajectory) - 2)]
        while simulation_time < control_period:
            simulation_time += sim.dt
            total_sim_time += sim.dt
            sim.input = u_setpoint
            sim.simulation_update()

    tout = np.asarray(tout)
    yout = {k: np.array(v) for k, v in yout.items()}
    yout = MultirotorTrajectory(yout["states"], yout["inputs"], tout)

    return yout, np.asarray(solve_time, dtype=np.float64) if profile_data else yout


def visualize_tracking_results(
    tout,
    yout,
    trajectory,
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

    expect_position = trajectory.position
    result_position = yout.position

    expect_attitude = trajectory.attitude
    result_attitude = yout.attitude

    expect_velocity = trajectory.velocity
    result_velocity = yout.velocity

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
    # ax.set_zlim(mean_z - 1, mean_z + 1)

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
