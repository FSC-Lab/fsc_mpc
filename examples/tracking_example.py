# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
from tqdm import tqdm
from visualization import trajectory_tracking_results

from quadrotor_mpc import acados_wrapper, quadrotor_model, trajectory_generator


def parse_cli():
    parser = ArgumentParser()

    parser.add_argument(
        "--codegen_dir",
        type=str,
        default="../lib",
        help="File containing the Acados configuration",
    )

    parser.add_argument(
        "--trajectory",
        type=str,
        default="loop",
        choices=["loop", "lemniscate"],
        help="path to other necessary data files (eg. vocabularies)",
    )

    parser.add_argument(
        "--max_speed",
        type=float,
        default=8,
        help=(
            "Maximum axial speed executed during the flight in m/s. For the `loop`"
            " trajectory, velocities are feasible up to 14 m/s, and for the"
            " `lemniscate` up to 8 m/s"
        ),
    )

    parser.add_argument(
        "--acceleration",
        type=float,
        default=1,
        help=(
            "Acceleration of the reference trajectory. Higher accelerations will"
            " shorten the executiontime of the tracking"
        ),
    )

    parser.add_argument(
        "--trajectory_radius",
        type=float,
        default=5,
        help="Radius of the reference trajectories",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_cli()

    # Load the disturbances for the custom offline simulator.

    t_horizon = 1.0
    # Simulation integration step (the smaller the more "continuous"-like simulation.
    simulation_dt = 5e-4

    # Number of MPC optimization nodes
    n_mpc_nodes = 10

    # Quadrotor simulator
    my_quad = quadrotor_model.QuadrotorModel(mass=1.0)

    # Recover some necessary variables from the MPC object
    reference_over_sampling = 5
    control_period = t_horizon / (n_mpc_nodes * reference_over_sampling)

    if args.trajectory == "loop":
        traj_ref, u_ref, t_ref = trajectory_generator.loop_trajectory(
            my_quad,
            control_period,
            radius=args.trajectory_radius,
            z=1,
            lin_acc=args.acceleration,
            clockwise=True,
            yawing=False,
            v_max=args.max_speed,
        )

    elif args.trajectory == "lemniscate":
        traj_ref, u_ref, t_ref = trajectory_generator.lemniscate_trajectory(
            my_quad,
            control_period,
            radius=args.trajectory_radius,
            z=1,
            lin_acc=args.acceleration,
            v_max=args.max_speed,
        )

    else:
        raise ValueError(
            f"Unknown trajectory {args.trajectory}. Options are `lemniscate` and `loop`"
        )

    codegen_dir = Path(args.codegen_dir)
    # Initialize quad MPC
    if codegen_dir.exists():
        quadrotor_mpc = acados_wrapper.AcadosWrapper.restore_from_file(str(codegen_dir))
    else:
        model = acados_wrapper.make_quadrotor_model("quadrotor")
        quadrotor_mpc = acados_wrapper.AcadosWrapper.make_new(
            1.0, 10, model, codegen_dst=str(codegen_dir)
        )

    # Set quad initial state equal to the initial reference trajectory state
    quad_current_state = traj_ref[0, :]
    my_quad.state = quad_current_state

    u_setpoint = u_ref[0, :]
    quad_trajectory = np.zeros((len(t_ref), len(quad_current_state)))
    u_optimized_seq = np.zeros((len(t_ref), 4))

    # Sliding reference trajectory initial index
    current_idx = 0

    # Measure the MPC optimization time
    mean_opt_time = 0.0

    # Measure total simulation time
    total_sim_time = 0.0

    print("\nRunning simulation...")
    for current_idx in tqdm(range(traj_ref.shape[0])):
        quad_current_state = my_quad.state

        quad_trajectory[current_idx, :] = quad_current_state

        # ##### Optimization runtime (outer loop) ##### #
        # Get the chunk of trajectory required for the current optimization
        traj_ref_block, u_ref_block = acados_wrapper.get_reference_chunk(
            traj_ref,
            u_ref,
            current_idx,
            n_mpc_nodes,
            reference_over_sampling,
        )

        # Set the reference for the OCP
        try:
            quadrotor_mpc.set_reference_trajectory(
                x_reference=traj_ref_block,
                u_reference=u_ref_block,
            )
        except acados_wrapper.AcadosWrapperException as exc:
            raise acados_wrapper.AcadosWrapperException(
                f"Failed to set reference on trajectory index {current_idx}"
            ) from exc

        # Optimize control input to reach pre-set target
        t_opt_init = time()
        u_optimized, _ = quadrotor_mpc.optimize(quad_current_state)
        mean_opt_time += time() - t_opt_init

        # MPC applies only first optimized input to the plant
        u_setpoint = u_optimized[0, :]
        u_optimized_seq[current_idx, :] = u_setpoint

        simulation_time = 0.0

        # ##### Simulation runtime (inner loop) ##### #
        while simulation_time < control_period:
            simulation_time += simulation_dt
            total_sim_time += simulation_dt
            my_quad.model_update(u_setpoint, simulation_dt)

    u_optimized_seq[current_idx, :] = u_setpoint

    quad_current_state = my_quad.state
    quad_trajectory[-1, :] = quad_current_state
    u_optimized_seq[-1, :] = u_setpoint

    # Average elapsed time per optimization
    mean_opt_time = mean_opt_time / current_idx * 1000
    tracking_rmse = np.mean(
        np.sqrt(np.sum((traj_ref[:, :3] - quad_trajectory[:, :3]) ** 2, axis=1))
    )

    v_max = float(np.max(traj_ref[:, 7:10]))

    title = rf"$v_{{max}}$={v_max:.2f} m/s | RMSE: {tracking_rmse:.4f} m"
    trajectory_tracking_results(
        t_ref,
        traj_ref,
        quad_trajectory,
        u_ref,
        u_optimized_seq,
        title,
    )

    v_max_abs = np.max(np.sqrt(np.sum(traj_ref[:, 7:10] ** 2, 1)))

    print(
        "\nReference: Executed trajectory",
        f"`{args.trajectory}`",
        f"with a peak axial velocity of {args.max_speed}"
        f"m/s, and a maximum speed of {v_max_abs:2.3f} m/s",
    )

    print(f"\n{'SIMULATION RESULTS'::^81s}\n")
    print(f"Mean optimization time: {mean_opt_time:.3f} ms")
    print(f"Tracking RMSE: {tracking_rmse:.4f} m\n")


if __name__ == "__main__":
    main()
