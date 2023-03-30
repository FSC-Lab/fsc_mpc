# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from quadrotor_mpc import acados_wrapper, quadrotor_model, trajectory_generator


def parse_cli():
    parser = ArgumentParser()

    parser.add_argument(
        "--codegen_dir",
        type=str,
        default="../lib/example",
        help="Output directory for codegen",
    )

    parser.add_argument(
        "--trajectory",
        type=str,
        default="loop",
        choices=["loop", "lemniscate", "straight"],
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

    t_horizon = 1.0

    # Number of MPC optimization nodes
    n_mpc_nodes = 10

    # Quadrotor simulator
    model = quadrotor_model.QuadrotorModel(mass=1.0)

    # Recover some necessary variables from the MPC object
    reference_over_sampling = 5
    control_period = t_horizon / (n_mpc_nodes * reference_over_sampling)

    params = {"body_frame_coordinates": True, "quadrotor_mass": model.mass}
    if args.trajectory == "loop":
        trajectory = trajectory_generator.loop_trajectory(
            control_period,
            radius=args.trajectory_radius,
            z=1,
            lin_acc=args.acceleration,
            clockwise=True,
            yawing=False,
            v_max=args.max_speed,
            trajectory_kw=params,
        )

    elif args.trajectory == "lemniscate":
        trajectory = trajectory_generator.lemniscate_trajectory(
            control_period,
            radius=args.trajectory_radius,
            z=1,
            lin_acc=args.acceleration,
            v_max=args.max_speed,
            trajectory_kw=params,
        )
    elif args.trajectory == "straight":
        trajectory = trajectory_generator.straight_trajectory(
            np.r_[0.0, 0.0],
            np.r_[150.0, 0.0],
            1.0,
            control_period,
            lin_acc=args.acceleration,
            v_max=args.max_speed,
            trajectory_kw=params,
        )

    else:
        raise ValueError(
            f"Unknown trajectory {args.trajectory}. Options are `lemniscate` and `loop`"
        )

    codegen_dir = Path(args.codegen_dir)
    # Initialize quad MPC
    if codegen_dir.exists():
        solver = acados_wrapper.AcadosWrapper.restore_from_file(str(codegen_dir))
    else:
        solver = acados_wrapper.AcadosWrapper.make_new(
            1.0,
            10,
            acados_wrapper.make_quadrotor_model("quadrotor"),
            codegen_dst=str(codegen_dir),
        )

    solver.set_constant_parameter([1.0])

    # Simulation integration step (the smaller the more "continuous"-like simulation.
    tout, yout = acados_wrapper.utils.run_simulation(
        model, solver, trajectory, reference_over_sampling, profile_data=True
    )
    tracking_rmse = np.mean(
        np.sqrt(np.sum((trajectory.states[:, :3] - yout["states"][:, :3]) ** 2, axis=1))
    )

    _ = acados_wrapper.utils.visualize_tracking_results(
        tout, yout, trajectory, autoshow=True
    )

    v_max_abs = np.max(np.sqrt(np.sum(trajectory.states[:, 7:10] ** 2, 1)))

    print(
        "\nReference: Executed trajectory",
        f"`{args.trajectory}`",
        f"with a peak axial velocity of {args.max_speed}"
        f"m/s, and a maximum speed of {v_max_abs:2.3f} m/s",
    )

    print(f"\n{'SIMULATION RESULTS'::^81s}\n")
    mean_optimization_time = yout["solve_time"].mean()
    print(f"Mean optimization time: {mean_optimization_time:.3f} ms")
    print(f"Tracking RMSE: {tracking_rmse:.4f} m\n")


if __name__ == "__main__":
    main()
