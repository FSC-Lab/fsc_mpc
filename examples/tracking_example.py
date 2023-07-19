# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from argparse import ArgumentParser
from pathlib import Path

import fscore.simulation as sim
import numpy as np
import trajectory_generator
import utils

import acados_wrapper


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


QUADROTOR_MASS = 1.0


def main():
    args = parse_cli()

    t_horizon = 1.0

    # Number of MPC optimization nodes
    n_mpc_nodes = 10

    # Recover some necessary variables from the MPC object
    reference_over_sampling = 5
    control_period = t_horizon / (n_mpc_nodes * reference_over_sampling)

    if args.trajectory == "loop":
        trajectory = trajectory_generator.loop_trajectory(
            control_period,
            radius=args.trajectory_radius,
            z=1,
            lin_acc=args.acceleration,
            clockwise=True,
            yawing=True,
            v_max=args.max_speed,
            vehicle_mass=QUADROTOR_MASS,
        )

    elif args.trajectory == "lemniscate":
        trajectory = trajectory_generator.lemniscate_trajectory(
            control_period,
            radius=args.trajectory_radius,
            z=1,
            lin_acc=args.acceleration,
            v_max=args.max_speed,
            vehicle_mass=QUADROTOR_MASS,
        )
    elif args.trajectory == "straight":
        trajectory = trajectory_generator.straight_trajectory(
            np.r_[0.0, 0.0, 1.0],
            np.r_[150.0, 0.0, 1.0],
            control_period,
            lin_acc=args.acceleration,
            v_max=args.max_speed,
            vehicle_mass=QUADROTOR_MASS,
        )

    else:
        raise ValueError(
            f"Unknown trajectory {args.trajectory}. Options are `lemniscate` and `loop`"
        )

    # Quadrotor simulator
    simulation_dt = 5e-4
    model = sim.SimpleQuadrotorSimulator(
        mass=QUADROTOR_MASS,
        base_dt=simulation_dt,
        init_time=0.0,
        init_state=np.array(trajectory.states[:, 0]),
        init_input=np.zeros((4,)),
        grav_accel=-9.81,
        quaternion_normalization_gain=2.0,
    )

    codegen_dir = Path(args.codegen_dir)
    # Initialize quad MPC
    params = {
        "t_horizon": t_horizon,
        "n_nodes": n_mpc_nodes,
        "q_cost": [10, 10, 10, 0.1, 0.1, 0.1, 0.0, 0.05, 0.05, 0.05],
        "r_cost": np.asarray([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
        "lbu": np.array([0.0, -8.0, -8.0, -8.0], dtype=np.float64),
        "ubu": np.array([80.0, 8.0, 8.0, 8.0], dtype=np.float64),
    }
    solver = acados_wrapper.AcadosWrapper(
        acados_wrapper.make_quadrotor_model("quadrotor"),
        params,
        codegen_dst=str(codegen_dir),
        clean_first=False,
    )

    solver.set_constant_parameter([QUADROTOR_MASS])

    # Simulation integration step (the smaller the more "continuous"-like simulation.
    tout, yout = utils.run_simulation(
        model, solver, trajectory, reference_over_sampling, profile_data=True
    )
    tracking_rmse = np.mean(
        np.sqrt(
            np.sum((trajectory.states[0:3, :] - yout["states"][0:3, :]) ** 2, axis=1)
        )
    )

    _ = utils.visualize_tracking_results(tout, yout, trajectory, autoshow=True)

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
