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

from argparse import ArgumentParser
from pathlib import Path

import fsc_mpc_py.simulation.simple_quadrotor_simulator as sim
import numpy as np
from fsc_mpc_py import mpc_interface, trajectory_generator, utils


def parse_cli():
    parser = ArgumentParser()

    parser.add_argument(
        "codegen_dir",
        type=str,
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
        n_order = 5
        generator = trajectory_generator.MinimumSnap(
            n_order,
            [0, 0, 1, 1],
            algorithm=trajectory_generator.MinimumSnapAlgorithm.CLOSED_FORM,
        )
        traj = generator.generate([0, 0, 1], [150, 0, 1], [0, 10])
        tt = np.arange(generator.t_ref[0], generator.t_ref[-1], control_period)

        trajectory = traj.to_real_trajectory(QUADROTOR_MASS, tt)

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
    solver = mpc_interface.MPCInterface(
        codegen_dst=str(codegen_dir),
    )

    solver.set_constant_parameter([QUADROTOR_MASS])
    solver.set_costs(params["q_cost"], params["r_cost"])

    # Simulation integration step (the smaller the more "continuous"-like simulation.
    yout, solve_time = utils.run_simulation(
        model, solver, trajectory, reference_over_sampling, profile_data=True
    )
    tracking_rmse = np.mean(
        np.sqrt(np.sum((trajectory.position - yout.position) ** 2, axis=1))
    )

    _ = utils.visualize_tracking_results(yout.time, yout, trajectory, autoshow=True)

    v_max_abs = np.max(np.sqrt(np.sum(trajectory.velocity**2, 1)))

    print(
        "\nReference: Executed trajectory",
        f"`{args.trajectory}`",
        f"with a peak axial velocity of {args.max_speed}"
        f"m/s, and a maximum speed of {v_max_abs:2.3f} m/s",
    )

    print(f"\n{'SIMULATION RESULTS'::^81s}\n")
    mean_optimization_time = solve_time.mean()
    print(f"Mean optimization time: {mean_optimization_time:.3f} ms")
    print(f"Tracking RMSE: {tracking_rmse:.4f} m\n")


if __name__ == "__main__":
    main()
