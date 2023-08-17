"""
Script to generate a json dataset for testing
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

import json
from argparse import ArgumentParser

import fsc_mpc_py.simulation.simple_quadrotor_simulator as sim
import numpy as np
from fsc_mpc_py import mpc_interface, trajectory_generator, utils

REFERENCE_OVER_SAMPLING = 1
T_HORIZON = 1.0
N_MPC_NODES = 10
QUADROTOR_MASS = 1.0
CONTROL_PERIOD = T_HORIZON / (N_MPC_NODES * REFERENCE_OVER_SAMPLING)
SIM_PERIOD = 5e-4
TRAJECTORY_RADIUS = 5
ACCELERATION = 1.0
MAX_SPEED = 8.0


def array2json(arr):
    if np.ndim(arr) == 1:
        arr = arr[:, None]
    return {"size": arr.shape, "values": np.ravel(arr, order="F").tolist()}


def parse_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "test_data_file", type=str, help="Name of data file to generate"
    )
    parser.add_argument(
        "codegen_dir",
        type=str,
        help="Directory of generated C++ code and acados_ocp_nlp.json",
    )
    return parser.parse_args()


def main():
    args = parse_cli()

    params = {
        "quadrotor_mass": QUADROTOR_MASS,
        "control_period": CONTROL_PERIOD,
        "sim_period": SIM_PERIOD,
    }

    save_data = {"params": params}

    trajectory = trajectory_generator.lemniscate_trajectory(
        CONTROL_PERIOD,
        radius=TRAJECTORY_RADIUS,
        z=1,
        lin_acc=ACCELERATION,
        v_max=MAX_SPEED,
        vehicle_mass=params["quadrotor_mass"],
    )

    save_data["trajectory"] = {
        "states": array2json(trajectory.states),
        "inputs": array2json(trajectory.inputs),
        "time": array2json(trajectory.time),
    }

    # Quadrotor simulator
    model = sim.SimpleQuadrotorSimulator(
        mass=QUADROTOR_MASS,
        base_dt=SIM_PERIOD,
        init_time=0.0,
        init_state=np.array(trajectory.states[:, 0]),
        init_input=np.array(trajectory.inputs[:, 0]),
        grav_accel=-9.81,
        quaternion_normalization_gain=1.0,
    )

    # Initialize quad MPC
    solver = mpc_interface.MPCInterface(codegen_dst=args.codegen_dir)

    solver.set_constant_parameter([params["quadrotor_mass"]])

    # Simulation integration step (the smaller the more "continuous"-like simulation.
    traj, _ = utils.run_simulation(
        model, solver, trajectory, REFERENCE_OVER_SAMPLING, show_progress=False
    )

    traj = vars(traj)

    save_data["sim_out"] = {"time": array2json(traj["time"])}
    save_data["sim_out"].update({k: array2json(v) for k, v in traj.items()})

    with open(args.test_data_file, "w", encoding="utf-8") as fp:
        json.dump(save_data, fp)


if __name__ == "__main__":
    main()
