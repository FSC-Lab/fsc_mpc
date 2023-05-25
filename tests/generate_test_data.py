# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import sys
from pathlib import Path

import fscore.simulation as sim
import numpy as np

sys.path.append("../src")

from quadrotor_mpc import acados_wrapper, trajectory_generator

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


def main():
    # Recover some necessary variables from the MPC object

    params = {
        "body_frame_coordinates": True,
        "quadrotor_mass": QUADROTOR_MASS,
        "t_horizon": T_HORIZON,
        "n_mpc_modes": N_MPC_NODES,
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
        trajectory_kw={
            k: params[k] for k in ["quadrotor_mass", "body_frame_coordinates"]
        },
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

    codegen_dir = Path(__file__).parent / "../lib/quadrotor_mpcpp"
    # Initialize quad MPC
    solver = acados_wrapper.AcadosWrapper.restore_from_file(str(codegen_dir))

    solver.set_constant_parameter([1.0])

    # Simulation integration step (the smaller the more "continuous"-like simulation.
    tout, yout = acados_wrapper.utils.run_simulation(
        model,
        solver,
        trajectory,
        REFERENCE_OVER_SAMPLING,
    )

    save_data["sim_out"] = {"time": array2json(tout)}
    save_data["sim_out"].update({k: array2json(v) for k, v in yout.items()})

    with open(Path(__file__).parent / "test_data.json", "w", encoding="utf-8") as fp:
        json.dump(save_data, fp)


if __name__ == "__main__":
    main()
