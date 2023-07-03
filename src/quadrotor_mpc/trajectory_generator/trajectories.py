# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Part of this work is derived from "data_driven_mpc"
# https://github.com/uzh-rpg/data_driven_mpc
# Licensed under the following terms

#  Trajectory generation functions. For the circle, lemniscate and random trajectories.

# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import dataclasses
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.typing import ArrayLike, NDArray


@dataclasses.dataclass
class Trajectory:
    states: NDArray[np.float64]
    inputs: NDArray[np.float64]
    time: NDArray[np.float64]

    def __len__(self):
        return self.time.size

    def __iadd__(self, other: Trajectory):
        self.states = np.concatenate((self.states, other.states))
        self.inputs = np.concatenate((self.inputs, other.inputs))
        self.time = np.concatenate((self.time, self.time[-1] + other.time))
        return self

    def __add__(self, other: Trajectory) -> Trajectory:
        res = dataclasses.replace(self)
        res += other
        return res

    @property
    def time_interval(self):
        return np.diff(self.time)

    def get_reference_chunk(
        self, idx: int, n_nodes: int, reference_over_sampling: int = 1
    ):
        # Dense references
        ref_traj_chunk = self.states[
            :, idx : idx + (n_nodes + 1) * reference_over_sampling
        ]
        ref_u_chunk = self.inputs[:, idx : idx + n_nodes * reference_over_sampling]

        # Indices for down-sampling the reference to number of MPC nodes
        downsample_ref_ind = np.arange(
            0,
            min(reference_over_sampling * (n_nodes + 1), ref_traj_chunk.shape[1]),
            reference_over_sampling,
            dtype=int,
        )

        # Sparser references (same dt as node separation)
        ref_traj_chunk = ref_traj_chunk[:, downsample_ref_ind]
        ref_u_chunk = ref_u_chunk[
            :, downsample_ref_ind[: max(len(downsample_ref_ind) - 1, 1)]
        ]

        return ref_traj_chunk, ref_u_chunk


def undo_quaternion_flip(q_past, q_current):
    if np.linalg.norm(q_past - q_current) > np.linalg.norm(q_past + q_current):
        return -q_current
    return q_current


def minimum_snap_trajectory_generator(
    t_ref: ArrayLike,
    traj_derivatives: ArrayLike,
    yaw_derivatives: Union[ArrayLike, None] = None,
    vehicle_mass: float = 1.0,
):
    """
    Follows the Minimum Snap Trajectory paper to generate a full trajectory given the position reference and its
    derivatives, and the yaw trajectory and its derivatives.

    :param traj_derivatives: np.array of shape 4x3xN. N corresponds to the length in
    samples of the trajectory, and:
        - The 4 components of the first dimension correspond to position, velocity,
        acceleration and jerk.
        - The 3 components of the second dimension correspond to x, y, z.
    :param yaw_derivatives: np.array of shape 2xN. N corresponds to the length in
    samples of the trajectory. The first
    row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
    :param t_ref: vector of length N, containing the reference times (starting from 0)
    for the trajectory.
    :param quad: vehicle3D object, corresponding to the vehicle model that will
    track the generated reference.
    :type quad: vehicle3D
    :param map_limits: dictionary of map limits if available, None otherwise.
    :param plot: True if show a plot of the generated trajectory.
    :return: tuple of 3 arrays:
        - Nx13 array of generated reference trajectory. The 13 dimension contains the
        components: position_xyz,
        attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
        - N array of reference timestamps. The same as in the input
        - Nx4 array of reference controls, corresponding to the four motors of the
        vehicle.
    """

    unit_z = np.array([[0], [0], [1]], dtype=np.float64)
    gravity = 9.81

    traj_derivatives = np.asarray(traj_derivatives, dtype=np.float64)

    n_refs, n_x, len_traj = traj_derivatives.shape
    if n_refs not in (3, 4):
        raise ValueError(
            "Expected 3 or 4 (position, velocity, acceleration[, jerk]) references"
        )

    if n_x != 3:
        raise ValueError("Trajectory must be 3-dimensional")

    t_ref = np.asarray(t_ref, dtype=np.float64)
    if t_ref.size != len_traj:
        raise ValueError("Mismatch between trajectory length and time references")

    full_pos, full_vel, full_acc, *full_jerk = map(
        np.squeeze, np.split(traj_derivatives, n_refs, axis=0)
    )

    discretization_dt = np.gradient(t_ref)

    # Add gravity to accelerations
    full_acc += unit_z * gravity
    # Compute body axes
    z_b = full_acc / np.linalg.norm(full_acc, axis=0, keepdims=True)

    rate = np.zeros((3, len_traj))
    f_t = vehicle_mass * np.sum(z_b * full_acc, axis=0)

    if yaw_derivatives is not None and n_refs == 4:
        (full_jerk,) = full_jerk
        yaw_derivatives = np.asarray(yaw_derivatives, dtype=np.float64)
        # yaw is defined as the projection of the body-x axis on the horizontal plane
        x_c = np.row_stack(
            (
                np.cos(yaw_derivatives[0, :]),
                np.sin(yaw_derivatives[0, :]),
                np.zeros(len_traj),
            ),
        )
        y_b = np.cross(z_b, x_c, axis=0)
        y_b = y_b / np.linalg.norm(y_b, axis=0, keepdims=True)
        x_b = np.cross(y_b, z_b, axis=0)

        # Rotation matrix (from body to world)
        b_r_w = np.moveaxis(np.dstack((x_b, y_b, z_b)), 1, -1)
        q = []
        for i in range(len_traj):
            # Transform to quaternion
            q.append(R.from_matrix(b_r_w[:, :, i]).as_quat())
            if i > 1:
                q[-1] = undo_quaternion_flip(q[-2], q[-1])
        q = np.column_stack(q)

        # Compute angular rate vector
        # Total thrust acceleration must be equal to the projection of the vehicle
        # acceleration into the Z body axis
        a_proj = np.sum(z_b * full_jerk, axis=0)

        h_omega = vehicle_mass / f_t * (full_jerk - a_proj * z_b)
        rate[0, :] = np.sum(-h_omega * y_b, axis=0)
        rate[1, :] = np.sum(h_omega * x_b, axis=0)
        rate[2, :] = -yaw_derivatives[1, :] * np.sum(unit_z * z_b, axis=0)

    else:
        # new way to compute attitude:
        # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
        q_w = 1.0 + np.sum(unit_z * z_b, axis=0)
        q_xyz = np.cross(unit_z, z_b, axis=0)
        q = 0.5 * np.row_stack((q_xyz, q_w))
        q = q / np.linalg.norm(q, axis=0, keepdims=True)

        # Use numerical differentiation of quaternions
        q_dot = np.gradient(q, axis=1) / discretization_dt
        w_int = np.zeros((3, len_traj))
        for i in range(len_traj):
            w_int[:, i] = (
                2.0
                * (R.from_quat(q[:, i]).inv() * R.from_quat(q_dot[:, i])).as_quat()[0:3]
            )
        rate[:] = w_int

        q_new = np.array(q)
        yaw_corr_acc = 0.0
        for i in range(1, len_traj):
            yaw_corr = -rate[2, i] * discretization_dt[i]
            yaw_corr_acc += yaw_corr
            q_corr = R.from_quat(
                [0.0, 0.0, np.sin(yaw_corr_acc / 2.0), np.cos(yaw_corr_acc / 2.0)]
            )
            q_new[:, i] = (R.from_quat(q[:, i]) * q_corr).as_quat()
            w_int[:, i] = (
                2.0
                * (R.from_quat(q[:, i]).inv() * R.from_quat(q_dot[:, i])).as_quat()[0:3]
            )

        q_new_dot = np.gradient(q_new, axis=1) / discretization_dt
        for i in range(1, len_traj):
            w_int[:, i] = (
                2.0
                * (
                    R.from_quat(q_new[:, i]).inv() * R.from_quat(q_new_dot[:, i])
                ).as_quat()[0:3]
            )

        q[:] = q_new
        rate[:] = w_int

    # Compute inputs
    u_ref = np.row_stack((f_t, rate))

    traj_ref = np.row_stack((full_pos, q, full_vel))

    return Trajectory(traj_ref, u_ref, t_ref)


def straight_trajectory(
    begin, end, discretization_dt, lin_acc, v_max, vehicle_mass=1.0
):
    position_diff = end - begin
    distance = np.linalg.norm(position_diff)
    heading_vector = position_diff[..., None] / distance

    ramp_up_t = v_max / lin_acc

    # Calculate simulation time to achieve desired maximum velocity with specified
    # acceleration
    t_cruise = distance / v_max - ramp_up_t

    refs = {}
    a = {}
    refs["ramp_up"] = np.r_[0:ramp_up_t:discretization_dt]
    a["ramp_up"] = lin_acc * np.ones_like(refs["ramp_up"])
    if t_cruise < 0.0:
        raise RuntimeError("Not enough time to accelerate to v_max and cruise!")
    refs["cruise"] = (
        refs["ramp_up"][-1] + np.r_[0:t_cruise:discretization_dt] + discretization_dt
    )
    a["cruise"] = np.zeros_like(refs["cruise"])
    refs["ramp_down"] = (
        refs["cruise"][-1] + np.r_[0:ramp_up_t:discretization_dt] + discretization_dt
    )
    a["ramp_down"] = -lin_acc * np.ones_like(refs["ramp_down"])
    t_ref = np.concatenate(tuple(refs.values()))
    a_vec = np.concatenate(tuple(a.values()))
    v_vec = np.cumsum(a_vec) * discretization_dt
    d_vec = np.cumsum(v_vec) * discretization_dt

    n = t_ref.size
    traj = np.zeros((4, 3, n))
    traj[0, 0:3, :] = begin[..., None] + heading_vector * d_vec
    traj[1, 0:3, :] = heading_vector * v_vec
    traj[2, 0:3, :] = heading_vector * a_vec
    yaw = np.zeros((2, n))
    yaw[0, :] = np.arctan2(position_diff[1], position_diff[0])

    return minimum_snap_trajectory_generator(t_ref, traj, yaw, vehicle_mass)


def loop_trajectory(
    discretization_dt,
    radius,
    z,
    v_max,
    lin_acc,
    clockwise=False,
    yawing=False,
    vehicle_mass=1.0,
):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.

    :param params: vehicle model
    :param discretization_dt: Sampling period of the trajectory.
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the vehicle yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
    """

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified
    # acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    refs = {}
    alphas = {}
    refs["ramp"] = np.arange(0, ramp_up_t, discretization_dt)
    alphas["ramp_up"] = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * refs["ramp"]) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    refs["coasting"] = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    alphas["coasting"] = np.ones_like(refs["coasting"]) * alpha_acc
    # Transition phase: decelerate
    refs["transition"] = np.arange(0, 2 * ramp_up_t, discretization_dt)
    alphas["transition"] = alpha_acc * np.cos(
        np.pi / (2 * ramp_up_t) * refs["transition"]
    )
    refs["transition"] += refs["coasting"][-1] + discretization_dt
    # Deceleration phase
    refs["downcoasting"] = (
        refs["transition"][-1]
        + np.arange(0, coasting_duration, discretization_dt)
        + discretization_dt
    )
    alphas["down_coasting"] = -np.ones_like(refs["downcoasting"]) * alpha_acc
    # Bring to rest phase
    refs["ramp_up"] = (
        refs["downcoasting"][-1]
        + np.arange(0, ramp_up_t, discretization_dt)
        + discretization_dt
    )
    alphas["ramp_up_end"] = alphas["ramp_up"] - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate(
        (
            refs["ramp"],
            refs["coasting"],
            refs["transition"],
            refs["downcoasting"],
            refs["ramp_up"],
        )
    )
    alpha_vec = np.concatenate(
        (
            alphas["ramp_up"],
            alphas["coasting"],
            alphas["transition"],
            alphas["down_coasting"],
            alphas["ramp_up_end"],
        )
    )

    # Calculate derivative of angular acceleration (alpha_vec)
    ramp_up_alpha_dt = (
        alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / ramp_up_t * refs["ramp"])
    )
    coasting_alpha_dt = np.zeros_like(alphas["coasting"])
    transition_alpha_dt = (
        -alpha_acc
        * np.pi
        / (2 * ramp_up_t)
        * np.sin(np.pi / (2 * ramp_up_t) * refs["transition"])
    )
    alpha_dt = np.concatenate(
        (
            ramp_up_alpha_dt,
            coasting_alpha_dt,
            transition_alpha_dt,
            coasting_alpha_dt,
            ramp_up_alpha_dt,
        )
    )

    if not clockwise:
        alpha_vec *= -1
        alpha_dt *= -1

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.sin(angle_vec)
    pos_traj_y = radius * np.cos(angle_vec)
    pos_traj_z = np.ones_like(pos_traj_x) * z

    vel_traj_x = radius * w_vec * np.cos(angle_vec)
    vel_traj_y = -(radius * w_vec * np.sin(angle_vec))

    acc_traj_x = radius * (
        alpha_vec * np.cos(angle_vec) - w_vec**2 * np.sin(angle_vec)
    )
    acc_traj_y = -radius * (
        alpha_vec * np.sin(angle_vec) + w_vec**2 * np.cos(angle_vec)
    )

    jerk_traj_x = radius * (
        alpha_dt * np.cos(angle_vec)
        - alpha_vec * np.sin(angle_vec) * w_vec
        - np.cos(angle_vec) * w_vec**3
        - 2 * np.sin(angle_vec) * w_vec * alpha_vec
    )
    jerk_traj_y = -radius * (
        np.cos(angle_vec) * w_vec * alpha_vec
        + np.sin(angle_vec) * alpha_dt
        - np.sin(angle_vec) * w_vec**3
        + 2 * np.cos(angle_vec) * w_vec * alpha_vec
    )

    if yawing:
        yaw_traj = -angle_vec
    else:
        yaw_traj = np.zeros_like(angle_vec)

    traj = np.concatenate(
        (
            np.row_stack((pos_traj_x, pos_traj_y, pos_traj_z))[None, ...],
            np.row_stack((vel_traj_x, vel_traj_y, np.zeros_like(vel_traj_x)))[
                None, ...
            ],
            np.row_stack((acc_traj_x, acc_traj_y, np.zeros_like(acc_traj_x)))[
                None, ...
            ],
            np.row_stack((jerk_traj_x, jerk_traj_y, np.zeros_like(jerk_traj_x)))[
                None, ...
            ],
        ),
        0,
    )

    yaw = np.row_stack((yaw_traj, w_vec))

    return minimum_snap_trajectory_generator(t_ref, traj, yaw, vehicle_mass)


def lemniscate_trajectory(
    discretization_dt,
    radius,
    z,
    lin_acc,
    v_max,
    vehicle_mass=1.0,
):
    """

    :param params:
    :param discretization_dt:
    :param radius:
    :param z:
    :param lin_acc:
    :param clockwise:
    :param yawing:
    :param v_max:
    :param map_name:
    :param plot:
    :return:
    """

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified
    # acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    refs = {}
    alphas = {}
    refs["ramp"] = np.arange(0, ramp_up_t, discretization_dt)
    alphas["ramp_up"] = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * refs["ramp"]) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    refs["coasting"] = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    alphas["coasting"] = np.ones_like(refs["coasting"]) * alpha_acc
    # Transition phase: decelerate
    refs["transition"] = np.arange(0, 2 * ramp_up_t, discretization_dt)
    alphas["transition"] = alpha_acc * np.cos(
        np.pi / (2 * ramp_up_t) * refs["transition"]
    )
    refs["transition"] += refs["coasting"][-1] + discretization_dt
    # Deceleration phase
    refs["downcoasting"] = (
        refs["transition"][-1]
        + np.arange(0, coasting_duration, discretization_dt)
        + discretization_dt
    )
    alphas["down_coasting"] = -np.ones_like(refs["downcoasting"]) * alpha_acc
    # Bring to rest phase
    refs["ramp_up"] = (
        refs["downcoasting"][-1]
        + np.arange(0, ramp_up_t, discretization_dt)
        + discretization_dt
    )
    alphas["ramp_up_end"] = alphas["ramp_up"] - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate(tuple(refs.values()))
    alpha_vec = np.concatenate(tuple(alphas.values()))

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Adaption: we achieve the highest spikes in the bodyrates when passing through the 'center' part of the figure-8
    # This leads to negative reference thrusts.
    # Let's see if we can alleviate this by adapting the z-reference in these parts to add some acceleration in the
    # z-component
    z_dim = 0.0

    # Compute position, velocity, acceleration, jerk
    traj = np.empty((3, 3, np.size(angle_vec)))
    traj[0, 0, :] = radius * np.cos(angle_vec)
    traj[0, 1, :] = radius * (np.sin(angle_vec) * np.cos(angle_vec))
    traj[0, 2, :] = -z_dim * np.cos(4.0 * angle_vec) + z

    traj[1, 0, :] = -radius * (w_vec * np.sin(angle_vec))
    traj[1, 1, :] = radius * (
        w_vec * np.cos(angle_vec) ** 2 - w_vec * np.sin(angle_vec) ** 2
    )
    traj[1, 2, :] = 4.0 * z_dim * w_vec * np.sin(4.0 * angle_vec)

    traj[2, 0, :] = -radius * (
        alpha_vec * np.sin(angle_vec) + w_vec**2 * np.cos(angle_vec)
    )
    traj[2, 1, :] = radius * (
        alpha_vec * np.cos(angle_vec) ** 2
        - 2.0 * w_vec**2 * np.cos(angle_vec) * np.sin(angle_vec)
        - alpha_vec * np.sin(angle_vec) ** 2
        - 2.0 * w_vec**2 * np.sin(angle_vec) * np.cos(angle_vec)
    )
    traj[2, 2, :] = (
        16.0
        * z_dim
        * (w_vec**2 * np.cos(4.0 * angle_vec) + alpha_vec * np.sin(4.0 * angle_vec))
    )

    return minimum_snap_trajectory_generator(t_ref, traj, vehicle_mass)
