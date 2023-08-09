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


Part of this work is derived from "data_driven_mpc"
https://github.com/uzh-rpg/data_driven_mpc
Licensed under the following terms

 Trajectory generation functions. For the circle, lemniscate and random trajectories.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import copy

import numpy as np
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R


class Trajectory:
    def __init__(self, states, inputs, time):
        self.time = np.asarray(time)
        self.states = np.asarray(states)
        self.inputs = np.asarray(inputs)

    def __len__(self):
        return self.time.size

    def __iadd__(self, other):
        self.states = np.concatenate((self.states, other.states))
        self.inputs = np.concatenate((self.inputs, other.inputs))
        self.time = np.concatenate((self.time, self.time[-1] + other.time))
        return self

    def __add__(self, other):
        res = copy.copy(self)
        res += other
        return res

    @property
    def time_interval(self):
        return np.diff(self.time)

    def get_reference_chunk(self, idx, n_nodes, reference_over_sampling=1):
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


def forward(traj_refs, yaw_refs, vehicle_mass, grav=9.81, drag_params=None):
    grav_vector = np.array([[0.0], [0.0], [grav]])
    traj_refs = np.asarray(traj_refs, dtype=np.float64)
    traj_refs = np.atleast_3d(traj_refs)

    n_ders, n_dims, len_traj = traj_refs.shape
    if (n_ders, n_dims) != (3, 3):
        raise ValueError(
            "Trajectory references must be 3D kinematical derivatives stacked"
            "columnwise"
        )

    vel = np.atleast_2d(traj_refs[0, ...])
    acc = np.atleast_2d(traj_refs[1, ...])
    jer = np.atleast_2d(traj_refs[2, ...])

    psi = yaw_refs[0, ...]
    dpsi = yaw_refs[1, ...]

    attitude = np.empty((4, len_traj), dtype=np.float64)
    inputs = np.empty((4, len_traj), dtype=np.float64)
    if drag_params is not None:
        cp_term = np.sqrt(np.sum(vel * vel, axis=0) + drag_params["veps"])
        w_term = 1.0 + drag_params["cp"] * cp_term
        v_dot_a = np.sum(vel * acc, axis=0)
        dw_term = drag_params["cp"] * v_dot_a / cp_term

        dw = w_term * acc + dw_term * vel
        w = w_term * vel
        dh_over_m = drag_params["dh"] / vehicle_mass

        z = acc + dh_over_m * w + grav_vector
        z_nrm = np.linalg.norm(z, axis=0, keepdims=True)
        z /= z_nrm

        dz = -np.cross(z, np.cross(z, jer + dh_over_m * dw, axis=0), axis=0) / z_nrm
        inputs[0, :] = np.sum(
            z * (vehicle_mass * (acc + grav_vector) + drag_params["dv"] * w), axis=0
        )
    else:
        z = acc + grav_vector
        z_nrm = np.linalg.norm(z, axis=0, keepdims=True)
        z /= z_nrm

        dz = -np.cross(z, np.cross(z, jer, axis=0), axis=0) / z_nrm
        inputs[0, :] = np.sum(z * (vehicle_mass * (acc + grav_vector)), axis=0)

    tilt_den = np.sqrt(2.0 * (1.0 + z[2, :]))
    tilt0 = 0.5 * tilt_den
    tilt1 = -z[1, :] / tilt_den
    tilt2 = z[0, :] / tilt_den
    c_half_psi = np.cos(0.5 * psi)
    s_half_psi = np.sin(0.5 * psi)
    attitude[0, :] = tilt1 * c_half_psi + tilt2 * s_half_psi
    attitude[1, :] = tilt2 * c_half_psi - tilt1 * s_half_psi
    attitude[2, :] = tilt0 * s_half_psi
    attitude[3, :] = tilt0 * c_half_psi
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    omg_den = z[2, :] + 1.0
    omg_term = dz[2, :] / omg_den
    inputs[1, :] = (
        dz[0, :] * s_psi
        - dz[1, :] * c_psi
        - (z[0, :] * s_psi - z[1, :] * c_psi) * omg_term
    )
    inputs[2, :] = (
        dz[0, :] * c_psi
        + dz[1, :] * s_psi
        - (z[0, :] * c_psi + z[1, :] * s_psi) * omg_term
    )
    inputs[3, :] = (z[1, :] * dz[0, :] - z[0, :] * dz[1, :]) / omg_den + dpsi

    return np.squeeze(attitude), np.squeeze(inputs)


def minimum_snap_trajectory_generator(
    t_ref,
    traj_derivatives,
    yaw_derivatives=None,
    vehicle_mass=1.0,
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

    traj_ref = np.zeros((10, len_traj))
    traj_ref[0:3, :] = np.squeeze(traj_derivatives[0, :, :])
    traj_ref[7:10, :] = np.squeeze(traj_derivatives[1, :, :])
    if yaw_derivatives is None:
        yaw_derivatives = np.zeros((2, len_traj), dtype=np.float64)

    traj_ref[3:7, :], u_ref = forward(
        traj_derivatives[1:, :, :],
        yaw_derivatives,
        vehicle_mass,
    )

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
    traj = np.zeros((4, 3, np.size(angle_vec)))
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

    return minimum_snap_trajectory_generator(t_ref, traj, vehicle_mass=vehicle_mass)
