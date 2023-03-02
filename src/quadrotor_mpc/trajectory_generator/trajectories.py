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


import numpy as np
from numpy.typing import ArrayLike

from quadrotor_mpc.rotation import (
    quaternion_conjugate,
    quaternion_product,
    quaternion_rotate_point,
    rotation_matrix_to_quaternion,
    undo_quaternion_flip,
)

from .trajectory_generator import fit_multi_segment_polynomial_trajectory, get_full_traj


def minimum_snap_trajectory_generator(
    traj_derivatives: ArrayLike,
    yaw_derivatives: ArrayLike,
    t_ref: ArrayLike,
    quad,
    frame="W2B",
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
    :param quad: Quadrotor3D object, corresponding to the quadrotor model that will
    track the generated reference.
    :type quad: Quadrotor3D
    :param map_limits: dictionary of map limits if available, None otherwise.
    :param plot: True if show a plot of the generated trajectory.
    :return: tuple of 3 arrays:
        - Nx13 array of generated reference trajectory. The 13 dimension contains the
        components: position_xyz,
        attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
        - N array of reference timestamps. The same as in the input
        - Nx4 array of reference controls, corresponding to the four motors of the
        quadrotor.
    """

    traj_derivatives = np.asarray(traj_derivatives)
    yaw_derivatives = np.asarray(yaw_derivatives)
    t_ref = np.asarray(t_ref)

    discretization_dt = t_ref[1] - t_ref[0]
    len_traj = traj_derivatives.shape[2]

    # Add gravity to accelerations
    gravity = 9.81
    thrust = (
        traj_derivatives[2, :, :].T
        + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
    )
    # Compute body axes
    z_b = thrust / np.linalg.norm(thrust, axis=1, keepdims=True)

    yawing = np.any(yaw_derivatives[0, :] != 0)

    rate = np.zeros((len_traj, 3))
    f_t = np.zeros((len_traj, 1))
    for i in range(len_traj):
        f_t[i, 0] = quad.mass * z_b[i].dot(thrust[i, :].T)

    if yawing:
        # yaw is defined as the projection of the body-x axis on the horizontal plane
        x_c = np.concatenate(
            (
                np.cos(yaw_derivatives[0, :])[:, np.newaxis],
                np.sin(yaw_derivatives[0, :])[:, np.newaxis],
                np.zeros(len_traj)[:, np.newaxis],
            ),
            1,
        )
        y_b = np.cross(z_b, x_c)
        y_b = y_b / np.linalg.norm(y_b, axis=1, keepdims=True)
        x_b = np.cross(y_b, z_b)

        # Rotation matrix (from body to world)
        b_r_w = np.dstack((x_b, y_b, z_b))
        q = []
        for i in range(len_traj):
            # Transform to quaternion
            q.append(rotation_matrix_to_quaternion(b_r_w[i]))
            if i > 1:
                q[-1] = undo_quaternion_flip(q[-2], q[-1])
        q = np.stack(q)

        # Compute angular rate vector
        # Total thrust acceleration must be equal to the projection of the quadrotor
        # acceleration into the Z body axis
        a_proj = np.zeros((len_traj, 1))

        for i in range(len_traj):
            a_proj[i, 0] = z_b[i].dot(traj_derivatives[3, :, i])

        h_omega = quad.mass / f_t * (traj_derivatives[3, :, :].T - a_proj * z_b)
        for i in range(len_traj):
            rate[i, 0] = -h_omega[i].dot(y_b[i])
            rate[i, 1] = h_omega[i].dot(x_b[i])
            rate[i, 2] = -yaw_derivatives[1, i] * np.array([0, 0, 1]).dot(z_b[i])

    else:
        # new way to compute attitude:
        # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
        e_z = np.array([[0.0, 0.0, 1.0]])
        q_w = 1.0 + np.sum(e_z * z_b, axis=1)
        q_xyz = np.cross(e_z, z_b)
        q = 0.5 * np.concatenate([q_xyz, np.expand_dims(q_w, axis=1)], axis=1)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)

        # Use numerical differentiation of quaternions
        q_dot = np.gradient(q, axis=0) / discretization_dt
        w_int = np.zeros((len_traj, 3))
        for i in range(len_traj):
            w_int[i, :] = (
                2.0 * quaternion_product(quaternion_conjugate(q[i, :]), q_dot[i])[:3]
            )
        rate[:, 0] = w_int[:, 0]
        rate[:, 1] = w_int[:, 1]
        rate[:, 2] = w_int[:, 2]

        print("Maximum yawrate before adaption: %.3f" % np.max(np.abs(rate[:, 2])))
        q_new = q
        yaw_corr_acc = 0.0
        for i in range(1, len_traj):
            yaw_corr = -rate[i, 2] * discretization_dt
            yaw_corr_acc += yaw_corr
            q_corr = np.array(
                [0.0, 0.0, np.sin(yaw_corr_acc / 2.0), np.cos(yaw_corr_acc / 2.0)]
            )
            q_new[i, :] = quaternion_product(q[i, :], q_corr)
            w_int[i, :] = (
                2.0 * quaternion_product(quaternion_conjugate(q[i, :]), q_dot[i])[:3]
            )

        q_new_dot = np.gradient(q_new, axis=0) / discretization_dt
        for i in range(1, len_traj):
            w_int[i, :] = (
                2.0
                * quaternion_product(quaternion_conjugate(q_new[i, :]), q_new_dot[i])[
                    :3
                ]
            )

        q = q_new
        rate[:, 0] = w_int[:, 0]
        rate[:, 1] = w_int[:, 1]
        rate[:, 2] = w_int[:, 2]
        print("Maximum yawrate after adaption: %.3f" % np.max(np.abs(rate[:, 2])))

    # Compute inputs
    u_ref = np.column_stack((f_t, rate))

    full_pos = traj_derivatives[0, :, :].T
    full_vel = traj_derivatives[1, :, :].T
    if frame == "W2B":
        q[:, 0:3] = -q[:, 0:3]
        for idx in range(len_traj):
            full_vel[idx, :] = quaternion_rotate_point(q[idx, :], full_vel[idx, :])
    traj_ref = np.concatenate((full_pos, q, full_vel), 1)

    # Locate starting point right at x=0 and y=0.
    traj_ref[:, 0] -= traj_ref[0, 0]
    traj_ref[:, 1] -= traj_ref[0, 1]

    return traj_ref, u_ref, t_ref


def straight_trajectory(quad, begin, end, z, discretization_dt, lin_acc, v_max):
    position_diff = end - begin
    distance = np.linalg.norm(position_diff)
    heading_vector = position_diff[..., None] / distance

    ramp_up_t = v_max / lin_acc

    # Calculate simulation time to achieve desired maximum velocity with specified
    # acceleration
    t_total = distance / v_max

    refs = {}
    a = {}
    refs["ramp_up"] = np.r_[0:ramp_up_t:discretization_dt]
    a["ramp_up"] = lin_acc * np.ones_like(refs["ramp_up"])
    t_cruise = t_total - 2 * ramp_up_t
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
    traj[0, 0:2, :] = begin[..., None] + heading_vector * d_vec
    traj[0, 2, :] = z
    traj[1, 0:2, :] = heading_vector * v_vec
    traj[2, 0:2, :] = heading_vector * a_vec
    yaw = np.zeros((2, n))
    yaw[0, :] = np.arctan2(position_diff[1], position_diff[0])

    reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(
        traj, yaw, t_ref, quad
    )
    return reference_traj, t_ref, reference_u


def loop_trajectory(
    quad,
    discretization_dt,
    radius,
    z,
    lin_acc,
    clockwise,
    yawing,
    v_max,
):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.

    :param quad: Quadrotor model
    :param discretization_dt: Sampling period of the trajectory.
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the quadrotor yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
    """

    # Apply map limits to radius
    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
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
    pos_traj_x = radius * np.sin(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_z = np.ones_like(pos_traj_x) * z

    vel_traj_x = (radius * w_vec * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = -(radius * w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]

    acc_traj_x = (
        radius
        * (alpha_vec * np.cos(angle_vec) - w_vec**2 * np.sin(angle_vec))[
            np.newaxis, np.newaxis, :
        ]
    )
    acc_traj_y = (
        -radius
        * (alpha_vec * np.sin(angle_vec) + w_vec**2 * np.cos(angle_vec))[
            np.newaxis, np.newaxis, :
        ]
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
    jerk_traj_x = jerk_traj_x[np.newaxis, np.newaxis, :]
    jerk_traj_y = jerk_traj_y[np.newaxis, np.newaxis, :]

    if yawing:
        yaw_traj = -angle_vec
    else:
        yaw_traj = np.zeros_like(angle_vec)

    traj = np.concatenate(
        (
            np.concatenate((pos_traj_x, pos_traj_y, pos_traj_z), 1),
            np.concatenate((vel_traj_x, vel_traj_y, np.zeros_like(vel_traj_x)), 1),
            np.concatenate((acc_traj_x, acc_traj_y, np.zeros_like(acc_traj_x)), 1),
            np.concatenate((jerk_traj_x, jerk_traj_y, np.zeros_like(jerk_traj_x)), 1),
        ),
        0,
    )

    yaw = np.concatenate((yaw_traj[np.newaxis, :], w_vec[np.newaxis, :]), 0)

    return minimum_snap_trajectory_generator(traj, yaw, t_ref, quad)


def lemniscate_trajectory(
    quad,
    discretization_dt,
    radius,
    z,
    lin_acc,
    v_max,
):
    """

    :param quad:
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

    yaw = np.zeros_like(traj)

    return minimum_snap_trajectory_generator(traj, yaw, t_ref, quad)
