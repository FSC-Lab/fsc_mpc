# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import itertools

import numpy as np
import pytest
from numpy.typing import ArrayLike

from quadrotor_mpc import quadrotor_model, trajectory_generator
from quadrotor_mpc.rotation import (
    quaternion_conjugate,
    quaternion_product,
    quaternion_rotate_point,
)


def check_trajectory(
    trajectory: ArrayLike, u_ref: ArrayLike, t_ref: ArrayLike, frame="W2B"
) -> None:
    """Checks the integrity of a trajectory produced by the minimum snap trajectory
    generator

    Parameters
    ----------
    trajectory : ArrayLike
        A N-by-10 array containing N trajectory points stacked row-wise
    u_ref : ArrayLike
        A N-by-4 array containing N inputs stacked row-wise
    t_ref : ArrayLike
        A N-element array containing N time references
    frame : str, optional
        Toggles whether the quaternion trajectory should be transformed into the
        world-to-body form, and whether velocity trajectory should be transformed into
        the world frame, by default "W2B"

    """

    # NO np.asarray! Make copies such that the checks will not mutate the original
    # arrays
    trajectory = np.array(trajectory)
    t_ref = np.array(t_ref)
    u_ref = np.array(u_ref)
    if frame == "W2B":
        for i in range(trajectory.shape[0]):
            trajectory[i, 3:7] = quaternion_conjugate(trajectory[i, 3:7])
            trajectory[i, 7:10] = quaternion_rotate_point(
                trajectory[i, 3:7], trajectory[i, 7:10]
            )

    dt = np.expand_dims(np.gradient(t_ref, axis=0), axis=1)
    numeric_derivative = np.gradient(trajectory, axis=0) / dt

    errors = np.zeros((dt.shape[0], 3))

    num_bodyrates = []

    for i in range(dt.shape[0]):
        # 1) check if velocity is consistent with position
        numeric_velocity = numeric_derivative[i, 0:3]
        analytic_velocity = trajectory[i, 7:10]
        errors[i, 0] = np.linalg.norm(numeric_velocity - analytic_velocity)
        assert np.allclose(analytic_velocity, numeric_velocity, atol=1e-2, rtol=1e-2)

        # 2) check if attitude is consistent with acceleration
        gravity = 9.81
        numeric_thrust = numeric_derivative[i, 7:10] + np.array([0.0, 0.0, gravity])
        numeric_thrust = numeric_thrust / np.linalg.norm(numeric_thrust)
        analytic_attitude = trajectory[i, 3:7]
        qnorm = np.linalg.norm(analytic_attitude)
        assert np.isclose(qnorm, 1.0)

        e_z = np.array([0.0, 0.0, 1.0])
        q_w = 1.0 + np.dot(e_z, numeric_thrust)
        q_xyz = np.cross(e_z, numeric_thrust)
        numeric_attitude = 0.5 * np.r_[q_xyz, q_w]
        numeric_attitude = numeric_attitude / np.linalg.norm(numeric_attitude)
        # the two attitudes can only differ in yaw --> check x,y component
        q_diff = quaternion_product(
            quaternion_conjugate(analytic_attitude), numeric_attitude
        )
        errors[i, 1] = np.linalg.norm(q_diff[0:2])
        assert np.allclose(
            q_diff[0:2],
            np.zeros(
                2,
            ),
            atol=1e-3,
            rtol=1e-3,
        )

        # 3) check if bodyrates agree with attitude difference
        numeric_bodyrates = (
            2.0
            * quaternion_product(
                quaternion_conjugate(trajectory[i, 3:7]), numeric_derivative[i, 3:7]
            )[:3]
        )
        num_bodyrates.append(numeric_bodyrates)
        analytic_bodyrates = u_ref[i, 1:4]
        errors[i, 2] = np.linalg.norm(numeric_bodyrates - analytic_bodyrates)
        assert np.allclose(numeric_bodyrates, analytic_bodyrates, atol=0.05, rtol=0.05)


@pytest.fixture(name="trajectories_test_data")
def quad_fixture():
    trajectory_radius_range = np.linspace(10, 40, 4)
    acceleration_range = np.linspace(0.5, 1.5, 4)
    max_speed_range = np.linspace(5.0, 15.0, 4)
    return quadrotor_model.QuadrotorModel(1.0), (
        trajectory_radius_range,
        acceleration_range,
        max_speed_range,
    )


def test_loop_trajectory(trajectories_test_data):
    control_period = 0.01
    quad, (
        trajectory_radius_range,
        acceleration_range,
        max_speed_range,
    ) = trajectories_test_data

    for trajectory_radius, acceleration, max_speed in itertools.product(
        trajectory_radius_range, acceleration_range, max_speed_range
    ):
        traj_ref, u_ref, t_ref = trajectory_generator.loop_trajectory(
            quad,
            control_period,
            radius=trajectory_radius,
            z=1,
            lin_acc=acceleration,
            clockwise=True,
            yawing=False,
            v_max=max_speed,
        )

        check_trajectory(traj_ref, u_ref, t_ref)


def test_lemniscate_trajectory(trajectories_test_data):
    control_period = 0.01
    quad, (
        trajectory_radius_range,
        acceleration_range,
        max_speed_range,
    ) = trajectories_test_data
    for trajectory_radius, acceleration, max_speed in itertools.product(
        trajectory_radius_range, acceleration_range, max_speed_range
    ):
        traj_ref, u_ref, t_ref = trajectory_generator.lemniscate_trajectory(
            quad,
            control_period,
            radius=trajectory_radius,
            z=1,
            lin_acc=acceleration,
            v_max=max_speed,
        )

        check_trajectory(traj_ref, u_ref, t_ref)
