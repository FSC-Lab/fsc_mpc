# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Part of this work is derived from "data_driven_mpc"
# https://github.com/uzh-rpg/data_driven_mpc
# Licensed under the following terms

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



def get_full_traj(poly_coeffs, target_dt, int_dt):
    dims = poly_coeffs.shape[-1]
    full_traj = np.zeros((4, dims, 0))
    t_total = np.zeros((0,))

    if isinstance(target_dt, float):
        # Adjust target_dt to make it divisible by int_dt
        target_dt = round(target_dt / int_dt) * int_dt

        # Assign target time for each keypoint using homogeneous spacing
        t_vec = np.arange(0, target_dt * (poly_coeffs.shape[0] + 1) - 1e-5, target_dt)

    else:
        # The time between each pair of points is assigned independently
        # First, also adjust each value of the target_dt vector to make it divisible by int_dt
        for i, dt in enumerate(target_dt):
            target_dt[i] = round(dt / int_dt) * int_dt

        t_vec = np.append(np.zeros(1), np.cumsum(target_dt[:-1]))

    for seg in range(len(t_vec) - 1):
        # Select time sampling (linear or quadratic) mode
        tau_dt = np.arange(t_vec[seg], t_vec[seg + 1] + 1e-5, int_dt)

        # Re-normalize time sampling vector between -1 and 1
        t1 = (tau_dt - t_vec[seg]) / (t_vec[seg + 1] - t_vec[seg]) * 2 - 1

        # Compression ratio
        compress = 2 / np.diff(t_vec)[seg]

        # Integrate current segment of trajectory
        traj = np.zeros((4, dims, len(t1)))

        for der_order in range(4):
            for i in range(dims):
                traj[der_order, i, :] = np.polyval(
                    np.polyder(poly_coeffs[seg, :, i], der_order), t1
                ) * (compress**der_order)

        if seg < len(t_vec) - 2:
            # Remove last sample (will be the initial point of next segment)
            traj = traj[:, :, :-1]
            t_seg = tau_dt[:-1]
        else:
            t_seg = tau_dt

        full_traj = np.concatenate((full_traj, traj), axis=-1)
        t_total = np.concatenate((t_total, t_seg))

    # Separate into p_xyz and yaw trajectories
    yaw_traj = full_traj[:, -1, :]
    full_traj = full_traj[:, :-1, :]

    return full_traj, yaw_traj, t_total


def fit_multi_segment_polynomial_trajectory(p_targets, yaw_targets):
    p_targets = np.concatenate((p_targets, yaw_targets[np.newaxis, :]), 0)
    m = multiple_waypoints(p_targets.shape[1] - 1)

    dims = p_targets.shape[0]
    n_segments = p_targets.shape[1]

    poly_coefficients = np.zeros((n_segments - 1, 8, dims))
    for dim in range(dims):
        b = rhs_generation(p_targets[dim, :])
        poly_coefficients[:, :, dim] = np.fliplr(
            np.linalg.solve(m, b).reshape(n_segments - 1, 8)
        )

    return poly_coefficients


def matrix_generation(ts):
    b = np.array(
        [
            [1, ts, ts**2, ts**3, ts**4, ts**5, ts**6, ts**7],
            [
                0,
                1,
                2 * ts,
                3 * ts**2,
                4 * ts**3,
                5 * ts**4,
                6 * ts**5,
                7 * ts**6,
            ],
            [0, 0, 2, 6 * ts, 12 * ts**2, 20 * ts**3, 30 * ts**4, 42 * ts**5],
            [0, 0, 0, 6, 24 * ts, 60 * ts**2, 120 * ts**3, 210 * ts**4],
            [0, 0, 0, 0, 24, 120 * ts, 360 * ts**2, 840 * ts**3],
            [0, 0, 0, 0, 0, 120, 720 * ts, 2520 * ts**2],
            [0, 0, 0, 0, 0, 0, 720, 5040 * ts],
            [0, 0, 0, 0, 0, 0, 0, 5040],
        ]
    )

    return b


def multiple_waypoints(n_segments):
    m = np.zeros((8 * n_segments, 8 * n_segments))

    for i in range(n_segments):
        if i == 0:
            # initial condition of the first curve
            b = matrix_generation(-1.0)
            m[8 * i : 8 * i + 4, 8 * i : 8 * i + 8] = b[:4, :]

            # intermediary condition of the first curve
            b = matrix_generation(1.0)
            m[8 * i + 4 : 8 * i + 7 + 4, 8 * i : 8 * i + 8] = b[:-1, :]

            # starting condition of the second curve position and derivatives
            b = matrix_generation(-1.0)
            m[8 * i + 4 + 1 : 8 * i + 4 + 7, 8 * (i + 1) : 8 * (i + 1) + 8] = -b[
                1:-1, :
            ]
            m[8 * i + 4 + 7 : 8 * i + 4 + 8, 8 * (i + 1) : 8 * (i + 1) + 8] = b[0, :]

        elif i != n_segments - 1:
            # starting condition of the ith curve position and derivatives
            b = matrix_generation(1.0)
            m[8 * i + 4 : 8 * i + 7 + 4, 8 * i : 8 * i + 8] = b[:-1, :]

            # end condition of the ith curve position and derivatives
            b = matrix_generation(-1.0)
            m[8 * i + 4 + 1 : 8 * i + 4 + 7, 8 * (i + 1) : 8 * (i + 1) + 8] = -b[
                1:-1, :
            ]
            m[8 * i + 4 + 7 : 8 * i + 4 + 8, 8 * (i + 1) : 8 * (i + 1) + 8] = b[0, :]

        if i == n_segments - 1:
            # end condition of the final curve position and derivatives (4 boundary conditions)
            b = matrix_generation(1.0)
            m[8 * i + 4 : 8 * i + 4 + 4, 8 * i : 8 * i + 8] = b[:4, :]

    return m


def fit_single_segment(
    p_start,
    p_end,
    v_start=None,
    v_end=None,
    a_start=None,
    a_end=None,
    j_start=None,
    j_end=None,
):
    if v_start is None:
        v_start = np.array([0, 0])
    if v_end is None:
        v_end = np.array([0, 0])
    if a_start is None:
        a_start = np.array([0, 0])
    if a_end is None:
        a_end = np.array([0, 0])
    if j_start is None:
        j_start = np.array([0, 0])
    if j_end is None:
        j_end = np.array([0, 0])

    poly_coefficients = np.zeros((8, len(p_start)))

    tf = 1
    ti = -1
    A = np.array(
        (
            [
                [
                    1 * tf**7,
                    1 * tf**6,
                    1 * tf**5,
                    1 * tf**4,
                    1 * tf**3,
                    1 * tf**2,
                    1 * tf**1,
                    1,
                ],
                [
                    7 * tf**6,
                    6 * tf**5,
                    5 * tf**4,
                    4 * tf**3,
                    3 * tf**2,
                    2 * tf**1,
                    1,
                    0,
                ],
                [
                    42 * tf**5,
                    30 * tf**4,
                    20 * tf**3,
                    12 * tf**2,
                    6 * tf**1,
                    2,
                    0,
                    0,
                ],
                [210 * tf**4, 120 * tf**3, 60 * tf**2, 24 * tf**1, 6, 0, 0, 0],
                [
                    1 * ti**7,
                    1 * ti**6,
                    1 * ti**5,
                    1 * ti**4,
                    1 * ti**3,
                    1 * ti**2,
                    1 * ti**1,
                    1,
                ],
                [
                    7 * ti**6,
                    6 * ti**5,
                    5 * ti**4,
                    4 * ti**3,
                    3 * ti**2,
                    2 * ti**1,
                    1,
                    0,
                ],
                [
                    42 * ti**5,
                    30 * ti**4,
                    20 * ti**3,
                    12 * ti**2,
                    6 * ti**1,
                    2,
                    0,
                    0,
                ],
                [210 * ti**4, 120 * ti**3, 60 * ti**2, 24 * ti**1, 6, 0, 0, 0],
            ]
        )
    )

    A = np.tile(A[:, :, np.newaxis], (1, 1, len(p_start)))

    b = np.concatenate(
        (p_end, v_end, a_end, j_end, p_start, v_start, a_start, j_start)
    ).reshape(8, -1)

    for i in range(len(p_start)):
        poly_coefficients[:, i] = np.linalg.inv(A[:, :, i]).dot(np.array(b[:, i]))

    return np.expand_dims(poly_coefficients, 0)


def rhs_generation(x):
    n = x.shape[0] - 1

    big_x = np.zeros((8 * n))
    big_x[:4] = np.array([x[0], 0, 0, 0]).T
    big_x[-4:] = np.array([x[-1], 0, 0, 0]).T

    for i in range(1, n):
        big_x[8 * (i - 1) + 4 : 8 * (i - 1) + 8 + 4] = np.array(
            [x[i], 0, 0, 0, 0, 0, 0, x[i]]
        ).T

    return big_x
