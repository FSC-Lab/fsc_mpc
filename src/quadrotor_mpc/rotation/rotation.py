# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import numba
from numpy.typing import ArrayLike


@numba.njit
def quaternion_normalize(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    q_norm = np.sqrt(q @ q)
    return 1 / q_norm * q


@numba.njit
def fast_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast cross product in 3 dimensions that does not broadcast or handle batch
    operation and does the minimum amount of checks

    Parameters
    ----------
    a : np.ndarray
        An array of 3 elements representing the left operand
    b : np.ndarray
        An array of 3 elements representing the right operand

    Returns
    -------
    np.ndarray
        The product a x b
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


@numba.njit
def quaternion_rotate_point(q: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Rotates a 3-vector `v` by a rotation represented by the unit quaternion `v`

    Parameters
    ----------
    q : ArrayLike
        An array of 4 elements representing a quaternion in real LAST order. Assumed to
        be normalized
    v : ArrayLike
        An array of 3 elements representing a 3-vector

    Returns
    -------
    np.ndarray
        The 3-vector after rotation
    """
    v = np.asarray(v)
    q = np.asarray(q)
    vec = q[0:3]
    w = q[3]
    uv = fast_cross(vec, v)
    uv += uv
    return v + w * uv + fast_cross(vec, uv)


def quaternion_to_rotation_matrix(q: ArrayLike) -> np.ndarray:
    """Converts a quaternion `q` into its equivalent 3-by-3 rotation matrix

    Parameters
    ----------
    q : ArrayLike
        An array of 4 elements representing a quaternion in real LAST order. Assumed to
        be normalized.
        Passing in a non-normal quaternion will result in a NON-orthogonal matrix

    Returns
    -------
    np.ndarray
        The equivalent 3-by-3 rotation matrix.
    """
    q = np.asarray(q)
    res = np.empty((3, 3))
    x, y, z, w = q[0], q[1], q[2], q[3]
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    res[0, 0] = 1.0 - (tyy + tzz)
    res[0, 1] = txy - twz
    res[0, 2] = txz + twy
    res[1, 0] = txy + twz
    res[1, 1] = 1.0 - (txx + tzz)
    res[1, 2] = tyz - twx
    res[2, 0] = txz - twy
    res[2, 1] = tyz + twx
    res[2, 2] = 1.0 - (txx + tyy)
    return res


@numba.njit
def quaternion_product(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Calculates the product of quaternion `a` and `b` using Hamiltonian quaternion
    product rules.

    Parameters
    ----------
    a : ArrayLike
        An array of 4 elements representing the left quaternion
    b : ArrayLike
        An array of 4 elements representing the right quaternion

    Returns
    -------
    np.ndarray
        The product a x b
    """

    a = np.asarray(a)
    b = np.asarray(b)
    res = np.empty((4,))
    res[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
    res[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2]
    res[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0]
    res[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    return res


def rotation_matrix_to_quaternion(mat: ArrayLike) -> np.ndarray:
    """Converts a 3-by-3 rotation matrix `mat` into its equivalent quaternion

    Parameters
    ----------
    mat : ArrayLike
        A 3-by-3 array representing a rotation matrix. Assumed to be orthonormal

    Returns
    -------
    np.ndarray
        The equivalent unit quaternion
    """
    mat = np.asarray(mat)
    t = mat.trace()
    q = np.empty((4,))
    if t > 0:
        t = np.sqrt(t + 1.0)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (mat[2, 1] - mat[1, 2]) * t
        q[1] = (mat[0, 2] - mat[2, 0]) * t
        q[2] = (mat[1, 0] - mat[0, 1]) * t

    else:
        i = 0
        if mat[1, 1] > mat[0, 0]:
            i = 1
        if mat[2, 2] > mat[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3

        t = np.sqrt(mat[i, i] - mat[j, j] - mat[k, k] + 1.0)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[0] = (mat[k, j] - mat[j, k]) * t
        q[j] = (mat[j, i] + mat[i, j]) * t
        q[k] = (mat[k, i] + mat[i, k]) * t
    return q


def undo_quaternion_flip(q_past, q_current):
    if np.sqrt(np.sum((q_past - q_current) ** 2)) > np.sqrt(
        np.sum((q_past + q_current) ** 2)
    ):
        return -q_current
    return q_current


@numba.njit
def quaternion_conjugate(q: ArrayLike) -> np.ndarray:
    """Computes the conjugate of the quaternion `q`. This operation is equivalent to
    taking the multiplication inverse if `q` is normalized

    Parameters
    ----------
    q : ArrayLike
        An array of 4 elements

    Returns
    -------
    np.ndarray
        The conjugated quaternion
    """
    q = np.asarray(q)
    w, x, y, z = q[3], q[0], q[1], q[2]
    return np.array([-x, -y, -z, w])


def angle_axis_to_quaternion(angle_axis: ArrayLike) -> np.ndarray:
    """Converts 3 angle-axis parameters `angle_axis` into its equivalent quaternion.

    Parameters
    ----------
    angle_axis : ArrayLike
        An array of 3 elements representing the angle axis parameters, which consist of
        the rotation angle multiplied to an unit-length rotation axis

    Returns
    -------
    np.ndarray
        The equivalent quaternion
    """
    angle_axis = np.asarray(angle_axis)
    theta_sq = angle_axis.dot(angle_axis)
    res = np.empty((4,))
    if theta_sq < 1e-10:
        real_factor = 1.0
        imag_factor = 0.5
    else:
        theta = np.sqrt(theta_sq)
        half_theta = 0.5 * theta
        real_factor = np.cos(half_theta)
        imag_factor = np.sin(half_theta) / theta
    res[0:3] = imag_factor * angle_axis
    res[3] = real_factor
    return res


def quaternion_to_angle_axis(quaternion: ArrayLike) -> np.ndarray:
    """Converts a quaternion into its equivalent angle-axis parameters

    Parameters
    ----------
    quaternion : ArrayLike
        An array of 4 elements representing a quaternion in real LAST order. Assumed to
        be normalized.

    Returns
    -------
    np.ndarray
        The equivalent angle-axis parameters
    """

    quaternion = np.asarray(quaternion)
    squared_n = quaternion[0:3].dot(quaternion[0:3])
    w = quaternion[3]

    if squared_n < 1e-10:
        squared_w = w * w
        two_atan_nbyw_by_n = 2 / w - 2.0 / 3.0 * squared_n / (w * squared_w)
    else:
        n = np.sqrt(squared_n)
        atan_nbyw = np.arctan2(-n, -w) if w < 0.0 else np.arctan2(n, w)
        two_atan_nbyw_by_n = 2.0 * atan_nbyw / n

    return two_atan_nbyw_by_n * quaternion[0:3]
