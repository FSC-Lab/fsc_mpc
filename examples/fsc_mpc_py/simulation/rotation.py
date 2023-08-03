"""
A minified version of the `rotation` library in use in the FSC
Copyright © 2023 Hs293Go

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

import numpy as np


def _fast_cross(a, b):
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def hat(vec):
    return np.array(
        [[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0.0]],
    )


def quaternion_product(lhs, rhs):
    """Computes the quaternion product betwen two quaternions

    Parameters
    ----------
    lhs
        Left operand quaternion consisting of 4 components in [x, y, z, w] (real
        component last) order
    rhs
        Right operand quaternion consisting of 4 components in [x, y, z, w] (real
        component last) order

    Returns
    -------
    np.ndarray
        The product quaternion

    Raises
    ------
    ValueError
        If any of the left or right operands is not an array of 4 elements
    """
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.size != 4 or rhs.size != 4:
        raise ValueError("Incompatible dimensions for quaternion product. Must be 4")

    return np.array(
        [
            lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[3] * rhs[1] + lhs[1] * rhs[3] + lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[3] * rhs[2] + lhs[2] * rhs[3] + lhs[0] * rhs[1] - lhs[1] * rhs[0],
            lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
        ]
    )


def quaternion_rotate_point(quaternion, point):
    """Applies a rotation parameterized by a unit-quaternion to a point or 3-vector

    Parameters
    ----------
    quaternion
        A unit quaternion consisting of 4 components in [x, y, z, w] (real component
        last) order

    point
        The point / 3-vector to be rotated

    Returns
    -------
    np.ndarray
        The rotated point or 3-vector

    Raises
    ------
    ValueError
        If the quaternion is not an array of 4 elements
    ValueError
        If the point is not an array of 3 elements
    """

    quaternion = np.asarray(quaternion)
    point = np.asarray(point)

    if quaternion.size != 4:
        raise ValueError("Invalid dimension for quaternion. Must be 4")
    if point.size != 3:
        raise ValueError("Invalid dimension for point. Must be 3")
    vec = quaternion[0:3]
    uv = _fast_cross(vec, point)
    uv += uv
    return point + quaternion[3] * uv + _fast_cross(vec, uv)
