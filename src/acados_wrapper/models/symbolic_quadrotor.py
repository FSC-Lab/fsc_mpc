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


from typing import Union
import casadi as cs
import numpy as np


DEFAULT_STATE = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
)

DEFAULT_INPUT = np.array([9.81, 0.0, 0.0, 0.0], dtype=np.float64)


def quaternion_conjugate(q: cs.MX) -> cs.MX:
    res = cs.MX(q)
    res[0:3] = -q[0:3]
    return res


def quaternion_rotate_point(q: cs.MX, v: cs.MX) -> cs.MX:
    w = q[3]
    vec = q[0:3]
    uv = cs.cross(vec, v, 1)
    uv += uv
    return v + w * uv + cs.cross(vec, uv)


def quaternion_product(a: cs.MX, b: cs.MX) -> cs.MX:
    res = cs.MX(4, 1)
    res[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
    res[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2]
    res[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0]
    res[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    return res


class SymbolicQuadrotor:
    """
    Implements the dynamics model of a quadrotor
    """

    NX = 10
    NU = 4

    SYM_GRAV_VECTOR = cs.vertcat(0, 0, -9.81)

    def __init__(self, mass: Union[cs.MX, float]):
        """Constructs the QuadrotorModel object

        Parameters
        ----------
        mass : float
            The mass of the quadrotor
        """
        # System state space

        self._mass = mass  # kg

    def model_derivatives(self, x: cs.MX, u: cs.MX) -> cs.MX:
        """Computes the symbolic, noiseless, model derivatives, i.e. evaluates the
        equations of motion, at given symbolic state `x` and input `u`v

        Parameters
        ----------
        x : cs.MX
            The operating state to evaluate the model derivatives. A 10-by-1 symbolic
            matrix in

                [position, attitude, velocity]

            order, where attitude is a unit quaternion in real LAST order
        u : cs.MX
            The control input to the quadrotor. A 4-by-1 symbolic matrix in

                [thrust, angular velocities]

            order

        Returns
        -------
        cs.MX
            The symbolic model derivatives as a 10-by-1 symbolic matrix
        """
        q = x[3:7]  # quaternion
        v_b = x[7:10]  # velocity

        f = u[0]  # Thrust force
        w = u[1:4]  # angular velocity

        augmented_w = cs.vertcat(w / 2, 0)
        a_thrust = cs.vertcat(0.0, 0.0, f / self._mass)

        # Return the core equations of motion
        return cs.vertcat(
            v_b,
            quaternion_product(q, augmented_w),
            quaternion_rotate_point(q, a_thrust) + self.SYM_GRAV_VECTOR,
        )
