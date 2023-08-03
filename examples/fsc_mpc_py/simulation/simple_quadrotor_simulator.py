"""
A minimalistic quadrotor dynamics simulator
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

from .rotation import quaternion_product, quaternion_rotate_point


class SimpleQuadrotorSimulator:
    """
    A minified simulator of a quadrotor UAV.
    This is used instead of a more full-featured simulator to preserve stability of
    tests
    """

    def __init__(
        self,
        mass,
        base_dt,
        init_time,
        init_state,
        init_input,
        grav_accel=None,
        quaternion_normalization_gain=1.0,
    ):
        self._mass = float(mass)
        if self._mass <= 0:
            raise ValueError("Vehicle mass must be positive")
        self._dt = base_dt
        self._time = init_time
        self._state = np.asarray(init_state)
        self._input = np.asarray(init_input)
        if grav_accel is not None:
            self._grav_accel = np.asarray(grav_accel)
        else:
            self._grav_accel = np.array([0.0, 0.0, -9.81])

        self._k_nrm = float(quaternion_normalization_gain)
        if self._k_nrm <= 0:
            raise ValueError("Quaternion normalization gain must be positive")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = np.asarray(val)

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, val):
        self._input = np.asarray(val)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        self._time = val

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        if self._dt <= 0.0:
            return ValueError("Timestep must be greater than 0")
        self._dt = val

    def simulation_update(self):
        k = np.zeros((10, 4))
        x = self._state
        u = self._input

        half_dt = self._dt / 2
        k[:, 0] = self.model_derivatives(x, u)
        k[:, 1] = self.model_derivatives(x + half_dt * k[:, 0], u)
        k[:, 2] = self.model_derivatives(x + half_dt * k[:, 1], u)
        k[:, 3] = self.model_derivatives(x + self._dt * k[:, 2], u)
        self._state = x + self._dt / 6 * k.dot(np.array([1.0, 2.0, 2.0, 1.0]))
        self._time += self._dt

    def model_derivatives(self, x, u):
        x = np.asarray(x)
        q = x[3:7]
        v = x[7:10]

        u = np.asarray(u)
        f = u[0]
        w = u[1:4]

        grav_vector = np.array([0.0, 0.0, self._grav_accel])
        body_thrust = np.array([0.0, 0.0, f])

        dx = np.zeros(10)
        dx[0:3] = v
        dx[3:7] = (
            quaternion_product(q, np.append(w, 0.0)) / 2.0
            + self._k_nrm * (1.0 - q.dot(q)) * q
        )
        dx[7:10] = quaternion_rotate_point(q, body_thrust / self._mass) + grav_vector
        return dx
