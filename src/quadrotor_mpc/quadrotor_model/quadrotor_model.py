# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from quadrotor_mpc.rotation import (
    fast_cross,
    quaternion_conjugate,
    quaternion_normalize,
    quaternion_product,
    quaternion_rotate_point,
)

DEFAULT_STATE = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.double
)

DEFAULT_INPUT = np.array([9.81, 0.0, 0.0, 0.0], dtype=np.double)


class QuadrotorModel:
    NX = 10
    NU = 4

    # Drag coefficients [kg / m]
    ROTOR_DRAG = np.array([0.3, 0.3, 0.0])
    AERO_DRAG = 0.08

    def __init__(
        self,
        mass,
        initial_state=DEFAULT_STATE,
        noisy=False,
        drag=False,
    ):
        # System state space
        self._state = initial_state

        self.mass = mass  # kg

        # Gravity vector
        self.g = np.array([0, 0, 9.81])  # m s^-2
        self._input = DEFAULT_INPUT  # N

        self.drag = drag
        self.noisy = noisy

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state[:] = np.asarray(val)

    @property
    def pos(self):
        return self._state[0:3]

    @pos.setter
    def pos(self, val):
        self._state[0:3] = np.asarray(val)

    @property
    def angle(self):
        return self._state[3:7]

    @angle.setter
    def angle(self, val):
        self._state[3:7] = np.asarray(val)

    @property
    def vel(self):
        return self._state[7:10]

    @vel.setter
    def vel(self, val):
        self._state[7:10] = np.asarray(val)

    @property
    def control(self):
        return self._input

    def model_derivatives(self, x, u, f_d=None):
        x = np.asarray(x)
        u = np.asarray(u)
        dx = np.empty((self.NX,))

        rate_q = np.zeros((4,))
        rate_q[0:3] = -0.5 * u[1:4]
        q = x[3:7]
        v_b = x[7:10]
        a_thrust = np.array([0, 0, u[0]])

        dx[0:3] = quaternion_rotate_point(quaternion_conjugate(q), x[7:10])
        dx[3:7] = quaternion_product(rate_q, q)
        dx[7:10] = (
            -fast_cross(u[1:4], v_b) + a_thrust + quaternion_rotate_point(q, -self.g)
        )
        if f_d is not None:
            dx[7:10] += f_d / self.mass
        if self.drag:
            # Compute aerodynamic drag acceleration in world frame
            a_drag = -self.AERO_DRAG * v_b**2 * np.sign(v_b) / self.mass
            # Add rotor drag
            r_drag = -self.ROTOR_DRAG * v_b / self.mass
            dx[7:10] += a_drag + r_drag
        return dx

    def model_update(self, u, dt):
        """
        Runge-Kutta 4th order dynamics integration

        :param u: 4-dimensional vector with components between [0.0, 1.0] that represent
        the activation of each motor.
        :param dt: time differential
        """

        self._input[:] = np.asarray(u)

        # Generate disturbance forces / torques
        if self.noisy:
            f_d = np.random.normal(size=(3,), scale=10 * dt)
        else:
            f_d = np.zeros((3,))

        x = self.state

        # RK4 integration

        k1 = self.model_derivatives(x, u, f_d)
        x_aux = x + dt / 2 * k1
        k2 = self.model_derivatives(x_aux, u, f_d)
        x_aux = x + dt / 2 * k2
        k3 = self.model_derivatives(x_aux, u, f_d)
        x_aux = x + dt * k3
        k4 = self.model_derivatives(x_aux, u, f_d)
        x = x + dt * (1.0 / 6.0 * k1 + 2.0 / 6.0 * k2 + 2.0 / 6.0 * k3 + 1.0 / 6.0 * k4)

        # Ensure unit quaternion
        x[3:7] = quaternion_normalize(x[3:7])

        self._state[:] = x
