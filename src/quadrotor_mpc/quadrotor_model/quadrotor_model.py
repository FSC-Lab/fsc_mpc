# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union

import casadi as cs
import numpy as np
from numpy.typing import ArrayLike

import quadrotor_mpc.rotation.symbolic as S
from quadrotor_mpc.rotation import (
    fast_cross,
    quaternion_conjugate,
    quaternion_normalize,
    quaternion_product,
    quaternion_rotate_point,
)

DEFAULT_STATE = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
)

DEFAULT_INPUT = np.array([9.81, 0.0, 0.0, 0.0], dtype=np.float64)


class QuadrotorModel:
    NX = 10
    NU = 4

    # Drag coefficients [kg / m]
    ROTOR_DRAG = np.array([0.3, 0.3, 0.0])
    AERO_DRAG = 0.08

    def __init__(
        self,
        mass: Union[float, cs.MX],
        initial_state: ArrayLike = DEFAULT_STATE,
        noisy: bool = False,
        drag: bool = False,
    ):
        """Constructs the QuadrotorModel object

        Parameters
        ----------
        mass : float
            The mass of the quadrotor
        initial_state : ArrayLike, optional
            The initial states of the quadrotor. An array of 10-elements in

            [position, attitude, velocity]

            order, where attitude is a unit quaternion in real LAST order, by default
            DEFAULT_STATE
        noisy : bool, optional
            Toggles simulating additive Gaussian process noise, by default False
        drag : bool, optional
            Toggles simulating aerodynamic and rotor drag, by default False
        """
        # System state space
        self._state = np.asarray(initial_state)

        self._mass = mass  # kg

        # Gravity vector
        self._g = np.array([0, 0, -9.81])  # m s^-2
        self._sym_g = cs.vertcat(0, 0, -9.81)
        self._input = DEFAULT_INPUT  # N

        self._drag = drag
        self._noisy = noisy

    @property
    def mass(self):
        return self._mass

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

    def symbolic_derivatives(self, x: cs.MX, u: cs.MX) -> cs.MX:
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

        augmented_w = cs.vertcat(-w / 2, 0)  # Minus half angular velocity as quaternion
        a_thrust = cs.vertcat(0.0, 0.0, f / self._mass)  # Thrust vector in body frame

        # Return the core equations of motion
        return cs.vertcat(
            S.quaternion_rotate_point(S.quaternion_conjugate(q), v_b),
            S.quaternion_product(augmented_w, q),
            -cs.cross(w, v_b, 1) + a_thrust + S.quaternion_rotate_point(q, self._sym_g),
        )

    def model_derivatives(
        self, x: ArrayLike, u: ArrayLike, f_d: Union[ArrayLike, None] = None
    ) -> np.ndarray:
        """Computes the model derivatives, i.e. evaluates the equations of motion, at
        given state `x` and input `u`

        Parameters
        ----------
        x : ArrayLike
            The operating state to evaluate the model derivatives. An array of
            10-elements in

                [position, attitude, velocity]

            order, where attitude is a unit quaternion in real LAST order
        u : ArrayLike
            The control input to the quadrotor. An array of 4-elements in

                [thrust, angular velocities]

            order
        f_d : Union[ArrayLike, None], optional
            Additive process noise. An array of 3-elements, by default None

        Returns
        -------
        np.ndarray
            The model derivatives as a 10-element array
        """
        x = np.asarray(x)
        u = np.asarray(u)

        q = x[3:7]  # quaternion
        v_b = x[7:10]  # velocity

        f = u[0]  # Thrust force
        w = u[1:4]  # angular velocity

        augmented_w = np.zeros((4,))
        augmented_w[0:3] = -0.5 * w  # Minus half angular velocity as quaternion
        a_thrust = np.array([0, 0, f / self._mass])  # Thrust vector in body frame

        # Evaluate the core equations of motion
        dx = np.empty((self.NX,))
        dx[0:3] = quaternion_rotate_point(quaternion_conjugate(q), v_b)
        dx[3:7] = quaternion_product(augmented_w, q)
        dx[7:10] = -fast_cross(w, v_b) + a_thrust + quaternion_rotate_point(q, self._g)

        # Add disturbances and drag
        if f_d is not None:
            f_d = np.asarray(f_d)
            dx[7:10] += f_d / self._mass
        if self._drag:
            # Compute aerodynamic drag acceleration in world frame
            a_drag = -self.AERO_DRAG * v_b**2 * np.sign(v_b) / self._mass
            # Add rotor drag
            r_drag = -self.ROTOR_DRAG * v_b / self._mass
            dx[7:10] += a_drag + r_drag
        return dx

    def model_update(self, u: ArrayLike, dt: float) -> np.ndarray:
        """Runs RK4 forward simulation of the quadrotor dynamics

        Parameters
        ----------
        u : ArrayLike
            The control input to the quadrotor. An array of 4-elements in

                [thrust, angular velocities]

            order
        dt : float
            The time step for integration

        Returns
        -------
        np.ndarray
            A copy of the updated state as a 10-element array
        """

        self._input[:] = np.asarray(u)

        # Generate disturbance forces / torques
        if self._noisy:
            f_d = np.random.normal(size=(3,), scale=10 * dt)
        else:
            f_d = np.zeros((3,))

        x = np.array(self.state)

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
        return x
