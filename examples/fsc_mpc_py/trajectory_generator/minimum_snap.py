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

import warnings
from typing import Union

import numpy as np
from fsc_mpc_py.trajectory_generator import PiecewisePolynomialTrajectory
from numpy.typing import ArrayLike
from scipy import optimize


class MinimumSnap:
    def __init__(self, degree: int, deriv_wts: ArrayLike) -> None:
        self._degree = degree
        self._n_cfs = degree + 1
        deriv_wts = np.asarray(deriv_wts, dtype=np.float64).squeeze()
        if deriv_wts.ndim > 1 or (deriv_wts < 0.0).any():
            raise ValueError(
                "Weights on derivatives must be a 1D list of nonnegative numbers"
            )
        self._deriv_wts = deriv_wts
        self._iszero_tol = 1e-8
        self._endpoints_order = 3
        self._t_ref = np.array([])
        self._pos_ref = np.array([])

    @property
    def pos_ref(self):
        if self._pos_ref.size == 0:
            raise ValueError("Position references are not yet initialized")
        return self._pos_ref

    @property
    def t_ref(self):
        if self._t_ref.size == 0:
            raise ValueError("Time references are not yet initialized")
        return self._t_ref

    def _process_endpoints(self, pt: ArrayLike, expected_dim: int = -1):
        pt = np.asarray(pt, dtype=np.float64)
        if pt.ndim == 1:
            pt = pt.reshape(-1, 1)
        dims, n_order = pt.shape
        if expected_dim > 0 and dims != expected_dim:
            raise ValueError("Mismatch in dimensions between endpoints")

        if n_order < self._endpoints_order:
            n_pad = self._endpoints_order - n_order
            pt = np.hstack([pt, np.zeros((dims, n_pad), dtype=np.float64)])
        elif n_order > self._endpoints_order:
            warnings.warn("Discarding derivatives with order > 3 at endpoints")
            pt = pt[:, 0 : self._endpoints_order]
        return pt

    def _process_waypoints(self, waypoints, waypoint_time, dims):
        waypoints = np.asarray(waypoints, dtype=np.float64)
        n_dims_wp, n_waypoints = waypoints.shape
        waypoint_time = np.asarray(waypoint_time, dtype=np.float64).squeeze()
        if waypoint_time.ndim > 1 or waypoint_time.size != n_waypoints:
            raise ValueError(
                "Time references must be a 1D list as long as number of waypoints"
            )
        if n_dims_wp != dims:
            raise ValueError("Mismatch in dimension between waypoints and endpoints")
        self._pos_ref = np.insert(self._pos_ref, [1], waypoints, axis=1)
        self._t_ref = np.insert(self._t_ref, 1, waypoint_time)

    def generate(
        self,
        init_point: ArrayLike,
        final_point: ArrayLike,
        t_span: ArrayLike,
        waypoints: Union[ArrayLike, None] = None,
        waypoint_time: Union[ArrayLike, None] = None,
    ):
        init_point = self._process_endpoints(init_point)
        dims = init_point.shape[0]
        final_point = self._process_endpoints(final_point, expected_dim=dims)
        self._pos_ref = np.column_stack([init_point[:, 0], final_point[:, 0]])

        t_span = np.asarray(t_span, dtype=np.float64).squeeze()
        if t_span.size != 2:
            raise ValueError(
                "Timespan must be two values corresponding to initial and final points"
            )
        self._t_ref = np.array(t_span)

        if waypoints is not None and waypoint_time is not None:
            self._process_waypoints(waypoints, waypoint_time, dims)
        elif not (waypoints is None and waypoint_time is None):
            raise ValueError(
                "Waypoints and times for waypoint traversal must be given together"
            )

        n_poly = self._pos_ref.shape[1] - 1
        n_vars = n_poly * self._n_cfs

        n_derivs = self._deriv_wts.size
        if n_derivs > self._degree:
            raise ValueError(
                "More derivatives than order of the polynomial are requested"
            )
        Q_all = np.zeros((n_derivs, n_vars, n_vars))
        for r, c_r in enumerate(self._deriv_wts):
            if c_r < self._iszero_tol:
                continue
            for i in range(n_poly):
                it, sent = i * self._n_cfs, (i + 1) * self._n_cfs
                q = self.compute_Q(r, self._t_ref[i : i + 2])
                Q_all[r, it:sent, it:sent] = q
        Q_all = np.sum(self._deriv_wts[:, None, None] * Q_all, axis=0).squeeze()

        Aeqs = {}
        beqs = {}
        polys = np.zeros((dims, self._n_cfs, n_poly))
        for d in range(dims):
            Aeqs["boundary"] = np.zeros((6, n_vars))
            beqs["boundary"] = np.zeros(6)

            for r in range(3):
                Aeqs["boundary"][r, 0 : self._n_cfs] = self.compute_tvec(
                    r, self._t_ref[0]
                )
                Aeqs["boundary"][r + 3, -self._n_cfs :] = self.compute_tvec(
                    r, self._t_ref[-1]
                )
            beqs["boundary"] = np.concatenate([init_point[d, :], final_point[d, :]])

            Aeqs["mid"] = np.zeros((n_poly - 1, n_vars))
            beqs["mid"] = np.zeros(n_poly - 1)
            for i in range(1, n_poly):
                it, sent = self._n_cfs * i, self._n_cfs * (1 + i)
                Aeqs["mid"][i - 1, it:sent] = self.compute_tvec(0, self._t_ref[i])
                beqs["mid"][i - 1] = self._pos_ref[d, i]

            Aeqs["cont"] = np.zeros(((n_poly - 1) * 3, n_vars))
            beqs["cont"] = np.zeros((n_poly - 1) * 3)
            for i in range(n_poly - 1):
                tvec_p = self.compute_tvec(0, self._t_ref[i + 1])
                tvec_v = self.compute_tvec(1, self._t_ref[i + 1])
                tvec_a = self.compute_tvec(2, self._t_ref[i + 1])
                itx, sentx = 3 * i, 3 * (i + 1)
                ity, senty = self._n_cfs * i, self._n_cfs * (i + 2)
                Aeqs["cont"][itx:sentx, ity:senty] = np.block(
                    [
                        [tvec_p, -tvec_p],
                        [tvec_v, -tvec_v],
                        [tvec_a, -tvec_a],
                    ]
                )

            Aeq = np.vstack([Aeqs["boundary"], Aeqs["mid"], Aeqs["cont"]])
            beq = np.concatenate([beqs["boundary"], beqs["mid"], beqs["cont"]])
            constr = optimize.LinearConstraint(Aeq, beq, beq)  # type: ignore
            soln = optimize.minimize(
                lambda x: (x @ Q_all @ x) / 2,
                np.zeros(n_vars),
                constraints=constr,
                method="trust-constr",
                jac=lambda x: Q_all @ x,
                hess=lambda _: Q_all,
            )
            polys[d, :, :] = np.reshape(soln.x, (self._n_cfs, n_poly), order="F")
        return PiecewisePolynomialTrajectory(self._t_ref, polys)

    def compute_Q(self, r, tspan):
        t1, t2 = tspan
        if t1 > t2:
            raise ValueError("Start time must be earlier than end time")
        Q = np.zeros((self._degree + 1, self._degree + 1))

        i = np.arange(r, self._degree + 1, dtype=np.int32)[None, :]
        l = i.T
        m_seq = np.arange(0, r)[None, None, :]
        k = i + l - 2 * r + 1
        Q[i, l] = (
            2.0
            * np.prod((i[..., None] - m_seq) * (l[..., None] - m_seq), axis=-1)
            * (t2**k - t1**k)
            / k
        )
        return Q

    def compute_tvec(self, r: int, t: float):
        tvec = np.zeros(self._degree + 1)
        n_seq = np.arange(r, self._degree + 1, dtype=np.int64)
        r_seq = np.arange(0, r, dtype=np.int64)
        tvec[n_seq] = np.prod(n_seq[None, :] - r_seq[:, None], axis=0) * t ** (
            n_seq - r
        )
        return tvec
