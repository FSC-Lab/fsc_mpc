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

import enum
import warnings
from typing import Union

import numpy as np
from fsc_mpc_py.trajectory_generator import PiecewisePolynomialTrajectory
from numpy.typing import ArrayLike
from scipy import optimize


class MinimumSnapAlgorithm(enum.Enum):
    CONSTRAINED = 0
    CLOSED_FORM = 1


class MinimumSnap:
    def __init__(
        self,
        degree: int,
        deriv_wts: ArrayLike,
        algorithm=MinimumSnapAlgorithm.CLOSED_FORM,
    ) -> None:
        self._degree = degree
        self._n_cfs = degree + 1
        self._n_poly = -1
        self._n_vars = -1
        self._dim = -1

        deriv_wts = np.asarray(deriv_wts, dtype=np.float64).squeeze()
        if deriv_wts.ndim > 1 or (deriv_wts < 0.0).any():
            raise ValueError(
                "Weights on derivatives must be a 1D list of nonnegative numbers"
            )
        self._deriv_wts = deriv_wts
        self._iszero_tol = 1e-8
        self._points_order = 3
        self._t_ref = np.array([])
        self._pos_ref = np.array([])
        self._algorithm = algorithm

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

    def _process_points(self, pt: ArrayLike):
        pt = np.asarray(pt, dtype=np.float64)
        if pt.ndim == 1:
            pt = pt.reshape(-1, 1)
        dim, n_order = pt.shape
        if self._dim < 0:
            self._dim = pt.shape[0]
        elif self._dim != dim:
            raise ValueError("Mismatch in dimensions between points")

        if n_order < self._points_order:
            n_pad = self._points_order - n_order
            pt = np.hstack([pt, np.zeros((self._dim, n_pad), dtype=np.float64)])
        elif n_order > self._points_order:
            warnings.warn("Discarding derivatives with order > 3 at points")
            pt = pt[:, 0 : self._points_order]
        return pt

    def _process_waypoints(self, waypoints, waypoint_time):
        waypoints = np.asarray(waypoints, dtype=np.float64)
        n_dims_wp, n_waypoints = waypoints.shape
        waypoint_time = np.asarray(waypoint_time, dtype=np.float64).squeeze()
        if waypoint_time.ndim > 1 or waypoint_time.size != n_waypoints:
            raise ValueError(
                "Time references must be a 1D list as long as number of waypoints"
            )
        if n_dims_wp != self._dim:
            raise ValueError("Mismatch in dimension between waypoints and points")
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
        init_point = self._process_points(init_point)
        final_point = self._process_points(final_point)
        self._pos_ref = np.column_stack([init_point[:, 0], final_point[:, 0]])

        t_span = np.asarray(t_span, dtype=np.float64).squeeze()
        if t_span.size != 2:
            raise ValueError(
                "Timespan must be two values corresponding to initial and final points"
            )
        self._t_ref = np.array(t_span)

        if waypoints is not None and waypoint_time is not None:
            self._process_waypoints(waypoints, waypoint_time)
        elif not (waypoints is None and waypoint_time is None):
            raise ValueError(
                "Waypoints and times for waypoint traversal must be given together"
            )

        self._n_poly = self._pos_ref.shape[1] - 1
        self._n_vars = self._n_poly * self._n_cfs

        n_derivs = self._deriv_wts.size
        if n_derivs > self._degree:
            raise ValueError(
                "More derivatives than order of the polynomial are requested"
            )
        Q_all = np.zeros((n_derivs, self._n_vars, self._n_vars))
        for r, c_r in enumerate(self._deriv_wts):
            if c_r < self._iszero_tol:
                continue
            for i in range(self._n_poly):
                it, sent = i * self._n_cfs, (i + 1) * self._n_cfs
                q = self.compute_Q(r, self._t_ref[i : i + 2])
                Q_all[r, it:sent, it:sent] = q
        Q_all = np.sum(self._deriv_wts[:, None, None] * Q_all, axis=0).squeeze()

        return (
            self._solve_constr(init_point, final_point, Q_all)
            if self._algorithm is MinimumSnapAlgorithm.CONSTRAINED
            else self._solve_unconstr(init_point, final_point, Q_all)
        )

    def _solve_unconstr(self, init_point, final_point, Q_all):
        polys = np.zeros((self._dim, self._n_cfs, self._n_poly))
        tk = self._t_ref[:, None] ** np.arange(0, self._n_cfs)
        for d in range(self._dim):
            # compute Tk   Tk(i,j) = ts(i)^(j-1)

            # compute A (n_cont*2*self._n_poly) * (self._n_cfs*self._n_poly)
            n_cont = 3
            # 1:p  2:pv  3:pva  4:pvaj  5:pvajs
            A = np.zeros((n_cont * 2 * self._n_poly, self._n_cfs * self._n_poly))
            for i in range(self._n_poly):
                for j in range(n_cont):
                    for k in range(j, self._n_cfs):
                        if k == j:
                            t1 = 1
                            t2 = 1
                        else:  # k>j
                            t1 = tk[i, k - j]
                            t2 = tk[i + 1, k - j]
                        A[n_cont * 2 * i + j, self._n_cfs * i + k] = (
                            np.prod(np.arange(k - j + 1, k + 1)) * t1
                        )
                        A[n_cont * 2 * i + n_cont + j, self._n_cfs * i + k] = (
                            np.prod(np.arange(k - j + 1, k + 1)) * t2
                        )

            # compute M
            M = np.zeros((self._n_poly * 2 * n_cont, n_cont * (self._n_poly + 1)))
            for i in range(self._n_poly * 2):
                j = np.floor((i + 1) / 2).astype(np.int64)
                rbeg = n_cont * i
                cbeg = n_cont * j
                M[rbeg : rbeg + n_cont, cbeg : cbeg + n_cont] = np.eye(n_cont)

            # compute C
            num_d = n_cont * (self._n_poly + 1)
            C = np.eye(num_d)
            df = np.concatenate(
                [self._pos_ref[d, :], init_point[d, 1:], final_point[d, 1:]]
            )
            # fix all pos(self._n_poly+1) + start va(2) +  va(2)
            fix_idx = np.concatenate(
                [np.arange(0, num_d, 3), [1, 2, num_d - 2, num_d - 1]]
            )
            free_idx = np.setdiff1d(np.arange(num_d), fix_idx)
            C = np.hstack([C[:, fix_idx], C[:, free_idx]])

            AiMC = np.linalg.solve(A, M @ C)
            R = AiMC.T @ Q_all @ AiMC

            n_fix = fix_idx.size
            # Rff = R[:n_fix, :n_fix]
            Rpp = R[n_fix:, n_fix:]
            Rfp = R[:n_fix, n_fix:]
            # Rpf = R[n_fix:, :n_fix]

            dp = -np.linalg.solve(Rpp, Rfp.T @ df)

            p = AiMC @ np.concatenate([df, dp])

            polys[d, :, :] = np.reshape(p, (self._n_cfs, self._n_poly), order="F")

        return PiecewisePolynomialTrajectory(self._t_ref, polys)

    def _solve_constr(self, init_point, final_point, Q_all):
        Aeqs = {}
        beqs = {}
        polys = np.zeros((self._dim, self._n_cfs, self._n_poly))
        for d in range(self._dim):
            Aeqs["boundary"] = np.zeros((6, self._n_vars))
            beqs["boundary"] = np.zeros(6)

            for r in range(3):
                Aeqs["boundary"][r, 0 : self._n_cfs] = self.compute_tvec(
                    r, self._t_ref[0]
                )
                Aeqs["boundary"][r + 3, -self._n_cfs :] = self.compute_tvec(
                    r, self._t_ref[-1]
                )
            beqs["boundary"] = np.concatenate([init_point[d, :], final_point[d, :]])

            Aeqs["mid"] = np.zeros((self._n_poly - 1, self._n_vars))
            beqs["mid"] = np.zeros(self._n_poly - 1)
            for i in range(1, self._n_poly):
                it, sent = self._n_cfs * i, self._n_cfs * (1 + i)
                Aeqs["mid"][i - 1, it:sent] = self.compute_tvec(0, self._t_ref[i])
                beqs["mid"][i - 1] = self._pos_ref[d, i]

            Aeqs["cont"] = np.zeros(((self._n_poly - 1) * 3, self._n_vars))
            beqs["cont"] = np.zeros((self._n_poly - 1) * 3)
            for i in range(self._n_poly - 1):
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
                np.zeros(self._n_vars),
                constraints=constr,
                method="trust-constr",
                jac=lambda x: Q_all @ x,
                hess=lambda _: Q_all,
            )
            polys[d, :, :] = np.reshape(soln.x, (self._n_cfs, self._n_poly), order="F")
        return PiecewisePolynomialTrajectory(self._t_ref, polys)

    def compute_Q(self, r, tspan):
        t1, t2 = tspan
        if t1 > t2:
            raise ValueError("Start time must be earlier than  time")
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
