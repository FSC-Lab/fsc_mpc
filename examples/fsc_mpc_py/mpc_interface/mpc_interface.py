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

import pathlib

import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver


class MPCInterfaceException(Exception):
    pass


class MPCInterface:
    """
    A wrapper over AcadosOcpSolver that abstracts away some operations such as
    construction / restoring from file, setting references, and running optimization
    """

    def __init__(
        self,
        codegen_dst="lib",
    ):
        codegen_dst = pathlib.Path(codegen_dst)

        if codegen_dst.exists() and codegen_dst.is_file():
            raise FileExistsError("Codegen destination can not be an existing file")

        json_file = codegen_dst / "acados_ocp_nlp.json"

        ocp = AcadosOcp()
        if not json_file.exists():
            raise MPCInterfaceException(
                "This wrapper requires a prebuilt acados solver"
            )
        build = False
        generate = False
        self._solver = AcadosOcpSolver(
            ocp, str(json_file), build=build, generate=generate
        )

        self._n_nodes = int(self._solver.N)
        self._nx = np.size(self._solver.get(0, "x"))
        self._nu = np.size(self._solver.get(0, "u"))
        self._ny = self._nx + self._nu

    def reset(self, reset_qp_solver_mem=1):
        self._solver.reset(reset_qp_solver_mem=reset_qp_solver_mem)

    def set_costs(self, q_cost, r_cost):
        q_cost = np.asarray(q_cost, dtype=np.float64)
        r_cost = np.asarray(r_cost, dtype=np.float64)

        if q_cost.ndim == 1 and r_cost.ndim == 1:
            cost_mat = np.diag(np.concatenate((q_cost, r_cost)))
            terminal_cost_mat = np.diag(q_cost)
        elif q_cost.ndim == 2 and r_cost.ndim == 2:
            cost_mat = scipy.linalg.blkdiag(q_cost, r_cost)
            terminal_cost_mat = q_cost
        else:
            raise ValueError(
                "q_cost and r_cost must both be 1D array of weights or 2D cost matrices"
            )

        for idx in range(self.n_nodes):
            self._solver.cost_set(idx, "W", cost_mat)
        self._solver.cost_set(self.n_nodes, "W", terminal_cost_mat)

    def set_constant_parameter(self, value):
        value = np.asarray(value)
        for it in range(self.n_nodes):
            self._solver.set(it, "p", value)

    @property
    def n_nodes(self):
        return self._n_nodes

    def set_reference_trajectory(self, x_reference, u_reference):
        """Sets a target trajectory for the Optimal Control Problem solver

        Parameters
        ----------
        acados_ocp_solver : AcadosOcpSolver
            An instance of the Acados Optimal Control Problem Solver
        N : int
            Number of shooting nodes
        x_reference : ArrayLike
            The state reference, stacked row-wise for each shooting node. If there are
            fewer states than shooting nodes, i.e. x_reference has fewer rows than N +
            1, then it will be padded by the last state
        u_reference : ArrayLike
            The input reference, stacked row-wise for each shooting node. The number of
            inputs must match the number of states or one less than the number of
            reference states

        Raises
        ------
        AcadosException
            When the number of input references does not match the number of state
            references
        """

        x_reference = np.asarray(x_reference, dtype=np.float64)
        u_reference = np.asarray(u_reference, dtype=np.float64)
        n_x_samples = x_reference.shape[0]
        n_u_samples = u_reference.shape[0]
        if n_x_samples not in (n_u_samples + 1, n_u_samples):
            raise MPCInterfaceException(
                f"Number of state ({n_x_samples}) and input ({n_u_samples}) references"
                " do not match"
            )

        # If not enough states in target sequence, append last state until required
        # length is met
        if n_x_samples < self.n_nodes + 1:
            x_reference = np.pad(
                x_reference, ((0, self.n_nodes + 1 - n_x_samples), (0, 0)), "edge"
            )
            u_reference = np.pad(
                u_reference, ((0, self.n_nodes - n_u_samples), (0, 0)), "edge"
            )

        for j in range(self.n_nodes):
            ref = np.concatenate([x_reference[j, :], u_reference[j, :]])
            self._solver.set(j, "yref", ref)
        # the last MPC node has only a state reference but no input reference
        self._solver.set(self.n_nodes, "yref", x_reference[self.n_nodes, :])

    def set_reference_state(self, x_reference, u_reference):
        ref = np.concatenate((np.asarray(x_reference), np.asarray(u_reference)))

        for j in range(self.n_nodes):
            self._solver.set(j, "yref", ref)
        self._solver.set(self.n_nodes, "yref", ref[:-4])

        return True

    def optimize(self, quad_current_state):
        # Set initial state. Add gp state if needed
        x_init = np.asarray(quad_current_state, dtype=np.float64)

        # Solve OCP
        self._solver.solve_for_x0(x_init)

        # Get u
        w_opt_acados = np.empty((4, self.n_nodes))
        x_opt_acados = np.empty((len(x_init), self.n_nodes + 1))
        x_opt_acados[:, 0] = self._solver.get(0, "x")
        for i in range(self.n_nodes):
            w_opt_acados[:, i] = self._solver.get(i, "u")
            x_opt_acados[:, i + 1] = self._solver.get(i + 1, "x")

        return w_opt_acados, x_opt_acados
