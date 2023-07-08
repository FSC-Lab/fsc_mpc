# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pathlib
from os import PathLike
from typing import Dict, Optional, Union

import casadi as cs
import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from numpy.typing import ArrayLike

from .models import symbolic_quadrotor

import dataclasses


class AcadosWrapperException(Exception):
    pass


def make_quadrotor_model(
    model_name: str, mass: Union[float, cs.MX] = cs.MX.sym("mass")
):
    # Declare model variables
    x = cs.MX.sym("x", 10)  # type: ignore
    u = cs.MX.sym("u", 4)  # type: ignore

    model = symbolic_quadrotor.SymbolicQuadrotor(mass)
    f_expl = model.model_derivatives(x, u)
    x_dot = cs.MX.sym("x_dot", f_expl.shape)  # type: ignore
    f_impl = x_dot - f_expl

    # Dynamics model
    acados_model = AcadosModel()
    acados_model.f_expl_expr = f_expl  # type: ignore
    acados_model.f_impl_expr = f_impl
    acados_model.x = x
    acados_model.xdot = x_dot
    acados_model.u = u
    if isinstance(mass, cs.MX):
        acados_model.p = mass  # type: ignore
    acados_model.name = model_name  # type: ignore
    return acados_model


@dataclasses.dataclass
class AcadosWrapperParams:
    t_horizon: float
    n_nodes: int
    q_cost: np.ndarray
    r_cost: np.ndarray
    lbu: np.ndarray
    ubu: np.ndarray
    solver_options: Optional[Dict[str, str]]


class AcadosWrapper:
    """
    A wrapper over AcadosOcpSolver that abstracts away some operations such as
    construction / restoring from file, setting references, and running optimization
    """

    def __init__(
        self,
        model: AcadosModel,
        params: AcadosWrapperParams,
        codegen_dst: Union[str, PathLike] = "lib",
        clean_first: bool = False,
    ):
        self._n_nodes = params.n_nodes

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model
        self._nx = cs.MX(model.x).size(1)
        self._nu = cs.MX(model.u).size(1)
        if isinstance(model.p, cs.MX) and model.p.size(1):
            ocp.parameter_values = np.zeros(model.p.size(1))
        self._ny = self._nx + self._nu

        codegen_dst = pathlib.Path(codegen_dst)

        if codegen_dst.exists() and codegen_dst.is_file():
            raise FileExistsError("Codegen destination can not be an existing file")

        if not codegen_dst.exists():
            pathlib.Path.mkdir(codegen_dst, parents=True)

        json_file = codegen_dst / "acados_ocp_nlp.json"

        if json_file.exists() and not clean_first:
            ocp = AcadosOcp()
            build = False
            generate = False
        else:
            build = True
            generate = True
            # Solver options
            ocp.dims.N = params.n_nodes
            ocp.solver_options.tf = params.t_horizon

            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"
            q_cost, r_cost = params.q_cost, params.r_cost
            if q_cost.size != self._nx:
                raise AcadosWrapperException(
                    f"Number of state weights does not match the state dimension {self._nx}"
                )

            if r_cost.size != self._nu:
                raise AcadosWrapperException(
                    f"Number of input weights does not match the input dimension {self._nu}"
                )
            ocp.cost.W = np.diag(np.r_[q_cost, r_cost])
            ocp.cost.W_e = np.diag(q_cost)

            ocp.cost.Vx = np.r_[np.eye(self._nx), np.zeros((self._nu, self._nx))]
            ocp.cost.Vu = np.r_[np.zeros((self._nx, self._nu)), np.eye(self._nu)]
            ocp.cost.Vx_e = np.eye(self._nx)
            # Initial reference trajectory (will be overwritten)
            ocp.cost.yref = np.concatenate(
                (symbolic_quadrotor.DEFAULT_STATE, symbolic_quadrotor.DEFAULT_INPUT)
            )
            ocp.cost.yref_e = symbolic_quadrotor.DEFAULT_STATE

            # Initial state (will be overwritten)
            ocp.constraints.x0 = symbolic_quadrotor.DEFAULT_STATE

            # Set constraints
            lbu, ubu = params.lbu, params.ubu
            if lbu.size != self._nu or ubu.size != self._nu:
                raise AcadosWrapperException(
                    "Number of input bounds does not match the input dimension 4"
                )
            ocp.constraints.lbu = lbu
            ocp.constraints.ubu = ubu
            ocp.constraints.idxbu = np.r_[0:4]
            ocp.code_export_directory = str(codegen_dst)
            if params.solver_options is not None:
                for k, v in params.solver_options.items():
                    if isinstance(v, str):
                        v = v.upper()
                    ocp.solver_options.set(k, v)

        self._solver = AcadosOcpSolver(
            ocp, str(json_file), build=build, generate=generate
        )

        self._n_nodes = int(self._solver.N)
        self._nx = np.size(self._solver.get(0, "x"))
        self._nu = np.size(self._solver.get(0, "u"))
        self._ny = self._nx + self._nu

    def reset(self, reset_qp_solver_mem=1):
        self._solver.reset(reset_qp_solver_mem=reset_qp_solver_mem)

    def set_costs(self, q_cost: ArrayLike, r_cost: ArrayLike):
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

    def set_reference_trajectory(
        self,
        x_reference: ArrayLike,
        u_reference: ArrayLike,
    ) -> None:
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
        n_x_samples = x_reference.shape[1]
        n_u_samples = u_reference.shape[1]
        if n_x_samples not in (n_u_samples + 1, n_u_samples):
            raise AcadosWrapperException(
                f"Number of state ({n_x_samples}) and input ({n_u_samples}) references"
                " do not match"
            )

        # If not enough states in target sequence, append last state until required
        # length is met
        if n_x_samples < self.n_nodes + 1:
            x_reference = np.pad(
                x_reference, ((0, 0), (0, self.n_nodes + 1 - n_x_samples)), "edge"
            )
            u_reference = np.pad(
                u_reference, ((0, 0), (0, self.n_nodes - n_u_samples)), "edge"
            )

        ref = np.empty(self._ny)
        for j in range(self.n_nodes):
            ref[0 : self._nx] = x_reference[:, j]
            ref[self._nx :] = u_reference[:, j]
            self._solver.set(j, "yref", ref)
        # the last MPC node has only a state reference but no input reference
        self._solver.set(self.n_nodes, "yref", x_reference[:, self.n_nodes])

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
