# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Dict, Optional

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from numpy.typing import ArrayLike

from quadrotor_mpc.quadrotor_model import DEFAULT_INPUT, DEFAULT_STATE, QuadrotorModel

DEFAULT_Q_COST = np.array(
    [10, 10, 10, 0.1, 0.1, 0.1, 0.0, 0.05, 0.05, 0.05], dtype=np.double
)

DEFAULT_R_COST = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.double)

DEFAULT_LB = np.array([0.0, -8.0, -8.0, -8.0], dtype=np.double)
DEFAULT_UB = np.array([80.0, 8.0, 8.0, 8.0], dtype=np.double)


class AcadosWrapperException(Exception):
    pass


def make_quadrotor_model(model_name: str, mass: float = 1.0):
    # Declare model variables
    x = cs.MX.sym("x", 10)  # type: ignore
    u = cs.MX.sym("u", 4)  # type: ignore

    model = QuadrotorModel(mass)
    f_expl = model.symbolic_derivatives(x, u)
    x_dot = cs.MX.sym("x_dot", f_expl.shape)  # type: ignore
    f_impl = x_dot - f_expl

    # Dynamics model
    acados_model = AcadosModel()
    acados_model.f_expl_expr = f_expl  # type: ignore
    acados_model.f_impl_expr = f_impl
    acados_model.x = x
    acados_model.xdot = x_dot
    acados_model.u = u
    acados_model.p = []
    acados_model.name = model_name  # type: ignore
    return acados_model


class AcadosWrapper:
    """
    A wrapper over AcadosOcpSolver that abstracts away some operations such as
    construction / restoring from file, setting references, and running optimization
    """

    def __init__(self, solver: AcadosOcpSolver):
        self._solver = solver
        self._n_nodes = int(self._solver.N)
        self._nx = np.size(solver.get(0, "x"))
        self._nu = np.size(solver.get(0, "u"))
        self._ny = self._nx + self._nu

    @classmethod
    def make_new(
        cls,
        t_horizon: float,
        n_nodes: int,
        model: AcadosModel,
        q_cost: ArrayLike = DEFAULT_Q_COST,
        r_cost: ArrayLike = DEFAULT_R_COST,
        lbu: ArrayLike = DEFAULT_LB,
        ubu: ArrayLike = DEFAULT_UB,
        solver_options: Optional[Dict[str, str]] = None,
        codegen_dir: Optional[str] = None,
        json_file: Optional[str] = None,
    ):
        cls._n_nodes = n_nodes

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model
        cls._nx = model.x.size()[0]  # type: ignore
        cls._nu = model.u.size()[0]  # type: ignore
        cls._ny = cls._nx + cls._nu
        ocp.dims.N = n_nodes
        ocp.solver_options.tf = t_horizon

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        q_cost = np.asarray(q_cost, dtype=np.double)
        if q_cost.size != cls._nx:
            raise AcadosWrapperException(
                "Number of state weights does not match the state dimension 10"
            )

        r_cost = np.asarray(r_cost, dtype=np.double)
        if r_cost.size != cls._nu:
            raise AcadosWrapperException(
                "Number of input weights does not match the input dimension 4"
            )
        ocp.cost.W = np.diag(np.concatenate((q_cost, r_cost)))
        ocp.cost.W_e = np.diag(q_cost)

        ocp.cost.Vx = np.zeros((cls._ny, cls._nx))
        ocp.cost.Vx[: cls._nx, : cls._nx] = np.eye(cls._nx)
        ocp.cost.Vu = np.zeros((cls._ny, cls._nu))
        ocp.cost.Vu[-4:, -4:] = np.eye(cls._nu)
        ocp.cost.Vx_e = np.eye(cls._nx)
        # Initial reference trajectory (will be overwritten)
        ocp.cost.yref = np.concatenate((DEFAULT_STATE, DEFAULT_INPUT))
        ocp.cost.yref_e = DEFAULT_STATE

        # Initial state (will be overwritten)
        ocp.constraints.x0 = DEFAULT_STATE

        # Set constraints
        lbu = np.asarray(lbu, dtype=np.double)
        ubu = np.asarray(ubu, dtype=np.double)
        if lbu.size != cls._nu or ubu.size != cls._nu:
            raise AcadosWrapperException(
                "Number of input bounds does not match the input dimension 4"
            )
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.r_[0:4]

        if codegen_dir is not None:
            ocp.code_export_directory = codegen_dir

        # Solver options
        if solver_options is not None:
            for k, v in solver_options.items():
                if isinstance(v, str):
                    v = v.upper()
                ocp.solver_options.set(k, v)

        solver_kw = {}
        if json_file is not None:
            solver_kw["json_file"] = json_file

        return cls(AcadosOcpSolver(ocp, build=True, generate=True, **solver_kw))

    @classmethod
    def restore_from_file(cls, json_file):
        return cls(AcadosOcpSolver(AcadosOcp(), json_file, build=False, generate=False))

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

        x_reference = np.asarray(x_reference)
        u_reference = np.asarray(u_reference)
        n_x_samples, nx = x_reference.shape
        n_u_samples, nu = u_reference.shape
        if n_x_samples not in (n_u_samples + 1, n_u_samples):
            raise AcadosWrapperException(
                f"Number of state ({n_x_samples}) and input ({n_u_samples}) references"
                " do not match"
            )

        # If not enough states in target sequence, append last state until required
        # length is met
        if n_x_samples < self.n_nodes + 1:
            x_reference_data, x_reference = x_reference.copy(), np.empty(
                (self.n_nodes + 1, nx)
            )
            u_reference_data, u_reference = u_reference.copy(), np.empty(
                (self.n_nodes, nu)
            )
            x_reference[:n_x_samples, :] = x_reference_data
            u_reference[:n_u_samples, :] = u_reference_data
            x_reference[n_x_samples:, :] = x_reference_data[-1, :]
            u_reference[n_u_samples:, :] = u_reference_data[-1, :]

        for j in range(self.n_nodes):
            ref = np.concatenate((x_reference[j, :], u_reference[j, :]))
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
        x_init = np.asarray(quad_current_state)

        # Set initial condition, equality constraint
        self._solver.set(0, "lbx", x_init)
        self._solver.set(0, "ubx", x_init)

        # Solve OCP
        self._solver.solve()

        # Get u
        w_opt_acados = np.empty((self.n_nodes, 4))
        x_opt_acados = np.empty((self.n_nodes + 1, len(x_init)))
        x_opt_acados[0, :] = self._solver.get(0, "x")
        for i in range(self.n_nodes):
            w_opt_acados[i, :] = self._solver.get(i, "u")
            x_opt_acados[i + 1, :] = self._solver.get(i + 1, "x")

        return w_opt_acados, x_opt_acados


def get_reference_chunk(
    reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling
):
    """
    Extracts the reference states and controls for the current MPC optimization given
    the over-sampled counterparts.

    :param reference_traj: The reference trajectory, which has been finely over-sampled
    by a factor of reference_over_sampling. It should be a vector of shape (Nx13), where
    N is the length of the trajectory in samples.
    :param reference_u: The reference controls, following the same requirements as
    reference_traj. Should be a vector of shape (Nx4).
    :param current_idx: Current index of the trajectory tracking. Should be an integer
    number between 0 and N-1.
    :param n_mpc_nodes: Number of MPC nodes considered in the optimization.
    :param reference_over_sampling: The over-sampling factor of the reference
    trajectories. Should be a positive integer.
    :return: Returns the chunks of reference selected for the current MPC iteration. Two
    numpy arrays will be returned: - An ((N+1)x13) array, corresponding to the reference
    trajectory. The first row is the state of current_idx.  - An (Nx4) array,
    corresponding to the reference controls.
    """

    # Dense references
    ref_traj_chunk = reference_traj[
        current_idx : current_idx + (n_mpc_nodes + 1) * reference_over_sampling, :
    ]
    ref_u_chunk = reference_u[
        current_idx : current_idx + n_mpc_nodes * reference_over_sampling, :
    ]

    # Indices for down-sampling the reference to number of MPC nodes
    downsample_ref_ind = np.arange(
        0,
        min(reference_over_sampling * (n_mpc_nodes + 1), ref_traj_chunk.shape[0]),
        reference_over_sampling,
        dtype=int,
    )

    # Sparser references (same dt as node separation)
    ref_traj_chunk = ref_traj_chunk[downsample_ref_ind, :]
    ref_u_chunk = ref_u_chunk[
        downsample_ref_ind[: max(len(downsample_ref_ind) - 1, 1)], :
    ]

    return ref_traj_chunk, ref_u_chunk
