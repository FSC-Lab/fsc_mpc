# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Any, Dict, Union

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import CMakeBuilder
from numpy.typing import ArrayLike

from quadrotor_mpc.quadrotor_model import DEFAULT_INPUT, DEFAULT_STATE, QuadrotorModel
from quadrotor_mpc.rotation.symbolic import (
    quaternion_conjugate,
    quaternion_product,
    quaternion_rotate_point,
)

DEFAULT_Q_COST = np.array(
    [10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05], dtype=np.double
)

DEFAULT_R_COST = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.double)

DEFAULT_U_BOUNDS = np.array([[0, 80], [-8, 8], [-8, 8], [-8, 8]])


class AcadosWrapperException(Exception):
    pass


def make_quadrotor_model(model_name: str):
    # Declare model variables
    p = cs.MX.sym("p", 3)  # type: ignore   position
    q = cs.MX.sym("a", 4)  # type: ignore   angle quaternion (wxyz)
    v = cs.MX.sym("v", 3)  # type: ignore   velocity

    x = cs.vertcat(p, q, v)  # Full state vector (10-dimensional)

    f = cs.MX.sym("f")  # type: ignore      thrust force
    r = cs.MX.sym("r", 3)  # type: ignore   angle rate

    u = cs.vertcat(f, r)  # Control input vector (4-dimensional)

    g = cs.vertcat(0.0, 0.0, -9.81)  # Gravity vector in world frame
    a_thrust = cs.vertcat(0.0, 0.0, f)  # Thrust vector in body frame

    f_expl = cs.vertcat(
        quaternion_rotate_point(quaternion_conjugate(q), v),  # Position
        1 / 2 * quaternion_product(cs.vertcat(-r, 0), q),  # Attitude
        -cs.cross(r, v, 1) + quaternion_rotate_point(q, g) + a_thrust,  # Velocity
    )

    x_dot = cs.MX.sym("x_dot", f_expl.shape)  # type: ignore
    f_impl = x_dot - f_expl

    # Dynamics model
    acados_model = AcadosModel()
    acados_model.f_expl_expr = f_expl
    acados_model.f_impl_expr = f_impl
    acados_model.x = x
    acados_model.xdot = x_dot
    acados_model.u = u
    acados_model.p = []
    acados_model.name = model_name  # type: ignore
    return acados_model


def make_acados_optimizer_from_config(config: Dict[str, Any]):
    model_name = str(config.get("model_name", "my_quad"))
    t_horizon = config["T"]
    n_nodes = config["param_scheme_N"]

    q_cost = np.asarray(config["cost_Q_weights"])
    # Add one more weight to the rotation (use quaternion norm weighting in acados)
    if q_cost.size == QuadrotorModel.NX - 1:
        q_cost = np.concatenate(
            (q_cost[:3], np.atleast_1d(q_cost[3:6].mean()), q_cost[3:])
        )
    try:
        q_mask = np.asarray(config["cost_Q_mask"], dtype=np.double)
        q_mask = np.concatenate((q_mask[:3], np.array([0.0]), q_mask[3:]))
        q_cost *= q_mask
    except KeyError:
        pass

    r_cost = np.asarray(config["cost_R_weights"])
    solver_options = config["opts"]

    bounds = np.column_stack(
        (np.asarray(config["constr_lbu"]), np.asarray(config["constr_ubu"]))
    )
    solver_kw = dict()
    try:
        if bool(config["use_cmake"]):
            builder = CMakeBuilder()
            builder.build_dir = config["cmake_build_dir"]
            builder.options_on = config["cmake_options_on"]
            builder.generator = config["cmake_generator"]
            solver_kw["cmake_builder"] = builder
    except KeyError:
        pass

    codegen_dir = None
    try:
        codegen_dir = str(config["codegen_dir"])
    except KeyError:
        pass

    return make_acados_optimizer(
        t_horizon,
        n_nodes,
        q_cost,
        r_cost,
        bounds,
        model_name,
        solver_options,
        solver_kw,
        codegen_dir,
    )


def make_acados_optimizer(
    t_horizon,
    n_nodes,
    q_cost=DEFAULT_Q_COST,
    r_cost=DEFAULT_R_COST,
    bounds=DEFAULT_U_BOUNDS,
    model_name="my_quad",
    solver_options: Union[None, Dict[str, str]] = None,
    solver_kw: Union[None, Dict[str, Any]] = None,
    codegen_dir: Union[str, None] = None,
):
    acados_model = make_quadrotor_model(model_name)
    nx = acados_model.x.size()[0]  # type: ignore
    nu = acados_model.u.size()[0]  # type: ignore
    ny = nx + nu
    n_param = acados_model.p.size()[0] if isinstance(acados_model.p, cs.MX) else 0

    # Create OCP object to formulate the optimization
    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.dims.N = n_nodes
    ocp.solver_options.tf = t_horizon

    # Initialize parameters
    ocp.dims.np = n_param
    ocp.parameter_values = np.zeros(n_param)

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    q_cost = np.asarray(q_cost)
    if q_cost.size != QuadrotorModel.NX:
        raise AcadosWrapperException(
            "Number of state weights does not match the state dimension 10"
        )

    r_cost = np.asarray(r_cost)
    if r_cost.size != QuadrotorModel.NU:
        raise AcadosWrapperException(
            "Number of input weights does not match the input dimension 4"
        )
    ocp.cost.W = np.diag(np.concatenate((q_cost, r_cost)))
    ocp.cost.W_e = np.diag(q_cost)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[-4:, -4:] = np.eye(nu)
    ocp.cost.Vx_e = np.eye(nx)

    # Initial reference trajectory (will be overwritten)
    ocp.cost.yref = np.concatenate((DEFAULT_STATE, DEFAULT_INPUT))
    ocp.cost.yref_e = DEFAULT_STATE

    # Initial state (will be overwritten)
    ocp.constraints.x0 = DEFAULT_STATE

    # Set constraints
    if bounds.shape != (QuadrotorModel.NU, 2):
        raise AcadosWrapperException(
            "Input bounds must be specified as a sequence of (LB, UB) pairs convertible"
            " to a 4 x  array"
        )
    ocp.constraints.lbu = bounds[:, 0]
    ocp.constraints.ubu = bounds[:, 1]
    ocp.constraints.idxbu = np.r_[0:4]

    if codegen_dir is not None:
        ocp.code_export_directory = codegen_dir

    # Solver options
    if solver_options is not None:
        for k, v in solver_options.items():
            if isinstance(v, str):
                v = v.upper()
            ocp.solver_options.set(k, v)
    if solver_kw is not None:
        return AcadosOcpSolver(ocp, **solver_kw)

    return AcadosOcpSolver(ocp)


def set_reference_trajectory(
    acados_ocp_solver: AcadosOcpSolver,
    N: int,
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
        The state reference, stacked row-wise for each shooting node. If there are fewer
        states than shooting nodes, i.e. x_reference has fewer rows than N + 1, then it
        will be padded by the last state
    u_reference : ArrayLike
        The input reference, stacked row-wise for each shooting node. The number of
        inputs must match the number of states or one less than the number of reference
        states

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
            f"Number of state ({n_x_samples}) and input ({n_u_samples}) references do"
            " not match"
        )

    # If not enough states in target sequence, append last state until required length
    # is met
    if n_x_samples < N + 1:
        x_reference_data, x_reference = x_reference.copy(), np.empty((N + 1, nx))
        u_reference_data, u_reference = u_reference.copy(), np.empty((N, nu))
        x_reference[:n_x_samples, :] = x_reference_data
        u_reference[:n_u_samples, :] = u_reference_data
        x_reference[n_x_samples:, :] = x_reference_data[-1, :]
        u_reference[n_u_samples:, :] = u_reference_data[-1, :]

    for j in range(N):
        ref = np.concatenate((x_reference[j, :], u_reference[j, :]))
        acados_ocp_solver.set(j, "yref", ref)
    # the last MPC node has only a state reference but no input reference
    acados_ocp_solver.set(N, "yref", x_reference[N, :])


def set_reference_state(acados_ocp_solver, N, x_reference, u_reference):
    ref = np.concatenate((np.asarray(x_reference), np.asarray(u_reference)))

    for j in range(N):
        acados_ocp_solver.set(j, "yref", ref)
    acados_ocp_solver.set(N, "yref", ref[:-4])

    return True


def optimize(acados_ocp_solver, N, quad_current_state):
    # Set initial state. Add gp state if needed
    x_init = np.asarray(quad_current_state)

    # Set initial condition, equality constraint
    acados_ocp_solver.set(0, "lbx", x_init)
    acados_ocp_solver.set(0, "ubx", x_init)

    # Solve OCP
    acados_ocp_solver.solve()

    # Get u
    w_opt_acados = np.empty((N, 4))
    x_opt_acados = np.empty((N + 1, len(x_init)))
    x_opt_acados[0, :] = acados_ocp_solver.get(0, "x")
    for i in range(N):
        w_opt_acados[i, :] = acados_ocp_solver.get(i, "u")
        x_opt_acados[i + 1, :] = acados_ocp_solver.get(i + 1, "x")

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
