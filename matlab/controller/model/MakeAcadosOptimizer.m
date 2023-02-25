function ocp = MakeAcadosOptimizer(t_horizon, n_nodes, q_cost, r_cost, bounds, model_name)

shooting_nodes = linspace(0, t_horizon, n_nodes + 1);
acados_model = MakeQuadrotorModel(model_name);
nx = size(acados_model.model_struct.sym_x, 1);
nu = size(acados_model.model_struct.sym_u, 1);
acados_model.set('T', t_horizon);

acados_model.set('cost_type', 'linear_ls');
acados_model.set('cost_type_e', 'linear_ls');

q_cost = reshape(q_cost, [], 1);
if length(q_cost) ~= nx
    error("Number of state weights does not match the state dimension 10");
end

r_cost = reshape(r_cost, [], 1);
if length(r_cost) ~= nu
    error("Number of input weights does not match the input dimension 4");
end
acados_model.set('cost_W', diag([q_cost; r_cost]));
acados_model.set('cost_W_e', diag(q_cost));

acados_model.set('cost_Vx', [eye(nx); zeros(nu, nx)]);
acados_model.set('cost_Vu', [zeros(nx, nu); eye(nu)]);
acados_model.set('cost_Vx_e', eye(nx));

% Initial reference trajectory (will be overwritten)
default_state = [0; 0; 0; 0; 0; 0; 1; zeros(3, 1)];
default_input = [9.81; 0; 0; 0];
acados_model.set('cost_y_ref', [default_state; default_input]);
acados_model.set('cost_y_ref_e', default_state);

% Initial state (will be overwritten)
acados_model.set('constr_type', 'bgh');
acados_model.set('constr_x0', default_state);

% Set constraints
if ~all(size(bounds) == [nu, 2])
    error('Input bounds must be specified as a sequence of (LB, UB) pairs convertible to a 4 x 2 array')
end
acados_model.set('constr_lbu', bounds(:, 1));
acados_model.set('constr_ubu', bounds(:, 2));
acados_model.set('constr_Jbu', eye(4));

acados_opts = acados_ocp_opts();
acados_opts.set('param_scheme_N', n_nodes);
acados_opts.set('shooting_nodes', shooting_nodes);
acados_opts.set('nlp_solver', 'sqp_rti');
acados_opts.set('sim_method', 'erk');
acados_opts.set('qp_solver', 'full_condensing_hpipm');

ocp = acados_ocp(acados_model, acados_opts);

end
