function [w_opt_acados, x_opt_acados] = Optimize(solver, N, initial_state)
%OPTIMIZE Summary of this function goes here
%   Detailed explanation goes here
if isempty(initial_state)
    initial_state = [zeros(3, 1); 0, 0, 0, 1, zeros(3, 1)];
end
% Set initial state. Add gp state if needed
x_init = reshape(initial_state, [], 1);

% Set initial condition, equality constraint
solver.set('constr_lbx', x_init, 0);
solver.set('constr_ubx', x_init, 0);

% Set parameters

% Solve OCP
solver.solve();

status = solver.get('status');
if status ~= 0
    error("Solver returned status %d\n", status);
end
% Get u
w_opt_acados = zeros(N, 4);
x_opt_acados = zeros(N + 1, length(x_init));
x_opt_acados(1, :) = solver.get('x', 0);
for i = (0:N - 1)
    %     disp(quad_mpc.solver.get('u', i));
    w_opt_acados(i + 1, :) = solver.get('u', i);
    x_opt_acados(i + 2, :) = solver.get('x', i + 1);
end
end
