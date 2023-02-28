addpath('controller/trajectories/');
addpath('controller/model/');
%
warning('off', 'all');
if exist(sprintf('%s/build', pwd), 'dir')
    rmdir('build', 's');
end
q_cost = [10 * ones(3, 1); ones(3, 1); 0.0; 0.05 * ones(3, 1)];
r_cost = 0.1 * ones(4, 1);
bounds = [0, 80; -8 * ones(3, 1), 8 * ones(3, 1)];
ocp = MakeAcadosOptimizer(1.0, 10, q_cost, r_cost, bounds, 'uav');

quad = struct('mass', 1, ...
    "frame", "W2B", ...
    "x", []);
% Simulation integration step (the smaller the more "continuous"-like simulation.
t_horizon = 1.0;
% Simulation integration step (the smaller the more "continuous"-like simulation.
simulation_dt = 5e-4;

% Number of MPC optimization nodes
n_mpc_nodes = 10;

% Recover some necessary variables from the MPC object
reference_over_sampling = 5;
control_period = t_horizon / (n_mpc_nodes * reference_over_sampling);

trajectory = struct( ...
    'radius', 5, ...
    'altitude', 1, ...
    'acceleration', 1, ...
    'max_vel', 8 ...
    );
trajectory_args = struct2cell(trajectory);

[traj_ref, u_ref, t_ref] = LemniscateTrajectory(quad, control_period, trajectory_args{:});

CheckTrajectory(traj_ref, u_ref, t_ref, false, quad.frame);

% Set quad initial state equal to the initial reference trajectory state
quad_current_state = traj_ref(1, :);
quad.x = quad_current_state;
% my_quad.set_state(quad_current_state)

ref_u = u_ref(1, :);
yout.t = zeros(length(t_ref), 1);
yout.x = zeros(length(t_ref), length(quad_current_state));
yout.u = zeros(length(t_ref), 4);

% Sliding reference trajectory initial index
current_idx = 0;

% Measure the MPC optimization time
mean_opt_time = 0.0;

% Measure total simulation time
total_sim_time = 0.0;

fprintf("\nRunning simulation...\n");
for current_idx = 1:size(traj_ref, 1)

    quad_current_state = quad.x;

    yout.x(current_idx, :) = quad_current_state;
    [ref_traj_chunk, ref_u_chunk] = GetReferenceChunk(traj_ref, ...
        u_ref, ...
        current_idx, ...
        n_mpc_nodes, ...
        reference_over_sampling ...
        );

    ocp = SetReference(ocp, n_mpc_nodes, ref_traj_chunk, ref_u_chunk);
    [u_opt, ~] = Optimize(ocp, n_mpc_nodes, quad_current_state);

    ref_u = u_opt(1, :);
    simulation_time = 0.0;
    while simulation_time < control_period
        simulation_time = simulation_time + simulation_dt;
        total_sim_time = total_sim_time + simulation_dt;
        quad = QuadrotorRK4Update(quad, ref_u, simulation_dt);
    end
    yout.t(current_idx) = total_sim_time;
    yout.u(current_idx, :) = ref_u;
    
end
fprintf('\n');
f1 = figure();
ax = gca;
view(ax, 3);
ax.NextPlot = 'add';
plot3(traj_ref(:, 1), traj_ref(:, 2), traj_ref(:, 3), '--r', 'LineWidth', 2, 'DisplayName', 'Reference');
plot3(yout.x(:, 1), yout.x(:, 2), yout.x(:, 3), 'b', 'LineWidth', 2, 'DisplayName', 'Executed');
zlim(ax, [0.0, 2.0]);
xlabel(ax, 'x (m)');
ylabel(ax, 'Y (m)');
zlabel(ax, 'Z (m)');
legend(ax);

f2 = figure('Position', [10, 10, 1024, 768]);

pos_err = traj_ref(:, 1:3) - yout.x(:, 1:3);

traj_len = size(yout.x, 1);
att_err = zeros(traj_len, 3);
for j = 1:traj_len
    att_err(j, :) = QuaternionToAngleAxis(...
        QuaternionProduct(traj_ref(j, 4:7).', ...
            QuaternionInverse(yout.x(j, 4:7).'))...
        );
end
vel_err = traj_ref(:, 8:10) - yout.x(:, 8:10);

label_opts = {'Interpreter', 'latex'};
for i = 1:3
ax = subplot(3, 3, sub2ind([3, 3], 1, i));

abs_pos_err = abs(pos_err(:, i));
mae_pos = mean(abs_pos_err);
plot(ax, yout.t, abs_pos_err, 'DisplayName', 'Absolute Position Error');
yline(ax, mae_pos, 'DisplayName', 'Mean Absolute Error');
xlabel(ax, 'Time (s)', label_opts{:});
ylabel(ax, 'Position (m)', label_opts{:});
legend(ax);

ax = subplot(3, 3, sub2ind([3, 3], 2, i));
abs_att_err = abs(att_err(:, i));
mae_att = mean(abs_att_err);
plot(ax, yout.t, abs_att_err, 'DisplayName', 'Absolute Attitude Error');
yline(ax, mae_att, 'DisplayName', 'Mean Attitude Error');
xlabel(ax, 'Time (s)', label_opts{:});
ylabel(ax, 'Angle ($s^{-1}$)', label_opts{:});
legend(ax);

ax = subplot(3, 3, sub2ind([3, 3], 3, i));
abs_vel_err = abs(vel_err(:, i));
mae_vel = mean(abs_vel_err);
plot(ax, yout.t, abs_vel_err, 'DisplayName', 'Absulte Velocity Error');
yline(ax, mae_vel, 'DisplayName', 'Mean Absolute Error');
xlabel(ax, 'Time (s)', label_opts{:});
ylabel(ax, 'Velocity (m/s)', label_opts{:});
legend(ax);
end
