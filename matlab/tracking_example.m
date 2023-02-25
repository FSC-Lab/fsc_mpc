addpath('controller/trajectories/');
addpath('controller/model/');
% 
warning('off', 'all');
if exist(sprintf('%s/build', pwd), 'dir')
    rmdir('build', 's');
end
q_cost = [10 * ones(3, 1); ones(3, 1); 0.0; 0.05*ones(3, 1)];
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

expected = load('lemniscate_trajectory.mat');
assert(all(ismembertol(expected.traj_ref, traj_ref), 'all'));
assert(all(ismembertol(expected.u_ref, u_ref), 'all'));
assert(all(ismembertol(expected.t_ref, t_ref), 'all'));
CheckTrajectory(traj_ref, u_ref, t_ref, false, quad.frame);

% Set quad initial state equal to the initial reference trajectory state
quad_current_state = traj_ref(1, :);
quad.x = quad_current_state;
% my_quad.set_state(quad_current_state)

ref_u = u_ref(1, :);
quad_trajectory = zeros(length(t_ref), length(quad_current_state));
u_optimized_seq = zeros(length(t_ref), 4);

% Sliding reference trajectory initial index
current_idx = 0;

% Measure the MPC optimization time
mean_opt_time = 0.0;

% Measure total simulation time
total_sim_time = 0.0;

fprintf("\nRunning simulation...\n");
last_prog = 0;
for current_idx = 1:size(traj_ref, 1)

    quad_current_state = quad.x;

    quad_trajectory(current_idx, :) = quad_current_state;
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
    u_optimized_seq(current_idx, :) = ref_u;
    

    prog = floor(10 * current_idx / size(traj_ref, 1));
    if prog > last_prog
        if current_idx > 1
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b');
        end
        fprintf('|%-10s|', repmat('*', 1, prog)); 
    end
    last_prog = prog;
end
fprintf('\n');