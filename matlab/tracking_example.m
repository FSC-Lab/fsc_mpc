addpath controller/trajectories/ controller/model/ utils/

setup ~/src/acados build_dir build/uav;
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

% Number of MPC optimization nodes
n_mpc_nodes = 10;

% Recover some necessary variables from the MPC object
reference_over_sampling = 5;
control_period = t_horizon / (n_mpc_nodes * reference_over_sampling);

radius = 5;
altitude = 1;
acceleration = 1;
max_vel = 8;

trajectory(2) = struct('states', [], 'inputs', [], 'time', []);
for i = 1:numel(trajectory)
    switch i
        case 1
            trajectory(i) = StraightTrajectory(quad, [0; 0; 1.0], [150; 0; 1.0], control_period, acceleration, max_vel);
        case 2
            trajectory(i) = LemniscateTrajectory(quad, control_period, radius, altitude, acceleration, max_vel);
    end

    fprintf("\nRunning simulation...\n");
    [tout, yout] = RunSimulation(ocp, trajectory(i));
    PlotTrackingResults(trajectory(i), tout, yout);
end
