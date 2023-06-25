addpath controller/trajectories/ controller/model/ utils/

setup ~/src/acados;
%
warning('off', 'all');

q_cost = [10 * ones(3, 1); ones(3, 1); 0.0; 0.05 * ones(3, 1)];
r_cost = 0.1 * ones(4, 1);
bounds = [0, 80; -8 * ones(3, 1), 8 * ones(3, 1)];

model = MakeQuadrotorModel();
ocp = MakeAcadosOptimizer(model, 1.0, 10, q_cost, r_cost, bounds);
ocp.set('p', 1.0);
quad = struct('mass', 1, "x", []);
% Simulation integration step (the smaller the more "continuous"-like simulation.
t_horizon = 1.0;

% Number of MPC optimization nodes
n_mpc_nodes = 10;

% Recover some necessary variables from the MPC object
reference_over_sampling = 5;
control_period = t_horizon / (n_mpc_nodes * reference_over_sampling);

trajectory = cell([2, 1]);

acceleration = 1;
max_vel = 8;

for i = 1:2
    switch i
        case 1
            trajectory{i} = StraightTrajectory([0; 0; 1.0], [150; 0; 1.0], control_period, acceleration, max_vel);
        case 2
            radius = 5;
            altitude = 1;
            trajectory{i} = LemniscateTrajectory(control_period, radius, altitude, acceleration, max_vel);
    end
end

for i = 1:numel(trajectory)
    fprintf("\nRunning simulation...\n");
    [tout, yout] = RunSimulation(ocp, trajectory{i}, @QuadrotorRK4Update);
    PlotTrackingResults(trajectory{i}, tout, yout);
end
