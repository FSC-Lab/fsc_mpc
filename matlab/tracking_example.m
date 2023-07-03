addpath controller/model/ utils/

setup ~/src/acados;

path_adjusted = false;
if ~path_adjusted
    py.sys.path().append("../src/quadrotor_mpc");
    path_adjusted = true;
end

trajectory_generator = py.importlib.import_module("trajectory_generator");
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
            begin_pos = py.numpy.array([0.0; 0.0; 1.0]);
            end_pos = py.numpy.array([150.0; 0.0; 1.0]);
            trajectory{i} = trajectory_generator.straight_trajectory(begin_pos, end_pos, control_period, acceleration, max_vel);
        case 2
            radius = 5;
            altitude = 1;
            trajectory{i} = trajectory_generator.lemniscate_trajectory(control_period, radius, altitude, acceleration, max_vel, 1.0);
    end
end

for i = 1:numel(trajectory)
    fprintf("\nRunning simulation...\n");
    [tout, yout] = RunSimulation(ocp, trajectory{i}, @QuadrotorRK4Update);
    PlotTrackingResults(trajectory{i}, tout, yout);
end
