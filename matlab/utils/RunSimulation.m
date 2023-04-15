function [tout, yout] = RunSimulation(ocp, trajectory, varargin)
%RUNSIMULATION Summary of this function goes here
%   Detailed explanation goes here
% Simulation integration step (the smaller the more "continuous"-like simulation.

p = inputParser;
p.addRequired('ocp', @(arg) isa(arg, 'acados_ocp'));
p.addRequired('trajectory', @ismatrix);
p.addOptional('update_fcn', @QuadrotorSO3Update);
p.addParameter('vehicle_mass', 1.0, @(arg) isscalar(arg) && arg >= 0);
p.addParameter('reference_over_sampling', 5, @(arg) isscalar(arg) && arg >= 0);
p.addParameter('simulation_dt', [], @(arg) isempty(arg) || (isscalar(arg) && arg > 0));

p.parse(ocp, trajectory, varargin{:});

update_fcn = p.Results.update_fcn;
reference_over_sampling = p.Results.reference_over_sampling;
simulation_dt = p.Results.simulation_dt;

n_mpc_nodes = ocp.opts_struct.param_scheme_N;
quad_current_state = trajectory.states(:, 1);

quad.mass = p.Results.vehicle_mass;
quad.x = quad_current_state;

t_length = length(trajectory.time);

yout.x = zeros(length(quad_current_state), t_length);
yout.u = zeros(4, t_length - 1);
tout = zeros(t_length, 1);
% Measure total simulation time
total_sim_time = 0.0;
yout.x(:, 1) = quad.x;

control_period = diff(trajectory.time);

for i = 1:t_length - 1
    [ref_traj_chunk, ref_u_chunk] = GetReferenceChunk(trajectory.states, ...
        trajectory.inputs, ...
        i, ...
        n_mpc_nodes, ...
        reference_over_sampling ...
        );
    
    ocp = SetReference(ocp, n_mpc_nodes, ref_traj_chunk, ref_u_chunk);
    [u_opt, ~] = Optimize(ocp, n_mpc_nodes, quad.x);

    ref_u = u_opt(:, 1);
    yout.u(:, i) = ref_u;
    simulation_time = 0.0;
    if ~isempty(simulation_dt)
        while simulation_time < control_period(i)
            simulation_time = simulation_time + simulation_dt;
            total_sim_time = total_sim_time + simulation_dt;
            quad = update_fcn(quad, ref_u, simulation_dt);
        end
    else
        quad = update_fcn(quad, ref_u, control_period(i));
        total_sim_time = total_sim_time + control_period(i);
    end
    tout(i + 1) = total_sim_time;
    yout.x(:, i + 1) = quad.x;
end
end
