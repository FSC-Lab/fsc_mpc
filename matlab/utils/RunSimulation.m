function [tout, yout] = RunSimulation(ocp, trajectory, update_fcn, options)
%RUNSIMULATION Summary of this function goes here
%   Detailed explanation goes here
% Simulation integration step (the smaller the more "continuous"-like simulation.

arguments
    ocp acados_ocp
    trajectory struct
    update_fcn = @QuadrotorSO3Update
    options.QuadMass {mustBePositive} = 1.0
    options.reference_over_sampling {mustBePositive} = 5
    options.simulation_dt double = []
end

reference_over_sampling = options.reference_over_sampling;
simulation_dt = options.simulation_dt;

n_mpc_nodes = ocp.opts_struct.param_scheme_N;
quad_current_state = trajectory.states(:, 1);

quad.mass = options.QuadMass;
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
