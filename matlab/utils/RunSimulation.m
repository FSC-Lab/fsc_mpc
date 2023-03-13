function [tout, yout] = RunSimulation(ocp, n_mpc_nodes, reference_over_sampling, quad, trajectory)
%RUNSIMULATION Summary of this function goes here
%   Detailed explanation goes here
% Simulation integration step (the smaller the more "continuous"-like simulation.
simulation_dt = 5e-4;

quad_current_state = trajectory.states(1, :);
quad.x = quad_current_state;
% my_quad.set_state(quad_current_state)

t_length = length(trajectory.time);
tout = zeros(t_length, 1);
yout.x = zeros(t_length, length(quad_current_state));
yout.u = zeros(t_length, 4);

% Measure total simulation time
total_sim_time = 0.0;

control_period = trajectory.time(2) - trajectory.time(1);

% update_fcn = @QuadrotorRK4Update;
update_fcn = @QuadrotorSO3Update;
for current_idx = 1:t_length

    quad_current_state = quad.x;

    yout.x(current_idx, :) = quad_current_state;
    [ref_traj_chunk, ref_u_chunk] = GetReferenceChunk(trajectory.states, ...
        trajectory.inputs, ...
        current_idx, ...
        n_mpc_nodes, ...
        reference_over_sampling ...
        );

    ocp = SetReference(ocp, n_mpc_nodes, ref_traj_chunk, ref_u_chunk);
    [u_opt, ~] = Optimize(ocp, n_mpc_nodes, quad_current_state);

    ref_u = u_opt(1, :);
    simulation_time = 0.0;
%     while simulation_time < control_period
%         simulation_time = simulation_time + simulation_dt;
%         total_sim_time = total_sim_time + simulation_dt;
    quad = QuadrotorSO3Update(quad, ref_u, control_period);
%     end
    total_sim_time = total_sim_time + control_period;
    tout(current_idx) = total_sim_time;
    yout.u(current_idx, :) = ref_u;
    
end
end

