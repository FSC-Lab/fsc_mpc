function trajectory = StraightTrajectory(quad, traj_begin, traj_end, z, discretization_dt, lin_acc, v_max)
    position_diff = traj_end - traj_begin;
    distance = norm(position_diff);
    heading_vector = position_diff / distance;

    ramp_up_t = v_max / lin_acc;

    % Calculate simulation time to achieve desired maximum velocity with specified
    % acceleration
    t_total = distance / v_max;

    refs = struct();
    a = struct();
    refs.ramp_up = 0:discretization_dt:(ramp_up_t - discretization_dt);
    a.ramp_up = lin_acc * ones(size(refs.ramp_up));
    t_cruise = t_total - 2 * ramp_up_t;
    refs.cruise = refs.ramp_up(end) + (0:discretization_dt:t_cruise);

    a.cruise = zeros(size(refs.cruise));
    refs.ramp_down = refs.cruise(end) + (0:discretization_dt:ramp_up_t) + discretization_dt;

    a.ramp_down = -lin_acc * ones(size(refs.ramp_down));
    t_ref = [refs.ramp_up, refs.cruise, refs.ramp_down];
    a_vec = [a.ramp_up, a.cruise, a.ramp_down];
    v_vec = cumsum(a_vec) * discretization_dt;
    d_vec = cumsum(v_vec) * discretization_dt;

    n = length(t_ref);
    traj = zeros(4, 3, n);
    traj(1, 1:2, :) = traj_begin + heading_vector .* d_vec;
    traj(1, 3, :) = z;
    traj(2, 1:2, :) = heading_vector * v_vec;
    traj(3, 1:2, :) = heading_vector * a_vec;
    yaw = zeros(2, n);
    yaw(1, :) = atan2(position_diff(2), position_diff(1));

    [traj_ref, u_ref, t_ref] = MinimumSnapTrajectoryGenerator(traj, yaw, t_ref, quad);
    trajectory = struct('states', traj_ref, 'inputs', u_ref, 'time', t_ref);
end
