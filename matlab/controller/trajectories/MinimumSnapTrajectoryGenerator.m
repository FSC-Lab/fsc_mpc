function [traj_ref, u_ref, t_ref] = MinimumSnapTrajectoryGenerator(traj_derivatives, yaw_derivatives, t_ref, quad)
%     Follows the Minimum Snap Trajectory paper to generate a full trajectory given the position reference and its
%     derivatives, and the yaw trajectory and its derivatives.
%
%     :param traj_derivatives: array of shape 4x3xN. N corresponds to the length in samples of the trajectory, and:
%         - The 4 components of the first dimension correspond to position, velocity, acceleration and jerk.
%         - The 3 components of the second dimension correspond to x, y, z.
%     :param yaw_derivatives: array of shape 2xN. N corresponds to the length in samples of the trajectory. The first
%     row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
%     :param t_ref: vector of length N, containing the reference times (starting from 0) for the trajectory.
%     :param quad: Quadrotor3D object, corresponding to the quadrotor model that will track the generated reference.
%     :type quad: Quadrotor3D
%     :param map_limits: dictionary of map limits if available, None otherwise.
%     :param plot: True if show a plot of the generated trajectory.
%     :return: tuple of 3 arrays:
%         - Nx13 array of generated reference trajectory. The 13 dimension contains the components: position_xyz,
%         attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
%         - N array of reference timestamps. The same as in the input
%         - Nx4 array of reference controls, corresponding to the four motors of the quadrotor.
proj_V = [eye(3), zeros(3, 1)];
GRAV_ACCEL = 9.81;

discretization_dt = t_ref(2) - t_ref(1);
len_traj = size(traj_derivatives, 3);

% Add gravity to accelerations
thrust = squeeze(traj_derivatives(3, :, :)).' + repmat([0, 0, GRAV_ACCEL], len_traj, 1);
% Compute body axes
z_b = thrust ./ vecnorm(thrust, 2, 2);

yawing = ~isempty(yaw_derivatives);

rate = zeros(len_traj, 3);
f_t = quad.mass * sum(z_b .* thrust, 2);
if yawing
    error("Not implemented");
    %         % yaw is defined as the projection of the body-x axis on the horizontal plane
    %         x_c = [cos(yaw_derivatives(1, :)).', sin(yaw_derivatives(1, :)).', zeros(len_traj, 1)];
    %         y_b = cross(z_b, x_c)
    %         y_b = y_b / sqrt(sum(y_b.^2, 2))
    %         x_b = cross(y_b, z_b)
    %
    %         % Rotation matrix (from body to world)
    %         b_r_w = cat((x_b, :, newaxis), y_b(:, :, newaxis), z_b(:, :, newaxis)), -1
    %         , 3)
    %         q = []
    %         for i in range(len_traj):
    %             % Transform to quaternion
    %             q.append(rotation_matrix_to_quat(b_r_w[i]))
    %             if i > 1:
    %                 q[-1] = undo_quaternion_flip(q[-2], q[-1])
    %         q = stack(q)
    %
    %         % Compute angular rate vector
    %         % Total thrust acceleration must be equal to the projection of the quadrotor acceleration into the Z body axis
    %         a_proj = zeros(len_traj, 1)
    %
    %         for i in range(len_traj):
    %             a_proj[i, 0] = z_b[i].dot(traj_derivatives(3, :, i))
    %
    %         h_omega = quad.mass / f_t * (traj_derivatives(3, :, :).T - a_proj * z_b)
    %         for i in range(len_traj):
    %             rate[i, 0] = -h_omega[i].dot(y_b[i])
    %             rate[i, 1] = h_omega[i].dot(x_b[i])
    %             rate[i, 2] = -yaw_derivatives(1, i) * array((0, 0, 1)).dot(z_b[i])

else
    % new way to compute attitude:
    % https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
    e_z = [0.0, 0.0, 1.0];
    q_w = 1.0 + sum(e_z .* z_b, 2);
    q_xyz = cross(repmat(e_z, len_traj, 1), z_b, 2);
    q = 0.5 * [q_xyz, q_w];
    q = q ./ vecnorm(q, 2, 2);

    qinv = [-q(:, 1:3), q(:, 4)];
    % Use numerical differentiation of quaternions
    q_dot = gradient(q.').' / discretization_dt;
    w_int = zeros(len_traj, 3);
    for i = 1:len_traj
        w_int(i, :) = proj_V * 2.0 * QuaternionProduct(qinv(i, :), q_dot(i, :));
    end
    rate(:) = w_int;

    fprintf("Maximum yawrate before adaption: %.3f\n", max(abs(rate(:, 3))));
    q_new = q;
    yaw_corr_acc = 0.0;
    for i = 2:len_traj
        yaw_corr = -rate(i, 3) * discretization_dt;
        yaw_corr_acc = yaw_corr_acc + yaw_corr;
        q_corr = AngleAxisToQuaternion([0; 0; yaw_corr_acc]);
        q_new(i, :) = QuaternionProduct(q(i, :), q_corr);
        w_int(i, :) = proj_V * 2.0 * QuaternionProduct(qinv(i, :), q_dot(i, :));
    end
    q_new_dot = gradient(q_new.').' / discretization_dt;
    for i = 2:len_traj
        w_int(i, :) = proj_V * 2.0 * QuaternionProduct(QuaternionInverse(q_new(i, :).').', q_new_dot(i, :));
    end

    q = q_new;
    rate(:) = w_int;
    fprintf("Maximum yawrate after adaption: %.3f\n", max(abs(rate(:, 3))));

end
% Compute inputs
u_ref = [f_t, rate];

full_pos = squeeze(traj_derivatives(1, :, :)).';
full_vel = squeeze(traj_derivatives(2, :, :)).';

if quad.frame == "W2B"
    q(:, 1:3) = -q(:, 1:3);
    for i = 1:len_traj
        full_vel(i, :) = QuaternionRotatePoint(q(i, :), full_vel(i, :));
    end
end
traj_ref = [full_pos, q, full_vel];

% Locate starting point right at x=0 and y=0.
traj_ref(:, 1:2) = traj_ref(:, 1:2) - traj_ref(1, 1:2);

end