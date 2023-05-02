function trajectory = LemniscateTrajectory(discretization_dt, radius, alt, lin_acc, v_max, options)

arguments
    discretization_dt double
    radius double
    alt double
    lin_acc double
    v_max double
    options.QuadMass double = 1.0
    options.UseBodyFrameDynamics logical = true
end

% Apply map limits to radius
ramp_up_t = 2; % s

% Calculate simulation time to achieve desired
% maximum velocity with specified acceleration
t_total = 2 * v_max / lin_acc + 2 * ramp_up_t;

% Transform to angular acceleration
alpha_acc = lin_acc / radius; % rad/s^2

% Generate time and angular acceleration sequences
% Ramp up sequence
ramp_t_vec = 0:discretization_dt:ramp_up_t - discretization_dt;
ramp_up_alpha = alpha_acc * sin(pi / (2 * ramp_up_t) * ramp_t_vec).^2;
% Acceleration phase
coasting_duration = (t_total - 4 * ramp_up_t) / 2;
coasting_t_vec = ramp_up_t + (0:discretization_dt:(coasting_duration - discretization_dt));
coasting_alpha = ones(size(coasting_t_vec)) * alpha_acc;
% Transition phase: decelerate
transition_t_vec = (0:discretization_dt:(2 * ramp_up_t - discretization_dt));
transition_alpha = alpha_acc * cos(pi / (2 * ramp_up_t) * transition_t_vec);
transition_t_vec = transition_t_vec + (coasting_t_vec(end) + discretization_dt);
% Deceleration phase
down_coasting_t_vec = transition_t_vec(end) + (0:discretization_dt:(coasting_duration - discretization_dt)) + discretization_dt;
down_coasting_alpha = -ones(size(down_coasting_t_vec)) * alpha_acc;
% Bring to rest phase
ramp_up_t_vec = (down_coasting_t_vec(end) + (0:discretization_dt:(ramp_up_t - discretization_dt))) + discretization_dt;

ramp_up_alpha_end = ramp_up_alpha - alpha_acc;

% Concatenate all sequences
t_ref = [ramp_t_vec, coasting_t_vec, transition_t_vec, down_coasting_t_vec, ramp_up_t_vec];
alpha_vec = [ramp_up_alpha, coasting_alpha, transition_alpha, down_coasting_alpha, ramp_up_alpha_end];

% Compute angular integrals
w_vec = cumsum(alpha_vec) * discretization_dt;
angle_vec = cumsum(w_vec) * discretization_dt;

% Adaption: we achieve the highest spikes in the bodyrates when passing through the 'center' part of the figure-8
% This leads to negative reference thrusts.
% Let's see if we can alleviate this by adapting the z-reference in these parts to add some acceleration in the
% z-component
z_dim = 0.0;

% Compute position, velocity, acceleration, jerk
traj = zeros(4, 3, length(angle_vec));

ca = cos(angle_vec);
sa = sin(angle_vec);
c4a = cos(4.0 * angle_vec);
s4a = sin(4.0 * angle_vec);

% position block
traj(1, 1, :) = radius * ca;
traj(1, 2, :) = radius * (sa .* ca);
traj(1, 3, :) = -z_dim * c4a + alt;

% velocity block
traj(2, 1, :) = -radius * (w_vec .* sa);
traj(2, 2, :) = radius * (w_vec .* ca.^2 - w_vec .* sa.^2);
traj(2, 3, :) = 4.0 * z_dim * w_vec .* s4a;

% acceleration block
traj(3, 1, :) = -radius * (alpha_vec .* sa + w_vec.^2 .* ca);
traj(3, 2, :) = radius * ( ...
    alpha_vec .* ca.^2 ...
    -2.0 * w_vec.^2 .* ca .* sa ...
    -alpha_vec .* sa.^2 ...
    -2.0 * w_vec.^2 .* sa .* ca ...
    );
traj(3, 3, :) = 16.0 * z_dim * (w_vec.^2 .* c4a + alpha_vec .* s4a);

yaw = [];

options = namedargs2cell(options);
[traj_ref, u_ref, t_ref] = MinimumSnapTrajectoryGenerator(traj, yaw, t_ref, options{:});
trajectory = struct('states', traj_ref, 'inputs', u_ref, 'time', t_ref);
end
