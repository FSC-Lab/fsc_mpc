function varargout = CheckTrajectory(trajectory, inputs, tvec, do_plot, frame)
    function res = allclose(a, b, atol, rtol)
        res = all(abs(a(:) - b(:)) <= atol + rtol * abs(b(:)), 'all');
    end
proj_V = [eye(3), zeros(3, 1)];

fprintf("Checking trajectory integrity...\n");

if frame == "W2B"
    trajectory(:, 4:6) = -trajectory(:, 4:6);
    for i = 1:size(trajectory, 1)
        trajectory(i, 8:10) = QuaternionRotatePoint(trajectory(i, 4:7), trajectory(i, 8:10));
    end
end

dt = gradient(tvec.').';
numeric_derivative = gradient(trajectory.').' ./ dt.';

errors = zeros(length(dt), 3);

num_bodyrates = zeros(length(dt), 3);

for i = 1:length(dt)
    % 1) check if velocity is consistent with position;
    numeric_velocity = numeric_derivative(i, 1:3);
    analytic_velocity = trajectory(i, 8:10);
    errors(i, 1) = norm(numeric_velocity - analytic_velocity);
    if ~allclose(analytic_velocity, numeric_velocity, 1e-2, 1e-2)
        error("inconsistent linear velocity\n [%f, %f, %f], [%f, %f, %f]\n", ...
            numeric_velocity, analytic_velocity);
    end

    % 2) check if attitude is consistent with acceleration;
    gravity = 9.81;
    numeric_thrust = numeric_derivative(i, 8:10) + [0.0, 0.0, gravity];
    numeric_thrust = numeric_thrust / norm(numeric_thrust);
    analytic_attitude = trajectory(i, 4:7);
    if abs(norm(analytic_attitude) - 1.0) > 1e-6
        error("quaternion does not have unit norm! [%f, %f, %f, %f], %f", ...
            analytic_attitude, norm(analytic_attitude));
    end

    e_z = [0.0, 0.0, 1.0];
    q_w = 1.0 + dot(e_z, numeric_thrust);
    q_xyz = cross(e_z, numeric_thrust);
    numeric_attitude = 0.5 * [q_xyz, q_w];
    numeric_attitude = numeric_attitude / norm(numeric_attitude);
    % the two attitudes can only differ in yaw --> check x,y component;
    q_diff = QuaternionProduct(QuaternionInverse(analytic_attitude.').', numeric_attitude);
    errors(i, 2) = norm(q_diff(1:2));
    if ~allclose(q_diff(1:2), zeros(2, 1), 1e-3, 1e-3)
        error("Attitude and acceleration do not match! [%f, %f, %f], [%f, %f, %f], [%f, %f, %f, %f],\n", ...
            analytic_attitude, numeric_attitude, q_diff);
    end

    % 3) check if bodyrates agree with attitude difference;
    numeric_bodyrates = 2.0 * QuaternionProduct(QuaternionInverse(trajectory(i, 4:7).').', numeric_derivative(i, 4:7));
    numeric_bodyrates = (proj_V * numeric_bodyrates).';
    num_bodyrates(i, :) = numeric_bodyrates;
    analytic_bodyrates = inputs(i, 2:4);
    errors(i, 3) = norm(numeric_bodyrates - analytic_bodyrates);
    if ~allclose(numeric_bodyrates, analytic_bodyrates, 0.05, 0.05)
        error("inconsistent angular velocity! [%f, %f, %f], [%f, %f, %f]", ...
            numeric_bodyrates, analytic_bodyrates);
    end

end
fprintf("Trajectory check successful\n");
fprintf("Maximum linear velocity error: %.5f\n", max(errors(:, 1)));
fprintf("Maximum attitude error: %.5f\n", max(errors(:, 2)));
fprintf("Maximum angular velocity error: %.5f\n", max(errors(:, 3)));

if do_plot
    f1 = figure();
    for i = 1:3
        ax = subplot(3, 2, sub2ind([2, 3], 1, i));
        ax.NextPlot = "add";
        plot(ax, tvec, numeric_derivative(:, i), "--r", "LineWidth", 2, "DisplayName", "numeric");
        plot(ax, tvec, trajectory(:, 7 + i), ":b", "LineWidth", 2, "DisplayName", "analytic");
        ylabel("m/s");
        if i == 1
            title(ax, "Velocity check");
        end
        legend(ax);
    end

    for i = 1:3
        ax = subplot(3, 2, sub2ind([2, 3], 2, i));
        ax.NextPlot = "add";
        plot(num_bodyrates(:, i), "--r", "LineWidth", 2, "DisplayName", "numeric");
        plot(inputs(:, 1 + i), ":b", "LineWidth", 2, "DisplayName", "analytic");
        ylabel("rad/s");
        if i == 1
            title(ax, "Body rate check");
        end
        legend(ax);
        sgtitle("Integrity check of reference trajectory");
    end
    varargout = {errors, f1};
else
    varargout = {errors};
end
end
