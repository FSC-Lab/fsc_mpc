function h = PlotTrackingResults(trajectory, tout, yout)

states = double(trajectory.states);
f1 = figure();
ax = gca;
view(ax, 3);
ax.NextPlot = 'add';
plot3(states(1, :), states(2, :), states(3, :), '--r', 'LineWidth', 2, 'DisplayName', 'Reference');
plot3(yout.x(1, :), yout.x(2, :), yout.x(3, :), 'b', 'LineWidth', 2, 'DisplayName', 'Executed');
zlim(ax, [0.0, 2.0]);
xlabel(ax, 'x (m)');
ylabel(ax, 'Y (m)');
zlabel(ax, 'Z (m)');
legend(ax);

f2 = figure('Position', [10, 10, 1024, 768]);

pos_err = states(1:3, :) - yout.x(1:3, :);

traj_len = size(yout.x, 2);
att_err = zeros(3, traj_len);
for j = 1:traj_len
    att_err(:, j) = QuaternionToAngleAxis(...
        QuaternionProduct(states(4:7, j), ...
            QuaternionInverse(yout.x(4:7, j)))...
        );
end
vel_err = states(8:10, :) - yout.x(8:10, :);

label_opts = {'Interpreter', 'latex'};
for i = 1:3
ax = subplot(3, 3, sub2ind([3, 3], 1, i));

abs_pos_err = abs(pos_err(i, :));
mae_pos = mean(abs_pos_err);
plot(ax, tout, abs_pos_err, 'DisplayName', 'Absolute Position Error');
yline(ax, mae_pos, 'DisplayName', 'Mean Absolute Error');
xlabel(ax, 'Time (s)', label_opts{:});
ylabel(ax, 'Position (m)', label_opts{:});
legend(ax);

ax = subplot(3, 3, sub2ind([3, 3], 2, i));
abs_att_err = abs(att_err(i, :));
mae_att = mean(abs_att_err);
plot(ax, tout, abs_att_err, 'DisplayName', 'Absolute Attitude Error');
yline(ax, mae_att, 'DisplayName', 'Mean Attitude Error');
xlabel(ax, 'Time (s)', label_opts{:});
ylabel(ax, 'Angle ($s^{-1}$)', label_opts{:});
legend(ax);

ax = subplot(3, 3, sub2ind([3, 3], 3, i));
abs_vel_err = abs(vel_err(i, :));
mae_vel = mean(abs_vel_err);
plot(ax, tout, abs_vel_err, 'DisplayName', 'Absulte Velocity Error');
yline(ax, mae_vel, 'DisplayName', 'Mean Absolute Error');
xlabel(ax, 'Time (s)', label_opts{:});
ylabel(ax, 'Velocity (m/s)', label_opts{:});
legend(ax);
end

h = {f1, f2};
end
