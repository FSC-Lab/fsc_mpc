function solver = SetReference(solver, N, x_reference, u_reference)
%SETSTATE Summary of this function goes here
%   Detailed explanation goes here

n_x_samples = size(x_reference, 2);
n_u_samples = size(u_reference, 2);

if n_x_samples == 1
    % WIP
else

    if n_x_samples < N + 1
        x_reference = [x_reference, ...
            repmat(x_reference(:, end), 1, N + 1 - n_x_samples)];

        u_reference = [u_reference, ...
            repmat(u_reference(:, end), 1, N - n_u_samples)];
    end

    % Determine which dynamics model to use based on the GP optimal input feature region
    for j = 0:(N - 1)
        ref = [x_reference(:, j + 1); u_reference(:, j + 1)];
        ref = reshape(ref, [], 1);
        solver.set('cost_y_ref', ref, j);
    end
    % the last MPC node has only a state reference but no input reference
    solver.set('cost_y_ref_e', reshape(x_reference(:, N), [], 1));
end
end