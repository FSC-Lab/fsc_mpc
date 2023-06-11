function quad = QuadrotorRK4Update(quad, u, dt)
%MULTIROTORRK4UPDATE Summary of this function goes here
%   Detailed explanation goes here

% AERO_DRAG = 0.8;
% ROTOR_DRAG = [0.3; 0.3; 0.0];


x = reshape(quad.x, [], 1);
u = reshape(u, [], 1);

k1 = QuadrotorModelDerivatives(quad.mass, x, u);
x_aux = x + dt / 2 * k1;
k2 = QuadrotorModelDerivatives(quad.mass, x_aux, u);
x_aux = x + dt / 2 * k2;
k3 = QuadrotorModelDerivatives(quad.mass, x_aux, u);
x_aux = x + dt * k3;
k4 = QuadrotorModelDerivatives(quad.mass, x_aux, u);

x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

q_sq_nrm = x(4:7).' * x(4:7);

if q_sq_nrm < 1e-10
    error("Quaternion norm is zero");
end

x(4:7) = x(4:7) ./ sqrt(q_sq_nrm);

quad.x(:) = x;
end
