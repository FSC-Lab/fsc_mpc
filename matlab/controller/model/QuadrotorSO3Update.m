function quad = QuadrotorSO3Update(quad, u, dt)
%MULTIROTORRK4UPDATE Summary of this function goes here
%   Detailed explanation goes here

% AERO_DRAG = 0.8;
% ROTOR_DRAG = [0.3; 0.3; 0.0];

x = reshape(quad.x, [], 1);
u = reshape(u, [], 1);
GRAV_ACCEL = 9.81;

p = x(1:3);
q = x(4:7);
v = x(8:10);

f = u(1);
w = u(2:4);

thrust = [0; 0; f / quad.mass];
g = [0; 0; -GRAV_ACCEL];

quad.x(:) = [p + dt .* v; ...
    QuaternionProduct(q, AngleAxisToQuaternion(dt .* w)); ...
    v + dt .* (QuaternionRotatePoint(q, thrust) + g)];
end
