function quad = MultirotorRK4Update(quad, u, dt)
%MULTIROTORRK4UPDATE Summary of this function goes here
%   Detailed explanation goes here

% AERO_DRAG = 0.8;
% ROTOR_DRAG = [0.3; 0.3; 0.0];
GRAV = [0.0; 0.0; -9.81];

    function vel = f_pos(x)
        vel = QuaternionRotatePoint(QuaternionInverse(x(4:7)), x(8:10));
    end

    function quat_rate = f_att(x, u)
        rate_q = [-0.5 * u(2:4); 0.0];
        quaternion = x(4:7);
        quat_rate = QuaternionProduct(rate_q, quaternion);
    end

    function acc = f_vel(x, u, f_d)
        a_thrust = [0.0; 0.0; u(1)];
        g_b = QuaternionRotatePoint(x(4:7), GRAV);
        acc = -cross(u(2:4), x(8:10)) + a_thrust + f_d / quad.mass + g_b;
    end

k1 = zeros(10, 1);
k2 = zeros(10, 1);
k3 = zeros(10, 1);
k4 = zeros(10, 1);

x = reshape(quad.x, [], 1);
u = reshape(u, [], 1);
f_d = zeros(3, 1);
k1(1:3) = f_pos(x);
k1(4:7) = f_att(x, u);
k1(8:10) = f_vel(x, u, f_d);
x_aux = x + dt / 2 * k1;
k2(1:3) = f_pos(x_aux);
k2(4:7) = f_att(x_aux, u);
k2(8:10) = f_vel(x_aux, u, f_d);
x_aux = x + dt / 2 * k2;
k3(1:3) = f_pos(x_aux);
k3(4:7) = f_att(x_aux, u);
k3(8:10) = f_vel(x_aux, u, f_d);
x_aux = x + dt * k3;
k4(1:3) = f_pos(x_aux);
k4(4:7) = f_att(x_aux, u);
k4(8:10) = f_vel(x_aux, u, f_d);

x = x + dt * (1.0 / 6.0 * k1 + 2.0 / 6.0 * k2 + 2.0 / 6.0 * k3 + 1.0 / 6.0 * k4);

q_sq_nrm = x(4:7).' * x(4:7);

if q_sq_nrm < 1e-10
    error("Quaternion norm is zero");
end

x(4:7) = x(4:7) ./ sqrt(q_sq_nrm);

quad.x(:) = x;
end
