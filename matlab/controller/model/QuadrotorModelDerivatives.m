function dx = QuadrotorModelDerivatives(mass, x, u)
%QUADROTORMODELDERIVATIVES Summary of this function goes here
%   Detailed explanation goes here
attitude = x(4:7);
velocity = x(8:10);

thrust = u(1);
ang_vel = u(2:4);

GRAV_ACCEL = -9.81;

gravity_vector = [0; 0; GRAV_ACCEL];
thrust_vector = [0; 0; thrust / mass];

q_sq_nrm = attitude.' * attitude;

dx = [...
    velocity;
    QuaternionProduct(attitude, [ang_vel / 2; 0.0]) + (1.0 - q_sq_nrm) * attitude; ...
    QuaternionRotatePoint(attitude, thrust_vector) + gravity_vector];

end

