function model = MakeQuadrotorModel(model_name, varargin)

import casadi.*

parser = inputParser();

addRequired(parser, 'model_name', @ischar);
addOptional(parser, 'constructor', @acados_ocp_model);

parse(parser, model_name, varargin{:});

p = MX.sym('p', 3, 1);
q = MX.sym('q', 4, 1);
v = MX.sym('v', 3, 1);

sym_x = vertcat(p, q, v);

f = MX.sym('f');
r = MX.sym('r', 3, 1);

sym_u = vertcat(f, r);

g = vertcat(0.0, 0.0, -9.81);
a_thrust = vertcat(0.0, 0.0, f);

f_expl = vertcat( ...
    QuaternionRotatePoint(QuaternionInverse(q), v), ...
    QuaternionProduct(vertcat(-r / 2, 0), q), ...
    -cross(r, v, 1) + QuaternionRotatePoint(q, g) + a_thrust...
);

model = parser.Results.constructor();
model.set('dyn_type', 'explicit');
model.set('dyn_expr_f', f_expl);
model.set('sym_x', sym_x);
model.set('sym_u', sym_u);
model.set('name', model_name);
end