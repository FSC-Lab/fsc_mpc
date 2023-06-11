function model = MakeQuadrotorModel(varargin)

import casadi.*

p = inputParser;
p.addOptional('model_name', 'uav');

p.parse(varargin{:});


sym_x = MX.sym('x', 10, 1);
sym_u = MX.sym('u', 4, 1);
sym_p = MX.sym('m');

f_expl = QuadrotorModelDerivatives(sym_p, sym_x, sym_u);

model = acados_ocp_model();
model.set('dyn_type', 'explicit');
model.set('dyn_expr_f', f_expl);
model.set('sym_x', sym_x);
model.set('sym_u', sym_u);
model.set('sym_p', sym_p)
model.set('name', p.Results.model_name);
end