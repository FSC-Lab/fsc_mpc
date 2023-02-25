function sim = MakeAcadosSimulator(t_horizon, n_nodes, model_name)
%MAKEACADOSSIMULATOR Summary of this function goes here
%   Detailed explanation goes here
h = t_horizon / n_nodes; % sampling time = length of first shooting interval
sim_model = MakeQuadrotorModel(model_name, @acados_sim_model);

sim_model.set('name', model_name);
sim_model.set('T', h);

% acados sim opts
sim_opts = acados_sim_opts();
sim_opts.set('method', 'erk');

sim = acados_sim(sim_model, sim_opts);
end

