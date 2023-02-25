function [ref_traj_chunk, ref_u_chunk] = GetReferenceChunk(reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling)
%    Extracts the reference states and controls for the current MPC optimization given the over-sampled counterparts.
%
%    :param reference_traj: The reference trajectory, which has been finely over-sampled by a factor of
%    reference_over_sampling. It should be a vector of shape (Nx13), where N is the length of the trajectory in samples.
%    :param reference_u: The reference controls, following the same requirements as reference_traj. Should be a vector
%    of shape (Nx4).
%    :param current_idx: Current index of the trajectory tracking. Should be an integer number between 0 and N-1.
%    :param n_mpc_nodes: Number of MPC nodes considered in the optimization.
%    :param reference_over_sampling: The over-sampling factor of the reference trajectories. Should be a positive
%    integer.
%    :return: Returns the chunks of reference selected for the current MPC iteration. Two numpy arrays will be returned:
%        - An ((N+1)x13) array, corresponding to the reference trajectory. The first row is the state of current_idx.
%        - An (Nx4) array, corresponding to the reference controls.

% Dense references
traj_sent = min((current_idx + (n_mpc_nodes + 1) * reference_over_sampling - 1), size(reference_traj, 1));
ref_traj_chunk = reference_traj(current_idx:traj_sent, :);
u_sent = min((current_idx + n_mpc_nodes * reference_over_sampling - 1), size(reference_u, 1));
ref_u_chunk = reference_u(current_idx:u_sent, :);

% Indices for down-sampling the reference to number of MPC nodes
downsample_ref_ind = 1:reference_over_sampling:min(reference_over_sampling * (n_mpc_nodes + 1), size(ref_traj_chunk, 1));

% Sparser references (same dt as node separation)
ref_traj_chunk = ref_traj_chunk(downsample_ref_ind, :);
ref_u_chunk = ref_u_chunk(downsample_ref_ind(1:max(length(downsample_ref_ind) - 1, 1)), :);
end