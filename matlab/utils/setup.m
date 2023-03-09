function setup(acados_root)
%SETUP Summary of this function goes here
%   Detailed explanation goes here

if nargin == 0
    acados_root = getenv("ACADOS_INSTALL_DIR");
end

if isempty(acados_root)
    error("Failed to get a valid acados installation location");
end

if ~isfolder(acados_root)
    error("Got acados installation location: %s, but it is not a valid folder");
end
setenv("ACADOS_INSTALL_DIR", acados_root);

acados_dirs = ["/external/casadi-matlab/", ...
    "/interfaces/acados_matlab_octave/", ...
    "/interfaces/acados_matlab_octave/acados_template_mex/"];
acados_dirs = arrayfun(@(it) strcat(acados_root, it), acados_dirs);
acados_dirs = arrayfun(@(it) string(what(it).path), acados_dirs);

curr_path = path;
for it = acados_dirs
    if contains(curr_path, it)
        fprintf("OK: %s is already on path\n", it);
    else
        addpath(it);
    end
end

ld_run_path = getenv("LD_RUN_PATH");
link_dirs = [strcat(acados_root, "/lib"), strcat(pwd, "/build")];
link_dirs = arrayfun(@(it) string(what(it).path), link_dirs);
for it = link_dirs
    if contains(ld_run_path, it)
        fprintf("OK: %s is already in LD_RUN_PATH\n", it);
    elseif isempty(ld_run_path)
        ld_run_path = it;
    else
        ld_run_path = strcat(ld_run_path, ":", it);
    end
end
setenv("LD_RUN_PATH", ld_run_path);
