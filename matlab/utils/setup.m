function setup(varargin)
%SETUP Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;
p.addOptional("acados_root", getenv("ACADOS_SOURCE_DIR"), @isstring);
p.addParameter("build_dir", "", @isstring);
[varargin{:}] = convertCharsToStrings(varargin{:});
p.parse(varargin{:});

acados_root = p.Results.acados_root;
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

if startsWith(p.Results.build_dir, "/")
    build_dir = p.Results.build_dir;
else
    build_dir = strcat("/", p.Results.build_dir);
end

ld_run_path = getenv("LD_RUN_PATH");
link_dirs = [strcat(acados_root, "/lib"), strcat(pwd, build_dir)];
link_dirs = arrayfun(@(it) string(what(it).path), link_dirs);
for it = link_dirs
    if isempty(it)
        warning("Attempting to add empty directory to LD_RUN_PATH");
    end
    if contains(ld_run_path, it)
        fprintf("OK: %s is already in LD_RUN_PATH\n", it);
    elseif isempty(ld_run_path)
        ld_run_path = it;
    else
        ld_run_path = strcat(ld_run_path, ":", it);
    end
end
setenv("LD_RUN_PATH", ld_run_path);
