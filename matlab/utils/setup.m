function setup(varargin)
%SETUP Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;
p.addOptional("acados_root", getenv("ACADOS_SOURCE_DIR"), @isstring);
p.addParameter("build_dir", "build", @isstring);
[varargin{:}] = convertCharsToStrings(varargin{:});
p.parse(varargin{:});

acados_root = p.Results.acados_root;
setenv("ACADOS_INSTALL_DIR", acados_root);

acados_dirs = ["/external/casadi-matlab/", ...
    "/interfaces/acados_matlab_octave/", ...
    "/interfaces/acados_matlab_octave/acados_template_mex/"];

curr_path = path;
for i = 1:numel(acados_dirs)
    acados_dir_path = fullfile(acados_root, acados_dirs(i));
    if ~exist(acados_dir_path, "dir")
        error("%s does not exist!", acados_dir_path);
    end    
    if contains(curr_path, acados_dir_path)
        fprintf("OK: %s is already on path\n", acados_dir_path);
    else
        addpath(acados_dir_path);
    end
end

build_dir = p.Results.build_dir;
ld_run_path = getenv("LD_RUN_PATH");
link_dirs = [fullfile(acados_root, "lib"), fullfile(pwd, build_dir)];
link_dirs = arrayfun(@(it) string(what(it).path), link_dirs);
for it = link_dirs
    if it == ""
        error("Attempting to add empty directory to LD_RUN_PATH");
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
