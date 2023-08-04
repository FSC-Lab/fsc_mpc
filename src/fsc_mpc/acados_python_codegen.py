#!/usr/bin/env python3
"""
Python driver script to generate sources for a MPC solver using acados
Copyright © 2023 FSC Lab

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import argparse
import importlib
import pathlib
import sys
import warnings

import acados_template
import numpy as np


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "codegen_dst", type=pathlib.Path, help="Destination directory for C++ codegen"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fsc_mpc.model.quadrotor_model",
        help="The module to import the MPC definition from",
    )
    parser.add_argument(
        "--name", type=str, default="codegen_model", help="Name of the model"
    )
    return parser.parse_args()


def main():
    args = parse_cli()
    try:
        mdl = importlib.import_module(args.model)
        mdl = vars(mdl)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    for it in ["DIMS", "MODEL", "MPC_HORIZON", "MPC_NUM_NODES"]:
        if it not in mdl:
            print(
                f"Model definition does not contain required {it} attribute."
                " Codegen Failed",
                file=sys.stderr,
            )
            sys.exit(1)

    dims = mdl["DIMS"]
    model = mdl["MODEL"]

    for it in [
        "DEFAULT_STATE",
        "DEFAULT_INPUT",
        "DEFAULT_PARAM",
        "DEFAULT_Q_WEIGHTS",
        "DEFAULT_R_WEIGHTS",
        "DEFAULT_LBU",
        "DEFAULT_UBU",
    ]:
        if it not in mdl:
            warnings.warn(
                f"Model definition does not contain {it}. Defaults will be used"
            )

    codegen_dst = args.codegen_dst
    if codegen_dst.exists() and codegen_dst.is_file():
        print("Codegen destination can not be an existing file", file=sys.stderr)
        sys.exit(1)

    if not codegen_dst.exists():
        codegen_dst.mkdir(parents=True)

    json_file = codegen_dst / "acados_ocp_nlp.json"
    ocp = acados_template.AcadosOcp()
    model.name = args.name
    ocp.model = model

    ocp.code_export_directory = str(codegen_dst)

    ocp.parameter_values = np.asarray(mdl.get("DEFAULT_PARAM", np.empty(0)))

    ocp.dims.N = int(mdl["MPC_NUM_NODES"])
    ocp.solver_options.tf = float(mdl["MPC_HORIZON"])

    ocp.cost.cost_type = mdl.get("COST_TYPE", "LINEAR_LS")
    ocp.cost.cost_type_e = mdl.get("COST_TYPE_E", ocp.cost.cost_type)

    q_weights = np.asarray(mdl.get("DEFAULT_Q_WEIGHTS", np.ones(dims["x"])))
    r_weights = np.asarray(mdl.get("DEFAULT_R_WEIGHTS", np.ones(dims["u"])))
    ocp.cost.W = np.diag(np.concatenate([q_weights, r_weights]))
    ocp.cost.W_e = np.diag(q_weights)

    ocp.cost.Vx = np.row_stack([np.eye(dims["x"]), np.zeros((dims["u"], dims["x"]))])
    ocp.cost.Vu = np.row_stack([np.zeros((dims["x"], dims["u"])), np.eye(dims["u"])])
    ocp.cost.Vx_e = np.eye(dims["x"])

    default_state = np.asarray(mdl.get("DEFAULT_STATE", np.ones(dims["x"])))
    default_input = np.asarray(mdl.get("DEFAULT_INPUT", np.ones(dims["u"])))

    ocp.cost.yref = np.concatenate((default_state, default_input))
    ocp.cost.yref_e = np.asarray(default_state)

    # Initial state (will be overwritten)
    ocp.constraints.x0 = np.asarray(default_state)

    default_lbu = np.asarray(mdl.get("DEFAULT_LBU", np.full(dims["u"], -1000)))
    default_ubu = np.asarray(mdl.get("DEFAULT_UBU", np.full(dims["u"], 1000)))
    ocp.constraints.lbu = np.asarray(default_lbu)
    ocp.constraints.ubu = np.asarray(default_ubu)
    ocp.constraints.idxbu = np.arange(0, dims["u"])

    _ = acados_template.AcadosOcpSolver.generate(ocp, str(json_file))


if __name__ == "__main__":
    main()
