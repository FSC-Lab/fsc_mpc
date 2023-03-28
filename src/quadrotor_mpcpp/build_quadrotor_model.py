#!/usr/bin/env python3

import argparse
from quadrotor_mpc import acados_wrapper


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "codegen_dst", type=str, help="Destination directory for C++ codegen"
    )
    parser.add_argument(
        "--name", type=str, default="codegen_model", help="Name of the model"
    )
    return parser.parse_args()


def main():
    args = parse_cli()
    t_horizon = 1.0
    n_nodes = 10
    model = acados_wrapper.make_quadrotor_model(args.name)
    _ = acados_wrapper.AcadosWrapper.make_new(
        t_horizon, n_nodes, model, codegen_dst=args.codegen_dst
    )


if __name__ == "__main__":
    main()
