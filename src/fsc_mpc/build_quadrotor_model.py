#!/usr/bin/env python3

import argparse

import acados_wrapper


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "codegen_dst", type=str, help="Destination directory for C++ codegen"
    )
    parser.add_argument(
        "--name", type=str, default="codegen_model", help="Name of the model"
    )
    parser.add_argument(
        "--horizon", type=float, default=1.0, help="Prediction horizon of the MPC"
    )
    parser.add_argument(
        "--n_nodes", type=int, default=10, help="Number of shooting nodes of the MPC"
    )
    return parser.parse_args()


def main():
    args = parse_cli()
    params = {
        "t_horizon": args.horizon,
        "n_nodes": args.n_nodes,
        "q_cost": [10, 10, 10, 1, 1, 1, 0, 0.1, 0.1, 0.1],
        "r_cost": [0.1, 0.1, 0.1, 0.1],
    }
    model = acados_wrapper.make_quadrotor_model(args.name)
    _ = acados_wrapper.AcadosWrapper(model, params, codegen_dst=args.codegen_dst)


if __name__ == "__main__":
    main()
