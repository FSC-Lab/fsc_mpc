"""
Simple and default implementation of the MPC for quadrotors
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

import acados_template
import casadi


def _quaternion_rotate_point(q, v):
    w = q[3]
    vec = q[0:3]
    uv = casadi.cross(vec, v, 1)
    uv += uv
    return v + w * uv + casadi.cross(vec, uv)


def _quaternion_product(a, b):
    res = casadi.MX(4, 1)
    res[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
    res[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2]
    res[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0]
    res[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    return res


ATTITUDE = slice(3, 7)
VELOCITY = slice(7, 10)

THROTTLE = 0
RATES = slice(1, 4)

MASS = 0

GRAV_ACCEL = -9.81


def model_derivatives(syms):
    q = syms["x"][ATTITUDE]
    v_b = syms["x"][VELOCITY]

    f = syms["u"][THROTTLE]
    w = syms["u"][RATES]

    mass = syms["p"][MASS]

    grav_vector = casadi.DM([0, 0, GRAV_ACCEL])

    augmented_w = casadi.vertcat(w / 2, 0)
    a_thrust = casadi.vertcat(0.0, 0.0, f / mass)

    # Return the core equations of motion
    return casadi.vertcat(
        v_b,
        _quaternion_product(q, augmented_w),
        _quaternion_rotate_point(q, a_thrust) + grav_vector,
    )


MPC_HORIZON = 2.0
MPC_NUM_NODES = 20

DEFAULT_STATE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
DEFAULT_INPUT = [9.81, 0.0, 0.0, 0.0]
DEFAULT_PARAM = [1]

DEFAULT_Q_WEIGHTS = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 0.0, 0.1, 0.1, 0.1]
DEFAULT_R_WEIGHTS = [0.1, 0.1, 0.1, 0.1]

DEFAULT_LBU = [0.0, -8.0, -8.0, -8.0]
DEFAULT_UBU = [80.0, 8.0, 8.0, 8.0]

SYM_NAMES = ["x", "xdot", "u", "p"]
SYM_DIMS = [10, 10, 4, 1]

DIMS = dict(zip(SYM_NAMES, SYM_DIMS))

_syms = {k: casadi.MX.sym(k, v, 1) for k, v in DIMS.items()}  # type: ignore

MODEL = acados_template.AcadosModel()
MODEL.f_expl_expr = model_derivatives(_syms)

for k in DIMS.keys():
    setattr(MODEL, k, _syms[k])
