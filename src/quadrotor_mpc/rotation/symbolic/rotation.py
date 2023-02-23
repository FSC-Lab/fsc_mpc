# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import casadi as cs


def fast_cross(a: cs.MX, b: cs.MX) -> cs.MX:
    return cs.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def quaternion_conjugate(q: cs.MX) -> cs.MX:
    res = cs.MX(q)
    res[0:3] = -q[0:3]
    return res


def quaternion_rotate_point(q: cs.MX, v: cs.MX) -> cs.MX:
    w = q[3]
    vec = q[0:3]
    uv = fast_cross(vec, v)
    uv += uv
    return v + w * uv + fast_cross(vec, uv)


def quaternion_product(a: cs.MX, b: cs.MX) -> cs.MX:
    res = cs.MX(4, 1)
    res[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
    res[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2]
    res[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0]
    res[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    return res
