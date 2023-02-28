# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def trajectory_tracking_results(
    t_ref,
    x_ref,
    x_executed,
    u_ref,
    u_executed,
    title,
):
    legend_labels = ["reference", "simulated"]

    fig3d, ax3d = plt.subplots(subplot_kw=dict(projection="3d"))

    fig, ax = plt.subplots(3, 3, sharex="all", figsize=(7, 9))
    ax = np.asarray(ax, dtype=plt.Axes)

    ax3d.plot(
        x_executed[:, 0],
        x_executed[:, 1],
        x_executed[:, 2],
        "b",
        linewidth=2,
        label="Actual",
    )
    ax3d.plot(
        x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], "--r", linewidth=2, label="Reference"
    )
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")  # type: ignore

    ax3d.set_zlim(0, np.mean(x_ref[:, 2]) * 1.5)  # type: ignore

    for i, label in enumerate("xyz"):
        ax[i, 0].plot(t_ref, x_executed[:, i], "b", linewidth=2, label=legend_labels[1])
        ax[i, 0].plot(t_ref, x_ref[:, i], "--r", linewidth=2, label=legend_labels[0])
        ax[i, 0].legend()
        ax[i, 0].set_ylabel(label)
    ax[0, 0].set_title(r"$p\:[m]$")
    ax[2, 0].set_xlabel(r"$t [s]$")

    q_euler = R.from_quat(x_executed[:, 3:7]).as_euler("XYZ")

    for i in range(3):
        ax[i, 1].plot(t_ref, q_euler[:, i], label=legend_labels[1])
    ref_euler = R.from_quat(x_ref[:, 3:7]).as_euler("XYZ")
    q_err = (
        R.from_quat(x_executed[:, 3:7]) * R.from_quat(x_ref[:, 3:7]).inv()
    ).as_euler("XYZ")

    for i in range(3):
        ax[i, 1].plot(t_ref, ref_euler[:, i], label=legend_labels[0])
        ax[i, 1].plot(t_ref, q_err[:, i], label="quat error")
    for i in range(3):
        ax[i, 1].legend()
    ax[0, 1].set_title(r"$\theta\:[rad]$")
    ax[2, 1].set_xlabel(r"$t [s]$")

    for i in range(3):
        ax[i, 2].plot(t_ref, x_executed[:, i + 7], label=legend_labels[1])
        ax[i, 2].plot(t_ref, x_ref[:, i + 7], label=legend_labels[0])
        ax[i, 2].legend()
    ax[0, 2].set_title(r"$v\:[m/s]$")
    ax[2, 2].set_xlabel(r"$t [s]$")

    plt.suptitle(title)

    if u_ref is not None and u_executed is not None:
        ax = plt.subplots(1, 4, sharex="all", sharey="all")[1]
        for i in range(4):
            ax[i].plot(t_ref, u_ref[:, i], label="ref")
            ax[i].plot(t_ref, u_executed[:, i], label="simulated")
            ax[i].set_xlabel(r"$t [s]$")
            tit = "Control %d" % (i + 1)
            ax[i].set_title(tit)
            ax[i].legend()
    plt.show()

    return fig, fig3d
