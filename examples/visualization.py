# Copyright (c) 2023 hs293go
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from quadrotor_mpc.rotation import (
    quaternion_conjugate,
    quaternion_product,
    quaternion_to_angle_axis,
)


def trajectory_tracking_results(
    t_ref,
    x_ref,
    x_executed,
    u_ref,
    u_executed,
    title,
):
    legend_labels = ["reference", "simulated"]

    with_ref = True if x_ref is not None else False

    fig3d, ax3d = plt.subplots(subplot_kw=dict(projection="3d"))

    fig, ax = plt.subplots(3, 3, sharex="all", figsize=(7, 9))
    ax = np.asarray(ax, dtype=plt.Axes)

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ax3d.plot(x_executed[:, 0], x_executed[:, 1], x_executed[:, 2], label="Actual")
    ax3d.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], label="Reference")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")  # type: ignore

    ax3d.set_zlim(0, np.mean(x_ref[:, 2]) * 1.5)  # type: ignore

    labels = ["x", "y", "z"]
    for i in range(3):
        ax[i, 0].plot(t_ref, x_executed[:, i], label=legend_labels[1])
        if with_ref:
            ax[i, 0].plot(t_ref, x_ref[:, i], label=legend_labels[0])
        ax[i, 0].legend()
        ax[i, 0].set_ylabel(labels[i])
    ax[0, 0].set_title(r"$p\:[m]$")
    ax[2, 0].set_xlabel(r"$t [s]$")

    q_euler = np.stack(
        [
            R.from_quat(x_executed[j, 3:7]).as_euler("XYZ")
            for j in range(x_executed.shape[0])
        ]
    )
    for i in range(3):
        ax[i, 1].plot(t_ref, q_euler[:, i], label=legend_labels[1])
    if with_ref:
        ref_euler = np.stack(
            [R.from_quat(x_ref[j, 3:7]).as_euler("XYZ") for j in range(x_ref.shape[0])]
        )
        traj_length = t_ref.shape[0]
        q_err = np.empty((traj_length, 3))
        for i in range(traj_length):
            q_err[i, :] = quaternion_to_angle_axis(
                quaternion_product(
                    x_executed[i, 3:7], quaternion_conjugate(x_ref[i, 3:7])
                )
            )

        for i in range(3):
            ax[i, 1].plot(t_ref, ref_euler[:, i], label=legend_labels[0])
            ax[i, 1].plot(t_ref, q_err[:, i], label="quat error")
    for i in range(3):
        ax[i, 1].legend()
    ax[0, 1].set_title(r"$\theta\:[rad]$")
    ax[2, 1].set_xlabel(r"$t [s]$")

    for i in range(3):
        ax[i, 2].plot(t_ref, x_executed[:, i + 7], label=legend_labels[1])
        if with_ref:
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
