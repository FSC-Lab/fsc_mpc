# Example of controlling a quadrotor by an MPC

This repo provides an example of controlling a quadrotor using the `acados` MPC toolbox.

The code is abridged from [Data Driven MPC](https://github.com/uzh-rpg/data_driven_mpc) with the GP components eliminated.

## Running the code (Python)

1. Install the acados toolbox, [following the instructions here](https://docs.acados.org/installation/

2. Install the acados Python interface

3. Open the `quadrotor_mpc` base folder in vscode, then press CTRL-F5 to run the basic tracking example.
    Investigate `.vscode/launch.json` to change run options

## Mathematical Conventions

This code adopts the following mathematical conventions

1. Quaternions are real-part LAST, i.e. `[x, y, z, w]`
2. The attitude quaternions is passive, world-to-body, i.e.

    $$
    \underrightarrow{\mathcal{F}}_b = \mathbf{R}\left(\mathbf{q}_{bi}\right) \underrightarrow{\mathcal{F}}_i
    $$

    or that the rotation represented by the attitude quaternion transforms the inertial frame to be aligned with the quadrotor body frame

3. The quadrotor velocity is expressed in the body frame, obeying

    $$
      \dot{\mathbf{v}}_b = -\boldsymbol{\omega}_b^\times\mathbf{v}_b + \frac{f}{m}\mathbf{1}_3 + g\mathbf{R}_{bi}\mathbf{1}_3
    $$
