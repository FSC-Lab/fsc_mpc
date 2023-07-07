# Example of controlling a quadrotor by an MPC

This repo provides an example of controlling a quadrotor using the `acados` MPC toolbox.

The code is abridged from [Data Driven MPC](https://github.com/uzh-rpg/data_driven_mpc) with the GP components eliminated. The API is also inspired by [rpg_mpc](https://github.com/uzh-rpg/rpg_mpc)

## Prerequisites

### FSCore

FSC Lab's core library. Not yet open-sourced. Ask H S Helson Go nicely (or bring him booze)

### CMake

We use [CMake Presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) to drive CMake configuration. Therefore, CMake >3.19 is required. Install the latest cmake following [instructions here](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line)

CMake configuration/build/install options can be put into `CMakeUserPresets.json`

``` jsonc
{
  "version": 3,
  "configurePresets": [
    {
      "name": "<your own name>",
      "displayName": "<your own name>",
      "inherits": [
        "vcpkg-linux-default"
      ],
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        // IMPORTANT: Put your own python executable here, especially if you are using Conda or virtualenvs
        "Python_EXECUTABLE": "${env:HOME}/anaconda3/envs/control_env/bin/python"
      },
      // These variables are defined in the following section. These variable definitions may go here or into bashrc
      "environment": {
        "ACADOS_SOURCE_DIR": "${env:HOME}/src/acados",
        "LD_LIBRARY_PATH": "${env:LD_LIBRARY_PATH}:/usr/local/lib"
      }
    }
  ]
}
```

### Acados toolbox and Python interface

1. Clone the acados toolbox with all its submodules

    ``` bash
    git clone --recurse-submodules https://github.com/acados/acados.git
    ```

2. Configure and install acados

    ```bash
    cmake -S . -B build -GNinja -DACADOS_WITH_QPOASES=ON -DACADOS_INSTALL_DIR=/usr/local 
    cmake --build build --config Release
    sudo cmake --install build
    ```
    where we set `ACADOS_INSTALL_DIR` to ensure that acados installs to the conventional location for cmake to pick it up

3. Adjust some environment variables 

    ``` bash
    export ACADOS_SOURCE_DIR=$LOCATION_WHERE_YOU_CLONED_ACADOS
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    ```

4. Install the acados python interface

    ``` bash
    cd $ACADOS_SOURCE_DIR/interfaces/acados_template
    pip3 install .
    ```

5. Run an acados example to trigger Tera renderer installation

    ```bash
    python $ACADOS_SOURCE_DIR/examples/acados_python/getting_started/minimal_example_ocp.py
    ```
    press `y` when prompted to install Tera renderer


### Vcpkg

1. Clone vcpkg

    ```bash
    git clone https://github.com/microsoft/vcpkg.git
    ```

2. Bootstrap and install

    ```bash
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ```

3. Adjust some environment variables

    ```bash
    export VCPKG_ROOT=$LOCATION_WHERE_YOU_CLONED_VCPKG
    export PATH=$PATH:$VCPKG_ROOT
    ```

## Mathematical Conventions

This code adopts the following mathematical conventions

1. Quaternions are real-part LAST, i.e. `[x, y, z, w]`
2. The attitude quaternions are passive, body-to-world, i.e.

    $$
    \underrightarrow{\mathcal{F}}_i = \mathbf{R}\left(\mathbf{q}_{ib}\right) \underrightarrow{\mathcal{F}}_b
    $$

    or that the rotation represented by the attitude quaternion transforms the quadrotor body frame to be aligned with the inertial frame

3. The quadrotor velocity is expressed in the inertial frame, obeying

    $$
      \dot{\mathbf{v}}_i = \frac{f}{m}\mathbf{R}\left(\mathbf{q}_{ib}\right)\mathbf{1}_3 + g\mathbf{1}_3
    $$
