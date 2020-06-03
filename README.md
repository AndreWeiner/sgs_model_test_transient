# Evaluation of the SGS-model performance in the initial transient stage of the bubble rise

## TODOs
- **IH**: Bundle and archive hydrodynamics data once simulations finish
- **AW**: Properly organise hydrodynamic data for analysis
- **AW**: Run machine learning simulations on hydrodynamic data
- **IH**: Start simulations with mass transfer for Sc number of 100


## Overview

This repository contains source code and documentation related to the research project (working title)

```
Assessment of a subgrid-scale model for convection-dominated mass transfer at rising bubbles during the initial transient rise
```
The project is a collaborative effort between the Technical Universities of Eindhoven and Braunschweig. (TODO: add some links to institutions and grants, etc.)

The repository is organized as follows:

- **data**: raw simulation data
- **notebooks**: Jupyter notebooks and Python scripts for machine-learning, post-processing, and visualization
- **output**: location for model-snapshots and final visualizations
- *create_jupyter_container.sh*: script to create a suitable Jupyter- container; read more in the sections below
- *Dockerfile*: Dockerfile to create the Jupyter/Python Docker container
- *start_notebooks.sh*: script to start Jupyter notebooks; read more in the sections below

## Dependencies

The provided Docker image contains a Jupyter and Python environment with all dependencies to run notebooks and scripts. This encapsulation of dependencies ensures that all results are reproducible independent of the underlying host operating system and the software stack installed thereon. Moreover, some workflows contain some *random* component, e.g. the initialization of model weights, and the Docker environment is essential to make the outcome of such workflows reproducible. These notebooks contain a *seed* for Numpy and PyTorch in the first notebook cell. One will obtain the published results by clicking on *Restart the kernel, then re-run the whole notebook* (>> symbol). Repeatedly executing some of the cells will lead to varying results.

Any installed version of [Docker](https://docs.docker.com/install/) larger than **1.10** is able to pull and execute the Docker image hosted on [Dockerhub](https://hub.docker.com/r/andreweiner/jupyter-environment). There are convenience scripts to create and start a suitable Docker container which require root privileges. To avoid running these scripts with *sudo*, follow the [post-installation steps](https://docs.docker.com/install/linux/linux-postinstall/) and create a dedicated Docker group.

Important libraries installed in the environment are:

- Ipython: 7.14.0
- Matplotlib: 3.1.2
- Numpy: 1.17.4
- Pandas: 0.25.3
- Python: 3.8.2
- PyTorch: 1.5.0+cpu
- Scikit Learn: 0.23.1
- Jupyter-lab: 2.1.3

## Getting started

### Downloading the data

To download the raw simulation data, click here (TODO: add link once data is available). Next, place the contents of the archive in the **data** folder. Note that the notebooks rely in the described naming and location to be functional. The following commands might be helpful to execute all steps in the command-line:

```
# this commands will be completed once the data is available
mkdir data
wget -q -O simulation_data.tar.gz https://url.to.archive
tar xzf simulation_data
```

### Starting Jupyter-lab

The Jupyter environment (container) has to be created only once on a new system. To create the container, run:
```
./create_jupyter_container.sh
```
The output of *docker container ls --all* should contain an entry similar to the following line:

```
docker container ls --all
... andreweiner/jupyter-environment:5d02515 ... 0.0.0.0:8000->8000/tcp, 8888/tcp  jupyter-5d02515
```
Once the container has been created successfully, the environment can be accessed using the *start_notebooks.sh* script:

```
./start_notebooks.sh
```
A url with the syntax [http://127.0.0.1:8000/?token=...]() will be displayed in the console. By opening the url in a web browser of your choice, the Jupyter notebooks can be accessed and executed.

## Licence

[GNU General Public License v3.0](https://github.com/AndreWeiner/sgs_model_test_transient/blob/master/LICENSE)
