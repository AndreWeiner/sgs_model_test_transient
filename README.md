# Evaluation of the SGS model performance in the initial transient stage of the bubble rise


## Introduction

This repository contains source code and documentation related to the research project:
```
Assessment of a subgrid-scale model for convection-dominated mass transfer at rising bubbles during the initial transient rise
```
This project is a collaborative effort between the [Multi-Scale Modeling of Multi-Phase Flows](https://www.tue.nl/en/research/research-groups/multi-scale-modelling-of-multi-phase-flows/) group at TU Eindhoven and the [Institute of Fluid Mechanics](https://www.tu-braunschweig.de/ism) at TU Braunschweig. The research project was planned and executed by **Irian Hierck**, **Claire Claassen**, **Hans Kuipers**, **Maike Balthussen**, all TU Eindhoven, and **Andre Weiner**, TU Braunschweig. A write-up of the results is published with open access at [https://doi.org/10.1002/aic.17641](https://doi.org/10.1002/aic.17641)

Cite as:
```
Weiner, A, Claassen, CMY, Hierck, IR, Kuipers, JAM, Baltussen, MW. Assessment of a subgrid-scale model for convection-dominated mass transfer for initial transient rise of a bubble. AIChE J. 2022;e17641. doi:10.1002/aic.17641
```
```
@article{https://doi.org/10.1002/aic.17641,
author = {Weiner, Andre and Claassen, Claire M. Y. and Hierck, Irian R. and Kuipers, J. A. M. and Baltussen, Maike W.},
title = {Assessment of a subgrid-scale model for convection-dominated mass transfer for initial transient rise of a bubble},
journal = {AIChE Journal},
volume = {n/a},
number = {n/a},
pages = {e17641},
year = {2022},
keywords = {high-Schmidt number problem, machine learning, mass transfer, multiphase flows, subgrid-scale modeling},
doi = {https://doi.org/10.1002/aic.17641},
url = {https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.17641}
}
```

## Acknowledgement

This work is part of the research program **First principles based multi-scale modeling of transport in reactive three phase flows** with project number 716.014.001, which is financed by the Netherlands Organization for Scientific Research (NWO) TOP grant. This work is also part of the research program **Reactive Gas-Liquid Two-Phase Flow in Porous Media**, with project number 716.018.001, which is financed by the Netherlands Organization for Scientific Research (NWO) TOP grant.

## Repository overview

The repository contains mostly software related to the *hybrid simulation approach*. In the hybrid approach, single-phase flow and species transport equations are solved. Interface deformation, interface velocity, and rise velocity are mapped from the two-phase to the single-phase simulations using machine learning.

[![All cases Sc 100](https://img.youtube.com/vi/qhxbkvM2jAI/0.jpg)](https://www.youtube.com/watch?v=qhxbkvM2jAI)

The mesh is surface aligned and deforms according to the interface deformation computed in the two-phase simulations. In contrast to the video below, the final meshes used for all computations reported in the article were created with *blockMesh*.

[![Mesh deformation](https://img.youtube.com/vi/ytD2Qs3fCxk/0.jpg)](https://www.youtube.com/watch?v=ytD2Qs3fCxk)

The repository is organized as follows:

- **data**: raw two-phase simulation data (interface data and log files)
- **notebooks**: Jupyter notebooks and Python scripts for machine-learning, post-processing, and visualization
- **openfoam**: OpenFOAM boundary conditions, apps, and test cases
- **output**: processed data, model snapshots, and visualizations
- *create_jupyter_container.sh*: script to create a suitable Jupyter container
- *create_openfoam_container.sh*: script to create a suitable container with OpenFOAM and PyTorch
- *start_notebooks.sh*: script to start Jupyter notebooks
- *start_openfoam.sh*: script to start an interactive shell in the OpenFOAM container
- *Dockerfile*: Dockerfile to create the Jupyter/Python Docker container

## Data

Raw simulation data, binary files, and visualizations are not stored on Github but have to be downloaded and extracted separately. The final data are available at:
```
http://doi.org/10.23728/b2share.d58c8710010f415c88608aa1d3b172b3
```
The following archives are available:

- *data* folder (~6.1GB): [link](https://b2share.eudat.eu/api/files/ac50ab77-cd0c-4903-bf0c-a0f9aef03b3f/data.tar.gz)
- *output* folder (~52MB): [link](https://b2share.eudat.eu/api/files/ac50ab77-cd0c-4903-bf0c-a0f9aef03b3f/output.tar.gz)
- *openfoam/run* folder (~2.1GB): [link](https://b2share.eudat.eu/api/files/ac50ab77-cd0c-4903-bf0c-a0f9aef03b3f/run.tar.gz)

Assuming that the *tar.gz* archives are located at the repository's top level, the data can be extracted as follows:
```
tar -xzf output.tar.gz
tar -xzf data.tar.gz
tar -xzf run.tar.gz -C openfoam
```

## Dependencies

The provided Docker images contain OpenFOAM, PyTorch, and all associated dependencies to run simulations, notebooks and scripts. This encapsulation of dependencies ensures that all results are reproducible independent of the underlying host operating system and the software stack installed thereon. Moreover, some workflows contain some *random* component, e.g. the initialization of network model weights, and the Docker environment is essential to make the outcome of such workflows reproducible. Machine learning notebooks contain a *seed* parameter for Numpy and PyTorch in the first notebook cell. One will obtain the published results by clicking on *Restart the kernel, then re-run the whole notebook* (>> symbol). Repeatedly executing some of the cells will lead to varying results.

Any installed version of [Docker](https://docs.docker.com/install/) larger than **1.10** is able to pull and execute the Docker images hosted on [Dockerhub](https://hub.docker.com/r/andreweiner/jupyter-environment). There are convenience scripts to create and start a suitable Docker container which require root privileges. To avoid running these scripts with *sudo*, follow the [post-installation steps](https://docs.docker.com/install/linux/linux-postinstall/) and create a dedicated Docker group.

Important libraries installed in the environments are:

- Ipython: 7.14.0
- Matplotlib: 3.1.2
- Numpy: 1.17.4
- Pandas: 0.25.3
- Python: 3.8.2
- PyTorch: 1.5.0+cpu
- Scikit Learn: 0.23.1
- Jupyter-lab: 2.1.3
- OpenFOAM-v2006
- LibTorch 1.6 (PyTorch C++ frontend)

To learn more about the container with OpenFOAM and PyTorch, refer to [this](https://github.com/AndreWeiner/of_pytorch_docker) repository.

## Notebooks

### Notebooks contained in this repository

If the *data* and *output* folder were downloaded and extracted as described above, it is not necessary to execute the notebooks before running OpenFOAM simulations.

- **data_processing.ipynb**: read and process raw log files and interface data coming from the two-phase simulations; computation of velocity projection; quick visualization of unprocessed data
- **model_creation.ipynb**: creation of machine learning models
- **export_models.ipynb**: export of machine learning models to TorchScript
- **sherwood_numbers.ipynb**: evaluation of Sherwood numbers; most figures reported in the article are created here

### Displaying the notebooks online

Static versions of the notebooks can be rendered online without the need to download or install any dependencies. The rendering on Github often does not work properly for large notebooks. [NBViewer](https://nbviewer.jupyter.org/) is typically the better choice:

- [data_processing.ipynb](https://nbviewer.jupyter.org/github/AndreWeiner/sgs_model_test_transient/blob/master/notebooks/data_processing.ipynb)
- [model_creation.ipynb](https://nbviewer.jupyter.org/github/AndreWeiner/sgs_model_test_transient/blob/master/notebooks/model_creation.ipynb)
- [export_models.ipynb](https://nbviewer.jupyter.org/github/AndreWeiner/sgs_model_test_transient/blob/master/notebooks/export_models.ipynb)
- [sherwood_numbers.ipynb](https://nbviewer.jupyter.org/github/AndreWeiner/sgs_model_test_transient/blob/master/notebooks/sherwood_numbers.ipynb)

### Starting Jupyterlab

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

## OpenFOAM simulations

### Creating the environment

The environment containing OpenFOAM and PyTorch can be created and used similarly to the Jupyterlab environment. To create the environment, run:

```
./create_openfoam_container.sh
```
To start an interactive shell, run:
```
./start_openfoam.sh
```

### Performing simulations

There are scripts in the *openfoam* folder to compile all applications and to run all test cases contained in the accompanying article. To reproduce the entire simulation data, run (assuming that the machine learning models are available in the *output* folder):
```
./start_openfoam
cd openfoam
./compile.sh
./run_simulations.sh
```
Running an individual case when starting from scratch might look as follows:

```
./start_openfoam.sh
# now we are inside the container
cd openfoam
# the run folder is not tracked by Git
mkdir run
cp -r test_cases/CB4_ref_0 run/
cd run/CB4_ref_0/
# runs pre-processing and the simulation itself
# solver, boundary conditions, and apps must have been compiled in advance
./Allrun
# once the simulation is done, we can compute local and global Sherwood numbers;
# unfortunately, a vast amount of warning messages due to the wedge mesh will be displayed;
# therefore, it is best to redirect all output
calcSh -patch bubble -field s1 &> /dev/null
```

## Licence

[GNU General Public License v3.0](https://github.com/AndreWeiner/sgs_model_test_transient/blob/master/LICENSE)
