# FEniCS workshop

The FEniCS project is a collection of scientific software for solving partial differential equations {term}`PDE`s with the Finite Element method {term}`FEM`.
The tutorial is currently built for `v0.10.x` of DOLFINx{cite}`DOLFINx2023`.

## Installation

For the tutorial it is recommended to use `conda`.
We recommend using the [conda-forge](https://conda-forge.org/) implementation of conda.
All dependencies used for the tutorial can be installed with the conda environment file [environment.yml](./environment.yml).

Store this file on your system, and from the folder with the file, run

```bash
conda env create -f environment.yml
```

The environment can then be activated with

```bash
conda activate workshop-env
```

```{admonition} Native Windows installation
:class: important
Note that on Windows one has to install [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/) for Just In Time-compilation.
```

```{admonition} PETSc on Windows
:class: important
PETSc is not available through conda on native Windows.
The first part of this tutorial does not require PETSc.
However, for the second part of the tutorial we require PETSc, and thus IntegralType.interior_facet: need to install DOLFINx on Windows using either WSL (and using conda inside WSL) or Docker.
```

```{bibliography}
:filter: cited and ({"README"} >= docnames)
```
