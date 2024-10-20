# FEniCS workshop

The FEniCS project is a collection of scientific software for solving partial differential equations {term}`PDE`s with the Finite Element method {term}`FEM`.
The tutorial is currently built for `v0.9.x` of DOLFINx{cite}`DOLFINx2023`.

## Installation

For the tutorial it is recommened to use `conda`, as described in: https://github.com/FEniCS/dolfinx/?tab=readme-ov-file#conda

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
