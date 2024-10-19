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
However, for the second part of the tutorial we require PETSc, and thus need to install DOLFINx on Windows using either WSL (and using conda inside WSL) or Docker.
```

- Introduction to DOLFINx

Custom one-sided integration to illustrate input of custom integration entities

- Non-linear problems

Show how to solve non-linear problems using NonLinearProblem and Newton solver

- Mesh generation

How one can read in meshes as array structures
Show how to use built in meshes or load from file/gmsh
How these can be manually partitioned
Show higher order geometry

- Multiphysics part 1

Introduce the notion of a submesh
Solve simple PDE on submesh as a starter
Exercises

- Multiphysics Part 2

Show how mixed assembly of co-dim 0 would work (sub mesh coupled to parent)
Interpolation to and from these meshes
Introduce notion of multiple meshes, go through example similar to the one of Remi
Exercises

- Co-dim 1 meshes, what are they, what are they useful for?

Co-dim 1 example
Other ideas?

- External operators in UFL

```{bibliography}
:filter: cited and ({"README"} >= docnames)
```
