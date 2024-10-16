# FEniCS workshop

The FEniCS project is a collection of scientific software for solving partial differential equations {term}`PDE`s with the Finite Element method {term}`FEM`.
The tutorial is currently built for `v0.9.x` of DOLFINx{cite}`DOLFINx2023`.

## Installation

For the tutorial it is recommened to use `conda`, as described in: https://github.com/FEniCS/dolfinx/?tab=readme-ov-file#conda

```{admonition} Native Windows installation
Note that on Windows one has to install [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/) for Just In Time-compilation.
```

```{admonition} PETSc on Windows
PETSc is not available through conda on native Windows.
The first part of this tutorial does not require PETSc.
However, for the second part of the tutorial we require PETSc, and thus need to install DOLFINx on Windows using either WSL (and using conda inside WSL) or Docker.
```


## Agenda:

- An introduction to finite elements


Quick refresher of Galkerin/Ritz-Galerkin methods
Going from a global polynomial basis to a basis defined on a subdivision of the domain (cell).
Define the functional used to ensure the “orthogonality” of the basis functions, used in the some Hilbert space
Maybe talk about the Ciarlet definition of a finite element?

- The finite element in basix

Introduce notion of a basix.ufl.finite_element
Describe tabulation, i.e. getting numerical values within a single reference cell.
Introduce the notion of push forward/pull back used for mapping values from/to the reference element to the physical element.
This will introduce some more “exotic” elements, like Nedelec (first kind) and Raviart Thomas, using co-variant/contravariant Piola to map to and from reference elements

- An introduction to the unified form language (UFL)

Explain how the unified form language works (what are the components needed).
Show examples:
Poisson
Navier-Stokes
Heat equation
EM formulation (TEAM 30?: https://github.com/Wells-Group/TEAM30)
Examples are only shown on a UFL domain, no mesh involved, to illustrate the generality of UFL).
Show differentiation tools, such that you can build up your adjoints automatically

- Form compilation (The FeniCS form compiler)

Given forms from the previous session, we now want to assemble scalars, vectors or matrices from these, given a mesh. Note the similarity of UFL to the mathematical formulation, where we could write out the pull-back and only insert:
Coefficient data for given cell
Constants for given cell
Nodes describing the geometry to compute the Jacobian
Show how UFL can do the pull-back for you, writing out the sums, and how FFCx interprets this as graph operations
These can then be transformed into C code, that can be accessed through the C++ interface of DOLFINx or the Python interface.
Maybe show how a single file can be used in either language?

- Introduction to DOLFINx

Introduction to how all of the above is glued together in DOLFINx
Show how to use built in meshes or load from file/gmsh
Show how we define functions and constants (holding data)
Show how solvers work (linear problems first)
Show how one can use PETSc directly
Exercises

- DOLFINx deep dive

Explain BCs (dirichlet and others)
Explain interpolation
Explain expression
Explain integration entities
Custom one-sided integration to illustrate input of custom integration entities
Maybe use “manual version” of DOLFINx here?

- Non-linear problems

Show how to solve non-linear problems using NonLinearProblem and Newton solver
Show how to implement custom Newton solver
Exercises?

- Mesh generation

How one can read in meshes as array structures
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
