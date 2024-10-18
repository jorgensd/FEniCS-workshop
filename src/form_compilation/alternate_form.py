# # Non-linear Poisson equation with alternative formulation

# In this section, we will focus on the approach most users use to interact
# with UFL, FFCx and basix.
# Here we will start by creating the domain we want to solve a problem on.
# In this case, we will use a unit cube

from mpi4py import MPI
import dolfinx

N = 5
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)

# Next, let's consider the problem
#
# $$
# \begin{align}
# -\nabla \cdot p(u) \nabla u &= f \quad \text{in } \Omega, \\
# u &= g \quad \text{on } \partial \Omega
# \end{align}
# $$
# where $p(u)=1+u^2$.
# We choose to use a third order Lagrange space for the unknown

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))
