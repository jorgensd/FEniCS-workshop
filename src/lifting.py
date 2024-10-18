# (lifting)=
# # Using the PETSc for solving linear problems
# In this section, we will cover how we can work directly with the PETSc Krylov subspace solvers.
#
# ## Problem specification
# We consider the equations of linear elasticity,
#
# $$
# \begin{align}
# -\nabla \cdot \sigma (u) &= f && \text{in } \Omega\\
# u &= u_D && \text{on } \partial\Omega_D\\
# \sigma(u) \cdot n &= T && \text{on } \partial \Omega_N
# \end{align}
# $$
#
# where
#
# $$
# \begin{align}
# \sigma(u)&= \lambda \mathrm{tr}(\epsilon(u))I + 2 \mu \epsilon(u)\\
# \epsilon(u) &= \frac{1}{2}\left(\nabla u + (\nabla u )^T\right)
# \end{align}
# $$
#
# where $\sigma$ is the stress tensor, $f$ is the body force per unit volume,
# $\lambda$ and $\mu$ are Lamé's elasticity parameters for the material in $\Omega$,
# $I$ is the identity tensor, $\mathrm{tr}$ is the trace operator on a tensor,
# $\epsilon$ is the symmetric strain tensor (symmetric gradient),
# and $u$ is the displacement vector field. Above we have assumed isotropic elastic conditions.
#
# We will consider a beam of dimensions $[0,0,0] \times [L,W,H]$, where
#
# $$
# \begin{align}
# u_D(0,y,z) &= (0,0,0)\\
# u_D(L,y,z) &= (0,0,-g)\\
# \end{align}
# $$
#
# where $g$ is a prescribed displacement.
# In other words we are clamping the beam on one end, and applying a given displacement on the
# other end.
# All other boundaries will be traction free, i.e. $T=(0,0,0)$.

# +
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import dolfinx
import dolfinx.fem.petsc
import ufl

L = 10.0
W = 3.0
H = 3.0
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [L, W, H]],
    [15, 7, 7],
    cell_type=dolfinx.mesh.CellType.hexahedron,
)
tdim = mesh.topology.dim
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
# -

# ## Locate exterior facets
# We start by locate the various facets for the different boundary conditions.
# First, we find all boundary facets (those facets that are connected to only one cell)

mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

# ## Locate subset of exterior facets

# Next we find those facets that should be clamped, and those that should have a non-zero traction on it.
# We pass in a Python function that takes in a `(3, num_points)` array, and returns an 1D array of booleans
# indicating if the point satisfies the condition or not.


# +
def left_facets(x):
    return np.isclose(x[0], 0.0)


clamped_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left_facets)
# -

# An equivalent way to find the facets is to use Python `lambda` functions, which are [anonymous functions](https://docs.python.org/3/glossary.html#term-lambda)
# (they are not bound to a variable name). Here we find the facets on the right boundary, where $x = L$

prescribed_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], L))

# As all mesh entities are represented as integers, we can find the boundary facets by
# remaining facets using numpy set operations

free_facets = np.setdiff1d(boundary_facets, np.union1d(clamped_facets, prescribed_facets))

# ## Defining a mesh marker
# Next, we can define a meshtag object for all the facets in the mesh

num_facets = mesh.topology.index_map(tdim - 1).size_local
markers = np.zeros(num_facets, dtype=np.int32)
clamped = 1
prescribed = 2
free = 3
markers[clamped_facets] = clamped
markers[prescribed_facets] = prescribed
markers[free_facets] = free
facet_marker = dolfinx.mesh.meshtags(mesh, tdim - 1, np.arange(num_facets, dtype=np.int32), markers)


# ## The variational formulation

# We have now seen this variational formulation a few times

# +
x = ufl.SpatialCoordinate(mesh)
T_0 = dolfinx.fem.Constant(mesh, (0.0, 0.0, 0.0))
E = dolfinx.fem.Constant(mesh, 1.4e3)
nu = dolfinx.fem.Constant(mesh, 0.3)
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
f = dolfinx.fem.Constant(mesh, (0.0, 0.0, 0.0))


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))


ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(T_0, v) * ds(3)
# -

# ## Alternative lifting procedure

# We locate the constrained degrees of freedom

clamped_dofs = dolfinx.fem.locate_dofs_topological(V, facet_marker.dim, facet_marker.find(clamped))
displaced_dofs = dolfinx.fem.locate_dofs_topological(V, facet_marker.dim, facet_marker.find(prescribed))

# Next, we define the prescribed displacement

u_prescribed = dolfinx.fem.Constant(mesh, (0.0, 0.0, -H / 2))
u_clamped = dolfinx.fem.Constant(mesh, (0.0, 0.0, 0.0))

# We define the Dirichlet boundary condition object as

bcs = [
    dolfinx.fem.dirichletbc(u_clamped, clamped_dofs, V),
    dolfinx.fem.dirichletbc(u_prescribed, displaced_dofs, V),
]

# The lifting procedure from {ref}`lifting`` is used in both C++ and Python, and what it does under the hood is to compute the local
# matrix-vector products of $A_{d, bc}$ and $g$ (no global matrix vector products are involved). However, we can use UFL
# to do this in a simpler fashion in Python

g = dolfinx.fem.Function(V)
g.x.array[:] = 0
dolfinx.fem.set_bc(g.x.array, bcs)
g.x.scatter_forward()
L_lifted = L - ufl.action(a, g)

# What happens here?
#
# `ufl.action` reduces the bi-linear form to a linear form (and would reduce a linear form to a scalar)
#  by replacing the trial function with the function $g$, that is only non-zero at the Dirichlet condition

# The new assembly of the linear and bi-linear form would be

# + tags=["hide-output"]
a_compiled = dolfinx.fem.form(a)
A = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=bcs)
A.assemble()
b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L_lifted))
dolfinx.fem.petsc.set_bc(b, bcs)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
# -

# ```{admonition} New assembly commands!
# :class: warning
# In all previous sections we have used `dolfinx.fem.assemble_matrix` and
# `dolfinx.fem.assemble_vector`, while we now use `dolfinx.fem.petsc.assemble_vector` and
# `dolfinx.fem.petsc.assemble_matrix`.
# The difference here is that we assemble into PETSc Matrix and Vector objects, which can
# be easily used with the PETSc solvers.
# Even if the Native DOLFINx matrices supports MPI distributed vectors and matrices, scipy doesn't.
# PETSc has a notion of MPI distributed matrices, which means that we can finally run our problems in parallel!
# ```

# Now that we have created our PETSc Matrix and PETSc Vector, we can create a PETSc Krylov subspace solver.

ksp = PETSc.KSP().create(mesh.comm)

# Next, we can choose from [all](https://petsc.org/main/manual/ksp/#tab-kspdefaults) the different methods PETSc support.
# We will limit ourself to a direct solver that can be executed in parallel.

ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")

# We attach the matrix to the operator, so that we can modify the entries later, and it will be reflected in a solve call.

ksp.setOperators(A)


# Next we can solve the linear system

uh = dolfinx.fem.Function(V)
ksp.solve(b, uh.x.petsc_vec)
assert ksp.getConvergedReason() > 0, "Solver did not converge"
uh.x.scatter_forward()

ksp.destroy()
_ = b.destroy()

# ```{admonition} Destruction of PETSc objects
# :class: warning
# PETSc does not handle the destruction of Python objects.
# Thus we manually have to call their destructor to avoid memory leaks.
# ```

# + tags=["hide-input"]
import sys, os

import pyvista

if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)
grid = dolfinx.plot.vtk_mesh(uh.function_space)
pyvista_grid = pyvista.UnstructuredGrid(*grid)
values = uh.x.array.reshape(-1, 3)
pyvista_grid.point_data["u"] = values
warped = pyvista_grid.warp_by_vector("u")
plotter = pyvista.Plotter()
plotter.show_axes()
plotter.add_mesh(pyvista_grid, style="points")
plotter.add_mesh(warped, scalars="u", lighting=True)
plotter.show()
# -


# ## Convenience wrapper for Linear problems
# As many users will solve linear problems over and over again, DOLFINx provides a simplfied user-interface,
# that takes care of creation, assembly and destruction of matrices and solvers.
# Given `a` and `L` from above, we show this interface:

u_new = dolfinx.fem.Function(V)
options = {"ksp_type": "preonly", "pc_type": "lu", "pc_mat_factor_type": "mumps"}
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs, u_new, petsc_options=options)
problem.solve()
assert problem.solver.getConvergedReason() > 0, "Solver did not converge"

# We verify that the solution is the same as above

np.testing.assert_allclose(uh.x.array, u_new.x.array, atol=1e-12)
