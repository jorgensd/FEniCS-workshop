# # Mixed finite element problems
#
# As indicated earlier, the finite element method can be used for {term}`PDE`s that consist
# of multiple physical quantities.
# In this section, we will consider the Stokes problem, which can be formulated as
#
# $$
# \begin{align}
# - \nabla \cdot ( \nabla \mathbf{u}) + \nabla p &= \mathbf{f} \quad \text{in } \Omega \\
# \nabla \cdot \mathbf{u} &= 0 \quad \text{in } \Omega \\
# \end{align}
# $$
#
# where $\mathbf{u}$ is the velocity field, $p$ is the pressure field, and $\mathbf{f}$ is a given source term.

# ```{admonition} Symmetric variational form
# :class: dropdown note
# We prefer problems that have a symmetric structure. For this reason, we introduce $\hat p=-p$ and rewrite the problem as
#
# $$
# \begin{align}
# - \nabla \cdot ( \nabla \mathbf{u} + \hat p I)  &= \mathbf{f} \quad \text{in } \Omega \\
# \nabla \cdot \mathbf{u} &= 0 \quad \text{in } \Omega \\
# \end{align}
# $$
# ```
#
# With this re-formulation, we can create the variational form by introducing a pair of test-functions, $\mathbf{v}\in V$
# $p \in Q$ multiply each equation by the corresponding test-function, and integrate over the domain.
#
# $$
# \begin{align}
#   \int_{\Omega} - \nabla \cdot  ( \nabla \mathbf{u} + \hat p I) \cdot \mathbf{v} ~\mathrm{d}x
#   &= \int_\Omega \mathbf{f} \cdot \mathbf{v}~\mathrm{d}x\\
#   \int_{\Omega} \nabla \cdot \mathbf{u} q ~\mathrm{d}x &= 0
# \end{align}
# $$
#
# We integrate the first equation by parts to obtain
#
# $$
# \begin{align}
#  \int_{\Omega} \nabla u : \nabla v ~\mathrm{d}x + \int_{\Omega} \hat p \nabla \cdot \mathbf{v} ~\mathrm{d}x
#  -\int_{\partial \Omega} ((\nabla \mathbf{u} + \hat p I) \cdot \mathbf{n}) \cdot \mathbf{v} ~\mathrm{d}s
# &= \int_\Omega \mathbf{f} \cdot \mathbf{v}~\mathrm{d}x\\
# \int_{\Omega} \nabla \cdot \mathbf{u} q ~\mathrm{d}x &= 0
# \end{align}
# $$
#
# ```{admonition} The boundary term
# :class: dropdown note
# From integration by parts, we obtain a boundary term that depends on the normal derivative of the velocity field.
# Thankfully, we use the **natural** boundary condition for the Stokes problem, whereever we do not have a Dirichlet boundary condition
# on the velocity.
# In other words, we apply the following conditions on the boundary $\partial \Omega = \partial\Omega_D\cup\Gamma$, where $\partial\Omega_D$
# and $\Gamma$ are two disjoint sets of the boundary.
#
# $$
# \begin{align}
# \mathbf{u} &= \mathbf{u}_D \quad \text{on } \partial\Omega_D\\
# \nabla \mathbf{u} \cdot \mathbf{n} + \bar p \mathbf{n} &= \mathbf{g} \quad \text{on } \Gamma
# \end{align}
# $$
#
#  This reduces the system to
#
# $$
# \begin{align}
#  \int_{\Omega} \nabla u : \nabla v ~\mathrm{d}x + \int_{\Omega} \hat p \nabla \cdot \mathbf{v} ~\mathrm{d}x
# &= \int_\Omega \mathbf{f} \cdot \mathbf{v}~\mathrm{d}x + \int_{\partial \Omega} \mathbf{g} \cdot \mathbf{v} ~\mathrm{d}s\\
# \int_{\Omega} \nabla \cdot \mathbf{u} q ~\mathrm{d}x &= 0
# \end{align}
# $$
# ```
#
# We start by setting up this variational formulation for a unit square domain $\Omega = [0,1]\times[0,1]$.

# +
from mpi4py import MPI
import dolfinx
import basix.ufl
import ufl
import numpy as np

M = 6
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, M, M)
# -

# ## Finite elements and mixed function space

# Next, we define the finite element spaces we would like to use for the velocity and pressure fields.
# We do this as descibed in {ref}`ufl-intro` and use `basix.ufl.mixed_element`
# to create a mixed element for the velocity and pressure fields.
# We choose the Taylor-Hood finite element pair for this problem.

el_u = basix.ufl.element("Lagrange", mesh.basix_cell(), 3, shape=(mesh.geometry.dim,))
el_p = basix.ufl.element("Lagrange", mesh.basix_cell(), 2)
el_mixed = basix.ufl.mixed_element([el_u, el_p])

# ## Test and trial functions in mixed spaces

# We can now define our mixed function space and the corresponding test and trial functions

W = dolfinx.fem.functionspace(mesh, el_mixed)
u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)

# ```{admonition} Test and trial functions in mixed spaces
# :class: dropdown note
# We observe that we use `ufl.TrialFunctions` and `ufl.TestFunctions` to define the trial and test functions for the mixed space,
# rather than `ufl.TestFunction` and `ufl.TrialFunction`.
# We could use the latter, but would then either have to index the corresponding functions or use `ufl.split` to extract the components.
# ```
# To see alternative approach for defining the test and trial functions expand the below cell

# + tags = ["hide-input"]
w = ufl.TrialFunction(W)
u, p = ufl.split(w)
u = ufl.as_vector([w[0], w[1]])
p = w[2]
# -

# ## Functions
# Next we define a function in `wh` in `W` to represent the solution.

wh = dolfinx.fem.Function(W)

# We can split this function into symbolic components for the velocity and pressure fields with `ufl.split`

uh, ph = ufl.split(wh)

# ## Boundary integrals
# So far, we have only considered the integrals over the domain $\Omega$. We also need to consider the integrals over the boundary.
# We do this by introducing the exterior facet measure, called `ds` in UFL.
# This measure will only consist of facet integrals over those facets that are connected to a signle cell.

ds = ufl.Measure("ds", domain=mesh)

# We create a function `g` to represent any any potential natural boundary conditions

g = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0, 0)))

# ## Variational form
# We can now define the variational formulation

# +
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0, 0)))

F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
F += ufl.inner(p, ufl.div(v)) * ufl.dx
F += ufl.inner(ufl.div(u), q) * ufl.dx
F -= ufl.inner(f, v) * ufl.dx
# -

# We can do as previosly, and split this into a bi-linear and linear form with

a, L = ufl.system(F)

# (boundary_subset)=
# ## Locating a subset of entities on a boundary
# We now want to apply a Dirichlet condition on the degrees of freedom on some subset of the boundary.
# We start by locating some sub-set of facets on the boundary, by use `dolfinx.mesh.locate_entities_boundary`.
# Let's say we want to prescibe the conditions:
#
# $$
# \begin{align}
# u(0, y) &= (ux, uy) \\
# u(x, 0) &= u(1, y) = (0, 0)
# \end{align}
# $$
# We would have to locate what degrees of freedom that are in the closure of this boundary.
# Previously, we used `dolfinx.mesh.exterior_facet_indices` to locate all the facets on the boundary.
# However, in this case, we only want to find a subset of facets on the boundary.
# We therefore use `dolfinx.mesh.locate_entities_boundary`:
# 1. Define a function `boundary_marker` that takes in a set of coordinates `x` on the form `(3, num_points)`
# and returns an array of length `num_points` indcating if the point is on the boundary.
# For instance, for the case $u(0, y) = (ux, uy)$, we would have


# +
def boundary_marker(x):
    return np.isclose(x[0], 0.0)


mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
left_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, boundary_marker)
# -

# + tags = ["remove-input"]
print(f"{left_facets=}")  # -
if MPI.COMM_WORLD.size == 0:
    assert len(left_facets) == M
# -

# ```{admonition} What points are sent into locate entitites boundary?
# :class: dropdown note
# The function checks that all vertices of a given entity satisfies the input constraint.
# ```
#
# Whenever we have a union of constraints, we can use the `numpy` bit operations `&` (and) and `|` (or) to combine the constraints.
#
# ```{admonition} Create a boundary marker for the top and bottom boundary and locate the facets on those boundaries
# :class: dropdown note
# We can use `numpy.isclose` for each of the cases, and combine them with `numpy.bitwise_or` or `|`
# ```
# Expand the below to see the solution

# + tags = ["hide-input"]


# +
def top_bottom_marker(x):
    return np.isclose(x[1], 1.0) | np.isclose(x[1], 0.0)


tb_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, top_bottom_marker)
# -

# + tags=["remove-input"]
if MPI.COMM_WORLD.size == 0:
    assert len(tb_facets) == 2 * M
# -

# For completeness, we find the remaining facets on the boundary

all_boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
remaining_facets = np.setdiff1d(all_boundary_facets, np.union1d(tb_facets, left_facets))

# + tags=["remove-input"]
if MPI.COMM_WORLD.size == 0:
    assert len(remaining_facets) == M
# -

# ## Dirichlet conditions in mixed spaces
#
# Now that we have found the boundaries of interest, we can create the Dirichlet conditions
# We start by considering what we have already seen:
# In the previous sections, Dirichlet conditions have been applied as input to
# `dolfinx.fem.dirichletbc` in the following way:
# 1. A function $w_D\in W$ from the space that contains the Dirichlet condition values
# 2. A list of the {term}`dofs` in the space $W$ that should be constrained
#
# However, what happens if we use this strategy on a mixed space?
# As `w_D` is a mixed function, we have both pressure and velocity components in this space.
# Thus, if we set all entries in the BC to a constant value, we would set **both** the pressure and velocity to the same value.
# We illustrate this below

# +
w_D = dolfinx.fem.Function(W)
w_D.x.array[:] = 0.43

wh = dolfinx.fem.Function(W)
dofs = dolfinx.fem.locate_dofs_topological(W, mesh.topology.dim - 1, tb_facets)
bc = dolfinx.fem.dirichletbc(w_D, dofs)
bc.set(wh.x.array)
wh.x.scatter_forward()
# -

# + tags=["hide-input"]
import pyvista
import os, sys

if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)


def visualize_mixed(mixed_function: dolfinx.fem.Function, scale=1.0):
    u_c = mixed_function.sub(0).collapse()
    p_c = mixed_function.sub(1).collapse()

    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_c.function_space))

    # Pad u to be 3D
    gdim = u_c.function_space.mesh.geometry.dim
    assert len(u_c) == gdim
    u_values = np.zeros((len(u_c.x.array) // gdim, 3), dtype=np.float64)
    u_values[:, :gdim] = u_c.x.array.real.reshape((-1, gdim))

    # Create a point cloud of glyphs
    u_grid["u"] = u_values
    glyphs = u_grid.glyph(orient="u", factor=scale)
    pyvista.set_jupyter_backend("static")
    plotter = pyvista.Plotter()
    plotter.add_mesh(u_grid, show_edges=False, show_scalar_bar=False)
    plotter.add_mesh(glyphs)
    plotter.view_xy()
    plotter.show()

    p_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(p_c.function_space))
    p_grid.point_data["p"] = p_c.x.array
    plotter_p = pyvista.Plotter()
    plotter_p.add_mesh(p_grid, show_edges=False)
    plotter_p.view_xy()
    plotter_p.show()

    pyvista.set_jupyter_backend("html")

# -

# A convenience function for visualizing the velocity and pressure fields
# is found by expanding the cell below

# + tags=["hide-input"]
visualize_mixed(wh)
# -

# What we want to do instead is to only apply the boundary condition to the velocity sub-space.
# We do this by first getting a function in the sub-space of the velocity field:

W0 = W.sub(0)
V, V_to_W0 = W0.collapse()

# ```{admonition} What is the difference between W0 and V?
# :class: dropdown note
# Whey you call the sub-command on a dolfinx function space (or function), you get a view into the sub-space,
# i.e. you will get a dofmap that only contains the degrees of freedom that are in the sub-space.
# However, the global dof numbering is still preserved, meaning that a dof in the sub-space will have
# the same index as in the global space.
# It also means that a function accessed through `u.sub(i)` will contain all the degrees of freedom in the global space.
# We use `W0.collapse()` to get a self contained function space of only the degrees of freedom in the sub-space.
# We also get back a map frome each degree of freedom in the sub space to the degree of freedom in the parent space.
# This can be super-useful when we want to assign data from collapsed sub spaces to the parent space!
# ```
# With the collapsed function space, we can create a function in the sub-space of the velocity field

u_D = dolfinx.fem.Function(V)
u_D.x.array[:] = 0.11

# Next, we want to fund the dofs of the collapsed sub space, and map them to the parent space

sub_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, tb_facets)

# However, these dofs are in a blocked (vector space) and one would have to expand them for block size

unrolled_sub_dofs = np.empty(len(sub_dofs) * V.dofmap.bs, dtype=np.int32)
for i, dof in enumerate(sub_dofs):
    for j in range(V.dofmap.bs):
        unrolled_sub_dofs[i * V.dofmap.bs + j] = dof * V.dofmap.bs + j

# We can now map them to the parent space

parent_dofs = np.asarray(V_to_W0)[unrolled_sub_dofs]

# We could also do this as a one-liner with

combined_dofs = dolfinx.fem.locate_dofs_topological((W0, V), mesh.topology.dim - 1, tb_facets)

# + tags=["remove-input"]

sort_order_mixed = np.argsort(combined_dofs[0])
sort_order_sub = np.argsort(parent_dofs)
np.testing.assert_allclose(combined_dofs[0][sort_order_mixed], parent_dofs[sort_order_sub])
np.testing.assert_allclose(combined_dofs[1][sort_order_mixed], unrolled_sub_dofs[sort_order_sub])

# -

# Let us create a Dirichlet condition with these degrees of freedom.
# We now pass in the prescribing function `u_D` first, then the tuple of `(parent_dofs, sub_dofs)`.
# As a final argument, we tell the Dirichlet condition what space we are working with, in this case `W0`

new_bc = dolfinx.fem.dirichletbc(u_D, combined_dofs, W0)
new_wh = dolfinx.fem.Function(W)
new_bc.set(new_wh.x.array)
new_wh.x.scatter_forward()

# + tags=["remove-input"]
visualize_mixed(new_wh)
# -

# ```{admonition} Sub-spaces of sub spaces
# :class: note
# As we in some cases only want to constrain one of the components of the vector space, say $\mathbf{u}=(u_x, u_y)$, $u_y=h(x,y)$, while $u_x$
# is unconstrained. We can do this by repeating the strategy above, but with `W.sub(0).sub(1)` and its corresponding collapsed space.
# ```
# We can of course do the same for the pressure conditions, by collapsing the second sub-space.

# For the cases above $(u_x, u_y)$ where constant in both directions, thus you might wonder why we did just send
# in a constant value. This was simply done for illustrative purposes.
# We will illustrate the final way of applying a Dirichlet condition

uy_D = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.3))
sub_dofs = dolfinx.fem.locate_dofs_topological(W.sub(0).sub(1), mesh.topology.dim - 1, left_facets)
bc_constant = dolfinx.fem.dirichletbc(uy_D, sub_dofs, W.sub(0).sub(1))

# Note that this looks quite similar to how we sent in collapsed sub functions.
# However, we do no longer require a map from the the collapsed space (reflected in the input to `locate_dofs_topological`).
# ```{warning} Dirichlet boundary conditions on sub-spaces with a block size
# In DOLFINx, we can not use the above syntax for a vector constant that is applied to W.sub(0).
# One can either constrain each component with a constant individually, or use a function as shown previously.
# ```

# + tags=["hide-input"]
newer_wh = dolfinx.fem.Function(W)
bc_constant.set(newer_wh.x.array)
newer_wh.x.scatter_forward()
visualize_mixed(newer_wh)
# -

# For the following exercise, we set `ux=1`, `uy=0` and $\mathbf{g}=\mathbf{f}=0$.
# ```{admonition} Solve the Stokes problem on the unit square
# :class: dropdown hint
# As the Stokes problem is linear, we can set up a solver similar to {ref}`scipy-lu`.
# ```
# Expand the below cell to see the solution

# + tags=["hide-input"]
W0 = W.sub(0)
V, V_to_W0 = W0.collapse()
u_inlet_x = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
u_inlet_y = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))
dofs_inlet_x = dolfinx.fem.locate_dofs_topological(W.sub(0).sub(0), mesh.topology.dim - 1, left_facets)
dofs_inlet_y = dolfinx.fem.locate_dofs_topological(W.sub(0).sub(1), mesh.topology.dim - 1, left_facets)
bc_inlet_x = dolfinx.fem.dirichletbc(u_inlet_x, dofs_inlet_x, W.sub(0).sub(0))
bc_inlet_y = dolfinx.fem.dirichletbc(u_inlet_y, dofs_inlet_y, W.sub(0).sub(1))

u_wall = dolfinx.fem.Function(V)
u_wall.x.array[:] = 0
dofs_wall = dolfinx.fem.locate_dofs_topological((W0, V), mesh.topology.dim - 1, tb_facets)
bc_wall = dolfinx.fem.dirichletbc(u_wall, dofs_wall, W0)

a_compiled = dolfinx.fem.form(a)
L_compiled = dolfinx.fem.form(L)
A = dolfinx.fem.create_matrix(a_compiled)
b = dolfinx.fem.create_vector(L_compiled)
A_scipy = A.to_scipy()
bcs = [bc_inlet_x, bc_inlet_y, bc_wall]
dolfinx.fem.assemble_matrix(A, a_compiled, bcs=bcs)

dolfinx.fem.assemble_vector(b.array, L_compiled)
dolfinx.fem.apply_lifting(b.array, [a_compiled], [bcs])
b.scatter_reverse(dolfinx.la.InsertMode.add)
[bc.set(b.array) for bc in bcs]

import scipy.sparse

A_inv = scipy.sparse.linalg.splu(A_scipy)

wh = dolfinx.fem.Function(W)
wh.x.array[:] = A_inv.solve(b.array)
visualize_mixed(wh, scale=0.1)
# -


# ```{admonition} A matrix block system
# We notice that we can write this system as a block matrix system

# $$
# \begin{bmatrix}
# A & B \\
# B^T & 0
# \end{bmatrix}
# \begin{bmatrix}
# \mathbf{u} \\
# \bar p
# \end{bmatrix}
# = \begin{bmatrix}
# L_u \\
# 0 \\
# \end{bmatrix}
# $$

# ```
