# # Using compiled forms in DOLFINx
#
# As DOLFINx is a C++ framework with a Python interface, we could use the generated C-code directly in our DOLFINx C++ programs.
# However, we could also use the abstract formulations within the DOLFINx framework.
# In this section, we will explore how we can use code written in the exact same way as in the previous sections within the
# DOLFINx Python framework.
# We start by creating the variational formulation of a Poisson problem in two dimensions
#
# $$
# \begin{align}
# -\nabla \cdot (\nabla u) &= f \qquad\text{in } \Omega\subset\mathbb{R}^2.\\
# u &= g \qquad \text{on } \partial\Omega.
# \end{align}
# $$

# We start as before by defining the abstract finite element formulation

# +
import ufl
import basix.ufl

c_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
domain = ufl.Mesh(c_el)
# -

# ## Manufactured solutions

# For the example we will solve in this section, we will use a manufactured solution.
# This means that we derive a source term $f$ and appropriate boundary conditions $g$ from a given $u$.
# For instance, if we choose $u_{ex}=\sin(\pi/2 x)\cos(\pi y)$, we can derive an $f$ by inserting this into
# our initial equation
#
# $$
# \begin{align}
# -\nabla \cdot \nabla u_{ex} = -\frac{\partial^2 u_{ex}}{\partial x^2} -\frac{\partial^2 u_{ex}}{\partial y^2} = 5/4\pi^2(\sin(\pi/2 x)\cos(\pi y))
# \end{align}
# $$

# We can formulate this directly in UFL with `SpatialCoordinate` and the different derivative operators

# +


def u_exact(module, x):
    return module.sin(module.pi / 2 * x[0]) * module.cos(module.pi * x[1])


x = ufl.SpatialCoordinate(domain)
u_ex = u_exact(ufl, x)
f = -ufl.div(ufl.grad(u_ex))
el = basix.ufl.element("Lagrange", "triangle", 2)
V = ufl.FunctionSpace(domain, el)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
# -

# ## Scalar, vector or tensor?
# Previously we have not focused too much on what the output of `F` would be.
#
# ```{admonition} Are the expressions above going to output scalars, vectors or tensors (matrices)?
# :class: dropdown tip
# Whenever there are only constants or coefficients in a form, it will be assembled to a scalar value.
# If there is either a `TestFunction` or a `TrialFunction` in the variational form, it will assemble into a
# vector in the appropriate space, as one compute the contributions for each $\phi_i$.
# If there is both a `TestFunction(V)` and a `TrialFunction(Q)` in the variational form, it will assemble into a
# matrix, as we compute contributions $\phi_i\in V$, $\psi_j\in Q$ for both spaces. Note that $V$ and $Q$ can be the
# same space. If they are not the same space we get a rectangular marix
# ```
#
# The form `F` above contains a mixture of terms, some involving just the test function and others involving both the
# test and trial function. We can use `ufl.system` to split this into a bi-linear and linear form.

a, L = ufl.system(F)

# We want to measure the accuracy in our solution, which we can do by computing the $L^2$-error

uh = ufl.Coefficient(V)
error = ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx

# ## The discrete problem
# We choose to use a unit-square for our problem, we create a mesh (a subdivision of the unit square into triangles)
# ```{admonition} What is mpi4py?
# :class: dropdown note
# DOLFINx supports distributed computing using the Message Passing Interface ({term}`MPI`) to pass data between processes.
# In the setting of this tutorial, we will use `MPI.COMM_WORLD`, which runs on as many processes as you specify when running
# a Python program with `mpiexec -n python3 name_of_script.py`. We use the Python interface for MPI, `mpi4py`
# to use this from Python.
# ```

# +
from mpi4py import MPI
import dolfinx

Nx = 13
Ny = 27
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny, cell_type=dolfinx.mesh.CellType.triangle)
# -

# As we previously have discussed, we are going to determine a set of coefficients $u_i$ for the given domain.
# We create a `dolfinx.fem.FunctionSpace` that contains the information for `mesh` with

V = dolfinx.fem.functionspace(mesh, el)

# ```{admonition} Consistency between ufl and DOLFINx
# :class: note
# Note that we have used the element defined in our abstract formulation above.
# ```
# To store the coefficients $u_i$, we create a `dolfinx.fem.Function`

x = dolfinx.fem.Function(V)

# We inspect the initial coefficients

# + tags=["remove-input"]
print(f"{x.x.array=}")
# -

# Next we create a function that can hold the boundary conditions.
# We use interpolation to set coefficients of the boundary condition.
#
# ```{admonition} What is interpolation?
# :class: dropdown note
# We can write the interpolant $\bar g$ of $g(x,y)$ as
#
# $$
# \bar g(x) = \sum_{i=0}^{n-1}l_i(g)\phi_i(x)
# $$
#
# For point evaluations this amounts to setting the coefficient for
# the $i$th basis function to the value $g(x)$ at that point in this physical element.
# ```
#
# We can compute the interpolated version of the exact solution with DOLFINx by
# passing in a Python function to `dolfinx.fem.Function.interpolate`.
# This function takes in a set of coordinates (x,y,z) written as a single two-dimensional
# array of shape `(3, num_points)`, meaning that we can perform vectorized operations on all
# of the `i`the coordinate with `x[i]`, i.e `np.sin(x[0])` evaluates an array of points at once.

# +
import numpy as np

gh = dolfinx.fem.Function(V)
gh.interpolate(lambda x: u_exact(np, x))
# -

# + tags=["remove-input"]
print(f"{gh.x.array=}")
# -

# ## Dirichlet boundary conditions
# Now that we have created a function that represent the boundary condition, we can create a
# Dirichlet boundary condition object.
# The specifics of what happens under the hood when applying such conditions will be covered in
# {ref}`lifting`.
# We will limit our-selfs to discuss how to decide what degrees of freedom we want to apply conditions to.
# We start by noting that for the problem at hand, we want to constrain all external boundaries.
# This means that we want to locate any degree of freedom that is associated with either a
# vertex, edge or a facet that can be defined as being on the boundary.
# We start by locating all facets that are only connected to a single cell.
# The fastest way to do this is with

mesh.topology.create_entities(mesh.topology.dim - 1)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

# ```{admonition} Creating entities
# :class: dropdown note
# When a mesh is created in DOLFINx we only compute the  **unique** global numbering
# and ownership of cells and vertices.
# However, one can compute this numbering for any sub-entity by calling
# `mesh.topology.create_entities(j)`.
# For instance computing the ownership and numbering of the facets can be done with
# `mesh.topology.create_entities(mesh.topology.dim-1)`.
# ```
# ```{admonition} Creating connectivity
# :class: dropdown note
# When a mesh is created in DOLFINx, the relation between the cells and the vertices is computed.
# However, one can compute the relationship between any set of entities in the mesh with
# `dolfinx.mesh.create_connectivity(i, j)` where `i` is the dimension of the index to map from
# and `j` is the dimension of the entities to map from.
# As an example, calling `mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)`
# will compute the relationship between all facets in the mesh and their cells.
# The inverse map (cell-to-facet) is computed with
# `mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)-1`
# ```
#
# With these two preliminaries computed, DOLFINx can determine which facets that are connected to
# a single cell only.

# We can access the connectivity with

facet_to_cell = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)

# and get the indices of the cells a facet is connected to with `facet_to_cell.links(i)`
#
# ```{admonition} Given a unit square, verify that the exterior facet indices are only connected to a single cell.
# :class: dropdown tip
# Loop through the exterior facet indices and check the length of `facet_to_cell.links(i)`
# ```
#
# Expand the code below for the answer

# + tags=["hide-input"]
for facet in boundary_facets:
    assert len(facet_to_cell.links(facet)) == 1
# -

# Now we can find all dofs in the closure of these facets, with

boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

# ```{admonition} The dof-closure
# :class: dropdown note
# When we want to find all degrees of freedom associated with an entity, and all sub entities of a lower topological dimension,
# we call this the closure dofs of that entity.
# ```
# And we can create the `dolfinx.fem.DirichletBC` object

bc = dolfinx.fem.dirichletbc(gh, boundary_dofs)
bcs = [bc]

# We are now ready to create the system matrix and vector. However, to do this,
# we need to compile the symbolic forms `a` and `L`.

a_compiled = dolfinx.fem.compile_form(MPI.COMM_WORLD, a)
L_compiled = dolfinx.fem.compile_form(MPI.COMM_WORLD, L)

# This has generated and compiled the C kernels for the local matrix assembly.
# We now associate these with the data we have created for the given mesh:

bilinear_form = dolfinx.fem.create_form(a_compiled, [V, V], mesh, {}, {}, {})

# ```{admonition} What data does the form require?
# :class: dropdown note
# We need to associate function spaces with the `TestFunction` and `TrialFunction`
# We also send in what mesh we want to integrate over.
# We will cover the other inputs later.
# ```
# We repeat the same procedure for the linear form

linear_form = dolfinx.fem.create_form(L_compiled, [V], mesh, {}, {}, {})

# We now create DOLFINx native stuctures for storing the sparse matrix and right hand side vector

A = dolfinx.fem.create_matrix(bilinear_form)
b = dolfinx.fem.create_vector(linear_form)

# ```{admonition} Why do we create the matrix explicitly?
# As the matrix is sparse, we want to pre-compute the sparsity pattern of the matrix before inserting
# data into it. `dolfinx.fem.create_matrix` estimates the sparsity pattern based on the variational
# form and creates the appropriate CSR matrix.
# ```
# We can get a view into `A` by creating a data wrapper compatible with scipy.

A_scipy = A.to_scipy()

# + tags=["remove-input"]
print(f"{A_scipy=}")
# -

# Next, we assemble the global matrix

dolfinx.fem.assemble_matrix(A, bilinear_form, bcs=bcs)

# and the global vector

dolfinx.fem.assemble_vector(b.array, linear_form)
dolfinx.fem.apply_lifting(b.array, [bilinear_form], [bcs])
b.scatter_reverse(dolfinx.la.InsertMode.add)
[bc.set(b.array) for bc in bcs]

# We can use scipy sparse LU solver to invert the matrix

# +
import scipy.sparse

A_inv = scipy.sparse.linalg.splu(A_scipy)
# -

# We can now apply this to any right hand side

# +
x.x.array[:] = A_inv.solve(b.array)
# -

# (error_estimation)=
# ## Error estimation
# We created a symbolic expression for computing the error above

compiled_error = dolfinx.fem.compile_form(MPI.COMM_WORLD, error)

# As we now need to supply `x` to replace `uh`, from the definition of error, we do this by adding
# {u: x} in the second of the three dictionaries in `dolfinx.fem.create_form`

L2_error_form = dolfinx.fem.create_form(compiled_error, [], mesh, {}, {uh: x}, {})

# We use `dolfinx.fem.assemble_scalar` to compute the integral over cells owned by the current process
# and accumulate the result with `mesh.comm.allreduce`

error_local = dolfinx.fem.assemble_scalar(L2_error_form)
error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)

# + tags=["remove-input"]
print(f"{np.sqrt(error_global)=}")
# -

# ## Visualizing the solution
# We use Pyvista to visualize the solution.
# DOLFINx provides data extraction interfaces that are compatible with Pyvista.

# +
import pyvista

mesh_pyvista = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(x.function_space))
# -

# We have now created a pyvista Unstructured grid compatible with the array
# from `u.x.array`.
# ```{warning} Pyvista compatible function spaces
# Pyvista supports arbitrary continuous and discontinuous Lagrange function spaces.
# If you have a function in another finite element space, please interpolate into a
# compatible continuous or discontinuous Lagrange space.
# ```

mesh_pyvista.point_data["x"] = x.x.array

# To visualize both in notebooks, on the web and when running the script,
# we launche a virtiual frame-buffer under certain conditions.

# +
import sys, os

if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)

# Next, we create a plotting instance, and warp the solution grid by the solution.
plotter = pyvista.Plotter()
warped_grid = mesh_pyvista.warp_by_scalar("x")

plotter.add_mesh(warped_grid, show_edges=True, edge_color="black")
plotter.show()
# -

# (rates)=
# ## Convergence rates
#
# For certain finite element formulations we have expected convergence rates.
# We can write this mathematically as
#
# $$
# E_h = \vert\vert u_h - u_{ex} \vert\vert_{L^2(\Omega)}\leq C h^r
# $$
#
# which means that as one refines the mesh, the approximate solution goes towards
# the exact solution with rate $r$.
#
# Given two sub-sequent solutions $u_{h_i}$, $u_{h_j}$ for resolution $h_i$, $h_j$,
# we can compute $E_{h_i}=Ch_i^r$, $E_{h_j}=Ch_j^r$ and solve for $r$
#
# $$
# r = \frac{\ln (E_{h_i}/E_{h_j})}{\ln(h_i/h_j)}
# $$
#
# ```{admonition} Compute the convergence rate for the example above
# :class: dropdown tip
# For a Pth order Lagrange space the convergence rate is $P+1$.
# ```
# Expand the code below to see the solution:

# + tags=["hide-input", "hide-output"]
Ns = np.array([8, 16, 32, 64])
hs = np.zeros_like(Ns, dtype=np.float64)
errors = np.zeros_like(Ns, dtype=np.float64)
for i, N in enumerate(Ns):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=dolfinx.mesh.CellType.triangle)
    V = dolfinx.fem.functionspace(mesh, el)
    x = dolfinx.fem.Function(V)

    gh = dolfinx.fem.Function(V)
    gh.interpolate(lambda x: u_exact(np, x))

    mesh.topology.create_entities(mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(gh, boundary_dofs)
    bcs = [bc]

    bilinear_form = dolfinx.fem.create_form(a_compiled, [V, V], mesh, {}, {}, {})
    linear_form = dolfinx.fem.create_form(L_compiled, [V], mesh, {}, {}, {})

    A = dolfinx.fem.create_matrix(bilinear_form)
    b = dolfinx.fem.create_vector(linear_form)

    A_scipy = A.to_scipy()
    dolfinx.fem.assemble_matrix(A, bilinear_form, bcs=bcs)

    dolfinx.fem.assemble_vector(b.array, linear_form)
    dolfinx.fem.apply_lifting(b.array, [bilinear_form], [bcs])
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    [bc.set(b.array) for bc in bcs]

    A_inv = scipy.sparse.linalg.splu(A_scipy)
    x.x.array[:] = A_inv.solve(b.array)
    L2_error_form = dolfinx.fem.create_form(compiled_error, [], mesh, {}, {uh: x}, {})

    error_local = dolfinx.fem.assemble_scalar(L2_error_form)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)

    errors[i] = np.sqrt(error_global)
    hs[i] = 1.0 / N


def compute_converence(hs, errors):
    return np.log(errors[1:] / errors[:-1]) / np.log(hs[1:] / hs[:-1])


print(f"{hs=}")
print(f"{errors=}")
print(f"{compute_converence(hs, errors)=}")
# -
