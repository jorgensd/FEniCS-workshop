# # Multiphysics: Solving PDEs on subdomains
# So far we have considered problems where the PDE is defined over the entire domain.
# However, in many cases this is not accurate. An example of this is fluid-structure interaction,
# where the fluid and solid domains are coupled.
# In this case, the PDEs are defined over different subdomains,
# and the coupling is done at the interface between the subdomains.
# In this section, we will show how to solve PDEs on subdomains using FEniCS.
#
# We will consider a simple problem where we have a domain $\Omega$ that is divided into two subdomains $\Omega_1$ and $\Omega_2$.
# In each of these domains we want to solve a PDE (that is not coupled to the other domain).
#
# We will consider the following PDEs:
#
# $$
# \begin{align*}
# - \nabla \cdot (\kappa \nabla T) &= f \quad \text{in } \Omega_1, \\
# \kappa \nabla T \cdot \mathbf{n} &= g \quad \text{on } \Gamma, \\
# T &= g \quad \text{ on } \partial \Omega_1\setminus\Gamma, \\
# - \nabla \cdot ( \nabla \mathbf{u}) - \nabla \bar p &= \mathbf{f} \quad \text{in } \Omega_2 \\
# \nabla \cdot \mathbf{u} &= 0 \quad \text{in } \Omega_2 \\
# \mathbf{u} &= \mathbf{h} \text{ on } \partial {\Omega_{2,D}} \\
# \nabla \mathbf{u} \cdot \mathbf{n} + \bar p \mathbf{n} &= \mathbf{0} \quad \text{on } \partial_{\Omega_{2, N}}
# \end{align*}
# $$
#
# We start by creating our mesh, a unit square, that we will split into two compartments
#
# $$
# \begin{align*}
# \Omega_1 &= [0.7, 1] \times [0, 1]\\
# \Omega_2 &= [0, 0.7] \times [0, 1]\\
# \end{align*}
# $$

# +
from mpi4py import MPI

import numpy as np

import dolfinx

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 40, 40)


def Omega1(x, tol=1e-14):
    return x[0] <= 0.7 + tol


def Omega0(x, tol=1e-14):
    return 0.7 - tol <= x[0]


tdim = mesh.topology.dim
cell_map = mesh.topology.index_map(tdim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
marker = np.empty(num_cells_local, dtype=np.int32)
heat_marker = 1
stokes_marker = 3

marker[dolfinx.mesh.locate_entities(mesh, tdim, Omega0)] = heat_marker
marker[dolfinx.mesh.locate_entities(mesh, tdim, Omega1)] = stokes_marker

cell_tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), marker)

# -

# + tags=["remove-input"]
assert len(np.unique(marker)) <= 2
# -

# This section will contain alot of figures, to illustrate the different steps.
# Expand to see the code for creating a plotter of meshes and meshtags.

# + tags=["hide-input"]

import pyvista


def plot_mesh(mesh: dolfinx.mesh.Mesh, tags: dolfinx.mesh.MeshTags = None):
    plotter = pyvista.Plotter()
    mesh.topology.create_connectivity(tdim - 1, tdim)
    if tags is None:
        ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    else:
        # Exclude indices marked zero
        exclude_entities = tags.find(0)
        marker = np.full_like(tags.values, True, dtype=np.bool_)
        marker[exclude_entities] = False
        ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, tags.dim, tags.indices[marker]))
        print(tags.indices[marker], tags.values[marker])
        ugrid.cell_data["Marker"] = tags.values[marker]

    plotter.add_mesh(ugrid, show_edges=True, line_width=3)
    plotter.show_axes()
    plotter.show()


plot_mesh(mesh, cell_tags)

# -

# Next we define the boundaries for our sub-domains

# + tags=["hide-input"]

mesh.topology.create_connectivity(tdim - 1, tdim)
facet_map = mesh.topology.index_map(tdim - 1)
num_facets_local = facet_map.size_local + facet_map.num_ghosts
facet_values = np.zeros(num_facets_local, dtype=np.int32)
outer_marker = 4
facet_values[dolfinx.mesh.exterior_facet_indices(mesh.topology)] = 4


def inlet(x, tol=1e-14):
    return np.isclose(x[0], 0.0) & ((x[1] >= 0.4 - tol) & (x[1] <= 0.6 + tol))


inlet_marker = 1
facet_values[dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, inlet)] = inlet_marker


def outlets(x, tol=1e-14):
    return (np.isclose(x[1], 0.0) & ((0.4 - tol <= x[0]) & (x[0] <= 0.6 + tol))) | (
        np.isclose(x[1], 1.0) & ((0.2 - tol <= x[0]) & (x[0] <= 0.35 + tol))
    )


outlet_marker = 2
facet_values[dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, outlets)] = outlet_marker

interface_marker = 3
facet_values[dolfinx.mesh.locate_entities(mesh, tdim - 1, lambda x: np.isclose(x[0], 0.7))] = interface_marker

facet_tags = dolfinx.mesh.meshtags(mesh, tdim - 1, np.arange(num_facets_local, dtype=np.int32), facet_values)
plot_mesh(mesh, facet_tags)

with dolfinx.io.XDMFFile(mesh.comm, "tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tags, mesh.geometry)

# -

# Next, we want to extract a sub-mesh containing only the the cellsthat will be used in the Stokes problem.
# We do this with `dolfinx.mesh.create_submesh`

stokes_mesh, stokes_cell_map, stokes_vertex_map, _ = dolfinx.mesh.create_submesh(
    mesh, cell_tags.dim, cell_tags.find(stokes_marker)
)

# ```{admonition} Creating a submesh
# :class: dropdown note
# `dolfinx.mesh.create_submesh` takes in three inputs:
# 1. The mesh we want to extract a mesh from
# 2. The dimension of the entities we want to make the mesh from. This can be any number $[0, tdim]$.
# We define a submesh consisting of a subset of cells from the input mesh is a submesh of **co-dimension 0**,
# while a submesh consisting fo facets from the input mesh is a submesh of **co-dimension 1**
# 3. A list of integers defining the entities (index local to process, and including ghosted entities)
#
# The function returns four objects
# 1. The submesh (as a `dolfinx.mesh.Mesh`)
# 2. A map from each entity cell in the submesh to the corresponding entity in the input mesh
# 3. A map from each **vertex** in the submesh to the corresponding vertex in the input mesh topology
# 4. A map from each **node** in the submesh to the corresponding node in the input mesh geometry
# ```

# ## Transfering meshtags to the sub-mesh
# There are many ways on could use the created sub-mesh.
# We will start by treating it as a normal mesh.
# For this, we would like to transfer the `cell_tags` and `facet_tags` defined on the whole mesh.
# We do this with the following function

# + tags=["hide-input"]

import numpy.typing as npt


def transfer_meshtags_to_submesh(
    mesh: dolfinx.mesh.Mesh,
    entity_tag: dolfinx.mesh.MeshTags,
    submesh: dolfinx.mesh.Mesh,
    sub_vertex_to_parent: npt.NDArray[np.int32],
    sub_cell_to_parent: npt.NDArray[np.int32],
) -> tuple[dolfinx.mesh.MeshTags, npt.NDArray[np.int32]]:
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.

    Args:
        mesh: Mesh containing the meshtags
        entity_tag: The meshtags object to transfer
        submesh: The submesh to transfer the `entity_tag` to
        sub_to_vertex_map: Map from each vertex in `submesh` to the corresponding
            vertex in the `mesh`
        sub_cell_to_parent: Map from each cell in the `submesh` to the corresponding
            entity in the `mesh`
    Returns:
        The entity tag defined on the submesh, and a map from the entities in the
        `submesh` to the entities in the `mesh`.

    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(len(sub_cell_to_parent), dtype=np.int32)
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(child_vertices_as_parent, p_f_to_v.links(facet)).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet
    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map


# -

stokes_facet_tags, stokes_facet_map = transfer_meshtags_to_submesh(
    mesh, facet_tags, stokes_mesh, stokes_vertex_map, stokes_cell_map
)

# + tags=["remove-input"]
plot_mesh(stokes_mesh, stokes_facet_tags)
# -

# ## Solve Stokes problem on submesh
# We can now solve the Stokes problem on this sub-mesh
#
# ```{admonition} Solve the Stokes problem on the submesh
# :class: dropdown tip
# See the section [Mixed problems](../deep_dive/mixed_problems.py) for hints on how to set up
# relevant boundary conditions and the variational form
# ```
# Expand the below dropdown to see the solution

# + tags=["hide-input"]

import scipy

import basix
import ufl

# Define variational form on submesh
el_u = basix.ufl.element("Lagrange", stokes_mesh.basix_cell(), 3, shape=(stokes_mesh.geometry.dim,))
el_p = basix.ufl.element("Lagrange", stokes_mesh.basix_cell(), 2)
el_mixed = basix.ufl.mixed_element([el_u, el_p])
W = dolfinx.fem.functionspace(stokes_mesh, el_mixed)
u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)
wh = dolfinx.fem.Function(W)
uh, ph = ufl.split(wh)
g = dolfinx.fem.Constant(stokes_mesh, dolfinx.default_scalar_type((0, 0)))
f = dolfinx.fem.Constant(stokes_mesh, dolfinx.default_scalar_type((0, 0)))
F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
F += ufl.inner(p, ufl.div(v)) * ufl.dx
F += ufl.inner(ufl.div(u), q) * ufl.dx
F -= ufl.inner(f, v) * ufl.dx
a, L = ufl.system(F)

# Create boundary conditions
W0 = W.sub(0)
V, V_to_W0 = W0.collapse()
u_wall = dolfinx.fem.Function(V)
u_wall.x.array[:] = 0
stokes_walls = np.union1d(stokes_facet_tags.find(outer_marker), stokes_facet_tags.find(interface_marker))
dofs_wall = dolfinx.fem.locate_dofs_topological((W0, V), mesh.topology.dim - 1, stokes_walls)
bc_wall = dolfinx.fem.dirichletbc(u_wall, dofs_wall, W0)
u_inlet = dolfinx.fem.Function(V)
u_inlet.interpolate(lambda x: (0.5 * x[1], 0 * x[0]))
dofs_inlet = dolfinx.fem.locate_dofs_topological((W0, V), mesh.topology.dim - 1, stokes_facet_tags.find(inlet_marker))
bc_inlet = dolfinx.fem.dirichletbc(u_inlet, dofs_inlet, W0)
bcs = [bc_wall, bc_inlet]

# Compile form and assemble system
a_compiled = dolfinx.fem.form(a)
L_compiled = dolfinx.fem.form(L)
A = dolfinx.fem.create_matrix(a_compiled)
b = dolfinx.fem.create_vector(L_compiled)
A_scipy = A.to_scipy()
dolfinx.fem.assemble_matrix(A, a_compiled, bcs=bcs)
dolfinx.fem.assemble_vector(b.array, L_compiled)
dolfinx.fem.apply_lifting(b.array, [a_compiled], [bcs])
b.scatter_reverse(dolfinx.la.InsertMode.add)
[bc.set(b.array) for bc in bcs]

# Solve with SPLU
import scipy.sparse

A_inv = scipy.sparse.linalg.splu(A_scipy)
wh = dolfinx.fem.Function(W)
wh.x.array[:] = A_inv.solve(b.array)
# -


# + tags=["hide-input"]
def visualize_function(function: dolfinx.fem.Function, scale=1.0):
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(function.function_space))
    plotter = pyvista.Plotter()
    if function.function_space.dofmap.bs == 1:
        u_grid.point_data[function.name] = function.x.array
        plotter_p = pyvista.Plotter()
        plotter_p.add_mesh(u_grid, show_edges=False)
        plotter_p.view_xy()
        plotter_p.show()

    else:
        # Pad u to be 3D
        gdim = function.function_space.mesh.geometry.dim
        assert len(function) == gdim
        u_values = np.zeros((len(function.x.array) // gdim, 3), dtype=np.float64)
        u_values[:, :gdim] = function.x.array.real.reshape((-1, gdim))

        # Create a point cloud of glyphs
        u_grid[function.name] = u_values
        glyphs = u_grid.glyph(orient=function.name, factor=scale)

        plotter = pyvista.Plotter()
        plotter.add_mesh(u_grid, show_edges=False, show_scalar_bar=False)
        plotter.add_mesh(glyphs)
        plotter.view_xy()
        plotter.show()


uh = wh.sub(0).collapse()
ph = wh.sub(1).collapse()
visualize_function(uh)
visualize_function(ph)
# -

# ## Solving with integration over full mesh
# As we have seen above, we can create a sub-mesh and use it as one would use any other
# mesh. However, as we want to use this mesh in multi-physics problems, we want to
# exploit the relation to the parent domain.
#
# In this section we will illustrate this for the Poisson problem defined at the top.
# We will define $\kappa$ on the whole domain $\Omega$, and not on the sub-mesh.

K = dolfinx.fem.functionspace(mesh, ("DG", 0))
kappa = dolfinx.fem.Function(K)
kappa.x.array[:] = 25
subset_cells = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[1] > 0.3)
kappa.interpolate(lambda x: 12 * x[0] + x[1], cells0=subset_cells)

# + tags=["hide-input"]
plotter = pyvista.Plotter()
u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
u_grid.cell_data["kappa"] = kappa.x.array
plotter.add_mesh(u_grid, show_edges=True)
plotter.view_xy()
plotter.show()
# -

# We create the submesh and the function space for the temperature on the submesh

heat_mesh, heat_cell_map, heat_vertex_map, _ = dolfinx.mesh.create_submesh(
    mesh, cell_tags.dim, cell_tags.find(heat_marker)
)
heat_facet_tags, _ = transfer_meshtags_to_submesh(mesh, facet_tags, heat_mesh, heat_vertex_map, heat_cell_map)

# However, in this case $\kappa$ lives on the parent mesh.
# There are two ways of handling this:
# 1. Interpolate $\kappa$ onto the `heat_mesh`
# 2. Integrate over a restricted section of the parent mesh

# ### Interpolate to submesh
# We can interpolate data onto a submesh.

K_sub = dolfinx.fem.functionspace(heat_mesh, ("DG", 0))
kappa_sub = dolfinx.fem.Function(K_sub)

# We do this by supplying to lists to interpolate.
# As we have seen before `cells0` relate to what cells in the incoming space (`K_sub`)
# we want to interpolate data onto.
# Now, we also need to pass the information about which cells in `mesh` that relates to
# `cells0`.
# We can retrieve all this information from the `heat_cell_map`

kappa_sub.interpolate(kappa, cells0=heat_cell_map, cells1=np.arange(len(heat_cell_map)))

# + tags=["hide-input"]
plotter = pyvista.Plotter()
u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(heat_mesh))
u_grid.cell_data["kappa_sub"] = kappa_sub.x.array
plotter.add_mesh(u_grid, show_edges=True)
plotter.view_xy()
plotter.show()
# -

# However, as we have already seen how to deal with problems that are pure related to
# quantities on a submesh, we will move to option number 2.

# ## Integration with function from parent and sub-mesh
# The first thing we have to decide on, is which of the domains `mesh` and `heat_mesh`
# that we want to associate with the integration.
# We will use `mesh` as it more easily generalizes to co-dimension 1 problems.

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

# Note that there is only a sub-set of cells in the `dx` measure that is also in `heat_mesh`.
# Therefore we restrict the integral

dx_heat = dx(heat_marker)

# Next, we set up the problem as done previously, with test-functions and trial-functions
# on the `heat_mesh`

# +
T = dolfinx.fem.functionspace(heat_mesh, ("Lagrange", 1))
t = ufl.TrialFunction(T)
dt = ufl.TestFunction(T)

x = ufl.SpatialCoordinate(mesh)
f = 10 * ufl.sin(5 * ufl.pi * x[1]) + x[0] ** 3
F_heat = ufl.inner(kappa * ufl.grad(t), ufl.grad(dt)) * dx_heat - ufl.inner(f, dt) * dx_heat
# -

# ```{warning}
# As we choose `mesh` as the integration domain, all spatial quantities such as `ufl.SpatialCoordinate`
# and `ufl.FacetNormal` should be defined wrt. this domain.
# ```

# We create the boundary condition as before

T_bndry = dolfinx.fem.Function(T)
T_bndry.x.array[:] = 0
heat_mesh.topology.create_connectivity(heat_mesh.topology.dim - 1, heat_mesh.topology.dim)
heat_bc_dofs = dolfinx.fem.locate_dofs_topological(T, heat_mesh.topology.dim - 1, heat_facet_tags.find(outer_marker))
bc_heat = dolfinx.fem.dirichletbc(T_bndry, heat_bc_dofs)
bcs_heat = [bc_heat]

# Note that so far very little have been different from just solving on the submesh.
# However, now an important distinction is introduced, namely the *entity map*
# ```{admonition} Entity maps
# :class: dropdown note
# Entity maps are an input to the `dolfinx.fem.form` function, and should be on the following form:
# The maps is a dictionary, where each key is a `dolfinx.mesh.Mesh` that is part of the form, but is not
# the chosen integration domain. For each key, we have a map that maps an entity from the integrating mesh
# to an entity in the mesh that is the key. *Note* that in the case where we use the full mesh as integration domain,
# this is the inverse map of the one returned by `dolfinx.mesh.create_submesh`.
# ```
# We therefore create this inverse map. As this map is a sparse map (not all cells in the full mesh is in the submesh),
# we use `-1` to indicate that a cell is not part of the submesh.

mesh_to_heat_entity = np.full(num_cells_local, -1, dtype=np.int32)
mesh_to_heat_entity[heat_cell_map] = np.arange(len(heat_cell_map), dtype=np.int32)
entity_maps = {heat_mesh: mesh_to_heat_entity}

# We can now compile our forms

a_heat, L_heat = dolfinx.fem.form(ufl.system(F_heat), entity_maps=entity_maps)

# Now we can solve assemble the system

# +
A_heat = dolfinx.fem.assemble_matrix(a_heat, bcs=bcs_heat)
A_heat_scipy = A_heat.to_scipy()

b_heat = dolfinx.fem.assemble_vector(L_heat)
dolfinx.fem.apply_lifting(b_heat.array, [a_heat], [bcs_heat])
b_heat.scatter_reverse(dolfinx.la.InsertMode.add)
[bc.set(b_heat.array) for bc in bcs_heat]
# -

# ```{admonition} Why did we not pass matrices to assemble vector/matrix?
# In all the previous examples we have seen, we have been explicitly creating the matrices
# and passing them to assemble. There are many good reasons for this if one calls
# assemble multiple times in a program:
# As the matrix is sparse, we want to pre-compute the sparsity pattern of the matrix before inserting
# data into it. `dolfinx.fem.create_matrix` estimates the sparsity pattern based on the variational
# form and creates the appropriate CSR matrix.
# It is cheaper to zero out the initial contributions in the matrix than creating a new one.
# ```

# We can now solve the system

# +
A_heat_inv = scipy.sparse.linalg.splu(A_heat_scipy)

th = dolfinx.fem.Function(T, name="Temperature")
th.x.array[:] = A_heat_inv.solve(b_heat.array)
# -

# + tags=["remove-input"]
visualize_function(th)
# -

# ## Combined assembly
# As we aim to solve multiphysics problems in a monolitic way, we will go through the basic steps of setting up such as system.
# We define the function spaces on the sub-meshes as done previously,
# however, instead of creating a `basix.ufl.mixed_element, we will create individual function spaces for `u` and `p`.

T = dolfinx.fem.functionspace(heat_mesh, ("Lagrange", 1))
V = dolfinx.fem.functionspace(stokes_mesh, ("Lagrange", 2, (stokes_mesh.topology.dim,)))
Q = dolfinx.fem.functionspace(stokes_mesh, ("Lagrange", 1))

# Next, we use `ufl.MixedFunctionSpace` to create a representation of the monolitic problem.

W = ufl.MixedFunctionSpace(T, V, Q)

# We can now create test and trial functions

t, u, p = ufl.TrialFunctions(W)
dt, du, dp = ufl.TestFunctions(W)

# We define the relevant integration meshes.
# For consistency, we will use the `mesh` as the integration domain for both problems.

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
dx_thermal = dx(heat_marker)
dx_fluid = dx(stokes_marker)

# Instead of creating separate definitions of each form, we use a single variational form

g = dolfinx.fem.Constant(stokes_mesh, dolfinx.default_scalar_type((0, 0)))
f_fluid = dolfinx.fem.Constant(stokes_mesh, dolfinx.default_scalar_type((0, 0)))
a = ufl.inner(ufl.grad(u), ufl.grad(du)) * dx_fluid
a += ufl.inner(p, ufl.div(du)) * dx_fluid
a += ufl.inner(ufl.div(u), dp) * dx_fluid
L = ufl.inner(f_fluid, du) * dx_fluid
L += ufl.inner(dolfinx.fem.Constant(stokes_mesh, 0.0), dp) * dx_fluid
x = ufl.SpatialCoordinate(mesh)
f_heat = 10 * ufl.sin(5 * ufl.pi * x[1]) + x[0] ** 3
a += ufl.inner(kappa * ufl.grad(t), ufl.grad(dt)) * dx_heat
L += ufl.inner(f, dt) * dx_thermal

# We can extract the matrix block structure by calling `ufl.extract_blocks`

a_blocked = ufl.extract_blocks(a)
L_blocked = ufl.extract_blocks(L)

# + tags=["remove-input"]
for i in range(3):
    for j in range(3):
        print(f"A_{i, j}={a_blocked[i][j]}")
for i in range(3):
    print(f"L_{i}", L_blocked[i])
# -

# ```{admonition} Extract blocks
# For a bi-linear form, the output of `extract_blocks` is an nested list
# of size M, where each sub-list is also of length M, representing the
# MxM blocked matrix.
# For extract blocks on a linear form, we need to supply an
# integral for each the M components.
# If we fail to do so, we will get a smaller blocked vector.
# Therefore we added a integral $\int_{\Omega_s} 0 * \delta p ~\mathrm{d}x$
# to ensure that we get the right block structure.
# ```

# We create the appropriate inverse maps and compile the forms

mesh_to_heat_entity = np.full(num_cells_local, -1, dtype=np.int32)
mesh_to_heat_entity[heat_cell_map] = np.arange(len(heat_cell_map), dtype=np.int32)
mesh_to_stokes_entity = np.full(num_cells_local, -1, dtype=np.int32)
mesh_to_stokes_entity[stokes_cell_map] = np.arange(len(stokes_cell_map), dtype=np.int32)
entity_maps = {heat_mesh: mesh_to_heat_entity, stokes_mesh: mesh_to_stokes_entity}
a_blocked_compiled = dolfinx.fem.form(a_blocked, entity_maps=entity_maps)
L_blocked_compiled = dolfinx.fem.form(L_blocked, entity_maps=entity_maps)

# We create the boundary condition as before

# +
T_bndry = dolfinx.fem.Function(T)
T_bndry.x.array[:] = 0
heat_mesh.topology.create_connectivity(heat_mesh.topology.dim - 1, heat_mesh.topology.dim)
heat_bc_dofs = dolfinx.fem.locate_dofs_topological(T, heat_mesh.topology.dim - 1, heat_facet_tags.find(outer_marker))
bc_heat = dolfinx.fem.dirichletbc(T_bndry, heat_bc_dofs)

stokes_walls = np.union1d(stokes_facet_tags.find(outer_marker), stokes_facet_tags.find(interface_marker))
dofs_wall = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, stokes_walls)
u_wall = dolfinx.fem.Function(V)
u_wall.x.array[:] = 0
bc_wall = dolfinx.fem.dirichletbc(u_wall, dofs_wall)
u_inlet = dolfinx.fem.Function(V)
u_inlet.interpolate(lambda x: (0.5 * x[1], 0 * x[0]))
dofs_inlet = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, stokes_facet_tags.find(inlet_marker))
bc_inlet = dolfinx.fem.dirichletbc(u_inlet, dofs_inlet)

bcs = [bc_heat, bc_wall, bc_inlet]
# -

# We could now use [scipy.sparse.vstack](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html)
# and [scipy.sparse.hstack]https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html to create the full matrix
# in a scipy compatible format, and solve as we have shown before.

# However, DOLFINx provides wrapper to assemble this into PETSc blocked and PETSc nest matrices.
# We will illustrate how to do this with PETSc blocked matrices.

# +
from petsc4py import PETSc

import dolfinx.fem.petsc

A = dolfinx.fem.petsc.create_matrix_block(a_blocked_compiled)
A.zeroEntries()
dolfinx.fem.petsc.assemble_matrix_block(A, a_blocked_compiled, bcs=bcs)
A.assemble()

b = dolfinx.fem.petsc.create_vector_block(L_blocked_compiled)
dolfinx.fem.petsc.assemble_vector_block(b, L_blocked_compiled, a_blocked_compiled, bcs=bcs)
b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
# -

# ```{admonition} Assembly of blocked matrices and vectors
# :class: dropdown note
# We observe that assembling blocked matrices looks like how we assemble native matrices.
# However, we observe that the lifting procedure is embedded in `assemble_block_vector`,
# as this code gets a bit compilcated for block systems.
# ```
#

# We create the KSP object and solve the system

# +
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.getPC().setFactorSolverType("mumps")

w_blocked = dolfinx.fem.petsc.create_vector_block(L_blocked_compiled)

ksp.solve(b, w_blocked)
assert ksp.getConvergedReason() > 0, "Solve failed"
# -

# We extract the individual functions from the blocked solution vector and visualize them

# +
blocked_maps = [(space.dofmap.index_map, space.dofmap.index_map_bs) for space in W.ufl_sub_spaces()]
local_values = dolfinx.cpp.la.petsc.get_local_vectors(w_blocked, blocked_maps)

Th = dolfinx.fem.Function(T, name="Temperature")
uh = dolfinx.fem.Function(V, name="Velocity")
ph = dolfinx.fem.Function(Q, name="Pressure")
Th.x.array[:] = local_values[0]
uh.x.array[:] = local_values[1]
ph.x.array[:] = local_values[2]
# -

# + tags=["remove-input"]
visualize_function(uh)
visualize_function(ph)
visualize_function(Th)
# -
