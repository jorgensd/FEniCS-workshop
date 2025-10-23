# # Integration over sub-domains
# So far we have only considered integration over the entire domain, or an entire boundary.
# However, in many PDEs we are interested in integrating over a sub-domain or a sub-set of the boundary.
# We will start by creating a mesh, and dividing it into two distinct sub-domains.

# +
from mpi4py import MPI

import numpy as np

import dolfinx

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 10, 10, dolfinx.mesh.CellType.triangle, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)


def inner_square(x, tol=1e-13):
    return (0.3 - tol <= x[0]) & (x[0] <= 0.8 + tol) & (x[1] <= 0.6 + tol) & (0.4 - tol <= x[1])


tdim = mesh.topology.dim
cell_map = mesh.topology.index_map(tdim)
num_cells = cell_map.size_local + cell_map.num_ghosts
all_cells = np.arange(num_cells, dtype=np.int32)
marker = np.ones(num_cells, dtype=np.int32)
marker[dolfinx.mesh.locate_entities(mesh, tdim, inner_square)] = 2

# + tags=["remove-input"]

import pyvista

ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
ugrid.cell_data["Marker"] = marker
plotter = pyvista.Plotter()
plotter.add_mesh(ugrid, show_edges=True)
plotter.view_xy()
plotter.show()
# -

# Previously we have only worked directly with mesh indices.
# Now we will wrap them in a `dolfinx.mesh.MeshTags` object.
# ```{admonition} The meshtags object
# :class: dropdown note
# A DOLFINx MeshTags object is a container for store integer values that one wants
# to associate with a set of entities in the mesh.
# The first input to the `meshtags` constructor is the `mesh` object associated with the
# `entities` (the third input) of a given dimension `dim` (the second input).
# The final input is a list of values where the `i`th entry in values is associated with the `i`th entity in `entities`.
# Note that the input entities should be a list of **sorted** and **unique** indices.
# The entities does not need to be contiguous, i.e. `entities = [0, 3, 5 9]` would be a valid input.
# ```

cell_tag = dolfinx.mesh.meshtags(mesh, tdim, np.arange(num_cells, dtype=np.int32), marker)

# We can attach the `MeshTags` object to an integration measure, and use it in the assembly process.

# +
import ufl

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tag)
# -

# We check that the measure is the same as `ufl.dx`:

default_assembly = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh)))
new_assembly = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * dx))
assert default_assembly == new_assembly

# + tags=["remove-input"]
print(f"{new_assembly=}")
# -

# The key is that we can now state that we only want to integrate over cells associated with a specific marker value.
# For instance, we can integrate over the inner domain with

inner_vol = dolfinx.fem.form(1 * dx(2))
local_volume = dolfinx.fem.assemble_scalar(inner_vol)

# + tags=["remove-input"]
print(f"{local_volume=}")
# -

# ```{admonition} Integrate over a subset of the boundary
# :class: tip
# Choose a sub-set of the boundary, and create a meshtags object for only the tagged entities.
# Create a `ufl.Measure` with subdomain data, and verify that the assembly over the whole domain and the domain
# restriction is correct.
# ```

# ## Custom integration entitites
# Some times, one wants to do something custom when integrating.
# For instance, one might want to do a one-sided integral over an internal surface.
# We start by creating such a surface

inner_facets = dolfinx.mesh.locate_entities(mesh, tdim - 1, lambda x: np.isclose(x[0], 0.5))

# Next, we want to find all cells that are on the left side of this surface.
# We could do this in many ways. One way is to check if the midpoint of the cell is on the left side of the surface.

midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, np.arange(num_cells, dtype=np.int32))

# Next, we loop over all the facets, find the connected cells, find the cell that is in the "correct" side of the boundary,
# compute its **integration entity** and store it in an array.

# +
mesh.topology.create_connectivity(tdim - 1, tdim)
mesh.topology.create_connectivity(tdim, tdim - 1)
c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
f_to_c = mesh.topology.connectivity(tdim - 1, tdim)

integration_entities = np.empty((len(inner_facets), 2), dtype=np.int32)
for i, facet in enumerate(inner_facets):
    cells = f_to_c.links(facet)
    assert len(cells) == 2, "Facet is not connected to two cells"
    for cell in cells:
        if midpoints[cell][0] < 0.5:
            local_facets = c_to_f.links(cell)
            facet_idx = np.flatnonzero(local_facets == facet)
            assert len(facet_idx) == 1, "Facet is not connected to cell"
            integration_entities[i] = (cell, facet_idx[0])
# -

# + tags=["remove-input"]
print(f"{integration_entities=}")
# -

# We can pass these integration entities into a `ufl.Measure` on a specific form:

subdomain_data = [(7, integration_entities.flatten())]
ds = ufl.Measure("ds", subdomain_data=subdomain_data, domain=mesh)

# which states that for exterior facet integrals written in ufl as `ds(7)` we will only integrate over the specified facets.

# We will verify that the integration entities are correct by integrating against the `ufl.FacetNormal` and check that it is correct.

# +
x = ufl.SpatialCoordinate(mesh)
expression = (x[0] + x[1] ** 2) * ufl.FacetNormal(mesh)
correct_normal = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((1, 0)))
facet_integral = dolfinx.fem.form(ufl.inner(expression, correct_normal) * ds(7))

integral = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(facet_integral), op=MPI.SUM)
# -

# + tags=["remove-input"]
print(f"{integral=}")
assert np.isclose(integral, 0.5 + 1 / 3)
# -
