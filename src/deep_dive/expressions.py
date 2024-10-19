# # Function and expression evaluation
# In this section, we will look at how we can extract data once we have solved our variational problem.
# To make this section concise, we will use functions with expressions that are already defined (not through a PDE),
# but through interpolation.

# We start by creating a 3D mesh

# +
from mpi4py import MPI
import dolfinx
import numpy as np

N = 2
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([2, 1.3, 0.8])],
    [N, N, N],
    dolfinx.mesh.CellType.tetrahedron,
    ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
)
# -

# We start by considering a scalar function in a discontinuous space

V = dolfinx.fem.functionspace(mesh, ("DG", 2))
u = dolfinx.fem.Function(V)

# ## Interpolation on a subset of cells
# We would like to interpolate the function
#
# $$
# \begin{align}
# u = \begin{cases}
# x[0] & \text{if } x[0] < 1\\
# x[1] & \text{otherwise}
# \end{cases}
# \end{align}
# $$

# We start by locating the cells that satisfies this condition.
# In {ref}`boundary_subset` we learnt how to use `dolfinx.mesh.locate_entities_boundary` to locate entites.
# In this section we will use `dolfinx.mesh.locate_entities` which does the same, but for all entities in the mesh.

tdim = mesh.topology.dim
left_cells = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[0] <= 1 + 1e-14)
right_cells = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[0] >= 1 - 1e-14)

# + tags=["remove-input"]
num_cells_local = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
assert np.union1d(left_cells, right_cells).size == num_cells_local
assert np.intersect1d(left_cells, right_cells).size == 0
# -

# We can now interpolate the function onto each of these subsets

u.interpolate(lambda x: x[0], cells0=left_cells)
u.interpolate(lambda x: x[1], cells0=right_cells)

# Whenever we interpolate on sub-sets of cells, we need to scatter forward the values

u.x.scatter_forward()

# + tags=["remove-input"]
import pyvista
import os, sys

if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)


def visualize_scalar(u: dolfinx.fem.Function, scale=1.0):
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data["u"] = u.x.array
    plotter = pyvista.Plotter()
    plotter.add_mesh(u_grid, show_edges=False)
    plotter.show_axes()
    plotter.show_bounds()
    plotter.show()


visualize_scalar(u)
# -

# ## Evalation at a point
# We want to evaluate the function at a point in the mesh that does not align with the nodes of the mesh.
# We do this through a sequence of steps:
#
# 1. As the mesh will be distributed on multiple processes, we have to determine which processes that has the point.
# 2. Then, for the processes that has the point, we need to figure out what cell the point is in.
# 3. Finally, we push this point back to the reference element, so that we can evaluate the basis functions,
# combine them with the coefficients on the given cell and push them forward to the physical space.

# ### Step 1: Determine which processes that has the cell
# As looping through all the cells, and compute exact collisions is expensive, we accelerate the search by using an axis-aligned
# bounding box tree. This is a tree that recursively divides the mesh into smaller and smaller boxes, such that we can quickly
# search through the tree to find the cells that might contain the point of interest.

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

# ```{note} Bounding boxes of other entities
# As seen below, we send in the topological dimension of the entities that we are interested in.
# This means that you can make a bounding box tree for facets, edges or vertices as well if it is needed.
# ```

# ```{admonition} Bounding boxes of subsets of entities
# :class: dropdown note
# In many scenarios, we already have a notion about which entities we are interested in, we can create a bounding box tree
# of only these entities by sending in the entities as a list.
# ```

sub_bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim, left_cells)

# We can now send the points into `dolfinx.geometry.compute_collisions_point` to find the cells that contain the point.
# ```{admonition} Point collision in parallel
# :class: dropdown note
# Note that each bounding box tree is local to the given process, and thus we can send in the same point to all processs,
# or unique points for each process.
# ```

points = np.array([[0.51, 0.32, 0], [1.3, 0.834, 0]], dtype=mesh.geometry.x.dtype)
potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, points)

# + tags=["remove-input"]
print(f"{potential_colliding_cells=}")
# -

# ```{admonition} Output of compute point collisions
# As we observe above, we get an `AdjacencyList` out of the function `compute_collisions_points`.
# This is a list of lists, where the $ith$ list contains the indices of the cells that has bounding
# boxes that collide with the $i$th point.
# ```

# + tags=["remove-input"]
print(f"{potential_colliding_cells.links(0)=}")
# -

# ### Step 2: Determine what cells the point is in
# Now that we have a reduced number of cells to more carefully examine, we can use
# the Gilbert–Johnson–Keerthi distance algorithm {term}`GJK`.
# To find the cells whose convex hull contains the point.
# ```{admonition} Higher order meshes
# :class: dropdown note
# As the GJK algorithm work on convex hulls, it is not 100 % accurate for higher order geometries (i.e. coordinate element is higher order,
# and the facets are curved). However, it is usually a sufficiently good enough approximation to be used in practice.
# For a more accurate algorithm, one could take all the points that satisfies the boundary box collision, pull them back to the reference element,
# and check that the resulting coordinate are inside the reference element. However, for non-affine grids this involves solving a non-linear problem
# (with a Newton type method).
# ```
# To do this efficiently, we use `dolfinx.geometry.compute_colliding_cells`
# which takes in the output of `compute_collisions_points` and the `points`.

colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, potential_colliding_cells, points)

# + tags=["remove-input"]
print(f"{colliding_cells=}")
# -

# If a point is on an interface between two cells, we can get multiple cells that contain the point.
# Now, to prepare for step 3, we reduce the set of points and cells to those that are colliding on the
# current process.

points_on_proc = []
cells = []
for i, point in enumerate(points):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

# ### Step 3: Evaluate function at a point in the cell
# We now have on step left, which is to evaluate the function at the point in the cell.
# As we saw above, this step involves quite a few operations.
# Thankfully, these are all encapsulated in the function `dolfinx.fem.Function.eval`.

points_on_proc = np.array(points_on_proc, dtype=np.float64).reshape(-1, 3)
cells = np.array(cells, dtype=np.int32)
u_values = u.eval(points_on_proc, cells)

# + tags=["remove-input"]

print(f"{points_on_proc=}")
print(f"{u_values=}")


def exact_function(x):
    left_cond = (x[0] <= 1 + 1e-14).astype(np.float64)
    right_cond = (x[0] >= 1 - 1e-14).astype(np.float64)
    return left_cond * x[0] + right_cond * x[1]


u_ex = exact_function(points_on_proc.T).T
np.testing.assert_allclose(u_ex.flatten(), u_values.flatten())
# -
