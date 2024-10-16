# # Motivation: Why use the finite element method?
#
# We want to start with a simple example to illustrate the basic idea behind the finite element method,
# which is sub-dividing the domain of interest $\Omega$ into smaller sub-domains, called elements, and
# on each element represent a function $u_h$ by a linear combination of simple basis functions, e.g. polynomials.
#
# Certain problems can be **solved analytically**, but only for specific boundary conditions,
# material properties, and geometries.
# ## Selection of approximation functions
#
# There are many different choices we could choose for approximating $g$:
# 1. Global polynomial approximation (e.g. piecewise linear, quadratic, cubic, etc.)
# 2. Finite Fourier series
# 3. Piecewise polynomial approximation (e.g. Taylor series)
#
# When solving {term}`PDE`s, we will encounter **singularities** and **non-smooth solutions** (e.g. kinks).
# Both these features make global polynomial approximation and Fourier series less attractive.
#
#
# We will start by trying to approximate $g(x) = x \sin(\pi x) \cos (3\pi x)$ on the interval $[0, 1]$.

# +tags=["hide-input"]
from mpi4py import MPI
import sys
import numpy as np
import pyvista
import scipy.sparse

import dolfinx
import ufl


def approximate_function(N: int, degree: int):
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)

    def g(x):
        return x[0] + np.sin(np.pi * x[0]) * np.cos(3 * np.pi * x[0])

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    u = dolfinx.fem.Function(V)
    u.interpolate(g)

    # Warp solution function and smoothen for higher order polynomial
    pv_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
    pv_grid.point_data["u"] = u.x.array
    warped = pv_grid.warp_by_scalar("u", normal=[0, 1, 0])
    warped_tessellate = warped.tessellate()

    # Compute reference solution
    x_ref = np.linspace(0, 1, 1000)
    g_ref = g(x_ref.reshape(1, -1))

    # warp grid nodes to match the solution
    lin_V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    lin_u = dolfinx.fem.Function(lin_V)
    lin_u.interpolate(u)
    lin_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(lin_V))
    lin_grid.point_data["u"] = lin_u.x.array
    lin_warped = lin_grid.warp_by_scalar("u", normal=[0, 1, 0])
    if sys.platform == "linux":
        pyvista.start_xvfb(0.05)
    pyvista.set_jupyter_backend("static")
    plotter = pyvista.Plotter()
    plotter.add_lines(
        np.vstack([x_ref, g_ref, np.zeros_like(x_ref)]).T, connected=True, color="red", label="Exact", width=3
    )
    plotter.add_mesh(warped_tessellate, color="b", style="wireframe", label="Approximation", line_width=3)
    plotter.add_mesh(lin_warped, color="b", style="points", point_size=10)
    plotter.view_xy()
    plotter.add_legend(face="triangle")

    if pyvista.OFF_SCREEN:
        plotter.screenshot("approximation.png")
    else:
        plotter.show()
    pyvista.set_jupyter_backend("html")


approximate_function(5, 1)
# -

# We could then increase the number of elements used

# + tags=["hide-input"]
approximate_function(10, 1)
# -


# ## Motivating example: Heat equation with different materials
# This can for instance be seen in heat transfer equation between different materials.
# We define a domain $\Omega\in \mathbb{R}^d$ as the union of two disjoint domains $\Omega_0$ and $\Omega_1$
#
# $$
# \begin{align*}
#  \Omega_0\cup\Omega_1&=\Omega,\\
#  \Omega_0\cap\Omega_1&=\Gamma\subset{R}^{d-1}.
# \end{align*}
# $$
#
# ### Material
# $c$ is the coefficient of thermal diffusivity
#
# $$
# c(x)=\begin{cases}
# c_0 & \text{if } x\in\Omega_0,\\
# c_1 & \text{if } x\in\Omega_1,
# \end{cases}
# $$
#
# which is discontinuous at $\Gamma$, while the governing partial differential equation yields
#
#

# + tags=["hide-input"]

# Define domain

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 100, 100, cell_type=dolfinx.mesh.CellType.quadrilateral)
Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
c = dolfinx.fem.Function(Q)
c.x.array[:] = 0.1
cells_left = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, lambda x: x[0] <= 0.7 + 1e-12)
c.x.array[cells_left] = 0.01
c.x.scatter_forward()
if sys.platform == "linux":
    pyvista.start_xvfb(0.05)
c_plotter = pyvista.Plotter()
c_mesh = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
c_mesh.cell_data["c"] = c.x.array
c_plotter.add_mesh(c_mesh)
if pyvista.OFF_SCREEN:
    c_plotter.screenshot("heat_coefficient.png")
else:
    c_plotter.show()
# -

# $$
# \begin{align*}
# \frac{\partial T}{\partial t}-\nabla \cdot(c \nabla ) T &= f \quad \text{in} \quad \Omega \\
#  u &= g \quad \text{on} \quad \partial \Omega \\
#  \left(c \nabla T \cdot n\right) \vert_{\Omega_1} &=\left( c \nabla T \cdot n\right) \vert_{\Omega_0}
# \quad \text{on} \quad \Gamma.
# \end{align*}
# $$
#

# + tags=["hide-input"]

# An illustration of such a problem can be found below

# Define function space, test and trial functions

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define previous time step solution/initial condition

T_n = dolfinx.fem.Function(V)
T_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

# Define time stepping and problem specific data
dt = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.01))
g = dolfinx.fem.Function(V)
g.x.array[:] = 0
f = dolfinx.fem.Function(V)
f.x.array[:] = 0


# Define boundary conditions
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
bcs = [dolfinx.fem.dirichletbc(g, boundary_dofs)]

# Define variational problem
F = (
    ufl.inner(T - T_n, v) * ufl.dx
    + dt * c * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    - dt * ufl.inner(f, v) * ufl.dx
)
a, L = dolfinx.fem.form(ufl.system(F))

# Define linear algebra structures
A = dolfinx.fem.create_matrix(a)
b = dolfinx.fem.create_vector(L)


t = 0
T_end = 0.1
Th = dolfinx.fem.Function(V)
Th.x.array[:] = T_n.x.array

# Assemble LHS matrix and invert once with scipy
A.data[:] = 0
dolfinx.fem.assemble_matrix(A, a, bcs=bcs)
A_scipy = A.to_scipy()
Ainv = scipy.sparse.linalg.splu(A_scipy)

while t < T_end:
    t += float(dt)

    # Assemble RHS vector and apply bcs through lifting
    b.array[:] = 0
    dolfinx.fem.assemble_vector(b.array, L)
    dolfinx.fem.apply_lifting(b.array, [a], [bcs])
    [bc.set(b.array) for bc in bcs]

    # Solve linear system
    Th.x.array[:] = Ainv.solve(b.array)
    T_n.x.array[:] = Th.x.array


plotter = pyvista.Plotter()
mesh_pyvista = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
mesh_pyvista.point_data["T"] = Th.x.array
warped_grid = mesh_pyvista.warp_by_scalar("T")
plotter.add_mesh(warped_grid, show_edges=True, edge_color="black")
if pyvista.OFF_SCREEN:
    plotter.screenshot("heat_transfer.png")
else:
    plotter.show()


# -
