# # Galerkin methods
#
# In the previous section, we observed that we can approximate a known function $g$
# by a set of piecewise polynomial functions $\phi_i$.
#
# We did not go much detail about how these functions would look like.
# In this section, we look at this is some more detail.

# We start by looking at an interval, subdivided into a set of elements.
# For each of these segments, we will define a **local** basis function and a set of **global** basis functions.

# ## Composition of $u_h$
#
# Given a finite dimensional function space $V_h$, with basis functions $\phi_i, i=0,\dots,N-1$, we can write any
# function $u_h$ in $V_h$ as
#
# $$
# \begin{align*}
#   u_h(x) &= \sum_{i=0}^{N-1} u_i \phi_i(x)
# \end{align*}
# $$
#
# where $u_i$ are the coefficients of function.
# These coefficients are also known as the degrees of freedom {term}`DOFs` of the function.

# + tags=["remove-input"]

import sys
import vtk  # noqa
import basix.ufl
import numpy as np
import pyvista
import matplotlib as mpl


def compute_physical_point(points, X, cell, degree):
    """
    Map coordinates `X` in reference element to a physical cells
    """
    el = basix.ufl.element("Lagrange", cell, degree)
    basis_values = el.tabulate(0, X)
    return basis_values[0] @ points


def plot_interval_basis_functions(N: int, degree: int, g):
    M = 4.5
    nodes = np.linspace(0, M, N + 1, endpoint=True).reshape(1, -1)

    reference_points = basix.create_lattice(basix.CellType.interval, 100, basix.LatticeType.equispaced, exterior=True)
    cells = np.vstack([np.arange(N), np.arange(1, N + 1)]).T
    physical_nodes = []
    for cell in cells:
        physical_nodes.append(compute_physical_point(nodes[0][cell], reference_points, basix.CellType.interval, 1))

    basis_functions = basix.ufl.element("Lagrange", basix.CellType.interval, degree).tabulate(0, reference_points)[0].T

    # Permute basis functions wrt vertices being first
    nbpc = basis_functions.shape[0]
    perm = np.hstack([np.array([0]), np.arange(2, nbpc), np.array([1])])
    ordered_basis_functions = basis_functions[perm]

    if sys.platform == "linux":
        pyvista.start_xvfb(0.05)
    pyvista.set_jupyter_backend("static")
    basis_plotter = pyvista.Plotter()
    colors = mpl.cm.tab20c(np.linspace(0, 1, (nbpc - 1) * N + nbpc))
    pv_nodes = np.vstack([nodes, np.zeros_like(nodes), np.zeros_like(nodes)]).T
    basis_plotter.add_points(pv_nodes, style="points", point_size=10, color="black")

    func_plotter = pyvista.Plotter()
    for i, (cell, p) in enumerate(zip(cells, physical_nodes)):
        g_approx = np.zeros_like(p)
        for j, basis in enumerate(ordered_basis_functions):
            # Find approximate dof position
            dof_pos = np.argmin(np.abs(ordered_basis_functions[j] - 1))
            dof_coord = p[dof_pos]
            g_exact = g(dof_coord)
            g_approx += basis * g_exact
            basis_index = (nbpc - 1) * i + j
            lines = np.vstack([p, basis, np.zeros_like(p)]).T
            basis_plotter.add_lines(lines, connected=True, width=3, color=pyvista.Color(colors[basis_index]))
        g_lines = np.vstack([p, g_approx, np.zeros_like(p)]).T
        func_plotter.add_lines(g_lines, connected=True, width=3, color="blue")

    legends = [[r"$\phi_{k}$".format(k="{" + f"{i}" + "}"), colors[i]] for i in range(len(colors))]
    basis_plotter.add_legend(legends, loc="center right")
    basis_plotter.view_xy()
    basis_plotter.show_grid()
    _x = np.linspace(0, M, 100)
    func_plotter.add_lines(
        np.vstack([_x, g(_x), np.zeros_like(_x)]).T, connected=True, label=r"$g$", color="red", width=3
    )
    func_plotter.add_points(pv_nodes, style="points", point_size=10, color="black")

    legends = [[r"$u_h$", "red"], [r"$g$", "blue"]]
    func_plotter.add_legend(legends, loc="center right")
    func_plotter.view_xy()
    func_plotter.show_grid()
    if not pyvista.OFF_SCREEN:
        basis_plotter.show()
        func_plotter.show()
    pyvista.set_jupyter_backend("html")


plot_interval_basis_functions(5, 1, lambda x: x + 3 * np.sin(np.pi * x))

# -

# We can do the same thing for a higher order set of polynomials

# + tags=["remove-input"]
plot_interval_basis_functions(5, 2, lambda x: x + 3 * np.sin(np.pi * x))
# -

#

# However, in the setting of a {term}`PDE`, we do not know the solution $u$.
#
# In this section, we will explain how to find the solution $u$ to a {term}`PDE` using the Galerkin method.
#
# We will consider the Poisson equation in 1D as a starting point
#
# $$
# \begin{align*}
# -\frac{\partial^2 u}{\partial x^2} &= f(x) \quad \text{in} \quad \Omega = [0, 1] \\
# u(0) &=0 \\
# u(1) &=0
# \end{align*}
# $$
#


# If we substitute $u_h$ into the Poisson equation, we get
#
# $$
# \begin{align*}
# -\sum_{i=0}^{N-1}u_i\frac{\partial^2 \phi_i}{\partial x^2} &= f(x) \quad \text{in} \quad \Omega = [0, 1]
# \end{align*}
# $$
