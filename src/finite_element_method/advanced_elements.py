# # Advanced finite elements

# There is a wide range of finite elements that are supported by ufl and basix.
# See for instance: [Supported elements in basix/ufl](https://defelement.com/lists/implementations/basix.ufl.html).

# ## Domain mapping

# The general idea of the finite element method is to sub-divide $\Omega$ into
# smaller (polygonal) elements $K_j$ such that
# 1) The triangulation covers $\Omega$: $\cup_{j=1}^{M}K_j=\bar{\Omega}$
# 2) No overlapping polyons: $\mathrm{int} K_i \cap \mathrm{int} K_j=\emptyset$ for $i\neq j$.
# 3) No vertex lines in the interior of a facet or edge of another element
#
# We will call our polygonal domain $\mathcal{K}={K_j}_{j=1}^{M}$.
# Next, we define a reference element $K_{ref}$, which is a simple polygon that we can map to any element $K_j$,
# using the mapping $F_j:K_{ref}\mapsto K_j$.
#
# We define the Jacobian of this mapping as $\mathbf{J_j}$.
#
# (straight_edge_triangle)=
# ### Example: Straight edged triangle
#
# As we saw in [the section on finite elements](./finite_element), we can use basix to get a
# sample of points within the reference element.

# + tags = ["hide-input"]
import numpy as np

import basix.ufl

reference_points = basix.create_lattice(
    basix.CellType.triangle, 8, basix.LatticeType.gll, exterior=False, method=basix.LatticeSimplexMethod.warp
)
# -

# Next, we realize that we can use the first order Lagrange space, to represent the mapping from the
# reference element to any physical element:
# Given three points, $\mathbf{p}_0=(x_0, y_0)^T, \mathbf{p}_1=(x_1,y_1)^T, \mathbf{p}_2=(x_2,y_2)^T$, we can represent any point $x$
# as the linear combination of the three basis functions on the reference element $X$.
#
# $$x = F_j(X)= \sum_{i=0}^3 \mathbf{p}_i \phi_i(X).$$
#
# In the next snippet we will create a function to compute `x` given the three points and a set of reference coordinates

def compute_physical_point(p0, p1, p2, X):
    """
    Map coordinates `X` in reference element to triangle defined by `p0`, `p1` and `p2`
    """
    el = basix.ufl.element("Lagrange", "triangle", 1)
    basis_values = el.tabulate(0, X)
    x = basis_values[0] @ np.vstack([p0, p1, p2])
    return x

# We can now experiment with this code

p0 = np.array([2.0, 1.8])
p1 = np.array([1.0, 1.2])
p2 = np.array([1.3, 1.0])
x = compute_physical_point(p0, p1, p2, reference_points)

# + tags=["remove-input"]
print(f"Reference points: {reference_points}")
print(f"Physical points: {x}")
# -

# We use matplotlib to visualize the reference points and the physical points

# + tags=["hide-input"]
import matplotlib.pyplot as plt

theta = 2 * np.pi
phi = np.linspace(0, theta, reference_points.shape[0])
rgb_cycle = (
    np.stack((np.cos(phi), np.cos(phi - theta / 4), np.cos(phi + theta / 4))).T + 1
) * 0.5  # Create a unique colors for each node

fig, (ax_ref, ax) = plt.subplots(2, 1)
# Plot reference points
reference_vertices = basix.cell.geometry(basix.CellType.triangle)
ref_triangle = plt.Polygon(reference_vertices, color="blue", alpha=0.2)
ax_ref.add_patch(ref_triangle)
ax_ref.scatter(reference_points[:, 0], reference_points[:, 1], c=rgb_cycle)
# Plot physical points
vertices = np.vstack([p0, p1, p2])
triangle = plt.Polygon(vertices, color="blue", alpha=0.2)
ax.add_patch(triangle)
_ = ax.scatter(x[:, 0], x[:, 1], c=rgb_cycle)
# -

# ### Exercises:

# ```{admonition} Can we use a similar kind of mapping on a quadrilateral/tetrahedral/hexahedral element?
# :class: dropdown tip
# Yes, for any polytope that we can describe with a Lagrange finite element, we can define a mapping from a reference to a physical element.
# ```
# ```{admonition} What happens if we change the order of the basis functions?
# :class: dropdown tip
# We require more nodes to describe our polygon. For instance, if we use a second order Lagrange element on a triangle, we would require 3 additional
# nodes to describe the midpoint of each edge.
# ```
# ```{admonition} How can we compute the Jacobian of the mapping?
# :class: dropdown tip
# As we now have an expression $F=x(X)$, we can compute the Jacobian by taking the derivative of $F$ with respect to $x$, i.e.
# $J = \frac{\partial F}{\partial X}=\sum_{i=0}^3 \mathbf{p}_i (\nabla \phi_i(X))^T$.
# ```

# ## Mapping of basis functions from the reference element
#
# As we have already seen, we can describe any cell in our subdivided domain with a mapping from the reference element.
# However, as we want to integrate over each element individually, we need to map the basis functions to and from the reference element.
# We call this map: $(\mathcal{F}_j\phi)(x)$.
#
# For scalar valued basis functions, such as the Lagrange basis function, this operation is simple
#
# $$
# (\mathcal{F}_j\phi)(x)= \phi(F_j^{-1}(x))=\phi(X).
# $$

# ## Vector-valued finite elements
#
# The basis functions we have considered so far have been scalar-valued, i.e. for each {term}`DOF`
# $u_i$ we have a scalar valued basis function $\phi_i$.

# In this section we will consider vector-valued basis functions
# 
# $$
# \phi_i(x): \mathbb{R}^n \mapsto \mathbb{R}^n.
# $$
#
#
# Instead of defining the dual basis $l_i$ as point evaluations, these dual-basis functions are often defined as integrals
# over a sub-entity of the reference cell, i.e. an edge, facet or the cell itself.
#
# We will considered the finite element called [Nédélec (first kind)](https://defelement.com/elements/nedelec1.html).
# For simplicity we will consider the first order element on a triangle, where:
#
# $$
# \begin{align*}
#  l_i: \mathbf{v}\mapsto \int_{e_i} \mathbf{v} \cdot \mathbf{t_i}\mathrm{d} s
# \end{align*}
# $$
#
# where $t_i$ is the tangent to the edge $e_i$.

# <center>
# <img src="https://defelement.com/img/element-Nedelec-variant-equispaced-triangle-1-0-large.png"
# width="250" height="250" />
# <img src="https://defelement.com/img/element-Nedelec-variant-equispaced-triangle-1-1-large.png"
# width="250" height="250" />
# <img src="https://defelement.com/img/element-Nedelec-variant-equispaced-triangle-1-2-large.png"
# width="250" height="250" /><br>
#  Nédélec (first kind) basis functions<br><br>
# </center>

# ```{admonition} What properties do the basis functions above have?
# :class: dropdown, tip
# We have that the basis functions are tangential to the edge that they are defined on.
# ```

# To preserve the properties of the basis functions from the reference element to the physical element,
# we use the covariant Piola map:
#
# $$
# \begin{align*}
# (\mathcal{F}_j^{\text{curl}}\phi)(x):= J_j^{-T} \phi(F_j^{-1}(x))
# \end{align*}
# $$

# The noteable feature of this map is that it preserves the **tangential** component of the basis function.
# We start by computing the Jacobian of the mapping from the reference element to the physical element.

# ```{admonition} `basix.tabulate` for vector valued basis functions
# :class: important
# When the basis function is vector-valued, basix returns the basis functions with the shape
# `(num_derivatives, num_points, vector_dimension, num_basis_functions)`.
# ```

### Exercise: Compute the Jacobian of the mapping from the reference element to the physical element at a single point in the reference elemnt
# Expand to reveal the solution

# + tags=["hide-input"]
# Get some points in the reference element
X = basix.create_lattice(basix.CellType.triangle, 8, basix.LatticeType.equispaced, exterior=True)
num_points, tdim = X.shape

# Tabulate basis function for the coordiante element
el = basix.ufl.element("Lagrange", "triangle", 1)
basis_derivatives = el.tabulate(1, X)[1:]

# Stack the vertices in a matrix
points = np.vstack([p0, p1, p2])

# Compute the basis derivatives in x and y direction at each point in X
dphi_dx = basis_derivatives[0] @ points
dphi_dy = basis_derivatives[1] @ points
# The Jacobian is a (2, 2) tensor at each point
jacobian = np.transpose(np.hstack([dphi_dx, dphi_dy]).reshape(num_points, tdim, tdim), (0, 2, 1))

# We could alternatively use Einstein summation notation
dphidx = el.tabulate(1, X[0:1])[1:].reshape(2, 3)
jac0 = np.einsum("ij,ki->jk", points, dphidx)
np.testing.assert_allclose(jacobian[0], jac0)
# -

# ```{admonition} If we use a straight edged triangle, is the Jacobian spatially dependent?
# :class: dropdown, tip
# No, the Jacobian is constant over the entire element, as the mapping is affine.
# If we instead considered a general quadrilateral or hexahedral element, the Jacobian would be spatially dependent.
# It is only constant in the setting where the quadrilateral is a parallelogram or the hexahedron is a parallelepiped.
# ```

### Exercise: Compute $J^{-T}$
# Expand to reveal the solution

# + tags=["hide-input"]
jacobian_inverse = np.linalg.inv(jacobian)
J_inv_T = np.transpose(jacobian_inverse, (0, 2, 1))
# -

### Exercise: Tabulate the basis functions of a first order Nedelec element on the reference element
# Expand to reveal the solution

# + tags=["hide-input"]
nedelec_el = basix.ufl.element("N1curl", "triangle", 1)
ref_basis = nedelec_el.tabulate(0, X).reshape(num_points, tdim, nedelec_el.dim,)

physical_basis = np.zeros((num_points, points.shape[1], nedelec_el.dim))
for i in range(num_points):
    physical_basis[i] = J_inv_T[i] @ ref_basis[i]
physical_basis = np.transpose(physical_basis, (2, 0, 1))

# -

# We can visualize the basis functions side by side

# + tags=["hide-input"]
x = compute_physical_point(p0, p1, p2, X)
import matplotlib.pyplot as plt
phi = np.linspace(0, theta, num_points)
theta = 2 * np.pi
rgb_cycle = (
    np.stack((np.cos(phi), np.cos(phi - theta / 4), np.cos(phi + theta / 4))).T + 1
) * 0.5  # Create a unique colors for each node
reference_vertices = basix.cell.geometry(basix.CellType.triangle)

for i in range(nedelec_el.dim):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Nedelec basis functions of reference and physical cell', fontsize=16)
    ref_triangle = plt.Polygon(reference_vertices, color="blue", alpha=0.2)
    triangle = plt.Polygon(points, color="blue", alpha=0.2)
    axes[0].add_patch(ref_triangle)
    axes[0].scatter(X[:,0], X[:,1], color=rgb_cycle, label="Reference points")
    axes[0].quiver(X[:, 0], X[:, 1], ref_basis[:, 0, i], ref_basis[:, 1, i])
    axes[0].set_title(r"$\phi_{i}$".format(i="{"+str(i)+"}"))
    axes[0].set_aspect('equal', 'box')

    axes[1].set_title(r"$\mathcal{F}"+r"\phi_{i}$".format(i="{"+str(i)+"}")) 
    axes[1].scatter(x[:,0], x[:,1], color=rgb_cycle, label="Physical points")
    axes[1].quiver(x[:, 0], x[:, 1], physical_basis[i, :, 0], physical_basis[i, :, 1])
    axes[1].add_patch(triangle)
    axes[1].set_aspect('equal', 'box')
    plt.show()
# -

# ### Exercise
# ```{admonition} What property does the covariant Piola mapping preserve?
# :class: dropdown tip
# We observe that the covariant Piola map maps normals to normals, since a 0 tangential component is mapped to a 0 tangent.
# ```

# ```{admonition} Why are these elements useful?
# :class: dropdown tip
# We consider the time-harmonic Maxwell equation {cite}`chenmaxwell` with no source term, 
# which simplifies into In a time-harmonic Maxwell equation for a magnetic field
#
# $$
# \begin{align*}
# \nabla \times (\tilde\epsilon^{-1} \nabla \times \mathbf{H}) - \omega^2\mu \mathbf{H} &= \nabla \times \mathbf{\tilde J},
# \qquad \text{in } \Omega\\
# \nabla \cdot (\mu \mathbf{H}) &= 0,  \qquad \text{in } \Omega\\
# \mathbf{H}\times \mathbf{n} &= 0 \qquad \text{on } \partial \Omega
# \end{align*}
# $$
# where $\omega$ is the frequency of the problem, $\tilde\epsilon$ is the effective permittivity
# ,$\mu$ is the permability, and $\mathbf{\tilde J}$ is the current density.
#
# As done previously, we integrate this equation by parts to obtain a weak formulation.
# By doing so, we end up with
#
# $$
# \begin{align*}
# \int_{\Omega} \nabla \times (\nabla \times \mathbf{H})\cdot \mathbf{v}~\mathrm{d}x &=
# \int_{\Omega} \nabla \times \mathbf{H} \cdot \nabla \times \mathbf{v}~\mathrm{d}x +
# \int_{\partial\Omega} \mathbf{n} \times (\nabla \times \mathbf{H}) \cdot \mathbf{v}~\mathrm{d}s
# \end{align*}
# $$
#
#
# $$
# H(\text{curl}, \Omega)=\{\mathbf{v}\vert \mathbf{v}\in L^2(\Omega), \nabla \times \mathbf{v}\in L^2(\Omega)\}
# $$
#
# to represent the magnetic field $\mathbf{H}$.
#
# It is known that the tangential component of a magnetic field is continuous across an interface between two materials,
# while the normal component is discontinuous.
# i.e. $\mathbf{H}_1\times \mathbf{n}_1 + \mathbf{H}_2 \times\mathbf{n}_2=0$,
# which is also suitable for the Nédélec element.
#
# Considering a problem where $\mathbf{\tilde J}=0$, it can be shown that this equation also
# implies that the field $\mathbf{H}$ is divergence free {cite}`dean2023` (Chapter 3.7).
# ```
#
# ## References
# ```{bibliography}
#    :filter: cited and ({"src/finite_element_method/advanced_elements"} >= docnames)
# ```