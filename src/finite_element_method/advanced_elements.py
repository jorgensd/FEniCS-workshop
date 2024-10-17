# # Advanced finite elements

# There is a wide range of finite elements that are supported by ufl and basix.
# See for instance: [Supported elements in basix/ufl](https://defelement.com/lists/implementations/basix.ufl.html).

# In this section, we will cover a sub-set of important element "classes".

# ## Mixed finite elements

# Not every function we want to represent is scalar valued.
# For instance, in fluid flow problems, the [Taylor-Hood](https://defelement.com/elements/taylor-hood.html)
# finite element pair is often used to represent the fluid velocity and pressure.
# For the velocity, each component (x, y, z) is represented with its own degrees of freedom in a Lagrange space..
# We represent this by adding a `shape` argument to the `basix.ufl.element` constructor.

import basix.ufl
vector_element = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
scalar_element = basix.ufl.element("Lagrange", "triangle", 1)

# Basix allows for a large variety of extra options to tweak your finite elements, see for instance
# [Variants of Lagrange elements](https://docs.fenicsproject.org/dolfinx/v0.8.0/python/demos/demo_lagrange_variants.html)
# for how to choose the node spacing in a Lagrange element.

# To create the Taylor-Hood finite element pair, we use the `basix.ufl.mixed_element`

m_el = basix.ufl.mixed_element([vector_element, scalar_element])

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
# For simplicity