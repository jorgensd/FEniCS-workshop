# (ufl-intro)=
# # Introduction to the Unified Form Language
#
# We have previously seen how to define a finite element, and evaluate its basis functions in points on the
# reference element.
# However, in this course we aim to solve problems from solid mechanics.
# Thus, we need more than the basis functions to efficiently solve the problems at hand.
#
# In this section, we will introduce the Unified Form Language {term}`UFL`, which is a domain-specific language for
# defining variational formulations for partial differential equations.

# ```{admonition} Symbolic problems
# :class: note
# In the following sections we will not be talking about a specific domain $\Omega$ or any boundary conditions.
# This is due to the fact the UFL is a symbolic language to represent variational problems, and the problems are
# domain inpdendent, as they are when you write the mathematics on paper.
# ```

# We start by creating a symbolic representation of a computational domain $\Omega$.

# ## The domain $\Omega$
# As discussed in the previous section, we can use a finite element to describe a mapping from a reference element to
# any element in the physical space.
# We call this element the *coordinate element*

# +
import basix.ufl

coordinate_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
# -

# To recap:
# - The first input to `basix.ufl.element` is the name of the finite element family we want to use.
# - The second input is what cells the computational domain will consist of.
# - The third input is the degree of the finite element space we want to use to describe the domain.
# - The fourth argument is a tuple telling basix what the dimension of the points in the physical space are in.
# Maybe add a forth argument to the code as well?
#
# For triangular elements, the last argument can either be `(2, )` or `(3, )`.
# If the input is `(3, )` we are describing a 2D manifold embedded in 3D.

# In the FEniCS project, we currently support the following cells:
# - `"vertex"`
# - `"interval"`
# - `"triangle"`
# - `"tetrahedron"`
# - `"hexahedron"`
# - `"prism"` (wedges)

# As discussed earlier, we could use higher order Lagrange elements to describe curved cells.

# We represent the computational domain with the `ufl.Mesh` class

# +

import ufl

domain = ufl.Mesh(coordinate_element)
# -

# ## The finite element
# Next, we want to describe the finite element our unknown $u_h$ will live in.
# I thinks it sounds strange that the unknown lives in the finite element
# In DOLFINx, we are not limited to *iso-parameteric* elements, that is elements that match the coordinate element.

# In this problem, we choose a *sub-parametric* element, that is, our unknown will have more degrees of freedom than
# the computational domain.

cell = str(domain.ufl_cell())
el = basix.ufl.element("Lagrange", cell, 3)

# We could also make a tensor space, for instance a Lagrange element describing a `(M, N)` tensor can be made with

M = 3
N = 2
tensor_el = basix.ufl.element("Lagrange", cell, 2, shape=(M, N))

# or a vector with 7 components with

vector_element = basix.ufl.element("Lagrange", cell, 3, shape=(7,))

# ```{admonition} Vector-valued finite elements
# :class: warning
# In the previous section, we encountered the "N1curl" element, which has vector valued basis functions.
# For these function spaces, it does not make sense to send in a `shape` variable. Instead, if we want multiple "N1curl spaces, we use
# the `basix.ufl.mixed_element` function
# ```

curl_el = basix.ufl.element("N1curl", cell, 1)
blocked_el = basix.ufl.mixed_element([curl_el for _ in range(4)])

# We can also make more advanced elements, for instance by enriching a linear Lagrange element.
# ```{admonition} Enrichment of a Lagrange function
# A piecewise linear Lagrange function on a triangle spans: $\{1, x, y\}$.
# We want to enrich this space with a degree [tree bubble element](https://defelement.com/elements/examples/triangle-bubble-3.html).
# A degree three bubble element is the element whose functional is $l_0: v\mapsto v\left(\frac{1}{3},\frac{1}{3}\right)$ and
# is zero at all edges of the reference triangle. This means that it spans $xy(1-x-y)$.
# I did not understand this part.
# We thus want to create an element with the dual basis $l_0$, $l_1$, $l_2$, $l_3$ that spans $\{1, x, y, xy(1-x-y)\}$.
# ```
# We do this with basix in the following way

enriched_element = basix.ufl.enriched_element(
    [basix.ufl.element("Lagrange", cell, 2), basix.ufl.element("Bubble", cell, 3)]
)

# This element is often used for fluid flow problems, where one has an element describing the velocity, and one describing the pressure.
# This means that we want to create an element with multiple blocks.

el_u = basix.ufl.blocked_element(enriched_element, shape=(2,))

# This was a bit confusing. el_u is the element for the velocity right? I didn't understand why we need to use blocked_element here. Is it becase we want to make a mixed element with an enriched element?

# We can create a mixed element for a mixed problem with

el_p = basix.ufl.element("Lagrange", cell, 1)
el_mixed = basix.ufl.mixed_element([el_u, el_p])

# This mixed element is known as the [MINI element](https://defelement.com/elements/mini.html)
# and is often used for the Stokes problem.

# Another example is the [Taylor-Hood](https://defelement.com/elements/taylor-hood.html) which we can create with

import basix.ufl

vector_element = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
scalar_element = basix.ufl.element("Lagrange", "triangle", 1)

# To create the Taylor-Hood finite element pair, we use the `basix.ufl.mixed_element`

m_el = basix.ufl.mixed_element([vector_element, scalar_element])

# ## Discontinuous (broken) elements
# Any finite element made with basix can be made discontinuous.
# This means that all degrees of freedom will be associated with the cell, and not with the vertices, edges or facets.
# I.e. consider a broken P1 space

el_dg = basix.ufl.element("Lagrange", cell, 1, discontinuous=True)

# This means that at any vertex shared between $M$ cells, there will be $M$ unique basis functions, that do not couple between the cells.
# In a finite element formulation, one would have to add extra integrals to ensure that these degrees of freedom are coupled.

# For the elements of the Lagrange family, we can use the family name "DG" to indicate that the space should be discontinous, i.e.

other_el_dg = basix.ufl.element("DG", cell, 1)
assert el_dg == other_el_dg
