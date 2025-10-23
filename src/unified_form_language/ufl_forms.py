# (variational_form)=
# # The UFL forms
# In this section we will show how to set up the following problem
#
# $$
# \min_{u\in V} G(u) = \frac{1}{2}\int_\Omega (u-g)\cdot (u-g)~\mathrm{d}x.
# $$
#
# for some known $g$.
# This problem is known as the $L^2$-projection of $g$ into $u$.
# We can observe this by looking at what solving this problem implies:
# To find the minimum, we need to find where $\frac{\mathrm{d}G}{\mathrm{d}u}$ is $0$.
# We define the residual $F$ as
#
# $$
# F(u, \delta u) := \frac{\mathrm{d}G}{\mathrm{d}u}[\delta u] = 0, \quad \forall \delta u \in V.
# $$
#
# We could compute this by hand and get:
# Find $u_h\in V$ such that
#
# $$
# \int_{\Omega} (u_h-g)  \cdot \delta u ~\mathrm{d} x = 0, \quad \forall \delta u \in V.
# $$
#
# Thus we can define the bi-linear form $a(u, v)$ and linear form $L(v)$
#
# $$
# \begin{align*}
# a(u, v) &= \int_{\Omega} u \cdot v ~\mathrm{d}x\\
# L(v) &= \int_{\Omega} g \cdot v ~ \mathrm{d}x
# \end{align*}
# $$

# We start by defining and abstract domain and a finite element

# +
import basix.ufl
import ufl

cell = "triangle"
c_el = basix.ufl.element("Lagrange", cell, 1, shape=(2,))
domain = ufl.Mesh(c_el)

el = basix.ufl.element("Lagrange", cell, 2)
# -

# ## The functions space
# We define the function space $V$, which will depend on the domain and the finite element

V = ufl.FunctionSpace(domain, el)

# Now we are ready to create the formulations above.
# We can start with defining our unknown $u_h$.
# A function in UFL is described as a {py:class}`ufl.Coefficient`, this means that it can be written as
# $u(x)=\sum_{i=0}^N u_i \phi_i(x)$.

uh = ufl.Coefficient(V)

# ## Spatial derivatives
# If we want to compute the derivative in the $i$th spatial direction,
# we can call {py:meth}`uh.dx<ufl.core.expr.Expr.dx>`

duh_dx = uh.dx(0)
duh_dy = uh.dx(1)

# If we want the gradient, we can call {py:func}`ufl.grad(operator)<ufl.grad>`

grad_uh = ufl.grad(uh)

# We can also use {py:func}`ufl.as_vector` to create our own vectors, for instance

alt_grad_uh = ufl.as_vector((duh_dx, duh_dy))

# ## Representing $g$
# Next, we define our $g$. We will consider four different cases for $g$:
# 1. $g$ is a constant over the whole domain
# 2. $g$ is an expression depending on the spatial coordinates of the domain, i.e. $g=g(x,y,z)$
# 3. $g$ is a function form another finite element function space
# 4. $g$ is a mixture of the above, combined with spatial derivatives

# For case 1., we create a symbolic representation of a constant

g = ufl.Constant(domain)

# For case 2., we use `ufl.SpatialCoordinate` to create a symbolic representation of the coordinates
# in the domain $\Omega$

x = ufl.SpatialCoordinate(domain)

# `x` is here a tuple of 2 components, representing the `x` and `y` coordinate in 2D, i.e. `x[0]`
# represents the x-coordinate, while `x[1]` represent the y-coordinate.
# The unified form language consist of [many](https://fenics.readthedocs.io/projects/ufl/en/latest/manual/form_language.html#basic-nonlinear-functions)
# geometrical expressions, for instance {py:func}`sin<ufl.sin>` and {py:func}`cos<ufl.cos>`.

g = ufl.sin(x[0]) + ufl.cos(ufl.pi * x[1])

# As the third case is easy to represent, we move directly to the fourth,
# where we choose $g=\nabla \cdot f where $f$ in a vector DG-3 space

el_f = basix.ufl.element("DG", cell, 3, shape=(2,))
Q = ufl.FunctionSpace(domain, el_f)
f = ufl.Coefficient(Q)
g = ufl.div(f)

# ## The functional
# Now that we have defined $u_h$ and $g$ we can define the functional $G$

G = (uh - g) ** 2 * ufl.dx

# ```{admonition} The integration measure
# :class: note
# In the code above, we have used {py:obj}`ufl.dx` to indicate that we want to integrate over the cells in our domain.
# ```
# An alternative definition is

dx = ufl.Measure("dx", domain=domain)

# We will come back to why this can be smart later.


# ## Symbolic differentiation
# Now that we have our $G$, we can use {py:func}`ufl.derivative` to compute the derivative of $G$ with respect to a coefficient.
# As we want to differentiate with respect to all functions $\delta u \in V$, we define a {py:func}`ufl.TestFunction`

du = ufl.TestFunction(V)
F = ufl.derivative(G, uh, du)

# ### Non-linear functional
# What if our functional had not been a projection, but something non-linear?
# How would we set up a symbolic representation of the operators needed to solve the problem?
# We could then compute the derivative of the residual, known as the Jacobian
# (not to be confused with the Jacobian of the mapping between the reference and physical cell).

# We could use a Newton method to solve the problem in an iterative fashion:
# Given $u_{k}$ solve
#
# $$
# \begin{align*}
# u_{k+1} &= u_{k} - \delta u\\
# H \delta u &= \frac{\mathrm{d}G}{\mathrm{d}u}[\delta u]
# \end{align*}
# $$
#
# where $ H= \frac{\mathrm{d}^2G}{\mathrm{d}u^2}[\delta u, \delta v]$
# We obtain $J$ with

dv = ufl.TrialFunction(V)
J = ufl.derivative(F, uh, dv)

# ```{admonition} Special case for quadratic functional
# :class: note
# For a quadratic functonal, the residual is linear, and thus the Newton method converges in one iteration
# ```

forms = [G, J, F]

# ## Further analysis of the variational form
#
# We next consider the steps we would have to implement if we wanted to solve this problem by hand.
# For illustrative purposes, we choose $g=\frac{f}{h}$ where $f$ and $h$ are two known functions in respective finite element
# spaces $Q$, $T$, where both $Q$ and $T$ uses the scalar valued elements.
#
# We next use the map $F_K:K_{ref}\mapsto K$ to map the integrals over each cell in the domain back to the reference cell.
# \begin{align}
# \int_\Omega u v~\mathrm{d}x&= \sum_{K\in\mathcal{K}}\int_K u(x) v(x)~\mathrm{d}x\\
# &= \sum_{K\in\mathcal{K}}\int_{F_K(K_{ref})} u(x)v(x)~\mathrm{d}x\\
# &= \sum_{K\in\mathcal{K}}\int_{K_{ref}}u(F_K(\bar x))v(F_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x\\
# \int_\Omega \frac{f}{h}v~\mathrm{d}x
# &=\sum_{K\in\mathcal{K}}\int_{K_{ref}}\frac{f(F_K(\bar x))}{h(F_K(\bar x))} v(F_K(\bar x)) \vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x
# \end{align}
# where $K$ is each element in the physical space, $J_K$ the Jacobian of the mapping.

#
# Next, we can insert the expansion of $u, v, f, h$ into the formulation:
# $u=\sum_{i=0}^{\mathcal{N}}u_i\phi_i(x)\qquad
# v=\sum_{i=0}^{\mathcal{N}}v_i\phi_i(x)\qquad
# f=\sum_{k=0}^{\mathcal{M}}f_k\psi_k(x)\qquad
# g=\sum_{l=0}^{\mathcal{T}}g_l\varphi_l(x)$
#

#
#  and identify the matrix system $Au=b$, where
# ```{math}
# :label: system
# \begin{align}
# A_{j, i} &= \int_{K_{ref}} \phi_i(F_K(\bar x))\phi_j(F_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x\\
# b_j &= \int_{K_{ref}} \frac{\Big(\sum_{k=0}^{\mathcal{M}}f_k\psi_i(F_K(\bar x))\Big)}
# {\Big(\sum_{l=0}^{\mathcal{T}}g_k\varphi_i(F_K(\bar x))\Big)}\phi_j(F_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x
# \end{align}
# ```
#
# ```{warning}
# Next, one can choose an appropriate quadrature rule with points and weights, include the
# correct mapping/restrictions of degrees of freedom for each cell.
# All of this becomes quite tedious and error prone work, and has to be repeated for every variational form!
# ```
#

Q = ufl.FunctionSpace(domain, basix.ufl.element("Lagrange", cell, 3))
f = ufl.Coefficient(Q)
T = ufl.FunctionSpace(domain, basix.ufl.element("DG", cell, 1))
h = ufl.Coefficient(T)
g = f / g
v = ufl.TestFunction(V)
L = f / g * v * ufl.dx

# We can use UFL to analyse the linear form above

pulled_back_L = ufl.algorithms.compute_form_data(
    L,
    do_apply_function_pullbacks=True,
    do_apply_integral_scaling=True,
    do_apply_geometry_lowering=True,
    preserve_geometry_types=(ufl.classes.Jacobian,),
)
print(pulled_back_L.integral_data[0])

# ## Inner products and dot products
# When working with vectors and tensors, we often want to compute the inner product or dot product
# between two values `u` and `v`.

V = ufl.FunctionSpace(domain, basix.ufl.element("N1curl", cell, 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#  The dot product of two vectors $u$ and $v$ is defined as `dot(u,v)=u[i]v[i]` using Einstein summation notation

i = ufl.Index()
f = u[i] * v[i] * dx
f_equivalent = ufl.dot(u, v) * dx

# For two tensors `r`, `s` we have that
#
# $$
# \mathrm{dot}(r, s)=r_{ik}s_{kl}i_{i}i_{l}
# $$
#
# The inner product is a contraction over all axes
#
# $$
# \mathrm{inner}(r, s)=r_{ij}s^*_{ij}
# $$
#
# where $s_{ij}^*$ is the complex conjugate of $s_{ij}$.
#
# ```{admonition} How would the above change if we choose "N1curl" elements as the receiving space?
# :class: dropdown note
# One would have to add in the covariant Piola mapping to map from the physical to reference basis function.
# ```
#
# ```{admonition} Can you get UFL to do the computations for you? (Expand for hint)
# :class: dropdown hint
# Try setting up the bi-linear form `a` for your problem and use `ufl.algorithms.compute_form_data` to see what expression you get out.
# Try turning various options on and off and see how the result looks like
# ```
# Expand the below to investigate the solution

# + tags=["hide-input", "hide-output"]
V = ufl.FunctionSpace(domain, basix.ufl.element("N1curl", cell, 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx
pulled_back_a = ufl.algorithms.compute_form_data(
    a,
    do_apply_function_pullbacks=True,
    do_apply_integral_scaling=False,
    do_apply_geometry_lowering=False,
    preserve_geometry_types=(ufl.classes.Jacobian,),
)
print(pulled_back_a.integral_data[0])
# -

# ## Quadrature rule
# The next step in the analysis of {eq}`system` is to choose a suitable [quadrature rule](https://en.wikipedia.org/wiki/Numerical_integration),
# and expand the integral as a sum of evaluations of the basis functions at points.
#
# $$
# A_{j, i} = \sum_{q=0}^M w_q \phi_i(F_K(\bar x_q))\phi_j(F_K(\bar x_q))\vert \mathrm{det} J_K(\bar x_q)\vert
# $$
#
# As you might have spotted in the previous exercise, UFL can estimate want degree you should use for your quadrature rule.
# We can get this number explicitly with the following code.

pulled_back_a = ufl.algorithms.compute_form_data(
    a,
    do_apply_function_pullbacks=True,
    do_apply_integral_scaling=True,
    do_apply_geometry_lowering=True,
    preserve_geometry_types=(ufl.classes.Jacobian,),
)
print(list(itg.metadata() for itg in pulled_back_a.integral_data[0].integrals))

# ```{admonition} Would the estimate change if we use a different cell for the domain?
# :class: dropdown note
# Yes, choosing a quadrilateral cell will increase the quadrature degree, as the map from the reference element is non-affine.
# ```

# Expand to see example

# + tags=["hide-input", "hide-output"]
cell = "quadrilateral"
domain = ufl.Mesh(basix.ufl.element("Lagrange", cell, 1, shape=(2,)))
V = ufl.FunctionSpace(domain, basix.ufl.element("N1curl", cell, 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx
pulled_back_a = ufl.algorithms.compute_form_data(
    a,
    do_apply_function_pullbacks=True,
    do_apply_integral_scaling=True,
    do_apply_geometry_lowering=True,
    preserve_geometry_types=(ufl.classes.Jacobian,),
)
print(list(itg.metadata() for itg in pulled_back_a.integral_data[0].integrals))
# -

# ```{admonition} Would the estimate change if use a higher order triangle?
# :class: dropdown note
# Yes, as above, the Jacobian becomes non-constant as the map is non-affine.
# ```

# + tags=["hide-input", "hide-output"]
cell = "triangle"
domain = ufl.Mesh(basix.ufl.element("Lagrange", cell, 2, shape=(2,)))
V = ufl.FunctionSpace(domain, basix.ufl.element("N1curl", cell, 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx
pulled_back_a = ufl.algorithms.compute_form_data(
    a,
    do_apply_function_pullbacks=True,
    do_apply_integral_scaling=True,
    do_apply_geometry_lowering=True,
    preserve_geometry_types=(ufl.classes.Jacobian,),
)
print(list(itg.metadata() for itg in pulled_back_a.integral_data[0].integrals))
# -

# We can fix the number of quadraure points by setting the `quadrature_degree` in the {py:class}`ufl.Measure`

dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})

# ```{warning} Variational crimes
# Reducing the accuracy of the integration by lowering the quadrature rule is considered to be a
# **variational crime** {cite}`sulli2012lecture` (Chapter 3.4) and should be done with caution.
# ```

# ## References
# ```{bibliography}
#    :filter: cited and ({"src/unified_form_language/ufl_forms"} >= docnames)
# ```
