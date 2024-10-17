# (functionals)=
# # Functionals and derivatives
# As mentioned above, many finite element problems can be rephrased as an optimization problem.
#
# For instance, we can write the equations of linear elasicity as an optimization problem:
#
# $$\min_{u_h\in V}J_h(u_h) = \int_\Omega C\epsilon(u_h): \epsilon(u_h)~\mathrm{d}x - \int_\Omega f\cdot v~\mathrm{d}x,$$
#
# where $C$ is the stiffness tensor given as $C_{ijkl} = \lambda \delta_{ij}\delta_{kl} + \mu(\delta_{ik}\delta_{jl}+\delta_{il}\delta{kj})$,
# $\epsilon$ is the symmetric strain tensor and $u_h$ a displacement field.
#
# We start by defining these quantities in UFL:

# The function space for displacement

import basix.ufl
import ufl

cell = "quadrilateral"
c_el = basix.ufl.element("Lagrange", cell, 1, shape=(2,))
domain = ufl.Mesh(c_el)

el = basix.ufl.element("Lagrange", cell, 2, shape=(2,))
Vh = ufl.FunctionSpace(domain, el)
uh = ufl.Coefficient(Vh)
f = ufl.Coefficient(Vh)

# Lame's elasticity parameters

mu = ufl.Constant(domain)
lmbda = ufl.Constant(domain)


def epsilon(u):
    return ufl.sym(ufl.grad(u))


# We define the stiffness tensor using [Einstein summation notation](https://mathworld.wolfram.com/EinsteinSummation.html).
# We start by defining the identity tensor which we will use as a [Kronecker Delta](https://mathworld.wolfram.com/KroneckerDelta.html)
# function.
# Next we define four indices that we will use to account for the four dimensions of the stiffness tensor.

Id = ufl.Identity(domain.geometric_dimension())
indices = ufl.indices(4)

# Secondly we define the product of two delta functions $\delta_{ij}\delta_{kl}$
# which results in a fourth order tensor.


def delta_product(i, j, k, l):
    return ufl.as_tensor(Id[i, j] * Id[k, l], indices)


# Finally we define the Stiffness tensor
i, j, k, l = indices
C = lmbda * delta_product(i, j, k, l) + mu * (delta_product(i, k, j, l) + delta_product(i, l, k, j))

# and the functional

Jh = 0.5 * (C[i, j, k, l] * epsilon(uh)[k, l]) * epsilon(uh)[i, j] * ufl.dx - ufl.inner(f, uh) * ufl.dx

# This syntax is remarkably similar to how it is written on [paper](https://en.wikipedia.org/wiki/Elasticity_tensor).

# ## Alternative formulation
# Instead of writing out all the indices with Einstein notation, one could write the same equation as


def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


Jh = 0.5 * ufl.inner(sigma(uh), epsilon(uh)) * ufl.dx - ufl.inner(f, uh) * ufl.dx

# ## Differentiating the energy functional
# We can differentiate the energy functional with respect to the displacement field $u_h$.

F = ufl.derivative(Jh, uh)

# Since we want to find the minimum of the functional, we set the derivative to zero.
# To solve this problem, we can for instance use a Newton method, where we solve a sequence of equations:
#
# $$ u_{k+1} = u_k - J_F(u_k)^{-1}F(u_k),$$
#
# where $J_F$ is the Jacobian matrix of $F$.
# We can rewrite this as:
#
# $$
# \begin{align}
# u_{k+1} &= u_k - \delta u_k\\
# J_F(u_k)\delta u_k &= F(u_k)
# \end{align}
# $$
#
# Which boils down to solving a linear system of equations for $\delta u_k$.
#
# We can compute the Jacobian using UFL:

J_F = ufl.derivative(F, uh)

# And with this and $F$ we can solve the minimization problem.
# See for instance:
# [Custom Newton Solver in DOLFINx](https://jsdokken.com/dolfinx-tutorial/chapter4/newton-solver.html)
# for more details about how you could implement this method by hand.
#
# # Extra material: Optimization problems with PDE constraints
#
# Another class of optimization problems is the so-called PDE-constrained optimization problems.
# For these problems, one usually have a problem on the form
#
# $$\min_{c}J_h(u_h, c)$$
#
# such that
#
# $$F(u_h,c)=0.$$
#
# We can use the adjoint method to compute the sensitivity of the functional with
# respect to the solution of the PDE.
# This is done by introducing the Lagrangian
#
# $$\min_{c}\mathcal{L}(u_h, c) = J_h(u_h,c) + (\lambda, F(u_h,c)).$$
#
# We now seek the minimum of the Lagrangian, i.e.
#
# $$\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}c}[\delta c] = 0,$$
#
# which we can write as
#
# $$
# \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}c}[\delta c]
# = \frac{\partial J}{\partial c}
# + \frac{\partial J}{\partial u}\frac{\mathrm{d}u}{\mathrm{d}c}[\delta c]
# + \left(\lambda, \frac{\partial F}{\partial u} \frac{\mathrm{d}u}{\mathrm{d}c}[\delta c]\right)
# + \left(\lambda, \frac{\partial F}{\partial c}[\delta c]\right).
# $$
#
# Since $\lambda$ is arbitrary, we choose $\lambda$ such that
#
# $$
# \frac{\partial J}{\partial u}\delta u
# = -\left(\lambda, \frac{\partial F}{\partial u} \delta u\right)
# $$
#
# for any $\delta u$ (including $\frac{\mathrm{d}u}{\mathrm{d}c}[\delta c]$).
#
# This would mean that
#
# $$
# \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}c}[\delta c]
# = \frac{\partial J}{\partial c}  + \left(\lambda, \frac{\partial F}{\partial c}[\delta c]\right).
# $$
#
# To find such a $\lambda$, we can solve the adjoint problem
#
# $$
# \left( \left(\frac{\partial F}{\partial u}\right)^* \lambda^*, \delta u\right) =
# -\left(\frac{\partial J}{\partial u}\right)^*\delta u.
# $$
#
# With UFL, we do not need to derive these derivatives by hand, and can use symbolic
# differentiation to get the left and right hand side of the adjoint problem.

dFdu_adj = ufl.adjoint(ufl.derivative(F, uh))
dJdu_adj = ufl.derivative(Jh, uh)
dJdf = ufl.derivative(Jh, f)
dFdf = ufl.derivative(F, f)
