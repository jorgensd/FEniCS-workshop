# (functionals)=
# # PDE-constrained optimization problems
# As seen in the previous section, we can use UFL to differentiate our variatonal forms.
# We can then also use UFL to represent PDE-constrained optimization problems.
#
# In this section, we will consider PDE-constrained optimization problems of the form
#
# $$
# \min_{c\in Q}J(u, c)
# $$
#
# subject to
#
# $$
# F(u, c) = 0.
# $$
# Perhaps write a little bit about what PDE-constrained optimization is? E.g give some real world examples and say that c is a control parameter.
#
# We can use the adjoint method to compute the sensitivity of the functional with
# respect to the solution of the PDE.
# This is done by introducing the Lagrangian
# Here it would also be good with a box explaining what the Lagrangian is and perhaps refer to some literature.
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
# = \frac{\partial J}{\partial c}[\delta c]  + \left(\lambda, \frac{\partial F}{\partial c}[\delta c]\right).
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

# (mother)=
# ## Example: The Poisson mother problem
#
# We will consider the following PDE constrained optimization problem:
# Maybe give some explanation of what this problem represents physically.
#
# $$
# \min_{f\in Q} \int_{\Omega} (u - d)^2 ~\mathrm{d}x + \frac{\alpha}{2}\int_{\Omega}  f \cdot f ~\mathrm{d}x
# $$
#
# such that
#
# $$
# \begin{align*}
# -\nabla \cdot \nabla u &= f \quad \text{in} \quad \Omega,\\
# u &= 0 \quad \text{on} \quad \partial \Omega.
# \end{align*}
# $$
#
# We start by formulating the forward problem, i.e, convert the PDE into a variational form

# +
import basix.ufl
import ufl

domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))

el_u = basix.ufl.element("Lagrange", "triangle", 1)
V = ufl.FunctionSpace(domain, el_u)

el_f = basix.ufl.element("DG", "triangle", 0)
Q = ufl.FunctionSpace(domain, el_f)
f = ufl.Coefficient(Q)

u = ufl.Coefficient(V)
v = ufl.TestFunction(V)
F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
# -

# Next we define the functional we want to minimize

d = ufl.Coefficient(V)
alpha = ufl.Constant(domain)
J = 1 / 2 * (u - d) ** 2 * ufl.dx + alpha / 2 * ufl.inner(f, f) * ufl.dx

# As seen previously, we can now differentiate the functional with respect to the solution of the PDE and the control to
# obtain the components required for the adjoint problem.

dJdu = ufl.derivative(J, u)
dJdf = ufl.derivative(J, f)
dFdu = ufl.derivative(F, u)
dFdf = ufl.derivative(F, f)

adj_rhs = -dJdu
adj_lhs = ufl.adjoint(dFdu)

# For solving the linear system, we prepare for using a Newton-method as before.

fwd_lhs = dFdu
fwd_rhs = F

# For the derivative of the functional with respect to the control we use
# the command `ufl.action` to replace the trial function with `lmbda` to create the matrix-vector product
# without forming the matrix

lmbda = ufl.Coefficient(V)
dLdf = ufl.derivative(J, f) + ufl.action(ufl.adjoint(dFdf), lmbda)

# I would add the mathematical equivalent of the code above here.

# We collect all the forms we care about in a list called `forms`, which will be explained in the next section.

forms = [adj_lhs, adj_rhs, fwd_lhs, fwd_rhs, dLdf, J]
