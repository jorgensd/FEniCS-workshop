# # Optimal control in DOLFINx interfacing with scipy
# In this section, we will solve the {ref}`mother`.
# We re-iterate the formulation of the problem:
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

# We start by creating the computational domain and the function space of $u$

# +
from mpi4py import MPI

import numpy as np
import scipy

import dolfinx
import ufl

M = 55
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, M, M)

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
# -

# Next we define the residual form of $F(u, v)=0 \quad \forall v \in V$
# where the control $f\in Q$ is defined as piecewise continuous.

# +
du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

W = dolfinx.fem.functionspace(domain, ("DG", 0))
f = dolfinx.fem.Function(W)

f.interpolate(lambda x: x[0] + x[1])

uh = dolfinx.fem.Function(V)
F = ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
# -

# Next we define the functional

alpha = dolfinx.fem.Constant(domain, 1e-3)
x = ufl.SpatialCoordinate(domain)
d = 1 / (2 * ufl.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
J = 1 / 2 * (uh - d) * (uh - d) * ufl.dx + alpha / 2 * f**2 * ufl.dx

# As seen in previous sections, can easily constrain all the degrees of freedom on the boundary

# +
tdim = domain.topology.dim

domain.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
u_bc = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.0))
bc = dolfinx.fem.dirichletbc(u_bc, boundary_dofs, V)
# -

# ## The forward problem
# Next, we use the `NonLinearSolver` we defined in {ref}`scipy_nonlinear`


# + tags = ["hide-input"]
class NonLinearSolver(scipy.sparse.linalg.LinearOperator):
    def __init__(self, F, uh, bcs):
        """
        Solve the problem F(uh, v)=0 forall v
        """
        jacobian = ufl.derivative(F, uh)
        self.J_compiled = dolfinx.fem.form(jacobian)
        self.F_compiled = dolfinx.fem.form(F)
        self.bcs = bcs
        self.A = dolfinx.fem.create_matrix(self.J_compiled)
        self.b = dolfinx.fem.Function(uh.function_space)
        self._A_scipy = self.A.to_scipy()
        self.uh = uh

        # Scipy specific parameters
        self.shape = (len(uh.x.array), len(uh.x.array))
        self.dtype = uh.x.array.dtype
        self.update(uh.x.array, None)

    def update(self, x, f):
        """Update and invert Jacobian"""
        self.A.data[:] = 0
        self.uh.x.array[:] = x
        dolfinx.fem.assemble_matrix(self.A, self.J_compiled, bcs=self.bcs)
        self._A_inv = scipy.sparse.linalg.splu(self._A_scipy)

    def _matvec(self, x):
        """Compute J^{-1}x"""
        return self._A_inv.solve(x)

    def _compute_residual(self, x):
        """
        Evaluate the residual F(x) = 0
        Args:
            x: Input vector with current solution
        Returns:
            Residual array
        """
        self.uh.x.array[:] = x
        self.b.x.array[:] = 0
        dolfinx.fem.assemble_vector(self.b.x.array, self.F_compiled)
        dolfinx.fem.apply_lifting(self.b.x.array, [self.J_compiled], [self.bcs], x0=[self.uh.x.array], alpha=-1.0)
        self.b.x.scatter_reverse(dolfinx.la.InsertMode.add)
        [bc.set(self.b.x.array, x0=self.uh.x.array, alpha=-1.0) for bc in self.bcs]
        return self.b.x.array

    def linSolver(self, _A, x, **kwargs):
        """
        The linear solver method we will use.
        Simply return `J^-1 x` for an input x
        """
        return kwargs["M"]._matvec(x), 0

    def solve(self, maxiter: int = 100, verbose: bool = False):
        """Call Newton-Krylov solver with direct solving (pre-conditioning only)"""
        self.uh.x.array[:] = scipy.optimize.newton_krylov(
            self._compute_residual,
            self.uh.x.array,
            method=self.linSolver,
            verbose=verbose,
            line_search=None,
            maxiter=maxiter,
            inner_M=self,
        )


# -

forward_problem = NonLinearSolver(F, uh=uh, bcs=[bc])

# Next, we repeat the symbolic differentiation to obtain the various forms we
# require for the functional sensitivity.

lmbda = dolfinx.fem.Function(V)
dFdu = ufl.derivative(F, uh, du)
dFdu_adj = ufl.adjoint(dFdu)
dJdu = ufl.derivative(J, uh, v)

# ## The adjoint problem
#
# Next, we create a small linear solver that caches the creation of the matrices, vectors
# and the compiled forms.


class LinearSolver:
    def __init__(self, a, L, uh, bcs):
        self.a_compiled = dolfinx.fem.form(a)
        self.L_compiled = dolfinx.fem.form(L)
        self.A = dolfinx.fem.create_matrix(self.a_compiled)
        self.b = dolfinx.fem.Function(uh.function_space)
        self.bcs = bcs
        self._A_scipy = self.A.to_scipy()
        self.uh = uh

    def solve(self):
        self._A_scipy.data[:] = 0

        dolfinx.fem.assemble_matrix(self.A, self.a_compiled, bcs=self.bcs)

        self.b.x.array[:] = 0
        dolfinx.fem.assemble_vector(self.b.x.array, self.L_compiled)
        dolfinx.fem.apply_lifting(self.b.x.array, [self.a_compiled], [self.bcs])
        self.b.x.scatter_reverse(dolfinx.la.InsertMode.add)
        [bc.set(self.b.x.array) for bc in self.bcs]

        A_inv = scipy.sparse.linalg.splu(self._A_scipy)
        self.uh.x.array[:] = A_inv.solve(self.b.x.array)
        return self.uh


# We use this to set up the adjoint problem

adj_problem = LinearSolver(ufl.replace(dFdu_adj, {uh: v}), -dJdu, lmbda, [bc])

# ```{note}
# The adjoint problem is always linear, and does not require a non-linear solver.
# ```

# ## The gradient of the functional
# Next we prepare the evaluation of the derivative of the functional.
# We split this in two components:

q = ufl.TrialFunction(W)
dJdf = ufl.derivative(J, f, q)
dFdf = ufl.action(ufl.adjoint(ufl.derivative(F, f, q)), lmbda)
dJdf_compiled = dolfinx.fem.form(dJdf)
dFdf_compiled = dolfinx.fem.form(dFdf)
dLdf = dolfinx.fem.Function(W)


# We create a convencience function for evaluating the functional for any specific
# control input.

# +
Jh = dolfinx.fem.form(J)


def eval_J(x):
    f.x.array[:] = x
    forward_problem.solve(verbose=False)
    local_J = dolfinx.fem.assemble_scalar(Jh)
    return domain.comm.allreduce(local_J, op=MPI.SUM)


# -

# We also create a convenience function for computing the gradient at a given point.
# Here we:
# 1. Compute the forward solution for given control
# 2. Compute the adjoint solution with the result from 1.
# 3. Assemble the derivative of the functional


def eval_gradient(x):
    f.x.array[:] = x
    forward_problem.solve()
    adj_problem.solve()
    dLdf.x.array[:] = 0
    dolfinx.fem.assemble_vector(dLdf.x.array, dJdf_compiled)
    dolfinx.fem.assemble_vector(dLdf.x.array, dFdf_compiled)
    return dLdf.x.array


# ## Solving the optimization problem
# We will use scipy to find the minimum.
# We create a call-back function to monitor the functional value


# +
def callback(intermediate_result):
    fval = intermediate_result.fun
    print(f"J: {fval}")


from scipy.optimize import minimize

opt_sol = minimize(
    eval_J,
    f.x.array,
    jac=eval_gradient,
    method="CG",
    tol=1e-9,
    options={"disp": True},
    callback=callback,
)
# -

# We inspect the optimal solution

# + tags=["remove-input"]
print(f"{opt_sol=}")
# -

# And assign it to `f` to get a final solution:

f.x.array[:] = opt_sol.x
forward_problem.solve()

# For our given `f` we have an analytical solution to the problem
#
# $$
# \begin{align*}
# f_{ex} &= \frac{1}{1+4\alpha \pi^4}\sin(\pi x)\sin(\pi y)\\
# u_{ex} &= \frac{1}{2\pi^2}f_{ex}
# \end{align*}
# $$

# We compute the error


def f_exact(mod, x):
    return 1 / (1 + 4 * float(alpha) * mod.pi**4) * mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1])


def u_exact(mod, x):
    return 1 / (2 * np.pi**2) * f_exact(mod, x)


u_ex = dolfinx.fem.Function(V)
u_ex.interpolate(lambda x: u_exact(np, x))

L2_error = dolfinx.fem.form(ufl.inner(uh - u_exact(ufl, x), uh - u_exact(ufl, x)) * ufl.dx)
local_error = dolfinx.fem.assemble_scalar(L2_error)
global_error = np.sqrt(domain.comm.allreduce(local_error, op=MPI.SUM))

# + tags =["hide-input"]
print(f"Error: {global_error:.2f}")
# -

# ## Verifying the gradient implementation
# When implementing the optimization problem, it is easy to do the wrong thing.
# We use a Taylor-expansion of the functional to verify it's accuracy
#
# $$
# \mathcal{L}(f_0+\alpha\delta f) = \mathcal{L}(f_0) + \alpha \delta f \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}f}(f_0) + \mathcal{O}((\alpha \delta f)^2)
# $$
#
# This means that we have that we have
#
# $$
# E_{\alpha} = \mathcal{L}(f_0+\alpha\delta f) - \mathcal{L}(f_0) - \alpha \delta f \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}f}(f_0) = D\alpha^2
# $$
#
# meaning that if we choose $\alpha_i$, $\alpha_j$ yields
#
# $$
# \ln(E_{\alpha_i}/E_{\alpha_j})= \ln(\alpha_i^2/\alpha_j^2) = 2 \ln(\alpha_i/\alpha_j)
# $$
#
# i.e. that if we compute perturbations with varying step size $\alpha_j$ we should get the rate 2.

# ```{admonition} Compute the convergence rate of the derivative with the gradient
# :class: tip dropdown
# For the convergence rate code see previous sections.
# ```
#
# Expand the below to see the solution


# + tags=["hide-input", "hide-output"]
def taylor_test(cost, grad, m_0, p=1e-2, n=5):
    """
    Compute a Taylor test for a cost function and gradient function from `m_0` in direction `p`

    Args:
        cost: Function taking in the control variable returning the functional value
        grad: Function computing the gradient for a given control input
        m_0: Inital condition of the control function to perturb around
        p: Step size, either a float or a vector scaling each entry in the control
        n: Number of steps. Halfiing steps from original step size `n` times.
    Returns:
        A triplet `(remainder, perturbance, conv_rate)` where remaineder is the error for each `perturbance` of `p`
        and the convergence rate.
    """
    l0 = cost(m_0)
    local_gradient = grad(m_0)
    global_gradient = np.hstack(MPI.COMM_WORLD.allgather(local_gradient))

    if isinstance(p, float):
        p = np.full_like(m_0, p)
    p_global = np.hstack(MPI.COMM_WORLD.allgather(p[: len(local_gradient)]))
    dJdm = np.dot(global_gradient, p_global)
    remainder = []
    perturbance = []
    for i in range(0, n):
        step = 0.5**i
        l1 = cost(m_0 + step * p)
        remainder.append(l1 - l0 - step * dJdm)
        perturbance.append(step)
    conv_rate = convergence_rates(remainder, perturbance)
    return remainder, perturbance, conv_rate


def convergence_rates(r, p):
    cr = []  # convergence rates
    for i in range(1, len(p)):
        cr.append(np.log(r[i] / r[i - 1]) / np.log(p[i] / p[i - 1]))
    return cr


f0 = np.zeros(len(f.x.array))
error, perturbance, rate = taylor_test(eval_J, grad=eval_gradient, m_0=f0)
if domain.comm.rank == 0:
    print(f"(2) Error: {error}")
    print(f"(2) Perturbance: {perturbance}")
    print(f"(2) Convergence rate: {rate}")
# -
