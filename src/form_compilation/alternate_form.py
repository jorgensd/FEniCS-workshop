# # The standard way of compiling code with DOLFINx

# In this section, we will focus on the approach most users use to interact
# with UFL, FFCx and basix.
# Here we will start by creating the domain we want to solve a problem on.
# In this case, we will use a unit square

# +
from mpi4py import MPI

import numpy as np
import scipy

import dolfinx
import ufl

N = 10
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
tdim = mesh.topology.dim
# -

# ## Problem specification: Non-linear Poisson.

# Next, let's consider the problem
#
# $$
# \begin{align}
# -\nabla \cdot p(u) \nabla u &= f \quad \text{in } \Omega, \\
# u &= g \quad \text{on } \partial \Omega
# \end{align}
# $$
# where $p(u)=1+u^2$.
# We choose to use a first order Lagrange space for the unknown.

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# ```{admonition} Alternative declaration of finite elements
# :class: dropdown info
# When working in DOLFINx, we can also supply the element information directly to the function space
# with a tuple `(family, degree)`, or a triplet `(family, degree, shape)`, i.e. if we want a M dimensional
# Lagrange 5th order vector space, one could specify this as `("Lagrange", 5, (M, ))`.
# ```
# We do as in the previous section, and define a manufactured solution


# +
def u_exact(module, x):
    return module.sin(2 * module.pi * x[1]) + x[0] * x[1] ** 2


x = ufl.SpatialCoordinate(mesh)
u_ex = u_exact(ufl, x)
# -

# As the problem is non-linear we need to prepare for solving this with a
# Newton-solver.

uh = dolfinx.fem.Function(V)

# We define the residual


# +
def p(u):
    return 1 + u**2


f = -ufl.div(p(u_ex) * ufl.grad(u_ex))
v = ufl.TestFunction(V)
F = ufl.inner(p(uh) * ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

gh = dolfinx.fem.Function(V)
gh.interpolate(lambda x: u_exact(np, x))

mesh.topology.create_entities(tdim - 1)
mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
bc = dolfinx.fem.dirichletbc(gh, boundary_dofs)
bcs = [bc]
# -

# We compute the Jacobian of the system using UFL.

J = ufl.derivative(F, uh)

# Now that we have associated the forms with the discrete problem already, we use
# {py:func}`dolfinx.fem.form`

residual = dolfinx.fem.form(F)
jacobian = dolfinx.fem.form(J)
A = dolfinx.fem.create_matrix(jacobian)
b = dolfinx.fem.create_vector(residual.function_spaces[0])

# We are now ready to use these forms to solve the variational problem.
#
# $$
# \begin{align}
# \frac{\partial F}{\partial u}[\delta u] &= F(u_k) \\
# u_{k+1} = u_k - \delta u
# \end{align}
# $$

# As we want $u_{k+1}=g$, but we do not know if $u_k=g$, we need to take this into account when assembling the
# right hand side for the Jacobian equation.
#
# We will go through how we apply this boundary condition using lifting
#
# ## Lifting
# Let us split the degrees of freedom into two disjoint sets, $u_d$, and $u_{bc}$, and set up the corresponding linear system
#
# $$
# \begin{align}
# \begin{pmatrix}
# A_{d,d} & A_{d, bc} \\
# A_{bc,d} & A_{bc, bc}
# \end{pmatrix}
# \begin{pmatrix}
# u_d \\
# u_{bc}
# \end{pmatrix}
# &=
# \begin{pmatrix}
# b_d \\
# b_{bc}
# \end{pmatrix}
# \end{align}
# $$ (A_split)
#
# In the identity row approach, we set the rows corresponding to the Dirichlet conditions to the identity row
# and set the appropriate dofs on the right hand side to contain the Dirichlet values:
#
# $$
# \begin{align}
# \begin{pmatrix}
# A_{d,d} & A_{d, bc} \\
# 0 & I
# \end{pmatrix}
# \begin{pmatrix}
# u_d \\
# u_{bc}
# \end{pmatrix}
# &=
# \begin{pmatrix}
# b_d \\
# g
# \end{pmatrix}
# \end{align}
# $$
#
# where $g$ is the vector satisfying the various Dirichlet conditions.
# We can now reduce this to a smaller system
#
# $$
# \begin{align}
# A_{d,d}
# u_d
# &=
# b_d
# -
# A_{d, bc}g
# \end{align}
# $$
#
# which is symmetric if $A_{d,d}$ is symmetric.
# However, we do not want to remove the degrees of freedom from the sparse matrix, as it makes the matrix
# less adaptable for varying boundary conditions, so we set up the system
#
# $$
# \begin{align}
# \begin{pmatrix}
# A_{d,d} & 0 \\
# 0 & I
# \end{pmatrix}
# \begin{pmatrix}
# u_d \\
# u_{bc}
# \end{pmatrix}
# &=
# \begin{pmatrix}
# b_d - A_{d,bc}g \\
# g
# \end{pmatrix}
# \end{align}
# $$

# We start by zeroing out existing entries in the right hand side.

b.array[:] = 0

#  Next we assemble the residual into `b`

dolfinx.fem.assemble_vector(b.array, residual)


# We want to apply the condition
#
# $$
# \delta u = u_k - g
# $$
#
# since $u_{k+1} = u_k - \delta u$
#
# This means that we would like to compute
#
# $$
# b - J (u_k - g) = b + J (g - u_k)
# $$

# ```{admonition} What does apply lifting do?
# :class: dropdown note
# Apply lifting will compute
#
# $$
# b-= \alpha \sum_j(A_j(g_j - x0_j))
# $$
#
# `dolfinx.fem.apply_lifting` takes in five arguments:
# 1. An array `b` that we will modify the entries of
# 2. A list of bi-linear forms `a_j` that assemble into the matrix `A_j`
# 3. A nested list of boundary conditions, where the `j` list of boundary conditions
# for the form `a_j`. We accumulate all these boundary conditions into a single vector $g_j$.
# 4. A list of vectors where the $j$th entry $x0_j$ will be subtracted from $g_j$.
# 5. A scalar value $\alpha$ that determines the magnitude of the modification.
# ```
# This means that we set $\alpha=-1$, $x0=uk$, $A=J$.

dolfinx.fem.apply_lifting(b.array, [jacobian], [bcs], x0=[uh.x.array], alpha=-1.0)
b.scatter_reverse(dolfinx.la.InsertMode.add)

# As for the last part of the application of Dirichlet condtitions, we want to set
# $u_k - g$ for the constrained dofs. We can do this with
# {py:meth}`dolfinx.fem.DirichletBC.set`.
# ```{admonition} What does DirichletBC.set do?
# :class: dropdown note
# We will replace the entries constrained by the Dirichlet bcs  with $\alpha * (g - x0)$
# where `b`, `x0` and `alpha` are the inputs to `bc.set`, where `g` is the boundary condition
# at those dofs.
# ```
# For the example at hand, we set $\alpha=-1$, $x0=u_k$.

[bc.set(b.array, x0=uh.x.array, alpha=-1.0) for bc in bcs]

# Next, we can compute the assemble the Jacobian

A_scipy = A.to_scipy()
dolfinx.fem.assemble_matrix(A, jacobian, bcs)

# ## Hand-coded Newton solver

# We are ready to do this iteratively
du = dolfinx.fem.Function(V)
correction_norm = None
for i in range(1, 20):
    # Assemble residual
    b.array[:] = 0
    dolfinx.fem.assemble_vector(b.array, residual)
    dolfinx.fem.apply_lifting(b.array, [jacobian], [bcs], x0=[uh.x.array], alpha=-1.0)
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    [bc.set(b.array, x0=uh.x.array, alpha=-1.0) for bc in bcs]

    print(f"Iteration {i}, |F|={np.linalg.norm(b.array)}, |du|={correction_norm}")
    # Assemble jacobian
    A.data[:] = 0
    dolfinx.fem.assemble_matrix(A, jacobian, bcs)

    A_inv = scipy.sparse.linalg.splu(A_scipy)
    du.x.array[:] = A_inv.solve(b.array)
    uh.x.array[:] -= du.x.array
    if (correction_norm := np.linalg.norm(du.x.array)) < 1e-6:
        print(f"Iteration {i}, Converged: |du|={correction_norm}")
        break

# We do as before, and compute the error between the exact and approximate solution.

error = dolfinx.fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
local_error = dolfinx.fem.assemble_scalar(error)
global_error = np.sqrt(mesh.comm.allreduce(local_error, op=MPI.SUM))

print(f"L2-error: {global_error:.2e}")


# + tags=["remove-input"]
import pyvista

mesh_pyvista = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(uh.function_space))
mesh_pyvista.point_data["uh"] = uh.x.array

Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u_linear = dolfinx.fem.Function(Q)
u_linear.interpolate(uh)


# Next, we create a plotting instance, and warp the solution grid by the solution.
plotter = pyvista.Plotter()

warped_grid = mesh_pyvista.warp_by_scalar("uh")
plotter.add_mesh(warped_grid, show_edges=False)
plotter.add_mesh(warped_grid, style="wireframe", color="black")
plotter.show()
# -

# (scipy_nonlinear)=
# ## Using scipy's Newton solver

# Of course we don't want to write out this kind of loop for every problem.
# We can for instance use the {py:func}`newton_krylov<scipy.optimize.newton_krylov>` solver from scipy, but instead of
# approximating the Jacobian, we will solve the non-linear problem directly.
# We do this by overloading the "ksp" method in scipy with a method that only does pre-conditioning,
# where the pre-conditioning is to solve the system for the exact Jacobian.
# We can do this with the following code:


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


# Inspect the code above, does it follow what we have described earlier?
# ```{admonition} Use the Newton solver above to compute convergence rates for the above problem.
# :class: dropdown tip
# See: {ref}`error_estimation` for instruction on how to compute convergence rates
# ```
# Expand the code above to see the solution.

# + tags=["hide-input", "hide-output"]
Ns = np.array([4, 8, 16, 32])
errors = np.zeros_like(Ns, dtype=np.float64)
hs = np.zeros_like(Ns, dtype=np.float64)
for i, N in enumerate(Ns):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = mesh.topology.dim
    x = ufl.SpatialCoordinate(mesh)
    u_ex = u_exact(ufl, x)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    uh = dolfinx.fem.Function(V)

    f = -ufl.div(p(u_ex) * ufl.grad(u_ex))
    v = ufl.TestFunction(V)
    F = ufl.inner(p(uh) * ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

    gh = dolfinx.fem.Function(V)
    gh.interpolate(lambda x: u_exact(np, x))

    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(gh, boundary_dofs)
    bcs = [bc]
    solver = NonLinearSolver(F, uh, bcs)
    solver.solve(verbose=True)

    error = dolfinx.fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
    local_error = dolfinx.fem.assemble_scalar(error)
    global_error = np.sqrt(mesh.comm.allreduce(local_error, op=MPI.SUM))
    errors[i] = global_error
    hs[i] = 1.0 / N


def compute_converence(hs, errors):
    return np.log(errors[1:] / errors[:-1]) / np.log(hs[1:] / hs[:-1])


print(f"{hs=}")
print(f"{errors=}")
print(f"{compute_converence(hs, errors)=}")
# -

# ```{admonition} Investigate various degrees for the unknown function space $V$, how does the convergence rate vary?
# :class: dropdown tip
# It follows the same pattern as we saw in {ref}`rates`, for a polynomial of degree $P$ the rate is $P+1$.
# ```

# ```{admonition} Which of the two strategies for creating variational forms did you prefer?
# :class: dropdown tup
# Depending on your use-case, and if you want to be compatible with C++ codes, the optimal choice is up to the user!
# ```
