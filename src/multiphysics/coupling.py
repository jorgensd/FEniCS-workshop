# # Coupling PDEs of multiple dimensions

# In the previous section, we considered problems where we solved non-coupled physics on two parts of a domain.
# However, the problems we actually want to considered are problems that are coupled across domains.

# In this section we will cover how to couple PDEs formulated in the domain (2D) with those living on a subset of
# facets (1D). This can naturally be extended to 3D.

# In this section, we will show how to solve the Signorini problem
#
# $$
# \begin{align*}
# \nabla \cdot (C \epsilon(\mathbf{u})) &= \mathbf{f} \text{ in } \Omega\\
# \mathbf{u} &= \mathbf{u}_D \text{ on } \delta\Omega_D \\
# C\epsilon(\mathbf{u})\mathbf{n} &= 0 \text{ on } \delta\Omega_N\\
# \mathbf{u}\cdot \hat{\mathbf{n}} &\leq g \text{ on } \Gamma\\
# \sigma_n(\mathbf{u}) &= (C\epsilon(\mathbf{u})\mathbf{n})\cdot \mathbf{n}\\
# \sigma_n(\mathbf{u}) \mathbf{n} &\leq 0 \text{ on } \Gamma\\
# \sigma_n(\mathbf{u})(\mathbf{u}\cdot \hat{\mathbf{n}}-g) &= 0 \text{ on } \Gamma
# \end{align*}
# $$
# 
# where $\mathbf{u}$ is the displacement, $C$ the stiffness tensor, $\epsilon$ the
# symmetric strain tensor and $\mathbf{f}$ the body force.

# In this tutorial we will consider a half circle, where we apply a displacement on the top boundary,
# and let the curved boundary be a potential contact boundary.
# We define a *rigid surface* as a plane at $y=-h$, where $h\in \mathbb{R}$

# As seen in {ref}`external_mesh`, we can use GMSH to create such a geometry.

# + tags=["hide-input", "hide-output"]
from mpi4py import MPI
import dolfinx
import gmsh
import numpy as np
import ufl

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
c_y = 1
R = 0.5
potential_contact_marker = 2
displacement_marker = 1
res = 0.2
order = 2
refinement_level = 2

# Initialize gmsh
gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    # Create disk and subtract top part
    membrane = gmsh.model.occ.addDisk(0, c_y, 0, R, R)
    square = gmsh.model.occ.addRectangle(-R, c_y, 0, 2 * R, 1.1 * R)
    gmsh.model.occ.synchronize()
    new_tags, _ = gmsh.model.occ.cut([(2, membrane)], [(2, square)])
    gmsh.model.occ.synchronize()

    # Split boundary into two components
    boundary = gmsh.model.getBoundary(new_tags, oriented=False)
    contact_boundary = []
    dirichlet_boundary = []
    for bnd in boundary:
        mass = gmsh.model.occ.getMass(bnd[0], bnd[1])
        if np.isclose(mass, np.pi * R):
            contact_boundary.append(bnd[1])
        elif np.isclose(mass, 2 * R):
            dirichlet_boundary.append(bnd[1])
        else:
            raise RuntimeError("Unknown boundary")

    # Tag physical groups for the surface
    for i, tags in enumerate(new_tags):
        gmsh.model.addPhysicalGroup(tags[0], [tags[1]], i+1)

    # Tag physical groups for the boundary
    gmsh.model.add_physical_group(1, contact_boundary, potential_contact_marker)
    gmsh.model.add_physical_group(1, dirichlet_boundary, displacement_marker)

    # Create higher resolution mesh at the contact boundary
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", contact_boundary)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", res)
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * res)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.075 * R)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.5 * R)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

    # Generate mesh, make second order and refine
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    for _ in range(refinement_level):
        gmsh.model.mesh.refine()
        gmsh.model.mesh.setOrder(order)

# -

# We inspect the generated mesh and markers
omega, ct, ft = dolfinx.io.gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
gmsh.finalize()

# + tags=["hide-input"]
import sys, os

import pyvista

if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)


def plot_mesh(mesh: dolfinx.mesh.Mesh, tags: dolfinx.mesh.MeshTags=None,
              style:str = "surface"):
    plotter = pyvista.Plotter()
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    if tags is None:
        ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
    else:
        # Exclude indices marked zero
        exclude_entities = tags.find(0)
        marker = np.full_like(tags.values, True, dtype=np.bool_)
        marker[exclude_entities] = False
        ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, tags.dim, tags.indices[marker]))
        ugrid.cell_data[ct.name] = tags.values[marker]

    plotter.add_mesh(ugrid, show_edges=True, line_width=3, style=style)
    plotter.show_axes()
    plotter.view_xy()
    plotter.show_bounds()
    plotter.show()

plot_mesh(omega, ct, style="wireframe")
plot_mesh(omega, ft)
# -

# ## Variational formulation
# We will use a formulation of this problem based on {cite}`keith2024` and {cite}`dokken2024`.
# We phrase this problem as a minimization problem, where we seek to find the displacement $\mathbf{u}$ that minimizes
# the functional
#
# $$
# \min_{\mathbf{u}\in \mathcal{K}} J(\mathbf{u}) = \frac{1}{2} \int_\Omega (C\epsilon(\mathbf{u}):\epsilon(\mathbf{v}))~\mathrm{d}x -
# \int_\Omega \mathbf{f}\cdot \mathbf{u}~\mathrm{d}x
# $$
#
# where
# $$
# \mathcal{K} = \{ \mathbf{u}\in V_{\mathbf{u}_D} \vert \mathbf{u}\cdot  \hat{\mathbf{n}} \leq g \}
# $$
#
# With this re-formulation, we can write a mixed finite element method, where we use two variables,
# the displacement $\mathbf{u}\in V(\Omega)$ and a latent variable $\psi \in Q(\Gamma)$/
# ```{admonition} Co-dimension 1 problem
# We note that since $\Gamma$ is the whole curved boundary, we need to solve
# a mixed dimensional finite element problem.
# ```
#
# We choose $V$ to be a P-th order vector Lagrange space, while $Q$ is a Pth order scalar Lagrange field.
# We start by defining the sub-mesh using the read in mesh markers

tdim = omega.topology.dim
fdim = tdim - 1
gdim = omega.geometry.dim
gamma, gamma_to_omega = dolfinx.mesh.create_submesh(omega, fdim, ft.find(potential_contact_marker))[
        0:2]

# Next, we define the function spaces, and combine them in a block structure
# using `ufl.MixedFunctionSpace`

V = dolfinx.fem.functionspace(omega, ("Lagrange", 1, (omega.geometry.dim, )))
Q = dolfinx.fem.functionspace(gamma, ("Lagrange", 1))
W = ufl.MixedFunctionSpace(V, Q)

# We can write the variational formulation as
# Given $\alpha_k$, $\psi_{k-1}$, solve:
# 
# $$
# \begin{align*}
# \alpha_k(\sigma(\mathbf{u}), \epsilon(\mathbf{v}))_\Omega - (\psi, \mathbf{v}\cdot \mathbf{n})_\Gamma &= -\alpha_k(\mathbf{f}, v)_\Omega - (\psi^{k-1}, \mathbf{v}\cdot \mathbf{n})_\Gamma\\
# (\mathbf{u}\cdot \mathbf{n}, w)_\Gamma - (e^{\psi}, w)_\Gamma &= (g, w)_\Gamma
# \end{align*}
# $$
#
# - Check for convergence.
# - Update latent variable $\psi^{k-1}$, $\alpha_k$.

# ```{admonition} Notes on the variational form
# :class: note dropdown
# - It is highly non-linear, as it contains $e^{\psi}$.
# - We can recognize the traditional part of the Signorini problem as the terms multiplied by $\alpha_k$.
# - One can update $\alpha_k$ at each iteration, but it is not a requirement for convergence,
#   and it is often sufficient to use a constant value.
# - We have that for the solved problem $e^{\psi} = \mathbf{u} \cdot \mathbf{n} - g$,
#   which we are guaranteed that this satisfies the Signorini condition.
# ```

# As discussed in the previous chapter, we need to choose a mesh to integrate over.
# As we would like to exploit the definition of the `n=ufl.FacetNormal(\Omega)` in our
# variational problem, we choose the integration domain to be $\Omega$.
# This means that we have to create a map from each facet in $\Omega$ to 
# the corresponding facet in $\Gamma$.

facet_imap = omega.topology.index_map(ft.dim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
omega_to_gamma = np.full(num_facets, -1)
omega_to_gamma[gamma_to_omega] = np.arange(len(gamma_to_omega))
entity_maps = {gamma: omega_to_gamma}
# -

# Next, we define the integration measures

dx = ufl.Measure("dx", domain=omega)
ds = ufl.Measure("ds", domain=omega, subdomain_data=ft,
                 subdomain_id=potential_contact_marker)

# ```{note} Integration over $\Gamma$
# Note that we have restricted the `ds` integration measure to the boundary $\Gamma$,
# where $Q$ is defined.
# ```
#
# Next, we define some problem specific parameters

E = dolfinx.fem.Constant(omega, dolfinx.default_scalar_type(2e4))
nu = dolfinx.fem.Constant(omega, 0.4)
alpha = dolfinx.fem.Constant(omega, dolfinx.default_scalar_type(0.1))

# As we define the rigid surface as a plane at $y=-h$, we can define the gap
# between the undeformed geometry with coordinates (x, y) and the surface as

h = 0.13
x, y = ufl.SpatialCoordinate(omega)
g = y + dolfinx.fem.Constant(omega, dolfinx.default_scalar_type(h))
uD = 0.72

# Similarly, we have that $\hat n = (0, -1)$

n = ufl.FacetNormal(omega)
n_g = dolfinx.fem.Constant(omega, np.zeros(gdim, dtype=dolfinx.default_scalar_type))
n_g.value[-1] = -1
f = dolfinx.fem.Constant(omega, np.zeros(gdim, dtype=dolfinx.default_scalar_type))

# We can write the residual of the variational form

# +
v, w = ufl.TestFunctions(W)
u = dolfinx.fem.Function(V, name="Displacement")
psi = dolfinx.fem.Function(Q, name="Latent_variable")
psi_k = dolfinx.fem.Function(Q, name="Previous_Latent_variable")

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def epsilon(w):
    return ufl.sym(ufl.grad(w))

def sigma(w, mu, lmbda):
    ew = epsilon(w)
    gdim = ew.ufl_shape[0]
    return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(ufl.grad(w)) * ufl.Identity(gdim)


F = alpha * ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * dx
F -= alpha * ufl.inner(f, v) * dx
F += -ufl.inner(psi - psi_k, ufl.dot(v, n)) * ds
F += ufl.inner(ufl.dot(u, n_g), w) * ds
F += ufl.inner(ufl.exp(psi), w) * ds - ufl.inner(g, w) * ds
residual = dolfinx.fem.form(ufl.extract_blocks(F), entity_maps=entity_maps)
# -

# Similarly, we can write the Jacobian

du, dpsi = ufl.TrialFunctions(W)
jac = ufl.derivative(F, u, du) + ufl.derivative(F, psi, dpsi)
J = dolfinx.fem.form(ufl.extract_blocks(jac), entity_maps=entity_maps)

# ```{admonition} Jacobian for mixed function spaces
# Note that we differentiate with respect to the function in the respective
# "sub space", $V$ and $Q$, but use trial functions form $W$. This is to be able
# to extract blocks when creating the form for the Jacobian.
# ```

# We define the displacement on the top boundary as we have done in previous tutorials.
# However, as we are using a `ufl.MixedFunctionSpace`, we can define the boundary condition
# with the appropriate sub-space without a mapping.

# +

def disp_func(x):
    values = np.zeros((gdim, x.shape[1]), dtype=dolfinx.default_scalar_type())
    values[1] = -uD
    return values

u_bc = dolfinx.fem.Function(V)
u_bc.interpolate(disp_func)
bc = dolfinx.fem.dirichletbc(
    u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(displacement_marker)))
bcs = [bc]

# -

# To solve this problem, we will use PETSc.
# We use the following wrapper to solve the problem

# + tags=["hide-input"]
from petsc4py import PETSc
import warnings
import dolfinx.fem.petsc

class NewtonSolver:
    max_iterations: int
    bcs: list[dolfinx.fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: dolfinx.fem.Form
    b: dolfinx.fem.Form
    dx: PETSc.Vec
    error_on_nonconvergence: bool

    def __init__(
        self,
        F: list[dolfinx.fem.form],
        J: list[list[dolfinx.fem.form]],
        w: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC] | None = None,
        max_iterations: int = 5,
        petsc_options: dict[str, str | float | int | None] = None,
        error_on_nonconvergence: bool = True,
    ):
        """Newton solver for blocked nonlinear problems.

        Note:
            Special feature of this solver is that it only measures the norm of the primal space
            increments when checking convergence (primal being the first space in the block).

        :param F: Residual on block form
        :param J: Block formulation of Jacobian
        :param w: List of solution vectors
        :param bcs: List of Dirichlet boundary conditions
        :param max_iterations: Max Newton iterations
        :param petsc_options: Krylov subspace solver options
        :param error_on_nonconvergence: Throw error if solver doesn't converge.
        """
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = dolfinx.fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = dolfinx.fem.petsc.create_matrix_block(J)
        self.dx = dolfinx.fem.petsc.create_vector_block(F)
        self.w = w
        self.x = dolfinx.fem.petsc.create_vector_block(F)
        self.norm_array = dolfinx.fem.Function(w[0].function_space)
        self.error_on_nonconvergence = error_on_nonconvergence
        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def _update_solution(self, beta):
        """Update solution vector ``w`` and internal variable ``x``.

        Two steps are performed:
        1. Update local arrays in ``w`` with the correction ``dx``.
        2. Scatter local arrays to global vector ``x``.
        """
        maps = [
            (
                si.function_space.dofmap.index_map,
                si.function_space.dofmap.index_map_bs,
            )
            for si in self.w
        ]
        # Get local vectors and update ``w`` with correction
        local_dx = dolfinx.cpp.la.petsc.get_local_vectors(self.dx, maps)
        for ldx, s in zip(local_dx, self.w):
            s.x.array[:] -= beta * ldx
            s.x.scatter_forward()

        # Scatter local vectors to blocked vector
        dolfinx.cpp.la.petsc.scatter_local_vectors(
            self.x, [si.x.petsc_vec.array_r for si in self.w], maps
        )
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def solve(self, tol=1e-6, beta=1.0) -> int:
        """Solve nonlinear problem

        Args:
            tol: Stopping tolerance for primal variable update
            beta: Step-size

        Raises:
            RuntimeError: If solver doesn't converge and ``error_on_nonconvergence=True``

        Returns:
            Number of iterations. If Krylov subspace solver doesn't converge, return 0.
        """
        i = 1
        blocked_maps = [
            (
                si.function_space.dofmap.index_map,
                si.function_space.dofmap.index_map_bs,
            )
            for si in self.w
        ]
        while i <= self.max_iterations:
            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            with self.b.localForm() as b_loc:
                b_loc.set(0)
            try:
                dolfinx.fem.petsc.assemble_vector_block(
                    self.b, self.F, self.J, bcs=self.bcs, x0=self.x, scale=-1.0
                )
            except TypeError:
                dolfinx.fem.petsc.assemble_vector_block(
                    self.b, self.F, self.J, bcs=self.bcs, x0=self.x, alpha=-1.0
                )
            self.b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()

            # Solve linear system for correction
            with self.dx.localForm() as dx_loc:
                dx_loc.set(0)
            self._solver.solve(self.b, self.dx)
            self.dx.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            # Check for convergence
            converged_reason = self._solver.getConvergedReason()
            if self.error_on_nonconvergence:
                assert (
                    converged_reason > 0
                ), f"Linear solver did not converge, received reason {converged_reason}"
            else:
                warnings.warn(f"Linear solver did not converge, reason {converged_reason} exiting", RuntimeWarning)
                return 0

            # Update solution
            self._update_solution(beta)


            # Compute norm of primal space diff
            local_du, _ = dolfinx.cpp.la.petsc.get_local_vectors(self.dx, blocked_maps)
            self.norm_array.x.array[:] = local_du
            self.norm_array.x.petsc_vec.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )
            self.norm_array.x.petsc_vec.normBegin(1)
            correction_norm = self.norm_array.x.petsc_vec.normEnd(1)

            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1
        if self.error_on_nonconvergence and i == self.max_iterations:
            raise RuntimeError("Newton solver did not converge")
        return i
# -

# We want to consider the Von-Mises stresses in post-processing, and
# use DOLFINx Expression to interpolate the stresses into an appropriate
# function space.

V_DG = dolfinx.fem.functionspace(omega, ("DG", 1, (omega.geometry.dim,)))
stress_space, stress_to_disp = V_DG.sub(0).collapse()
von_mises = dolfinx.fem.Function(stress_space, name="von_Mises")
u_dg = dolfinx.fem.Function(V_DG, name="u")
s = sigma(u, mu, lmbda) - 1.0 / 3 * ufl.tr(sigma(u, mu, lmbda)) * ufl.Identity(
    len(u)
)
von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s, s))
stress_expr = dolfinx.fem.Expression(von_Mises, stress_space.element.interpolation_points())

# We can now set up the solver and solve the problem

solver = NewtonSolver(
    residual,
    J,
    [u, psi],
    bcs=bcs,
    max_iterations=25,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": 120,
        "ksp_error_if_not_converged": True,
    },
    error_on_nonconvergence=True,
)

# Note that all memory is assigned outside the for-lopp.
# In this problem, we measure the norm of the change in the primal space,
# rather than the for the mixed function.

max_iterations = 25
normed_diff = 0
tol = 1e-5
u_prev = dolfinx.fem.Function(V)
diff = dolfinx.fem.Function(V)
for it in range(max_iterations):
    print(f"{it=}/{max_iterations} {normed_diff:.2e}")
    # Solve the first iterations inaccurately
    solver_tol = 100*tol if it < 3 else tol
    converged = solver.solve(solver_tol, 1)

    diff.x.array[:] = u.x.array - u_prev.x.array
    diff.x.petsc_vec.normBegin(2)
    normed_diff = diff.x.petsc_vec.normEnd(2)
    if normed_diff <= tol and it >=3:
        print(f"Converged at {it=} with increment norm {normed_diff:.2e}<{tol:.2e}")
        break
    u_prev.x.array[:] = u.x.array
    psi_k.x.array[:] = psi.x.array
    if not converged:
        print(f"Solver did not convert at {it=}, exiting with {converged=}")
        break

if it == max_iterations - 1:
    print(f"Did not converge within {max_iterations} iterations")

# We compute the von-Mises stresses for the final solution

von_mises.interpolate(stress_expr)

# Additionally, we interpolate the displacement onto a DG function space
# for compatible visualization in Pyvista.

u_dg.interpolate(u)


# -

# + tags=["hide-input"]
import sys, os

import pyvista

if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)
grid = dolfinx.plot.vtk_mesh(u_dg.function_space)
pyvista_grid = pyvista.UnstructuredGrid(*grid)
values = u_dg.x.array.reshape(-1, omega.geometry.dim)
values_padded = np.zeros((values.shape[0], 3))
values_padded[:, :omega.geometry.dim] = values
pyvista_grid.point_data["u"] = values_padded
warped = pyvista_grid.warp_by_vector("u")
stresses = np.zeros_like(u_dg.x.array)
stresses[stress_to_disp]= von_mises.x.array
stresses = stresses.reshape(-1, omega.geometry.dim)[:, 0]
warped.point_data["von_Mises"] = stresses

plotter = pyvista.Plotter()
plotter.add_mesh(pyvista_grid, style="wireframe", color="b")
plotter.add_mesh(warped, scalars="von_Mises", lighting=True, show_edges=True)
plotter.show_bounds()
plotter.view_xy()
plotter.show()
# -


# ## References
# ```{bibliography}
#    :filter: cited and ({"src/multiphysics/coupling"} >= docnames)
# ```