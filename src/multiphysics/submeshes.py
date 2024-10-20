# # Multiphysics: Solving PDEs on subdomains
# So far we have considered problems where the PDE is defined over the entire domain.
# However, in many cases this is not accurate. An example of this is fluid-structure interaction,
# where the fluid and solid domains are coupled. 
# In this case, the PDEs are defined over different subdomains,
# and the coupling is done at the interface between the subdomains. 
# In this section, we will show how to solve PDEs on subdomains using FEniCS.

# We will consider a simple problem where we have a domain $\Omega$ that is divided into two subdomains $\Omega_1$ and $\Omega_2$.
# In each of these domains we want to solve a PDE (that is not coupled to the other domain).

# We will consider the following PDEs:
#
# $$
# \begin{align*}
# - \nabla \cdot (\kappa \nabla T) &= f \quad \text{in } \Omega_1, \\
# \kappa \nabla T \cdot \mathbf{n} &= g \quad \text{on } \Gamma, \\
# T &= g \quad \text{ on } \partial \Omega_1\setminus\Gamma, \\
# - \nabla \cdot ( \nabla \mathbf{u}) - \nabla \bar p &= \mathbf{f} \quad \text{in } \Omega_2 \\
# \nabla \cdot \mathbf{u} &= 0 \quad \text{in } \Omega_2 \\
# \mathbf{u} &= \mathbf{g} \text{ on } \partial {\Omega_{2,D}} \\
# \nabla \mathbf{u} \cdot \mathbf{n} + \bar p \mathbf{n} &= \mathbf{0} \quad \text{on } \partial_{\Omega_{2, N}}
# \end{align*}
# $$
