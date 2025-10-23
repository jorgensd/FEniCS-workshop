# # Visualization and outputting formats

# ## Visualization

# In this workshop, we will often need to inspect the meshes, markers and functions that we create,
# beyond what is possible with the print function in Python.

# As seen for the most basic interactions on an element level, or things such as line plots,
# we can use the `matplotlib` library to create plots within our scripts.
#
# However, we will often deal with unstructured grids, and sometimes with curved elements, which
# is not easy to visualize with `matplotlib`.
#
# We therefore recommend using `pyvista` for visualization within your codes.
# Pyvista can either be used for interactive plotting, or to create screen-shots/pngs.
# In this tutorial the primary focus is interactive plotting.
#

# The easiest way to get going with pyvista is to use the following commands

# +
import pyvista

plotter = pyvista.Plotter()
plotter.show()
plotter.screenshot("test.png")
# -

# The environment variables that has been set for DOLFINx in the provided conda environment is listed below
# ```bash
#  PYVISTA_TRAME_SERVER_PROXY_PREFIX="/proxy/"
#  PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"
#  PYVISTA_OFF_SCREEN=false
#  PYVISTA_JUPYTER_BACKEND="html"
# ```

# ## Outputting
#
# Sometimes one doesn't want to inspect or visualize the result right away.
# Then it is convenient to save the solution to a file that can be opened in
# an external visualization software, such as {term}`Paraview`.

# DOLFINx supports a variety of output formats compatible formats, each with its own
# benefits and drawbacks.

# ### XDMFFile
#
# #### Pros
#  - Preferred format for reading in meshes and mesh tags
#  - Can store multiple meshes in a single file
#  - Binary format (`.h5`) allows efficient parallel reading
#  - `h5py` from Python can be used to inspect the h5-files
#  - `h5dump` can be used to inspect the files in the terminal
# #### Cons
# - Only supports first and second order geometries
# - Functions can only be outputted as continuous Lagrange functions of
#   the same degree as the mesh coordinate element
# - No longer maintained by Kitware

# ### VTXWriter
#
# #### Pros
# - Flexible format that can output arbitrary order continuous and discontinuous
#   Lagrange functions (similar to Pyvista)
# - Uses a binary format (binary pack) from ADIOS2 as backend, which is easy to adapt to your own purposes.
# - `bpls` can be used to inspect the files from terminal
# #### Cons
# - Some limited support for DG-0 functions (time-dependent functions currently not working)
# - Storage of a single mesh

# ### FidesWriter
# #### Pros
# - Flexible format that can output arbitrary order continuous and discontinuous
#   Lagrange functions (similar to Pyvista)
# - Uses a binary format (binary pack) from ADIOS2 as backend, which is easy to adapt to your own purposes.
# - `bpls` can be used to inspect the files from terminal
#
# #### Cons
# - Can only store linear (first order) meshes

# ### VTKFile
#
# #### Pros
# - Flexible format that can output arbitrary order continuous and discontinuous
#   Lagrange functions (similar to Pyvista)
# #### Cons
# - ASCII (text) based output that creates many files when used with multiple MPI processes

# ## Checkpointing
# Checkpointing in DOLFINx is supported with the extension [ADIOS4DOLFINx](https://github.com/jorgensd/adios4dolfinx).
# See the online [documentation](https://jsdokken.com/adios4dolfinx) for illustrative use-cases.
