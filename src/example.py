import basix.ufl
import ufl

cell = "triangle"
c_el = basix.ufl.element("Lagrange", cell, 1, shape=(2,))
domain = ufl.Mesh(c_el)

el = basix.ufl.element("Lagrange", cell, 2)
V = ufl.FunctionSpace(domain, el)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(u, v) * ufl.dx

forms = [a]
