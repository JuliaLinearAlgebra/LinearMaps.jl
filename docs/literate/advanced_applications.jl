# # Advanced applications
#
# ```@meta
# Draft = false
# ```
#
# On this page we demonstrate the use of the LinearMaps package for more for some more
# advanced example and/or use cases. If you have a nice example to add to the list, please
# feel free to contribute!
#
# ## Solving indefinite system with iterative methods
#
# This example demonstrates how LinearMaps.jl with can be combined with
# [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
# to solve a blocked linear system of the form:
#
# ```math
# \begin{pmatrix}
# M & B \\
# B^\mathrm{T} & 0
# \end{pmatrix}
# \begin{pmatrix}
# U \\ P
# \end{pmatrix}
# =
# \begin{pmatrix}
# F \\ G
# \end{pmatrix}
# ```
#
# where ``M`` is a positive definite matrix, and ``B`` has full rank. This example is
# translated from [this tutorial
# program](https://www.dealii.org/current/doxygen/deal.II/step_20.html) from the
# documentation of the [deal.II](https://www.dealii.org/) finite element library. The system
# originates from a mixed finite element discretization of the Laplace equation. However, in
# this example we will focus only on the solution of the system above, and refer to the
# deal.II example for more background information.
#

using LinearMaps, LinearAlgebra, SparseArrays, IterativeSolvers


# First we create the involved block matrices.

s = 1000
m, n = 15*s, 10*s

nz = 0.001

M = sprand(m, m, nz) + 2I; M = M'M
B = spdiagm(m, n, 0 => rand(n) .+ 3, 1 => rand(n-1), -1 => rand(n-1))
B = spdiagm(m, n, 0 => rand(n) .+ 0, 1 => rand(n-1), -1 => rand(n-1))

F = rand(m)
G = rand(n)


A = [M B; B' 0I]
b = [F; G]

x = A \ b

# cgz!(y, A, x) = IterativeSolvers.cg!(fill!(y, 0), A, x; abstol=10-6)
cgz!(y, A, x) = IterativeSolvers.cg!(fill!(y, 0), A, x)

iM = InverseMap(M; solver=cgz!)

S = B' * iM * B


# Preconditioner for S
# P = B' * LinearMap(inv(Diagonal(convert(Vector, diag(M))))) * B
P = B' * inv(Diagonal(convert(Vector, diag(M)))) * B

#-

G′ = B' * iM * F - G
P = IterativeSolvers.cg(S, G′; Pl=P)
P = IterativeSolvers.cg(S, G′; verbose=true)

F′ = F - B*P
U = IterativeSolvers.cg(M, F′)


#-

norm(U - x[1:m]) / norm(U)

#-
norm(P - x[(m+1):end]) / norm(P)
