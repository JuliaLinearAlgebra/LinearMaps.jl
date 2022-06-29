# # Advanced applications
#
# ```@meta
# Draft = false
# ```
#
# On this page we demonstrate the use of the LinearMaps package for more for some more
# advanced example and/or use cases. If you have a nice example to add to the list, please
# feel free to contribute! [^1]
#
# ```@contents
# Pages = ["advanced_applications.md"]
# ```
#
# ## Solving indefinite system with iterative methods
#
# This example demonstrates how LinearMaps.jl with can be combined with
# [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
# to solve an indefinite blocked linear system of the form:
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
# \end{pmatrix},
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
# Since the system is not positive definite it is not possible to solve it using conjugate
# gradients (CG). While other iterative methods can handle indefinite systems, having zeroes
# on the diagonal is problematic for some preconditioners. It is of course possible to use a
# direct solver using e.g. LU factorization from UMFPACK, which is provided by the
# SparseArrays standard library. However, in this example we will take a different approach
# and solve the system using the [Schur
# complement](https://en.wikipedia.org/wiki/Schur_complement).
#
# We first rewrite the system above by multiplying the first row with ``B^\mathrm{T}M^{-1}``
# and subtracting the second row. We then obtain the following equivalent system:
#
# ```math
# \begin{align*}
# B^\mathrm{T} M^{-1} B P &= B^\mathrm{T} M^{-1} F - G,\\
# MU &= F - B P.
# \end{align*}
# ```
#
# The Schur complement ``S := B^\mathrm{T} M^{-1} B`` is symmetric and positive definite
# (since ``M`` is positive definite and ``B`` have full rank) and we can thus apply CG to
# the first equation to solve for ``P``, and then apply CG again to the second equation to
# solve for ``U``. However, computing ``S`` is expensive, in particular since the inverse
# ``M^{-1}`` won't be sparse, in general. Fortunately CG doesn't require us to materialize
# the matrix, we only need a way to evaluate ``S v = B^\mathrm{T} M^{-1} B v`` for a vector
# ``v``. We can do this as follows:
#
#  1. Compute ``w = B v``,
#  2. Solve ``M y = w`` for ``y`` using CG (``M`` is positive definite),
#  2. Compute ``z = S v = B^\mathrm{T} y``.
#

# First we load the packages that we need, and generate matrices the data.

using LinearMaps, LinearAlgebra, SparseArrays, IterativeSolvers
using Preconditioners

s = 100
m, n = 15*s, 10*s

nz = 0.001

Mt = sprand(m, m, nz) + 5I
const M = Mt'Mt
B = spdiagm(m, n, 0 => rand(n) .+ 3, 1 => rand(n-1), -1 => rand(n-1))
const B = spdiagm(m, n, 0 => rand(n) .+ 1, 1 => rand(n-1), -1 => rand(n-1))

const F = rand(m)
const G = rand(n)


A = [M B; B' 0I]
b = [F; G]

x = A \ b


const precond_M = DiagonalPreconditioner(M)

# cgz!(y, A, x) = IterativeSolvers.cg!(fill!(y, 0), A, x; abstol=10-6)
cgz! = (y, A, x) -> IterativeSolvers.cg!(fill!(y, 0), A, x; Pl=precond_M, verbose=true)

const iM = InverseMap(M; solver=cgz!)

const S = B' * iM * B


# Preconditioner for S
# P = B' * LinearMap(inv(Diagonal(convert(Vector, diag(M))))) * B

struct PreconditionS{A}
    P::A
end

function LinearAlgebra.ldiv!(y, p::PreconditionS)
    yc = copy(y)
    ldiv!(y, p, yc)
end
function LinearAlgebra.ldiv!(y, p::PreconditionS, x)
    ldiv!(y, p.P, x)
end


const precond_S = PreconditionS(lu(B' * inv(Diagonal(convert(Vector, diag(M)))) * B))
const precond_S2 = DiagonalPreconditioner(B' * identity(Diagonal(convert(Vector, diag(M)))) * B)

#-

const G′ = B' * iM * F - G
P = IterativeSolvers.cg(S, G′; verbose=false)
P = IterativeSolvers.cg(S, G′; Pl=precond_S, verbose=true)
P = IterativeSolvers.cg(S, G′; Pl=precond_S2, verbose=true)

F′ = F - B*P
U = IterativeSolvers.cg(M, F′; verbose=true)
U = IterativeSolvers.cg(M, F′; verbose=true, Pl=precond_M)


#-

norm(U - x[1:m]) / norm(U)

#-
norm(P - x[(m+1):end]) / norm(P)
