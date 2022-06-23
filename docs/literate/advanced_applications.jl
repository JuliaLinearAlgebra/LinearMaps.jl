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

using LinearMaps, LinearAlgebra, SparseArrays, IterativeSolvers


# First we create the involved block matrices.

m, n = 15, 10

M = sprand(m, m, 0.1) + 2I; M = M'M
B = spdiagm(m, n, 0 => rand(n) .+ 3, 1 => rand(n-1), -1 => rand(n-1))

F = rand(m)
G = rand(n)


A = [M B; B' 0I]
b = [F; G]

x = A \ b

# cgz!(y, A, x) = IterativeSolvers.cg!(fill!(y, 0), A, x; abstol=10-6)
cgz!(y, A, x) = IterativeSolvers.cg!(fill!(y, 0), A, x)

iM = InverseMap(M; solver=cgz!)

S = B' * iM * B

G′ = B' * iM * F - G
P = IterativeSolvers.cg(S, G′)

F′ = F - B*P
U = IterativeSolvers.cg(M, F′)


#-

norm(U - x[1:m]) / norm(U)

#-
norm(P - x[(m+1):end]) / norm(P)
