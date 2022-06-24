struct InverseMap{T, F, S} <: LinearMap{T}
    A::F
    ldiv!::S
end

"""
    InverseMap(A; solver = ldiv!)

Lazy inverse of `A` such that `InverseMap(A) * x` is the same as `A \\ x`. Letting an
`InverseMap` act on a vector thus requires solving a linear system.

A solver function can be passed with the `solver` keyword argument. The solver should
be of the form `f(y, A, x)` where `A` is the wrapped map, `x` the right hand side, and
`y` a preallocated output vector in which the result should be stored. The default solver
is `LinearAlgebra.ldiv!`.

Note that `A` must be compatible with the solver function. `A` can, for example, be a
factorization of a matrix, or another `LinearMap` (in combination with an iterative solver
such as conjugate gradient).

# Examples
```julia
julia> using LinearMaps, LinearAlgebra

julia> A = rand(2, 2); b = rand(2);

julia> InverseMap(lu(A)) * b
2-element Vector{Float64}:
  1.0531895201271027
 -0.4718540250893251

julia> A \\ b
2-element Vector{Float64}:
  1.0531895201271027
 -0.4718540250893251
```
"""
function InverseMap(A::F; solver::S=LinearAlgebra.ldiv!) where {F, S}
    T = eltype(A)
    InverseMap{T,F,S}(A, solver)
end

Base.size(imap::InverseMap) = size(imap.A)
Base.transpose(imap::InverseMap) = InverseMap(transpose(imap.A); solver=imap.ldiv!)
Base.adjoint(imap::InverseMap) = InverseMap(adjoint(imap.A); solver=imap.ldiv!)

LinearAlgebra.issymmetric(imap::InverseMap) = issymmetric(imap.A)
LinearAlgebra.ishermitian(imap::InverseMap) = ishermitian(imap.A)
LinearAlgebra.isposdef(imap::InverseMap) = isposdef(imap.A)

# Two separate methods to deal with method ambiguities
function _unsafe_mul!(y::AbstractVector, imap::InverseMap, x::AbstractVector)
    imap.ldiv!(y, imap.A, x)
    return y
end
function _unsafe_mul!(y::AbstractMatrix, imap::InverseMap, x::AbstractMatrix)
    imap.ldiv!(y, imap.A, x)
    return y
end
