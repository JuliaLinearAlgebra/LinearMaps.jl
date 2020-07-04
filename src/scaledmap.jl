#=
scaledmap.jl
Efficiently handle "scale * A" when the scale factor is Real or Complex.
In principle, the map A itself could be any type,
but for simplicity this code focuses "solely" on Real or Complex LinearMaps.
=#

using LinearAlgebra: UniformScaling
import LinearAlgebra: issymmetric, ishermitian, isposdef

#=
todo: i would prefer to let "scale" be a different type than the map.
e.g., if A is ComplexF32, allow scale to be Real so that multiplying
by scale would require only a real multiply.  but i could not figure
out how to get @inferred to pass when i had scale::S
=#

"""
    ScaledMap{T, A<:LinearMapRC} <: LinearMap{T}
"""
struct ScaledMap{T, A<:LinearMapRC} <: LinearMap{T}
    scale::T
    map::A
    function ScaledMap{T, A}(scale::S, map::A) where {T, S <: RealOrComplex, A <: LinearMapRC}
        promote_type(S, eltype(map)) == T || throw(InexactError())
        new{T, A}(scale, map)
    end
end

# constructor
ScaledMap(scale::S, map::A) where {S <: RealOrComplex, A <: LinearMapRC} =
    ScaledMap{promote_type(S,eltype(map)), A}(scale, map)

# show
function Base.show(io::IO, A::ScaledMap{T}) where {T}
    println(io, "LinearMaps.ScaledMap{$T}, scale = $(A.scale)")
    show(io, A.map)
end

# basic methods
Base.size(A::ScaledMap) = size(A.map)
Base.isreal(A::ScaledMap) = isreal(A.scale) && isreal(A.map)
LinearAlgebra.issymmetric(A::ScaledMap) = isreal(A.scale) && issymmetric(A.map)
LinearAlgebra.ishermitian(A::ScaledMap) = ishermitian(A.map)

# subtle issue here that if map is Complex, then so is scale
LinearAlgebra.isposdef(A::ScaledMap) =
    isposdef(A.map) && isreal(A.scale) && real(A.scale) > 0

Base.transpose(A::ScaledMap) = ScaledMap(A.scale, transpose(A.map))
Base.adjoint(A::ScaledMap) = ScaledMap(conj(A.scale), adjoint(A.map))

# comparison (sufficient, not necessary)
Base.:(==)(A::ScaledMap, B::ScaledMap) =
    (eltype(A) == eltype(B) && A.map == B.map) && A.scale == B.scale

# approximate comparison (because == for real scalars is dubious)
Base.:(≈)(A::ScaledMap, B::ScaledMap) =
    (eltype(A) == eltype(B) && A.map == B.map) && A.scale ≈ B.scale

# x * conj(x) is real in math, but not perfectly so in computation,
# but we want it to be real here (when possible) for isposdef()
# and so that α*conj(α)*A is has a real scale when A is Real
@inline _scalar_product(a, b) = isreal(a*b) ? real(a*b) : a*b

# scalar multiplication and division
Base.:(*)(α::RealOrComplex, A::ScaledMap) =
    ScaledMap(_scalar_product(α, A.scale), A.map)
Base.:(*)(α::RealOrComplex, A::LinearMapRC) = ScaledMap(α, A)
Base.:(*)(A::LinearMapRC, α::RealOrComplex) = α * A
Base.:(*)(A::ScaledMap, α::RealOrComplex) = α * A

Base.:(*)(α::RealOrComplex, A::CompositeMapRC) = ScaledMap(α, A)
Base.:(*)(A::CompositeMapRC, α::RealOrComplex) = ScaledMap(α, A)

Base.:(\)(α::RealOrComplex, A::LinearMapRC) = inv(α) * A
Base.:(/)(A::LinearMapRC, α::RealOrComplex) = inv(α) * A
Base.:(-)(A::LinearMapRC) = -1 * A # possibly redundant with compositemap.jl

# I
Base.:(*)(A::LinearMapRC, B::UniformScaling{<:RealOrComplex}) = B.λ * A
Base.:(*)(A::UniformScaling{<:RealOrComplex}, B::LinearMapRC) = A.λ * B


# composition (not essential, but might save multiple scaling operations)
Base.:(*)(A::ScaledMap, B::ScaledMap) =
     ScaledMap(_scalar_product(A.scale, B.scale), A.map * B.map)

# composition: keep "scale" as the outer-most type
# this helps with cases like "transpose(F) * 3 * F"
Base.:(*)(A::ScaledMap, B::LinearMapRC) = ScaledMap(A.scale, A.map * B)
Base.:(*)(A::LinearMapRC, B::ScaledMap) = ScaledMap(B.scale, A * B.map)

# multiplication with vectors
function A_mul_B!(y::AbstractVector, A::ScaledMap, x::AbstractVector)
    # no size checking, will be done by map
    A_mul_B!(y, A.map, x)
    y .*= A.scale # todo: any more efficient way using 4-arg mul! ?
end

# todo: 5-arg mul! ?

# todo: these next two seem unnecessary
#At_mul_B!(y::AbstractVector, A::ScaledMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)
#Ac_mul_B!(y::AbstractVector, A::ScaledMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)
