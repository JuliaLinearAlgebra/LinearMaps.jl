struct KhatriRaoMap{T,A<:Tuple{MapOrVecOrMat,MapOrVecOrMat}} <: LinearMap{T}
    maps::A
    function KhatriRaoMap{T,As}(maps::As) where {T,As<:Tuple{MapOrVecOrMat,MapOrVecOrMat}}
        @assert promote_type(T, map(eltype, maps)...) == T  "eltype $(eltype(A)) cannot be promoted to $T in KhatriRaoMap constructor"
        @inbounds size(maps[1], 2) == size(maps[2], 2) || throw(ArgumentError("matrices need equal number of columns"))
        new{T,As}(maps)
    end
end
KhatriRaoMap{T}(maps::As) where {T, As} = KhatriRaoMap{T, As}(maps)

"""
    khatrirao(A::MapOrVecOrMat, B::MapOrVecOrMat) -> KhatriRaoMap

Construct a lazy representation of the Khatri-Rao (or column-wise Kronecker) product of two
maps or arrays `A` and `B`. For the application to vectors, the tranpose action of `A` on
vectors needs to be defined.
"""
khatrirao(A::MapOrVecOrMat, B::MapOrVecOrMat) =
    KhatriRaoMap{Base.promote_op(*, eltype(A), eltype(B))}((A, B))

struct FaceSplittingMap{T,A<:Tuple{AbstractMatrix,AbstractMatrix}} <: LinearMap{T}
    maps::A
    function FaceSplittingMap{T,As}(maps::As) where {T,As<:Tuple{AbstractMatrix,AbstractMatrix}}
        @assert promote_type(T, map(eltype, maps)...) == T  "eltype $(eltype(A)) cannot be promoted to $T in KhatriRaoMap constructor"
        @inbounds size(maps[1], 1) == size(maps[2], 1) || throw(ArgumentError("matrices need equal number of columns, got $(size(maps[1], 1)) and $(size(maps[2], 1))"))
        new{T,As}(maps)
    end
end
FaceSplittingMap{T}(maps::As) where {T, As} = FaceSplittingMap{T, As}(maps)

"""
    facesplitting(A::AbstractMatrix, B::AbstractMatrix) -> FaceSplittingMap

Construct a lazy representation of the face-splitting (or row-wise Kronecker) product of
two matrices `A` and `B`.
"""
facesplitting(A::AbstractMatrix, B::AbstractMatrix) =
    FaceSplittingMap{Base.promote_op(*, eltype(A), eltype(B))}((A, B))

Base.size(K::KhatriRaoMap) = ((A, B) = K.maps; (size(A, 1) * size(B, 1), size(A, 2)))
Base.size(K::FaceSplittingMap) = ((A, B) = K.maps; (size(A, 1), size(A, 2) * size(B, 2)))
Base.adjoint(K::KhatriRaoMap) = facesplitting(map(adjoint, K.maps)...)
Base.adjoint(K::FaceSplittingMap) = khatrirao(map(adjoint, K.maps)...)
Base.transpose(K::KhatriRaoMap) = facesplitting(map(transpose, K.maps)...)
Base.transpose(K::FaceSplittingMap) = khatrirao(map(transpose, K.maps)...)

LinearMaps.MulStyle(::Union{KhatriRaoMap,FaceSplittingMap}) = FiveArg()

function _unsafe_mul!(y, K::KhatriRaoMap, x::AbstractVector)
    A, B = K.maps
    Y = reshape(y, (size(B, 1), size(A, 1)))
    if size(B, 1) <= size(A, 1)
        mul!(Y, convert(Matrix, B * Diagonal(x)), transpose(A))
    else
        mul!(Y, B, transpose(convert(Matrix, A * transpose(Diagonal(x)))))
    end
    return y
end
function _unsafe_mul!(y, K::KhatriRaoMap, x::AbstractVector, α, β)
    A, B = K.maps
    Y = reshape(y, (size(B, 1), size(A, 1)))
    if size(B, 1) <= size(A, 1)
        mul!(Y, convert(Matrix, B * Diagonal(x)), transpose(A), α, β)
    else
        mul!(Y, B, transpose(convert(Matrix, A * transpose(Diagonal(x)))), α, β)
    end
    return y
end

function _unsafe_mul!(y, K::FaceSplittingMap, x::AbstractVector)
    A, B = K.maps
    @inbounds for m in eachindex(y)
        y[m] = zero(eltype(y))
        l = firstindex(x)
        for i in axes(A, 2)
            ai = A[m,i]
            @simd for k in axes(B, 2)
                y[m] += ai*B[m,k]*x[l]
                l += 1
            end
        end
    end
    return y
end
function _unsafe_mul!(y, K::FaceSplittingMap, x::AbstractVector, α, β)
    A, B = K.maps
    @inbounds for m in eachindex(y)
        y[m] *= β
        l = firstindex(x)
        for i in axes(A, 2)
            ai = A[m,i]
            @simd for k in axes(B, 2)
                y[m] += ai*B[m,k]*x[l]*α
                l += 1
            end
        end
    end
    return y
end
