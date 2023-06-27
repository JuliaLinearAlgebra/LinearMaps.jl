# Matrix: create matrix representation of LinearMap
function Base.Matrix{T}(A::LinearMap) where {T}
    mat = Matrix{T}(undef, size(A))
    _unsafe_mul!(mat, A, true)
end

function Base.AbstractMatrix{T}(A::LinearMap) where {T}
    mat = similar(Array{T}, axes(A))
    _unsafe_mul!(mat, A, true)
end

Base.Matrix(A::LinearMap{T}) where {T} = Matrix{T}(A)
Base.AbstractMatrix(A::LinearMap{T}) where {T} = AbstractMatrix{T}(A)

Base.Array(A::LinearMap) = Matrix(A)
Base.convert(::Type{T}, A::LinearMap) where {T<:Matrix} = T(A)
Base.convert(::Type{Array}, A::LinearMap) = convert(Matrix, A)
# Base.convert(::Type{AbstractMatrix}, A::LinearMap) = convert(Matrix, A)
Base.convert(::Type{AbstractMatrix}, A::LinearMap) = AbstractMatrix(A)
Base.convert(::Type{AbstractArray}, A::LinearMap) = convert(AbstractMatrix, A)

# special cases

# UniformScalingMap
Base.convert(::Type{AbstractMatrix}, J::UniformScalingMap) = Diagonal(fill(J.λ, J.M))

# WrappedMap
Base.Matrix{T}(A::WrappedMap) where {T} = Matrix{T}(A.lmap)
Base.convert(::Type{T}, A::WrappedMap) where {T<:Matrix} = convert(T, A.lmap)
Base.Matrix{T}(A::VectorMap) where {T} = copyto!(Matrix{eltype(T)}(undef, size(A)), A.lmap)
Base.convert(::Type{T}, A::VectorMap) where {T<:Matrix} = T(A)
Base.convert(::Type{AbstractMatrix}, A::WrappedMap) = convert(AbstractMatrix, A.lmap)

# TransposeMap & AdjointMap
for (T, t) in ((AdjointMap, adjoint), (TransposeMap, transpose))
    @eval Base.convert(::Type{AbstractMatrix}, A::$T) = $t(convert(AbstractMatrix, A.lmap))
end

# CompositeMap
function Base.Matrix{T}(AB::CompositeMap{<:Any, <:Tuple{VecOrMatMap, LinearMap}}) where {T}
    B, A = AB.maps
    require_one_based_indexing(B)
    Y = Matrix{T}(undef, size(AB))
    for (yi, bi) in zip(eachcol(Y), eachcol(B.lmap))
        _unsafe_mul!(yi, A, bi)
    end
    return Y
end
for ((TA, fieldA), (TB, fieldB)) in (((VecOrMatMap, :lmap), (VecOrMatMap, :lmap)),
                                     ((VecOrMatMap, :lmap), (UniformScalingMap, :λ)),
                                     ((UniformScalingMap, :λ), (VecOrMatMap, :lmap)))
    @eval function Base.convert(::Type{AbstractMatrix},
                                AB::CompositeMap{<:Any,<:Tuple{$TB,$TA}})
        B, A = AB.maps
        return A.$fieldA*B.$fieldB
    end
end
function Base.Matrix{T}(AB::CompositeMap{<:Any, <:Tuple{VecOrMatMap, VecOrMatMap}}) where {T}
    B, A = AB.maps
    return mul!(Matrix{T}(undef, size(AB)), A.lmap, B.lmap)
end
function Base.Matrix{T}(λA::CompositeMap{<:Any, <:Tuple{VecOrMatMap, UniformScalingMap}}) where {T}
    A, J = λA.maps
    return mul!(Matrix{T}(undef, size(λA)), J.λ, A.lmap)
end
function Base.Matrix{T}(Aλ::CompositeMap{<:Any, <:Tuple{UniformScalingMap, VecOrMatMap}}) where {T}
    J, A = Aλ.maps
    return mul!(Matrix{T}(undef, size(Aλ)), A.lmap, J.λ)
end

# KroneckerMap & KroneckerSumMap
Base.Matrix{T}(A::KroneckerMap) where {T} = kron(convert.(Matrix{T}, A.maps)...)
Base.convert(::Type{AbstractMatrix}, A::KroneckerMap) =
    kron(convert.(AbstractMatrix, A.maps)...)

function Base.Matrix{T}(L::KroneckerSumMap) where {T}
    A, B = L.maps
    IA = Diagonal(ones(Bool, size(A, 1)))
    IB = Diagonal(ones(Bool, size(B, 1)))
    return kron(Matrix{T}(A), IB) + kron(IA, Matrix{T}(B))
end
function Base.convert(::Type{AbstractMatrix}, L::KroneckerSumMap)
    A, B = L.maps
    IA = Diagonal(ones(Bool, size(A, 1)))
    IB = Diagonal(ones(Bool, size(B, 1)))
    return kron(convert(AbstractMatrix, A), IB) + kron(IA, convert(AbstractMatrix, B))
end

# FillMap
Base.Matrix{T}(A::FillMap) where {T} = fill(T(A.λ), size(A))
