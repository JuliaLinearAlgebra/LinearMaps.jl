# Matrix: create matrix representation of LinearMap
function Base.Matrix(A::LinearMap)
    M, N = size(A)
    T = eltype(A)
    mat = Matrix{T}(undef, (M, N))
    v = fill(zero(T), N)
    @inbounds for i in 1:N
        v[i] = one(T)
        mul!(view(mat, :, i), A, v)
        v[i] = zero(T)
    end
    return mat
end
Base.Array(A::LinearMap) = Matrix(A)
Base.convert(::Type{Matrix}, A::LinearMap) = Matrix(A)
Base.convert(::Type{Array}, A::LinearMap) = convert(Matrix, A)
Base.convert(::Type{AbstractMatrix}, A::LinearMap) = convert(Matrix, A)
Base.convert(::Type{AbstractArray}, A::LinearMap) = convert(AbstractMatrix, A)

# sparse: create sparse matrix representation of LinearMap
function SparseArrays.sparse(A::LinearMap{T}) where {T}
    M, N = size(A)
    rowind = Int[]
    nzval = T[]
    colptr = Vector{Int}(undef, N+1)
    v = fill(zero(T), N)
    Av = Vector{T}(undef, M)

    @inbounds for i in 1:N
        v[i] = one(T)
        mul!(Av, A, v)
        js = findall(!iszero, Av)
        colptr[i] = length(nzval) + 1
        if length(js) > 0
            append!(rowind, js)
            append!(nzval, Av[js])
        end
        v[i] = zero(T)
    end
    colptr[N+1] = length(nzval) + 1

    return SparseMatrixCSC(M, N, colptr, rowind, nzval)
end
Base.convert(::Type{SparseMatrixCSC}, A::LinearMap) = sparse(A)

# special cases

# UniformScalingMap
Base.Matrix(J::UniformScalingMap) = Matrix(J.λ*I, size(J))
Base.convert(::Type{AbstractMatrix}, J::UniformScalingMap) = Diagonal(fill(J.λ, size(J, 1)))

# WrappedMap
Base.Matrix(A::WrappedMap) = Matrix(A.lmap)
Base.convert(::Type{AbstractMatrix}, A::WrappedMap) = convert(AbstractMatrix, A.lmap)
Base.convert(::Type{Matrix}, A::WrappedMap) = convert(Matrix, A.lmap)
SparseArrays.sparse(A::WrappedMap) = sparse(A.lmap)
Base.convert(::Type{SparseMatrixCSC}, A::WrappedMap) = convert(SparseMatrixCSC, A.lmap)

# TransposeMap & AdjointMap
for (TT, T) in ((AdjointMap, adjoint), (TransposeMap, transpose))
    @eval Base.convert(::Type{AbstractMatrix}, A::$TT) = $T(convert(AbstractMatrix, A.lmap))
    @eval SparseArrays.sparse(A::$TT) = $T(convert(SparseMatrixCSC, A.lmap))
end

# LinearCombination
for (TT, T) in ((Type{Matrix}, Matrix), (Type{SparseMatrixCSC}, SparseMatrixCSC))
    @eval function Base.convert(::$TT, ΣA::LinearCombination{<:Any,<:Tuple{Vararg{MatrixMap}}})
        maps = ΣA.maps
        mats = map(A->getfield(A, :lmap), maps)
        return convert($T, sum(mats))
    end
end

# CompositeMap
for (TT, T) in ((Type{Matrix}, Matrix), (Type{SparseMatrixCSC}, SparseMatrixCSC))
    @eval function Base.convert(::$TT, AB::CompositeMap{<:Any,<:Tuple{MatrixMap,MatrixMap}})
        B, A = AB.maps
        return convert($T, A.lmap*B.lmap)
    end
    @eval function Base.convert(::$TT, λA::CompositeMap{<:Any,<:Tuple{MatrixMap,UniformScalingMap}})
        A, J = λA.maps
        return convert($T, J.λ*A.lmap)
    end
    @eval function Base.convert(::$TT, Aλ::CompositeMap{<:Any,<:Tuple{UniformScalingMap,MatrixMap}})
        J, A = Aλ.maps
        return convert($T, A.lmap*J.λ)
    end
end

# BlockMap & BlockDiagonalMap
Base.Matrix(A::BlockMap) = hvcat(A.rows, convert.(Matrix, A.maps)...)
Base.convert(::Type{AbstractMatrix}, A::BlockMap) = hvcat(A.rows, convert.(AbstractMatrix, A.maps)...)
function Base.convert(::Type{SparseMatrixCSC}, A::BlockMap)
    return hvcat(
        A.rows,
        convert(SparseMatrixCSC, first(A.maps)),
        convert.(AbstractMatrix, Base.tail(A.maps))...
    )
end
Base.Matrix(A::BlockDiagonalMap) = cat(convert.(Matrix, A.maps)...; dims=(1,2))
Base.convert(::Type{AbstractMatrix}, A::BlockDiagonalMap) = sparse(A)
function SparseArrays.sparse(A::BlockDiagonalMap)
    return blockdiag(convert.(SparseMatrixCSC, A.maps)...)
end

# KroneckerMap & KroneckerSumMap
Base.Matrix(A::KroneckerMap) = kron(convert.(Matrix, A.maps)...)
Base.convert(::Type{AbstractMatrix}, A::KroneckerMap) = kron(convert.(AbstractMatrix, A.maps)...)
function SparseArrays.sparse(A::KroneckerMap)
    return kron(
        convert(SparseMatrixCSC, first(A.maps)),
        convert.(AbstractMatrix, Base.tail(A.maps))...
    )
end
function Base.Matrix(L::KroneckerSumMap)
    A, B = L.maps
    IA = Diagonal(ones(Bool, size(A, 1)))
    IB = Diagonal(ones(Bool, size(B, 1)))
    return kron(Matrix(A), IB) + kron(IA, Matrix(B))
end
function Base.convert(::Type{AbstractMatrix}, L::KroneckerSumMap)
    A, B = L.maps
    IA = Diagonal(ones(Bool, size(A, 1)))
    IB = Diagonal(ones(Bool, size(B, 1)))
    return kron(convert(AbstractMatrix, A), IB) + kron(IA, convert(AbstractMatrix, B))
end
function SparseArrays.sparse(L::KroneckerSumMap)
    A, B = L.maps
    IA = sparse(Diagonal(ones(Bool, size(A, 1))))
    IB = sparse(Diagonal(ones(Bool, size(B, 1))))
    return kron(convert(AbstractMatrix, A), IB) + kron(IA, convert(AbstractMatrix, B))
end
