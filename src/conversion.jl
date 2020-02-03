# Matrix: create matrix representation of LinearMap
function Base.Matrix(A::LinearMap)
    M, N = size(A)
    T = eltype(A)
    mat = Matrix{T}(undef, (M, N))
    v = fill(zero(T), N)
    @inbounds for i = 1:N
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

# special cases
Base.convert(::Type{AbstractMatrix}, A::WrappedMap) = convert(AbstractMatrix, A.lmap)
Base.convert(::Type{Matrix}, A::WrappedMap) = convert(Matrix, A.lmap)
function Base.convert(::Type{Matrix}, ΣA::LinearCombination{<:Any,<:Tuple{Vararg{MatrixMap}}})
    if length(ΣA.maps) <= 10
        return (+).(map(A->getfield(A, :lmap), ΣA.maps)...)
    else
        S = zero(first(ΣA.maps).lmap)
        for A in ΣA.maps
            S .+= A.lmap
        end
        return S
    end
end
function Base.convert(::Type{Matrix}, AB::CompositeMap{<:Any,<:Tuple{MatrixMap,MatrixMap}})
    B, A = AB.maps
    return A.lmap*B.lmap
end
function Base.convert(::Type{Matrix}, λA::CompositeMap{<:Any,<:Tuple{MatrixMap,UniformScalingMap}})
    A, J = λA.maps
    return J.λ*A.lmap
end
function Base.convert(::Type{Matrix}, Aλ::CompositeMap{<:Any,<:Tuple{UniformScalingMap,MatrixMap}})
    J, A = Aλ.maps
    return A.lmap*J.λ
end

Base.Matrix(A::BlockMap) = materialize!(Matrix{eltype(A)}(undef, size(A)), A)

# sparse: create sparse matrix representation of LinearMap
function SparseArrays.sparse(A::LinearMap{T}) where {T}
    M, N = size(A)
    rowind = Int[]
    nzval = T[]
    colptr = Vector{Int}(undef, N+1)
    v = fill(zero(T), N)
    Av = Vector{T}(undef, M)

    for i = 1:N
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
Base.convert(::Type{SparseMatrixCSC}, A::WrappedMap) = convert(SparseMatrixCSC, A.lmap)
Base.convert(::Type{SparseMatrixCSC}, A::BlockMap) = materialize!(spzeros(eltype(A), size(A)...), A)
