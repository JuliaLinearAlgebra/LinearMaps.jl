function tr(A::LinearMap)
    _issquare(A) || throw(ArgumentError("operator needs to be square in tr"))
    _tr(A)
end

function _tr(A::LinearMap{T}) where {T}
    S = typeof(oneunit(eltype(A)) + oneunit(eltype(A)))
    ax1, ax2 = axes(A)
    xi = zeros(eltype(A), ax2)
    y = similar(xi, T, ax1)
    o = one(T)
    z = zero(T)
    s = zero(S)
    @inbounds for (i, j) in zip(ax1, ax2)
        xi[j] = o
        mul!(y, A, xi)
        xi[j] = z
        s += y[i]
    end
    return s
end
function _tr(A::OOPFunctionMap{T}) where {T}
    S = typeof(oneunit(eltype(A)) + oneunit(eltype(A)))
    ax1, ax2 = axes(A)
    xi = zeros(eltype(A), ax2)
    o = one(T)
    z = zero(T)
    s = zero(S)
    @inbounds for (i, j) in zip(ax1, ax2)
        xi[j] = o
        s += (A * xi)[i]
        xi[j] = z
    end
    return s
end
# specialiations
_tr(A::AbstractVecOrMat) = tr(A)
_tr(A::WrappedMap) = _tr(A.lmap)
_tr(A::TransposeMap) = _tr(A.lmap)
_tr(A::AdjointMap) = conj(_tr(A.lmap))
_tr(A::UniformScalingMap) = A.M * A.λ
_tr(A::ScaledMap) = A.λ * _tr(A.lmap)
function _tr(L::KroneckerMap)
    if all(_issquare, L.maps)
        return prod(_tr, L.maps)
    else
        return invoke(_tr, Tuple{LinearMap}, L)
    end
end
function _tr(L::OuterProductMap{<:RealOrComplex})
    a, bt = L.maps
    return bt.lmap*a.lmap
end
function _tr(L::OuterProductMap)
    a, bt = L.maps
    mapreduce(*, +, a.lmap, bt.lmap)
end
function _tr(L::KroneckerSumMap)
    A, B = L.maps # A and B are square by construction
    return _tr(A) * size(B, 1) + _tr(B) * size(A, 1)
end
_tr(A::FillMap) = A.size[1] * A.λ
