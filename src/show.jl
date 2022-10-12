# summary
Base.summary(io::IO, A::LinearMap) = print(io, map_summary(A))
function map_summary(A::LinearMap)
    Base.dims2string(size(A)) * ' ' * _show_typeof(A)
end

# show
Base.show(io::IO, A::LinearMap) = print(io, map_show(io, A, 0))

map_show(io::IO, A::LinearMap, i) = ' '^i * map_summary(A) * _show(io, A, i)
map_show(io::IO, A::AbstractVecOrMatOrQ, i) = ' '^i * summary(A)
_show(io::IO, ::LinearMap, _) = ""
function _show(io::IO, A::FunctionMap{T,F,Nothing}, _) where {T,F}
    "($(A.f); issymmetric=$(A._issymmetric), ishermitian=$(A._ishermitian), isposdef=$(A._isposdef))"
end
function _show(io::IO, A::FunctionMap, _)
    "($(A.f), $(A.fc); issymmetric=$(A._issymmetric), ishermitian=$(A._ishermitian), isposdef=$(A._isposdef))"
end
function _show(io::IO, A::Union{CompositeMap,LinearCombination,KroneckerMap,KroneckerSumMap}, i)
    n = length(A.maps)
    " with $n map" * (n>1 ? "s" : "") * ":\n" * print_maps(io, A.maps, i+2)
end
function _show(io::IO, A::Union{AdjointMap,TransposeMap,WrappedMap}, i)
    " of\n" * map_show(io, A.lmap, i+2)
end
function _show(io::IO, A::BlockMap, i)
    nrows = length(A.rows)
    n = length(A.maps)
    " with $n block map" * (n>1 ? "s" : "") *
        " in $nrows block row" * (nrows>1 ? "s" : "") * '\n' * print_maps(io, A.maps, i+2)
end
function _show(io::IO, A::BlockDiagonalMap, i)
    n = length(A.maps)
    " with $n diagonal block map" * (n>1 ? "s" : "") * ":\n" * print_maps(io, A.maps, i+2)
end
function _show(io::IO, J::UniformScalingMap, _)
    " with scaling factor: $(J.λ)"
end
function _show(io::IO, A::ScaledMap{T}, i) where {T}
    " with scale: $(A.λ) of\n" * map_show(io, A.lmap, i+2)
end
function _show(io::IO, A::FillMap{T}, _) where {T}
    " with fill value: $(A.λ)"
end

# helper functions
function _show_typeof(A::LinearMap{T}) where {T}
    split(string(typeof(A)), '{')[1] * '{' * string(T) * '}'
end
function _show_typeof(A::FunctionMap{T,<:Any,<:Any,iip}) where {T,iip}
    split(string(typeof(A)), '{')[1] * '{' * string(T) * ',' * string(iip) * '}'
end

function print_maps(io::IO, maps, k)
    n = length(maps)
    str = ""
    if get(io, :limit, true) && n > 10
        s = 1:5
        e = n-5:n
        if e[1] - s[end] > 1
            for i in s
                str *= map_show(io, maps[i], k) * '\n'
            end
            str *= ' '^k * '⋮'
            for i in e
                str *= '\n' * map_show(io, maps[i], k)
            end
        else
            for i in 1:n-1
                str *= map_show(io, maps[i], k) * '\n'
            end
            str *= map_show(io, last(maps), k)
        end
    else
        for i in 1:n-1
            str *= map_show(io, maps[i], k) * '\n'
        end
        str *= map_show(io, last(maps), k)
    end
    return str
end
