# summary
function Base.summary(io::IO, A::LinearMap)
    print(io, Base.dims2string(size(A)))
    print(io, ' ')
    _show_typeof(io, A)
end

# show
Base.show(io::IO, A::LinearMap) = (summary(io, A); _show(io, A))
_show(io::IO, ::LinearMap) = nothing
function _show(io::IO, A::FunctionMap{T,F,Nothing}) where {T,F}
    print(io, "($(A.f); ismutating=$(A._ismutating), issymmetric=$(A._issymmetric), ishermitian=$(A._ishermitian), isposdef=$(A._isposdef))")
end
function _show(io::IO, A::FunctionMap)
    print(io, "($(A.f), $(A.fc); ismutating=$(A._ismutating), issymmetric=$(A._issymmetric), ishermitian=$(A._ishermitian), isposdef=$(A._isposdef))")
end
function _show(io::IO, A::Union{CompositeMap,LinearCombination,KroneckerMap,KroneckerSumMap})
    n = length(A.maps)
    println(io, " with $n map", n>1 ? "s" : "", ":")
    print_maps(io, A.maps)
end
function _show(io::IO, A::Union{AdjointMap,TransposeMap,WrappedMap})
    print(io, " of ")
    L = A.lmap
    if A isa MatrixMap
        # summary(io, L)
        # println(io, ":")
        # Base.print_matrix(io, L)
        print(io, typeof(L))
    else
        show(io, L)
    end
end
function _show(io::IO, A::BlockMap)
    nrows = length(A.rows)
    n = length(A.maps)
    println(io, " with $n block map", n>1 ? "s" : "", " in $nrows block row", nrows>1 ? "s" : "")
    print_maps(io, A.maps)
end
function _show(io::IO, A::BlockDiagonalMap)
    n = length(A.maps)
    println(io, " with $n diagonal block map", n>1 ? "s" : "")
    print_maps(io, A.maps)
end
function _show(io::IO, J::UniformScalingMap)
    s = "$(J.λ)"
    print(io, " with scaling factor: $s")
end
function _show(io::IO, A::ScaledMap{T}) where {T}
    println(io, " with scale: $(A.λ) of")
    show(io, A.lmap)
end

# helper functions
function _show_typeof(io::IO, A::LinearMap{T}) where {T}
    Base.show_type_name(io, typeof(A).name)
    print(io, '{')
    show(io, T)
    print(io, '}')
end

function print_maps(io::IO, maps::Tuple{Vararg{LinearMap}})
    n = length(maps)
    if get(io, :limit, true) && n > 10
        s = 1:5
        e = n-5:n
        if e[1] - s[end] > 1
            for i in s
                # print(io, ' ')
                show(io, maps[i])
                println(io, "")
            end
            print(io, "⋮")
            for i in e
                println(io, "")
                show(io, maps[i])
            end
        else
            for i in 1:n-1
                # print(io, ' ')
                show(io, maps[i])
                println(io, "")
            end
            # print(io, ' ')
            show(io, last(maps))
        end
    else
        for i in 1:n-1
            # print(io, ' ')
            show(io, maps[i])
            println(io, "")
        end
        # print(io, ' ')
        show(io, last(maps))
    end
end
