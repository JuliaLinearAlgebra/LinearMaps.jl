function rrule(::typeof(*), A::LinearMap, x::AbstractVector)
    y = A*x
    function pullback(dy)
        DY = unthunk(dy)
        return NoTangent(), @not_implemented("Gradient with respect to linear map itself not implemented."), @thunk(A' * DY)
    end
    return y, pullback
end

function rrule(A::LinearMap, x::AbstractVector)
    y = A*x
    function pullback(dy)
        DY = unthunk(dy)
        return @not_implemented("Gradient with respect to linear map itself not implemented."), @thunk(A' * DY)
    end
    return y, pullback
end
