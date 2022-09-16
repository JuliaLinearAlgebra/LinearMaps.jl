function rrule(::typeof(*), A::LinearMap, x::AbstractVector)
    y = A*x
    function pullback(dy)
        DY = unthunk(dy)
        # Because A is an abstract map, the product is only differentiable w.r.t the input
        return NoTangent(), NoTangent(), @thunk(A' * DY)
    end
    return y, pullback
end

function rrule(A::LinearMap, x::AbstractVector)
    y = A*x
    function pullback(dy)
        DY = unthunk(dy)
        # Because A is an abstract map, the product is only differentiable w.r.t the input
        return NoTangent(), @thunk(A' * DY)
    end
    return y, pullback
end
