type LinearCombination{T}<:LinearMap{T}
    maps::Vector{LinearMap}
    coeffs::Vector{T}
    function LinearCombination(maps::Vector{LinearMap},coeffs::Vector{T})
        N=length(maps)
        N==length(coeffs) || error("Number of coefficients doesn't match number of terms")
        sz=size(maps[1])
        for n=1:N
            size(maps[n])==sz || throw(DimensionMismatch("LinearCombination"))
            promote_type(T,eltype(maps[n]))==T || throw(InexactError())
        end
        new(maps,coeffs)
    end
end

# basic methods
Base.size(A::LinearCombination,n)=size(A.maps[1],n)
Base.size(A::LinearCombination)=size(A.maps[1])
Base.isreal(A::LinearCombination)=all(isreal,A.maps) && all(isreal,A.coeffs) # sufficient but not necessary
Base.issym(A::LinearCombination)=all(issym,A.maps) # sufficient but not necessary
Base.ishermitian(A::LinearCombination)=all(ishermitian,A.maps) && all(isreal,A.coeffs) # sufficient but not necessary
Base.isposdef(A::LinearCombination)=all(isposdef,A.maps) && all(isposdef,A.coeffs) # sufficient but not necessary

# adding linear maps
function +(A1::LinearCombination,A2::LinearCombination)
    size(A1)==size(A2) || throw(DimensionMismatch("+"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{T}(LinearMap[A1.maps...,A2.maps...],T[A1.coeffs...,A2.coeffs...])
end
function +(A1::LinearMap,A2::LinearCombination)
    size(A1)==size(A2) || throw(DimensionMismatch("+"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{T}(LinearMap[A1,A2.maps...],T[one(T),A2.coeffs...])
end
+(A1::LinearCombination,A2::LinearMap)=+(A2,A1)
function +(A1::LinearMap,A2::LinearMap)
    size(A1)==size(A2) || throw(DimensionMismatch("+"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{T}(LinearMap[A1,A2],T[one(T),one(T)])
end
function -(A1::LinearCombination,A2::LinearCombination)
    size(A1)==size(A2) || throw(DimensionMismatch("-"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{T}(LinearMap[A1.maps...,A2.maps...],T[A1.coeffs...,map(-,A2.coeffs)...])
end
function -(A1::LinearMap,A2::LinearCombination)
    size(A1)==size(A2) || throw(DimensionMismatch("-"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{T}(LinearMap[A1,A2.maps...],T[one(T),map(-,A2.coeffs)...])
end
function -(A1::LinearCombination,A2::LinearMap)
    size(A1)==size(A2) || throw(DimensionMismatch("-"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{T}(LinearMap[A1.maps...,A2],T[A1.coeffs...,-one(T)])
end
function -(A1::LinearMap,A2::LinearMap)
    size(A1)==size(A2) || throw(DimensionMismatch("-"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{T}(LinearMap[A1,A2],T[one(T),-one(T)])
end

# scalar multiplication
-(A::LinearMap)=LinearCombination{eltype(A)}(LinearMap[A],[-one(eltype(A))])
-(A::LinearCombination)=LinearCombination{eltype(A)}(A.maps,map(-,A.coeffs))

function *(alpha::Number,A::LinearMap)
    T=promote_type(eltype(alpha),eltype(A))
    return LinearCombination{T}(LinearMap[A],T[alpha])
end
*(A::LinearMap,alpha::Number)=*(alpha,A)
function *(alpha::Number,A::LinearCombination)
    T=promote_type(eltype(alpha),eltype(A))
    return LinearCombination{T}(A.maps,alpha*A.coeffs)
end
*(A::LinearCombination,alpha::Number)=*(alpha,A)

function \(alpha::Number,A::LinearMap)
    T=promote_type(eltype(alpha),eltype(A))
    return LinearCombination{T}(LinearMap[A],T[one(T)/alpha])
end
/(A::LinearMap,alpha::Number)=\(alpha,A)
function \(alpha::Number,A::LinearCombination)
    T=promote_type(eltype(alpha),eltype(A))
    return LinearCombination{T}(A.maps,A.coeffs/alpha)
end
/(A::LinearCombination,alpha::Number)=\(alpha,A)

# comparison of LinearCombination objects
==(A::LinearCombination,B::LinearCombination)=(eltype(A)==eltype(B) && A.maps==B.maps && A.coeffs==B.coeffs)

# special transposition behavior
transpose(A::LinearCombination)=LinearCombination{eltype(A)}(LinearMap[transpose(l) for l in A.maps],A.coeffs)
ctranspose(A::LinearCombination)=LinearCombination{eltype(A)}(LinearMap[ctranspose(l) for l in A.maps],conj(A.coeffs))

# multiplication with vectors
function Base.A_mul_B!(y::AbstractVector,A::LinearCombination,x::AbstractVector)
    # no size checking, will be done by individual maps
    Base.A_mul_B!(y,A.maps[1],x)
    scale!(A.coeffs[1],y)
    z=similar(y)
    for n=2:length(A.maps)
        Base.A_mul_B!(z,A.maps[n],x)
        Base.axpy!(A.coeffs[n],z,y)
    end
    return y
end
function Base.At_mul_B!(y::AbstractVector,A::LinearCombination,x::AbstractVector)
    # no size checking, will be done by individual maps
    Base.At_mul_B!(y,A.maps[1],x)
    scale!(A.coeffs[1],y)
    z=similar(y)
    for n=2:length(A.maps)
        Base.At_mul_B!(z,A.maps[n],x)
        Base.axpy!(A.coeffs[n],z,y)
    end
    return y
end
function Base.Ac_mul_B!(y::AbstractVector,A::LinearCombination,x::AbstractVector)
    # no size checking, will be done by individual maps
    Base.Ac_mul_B!(y,A.maps[1],x)
    scale!(conj(A.coeffs[1]),y)
    z=similar(y)
    for n=2:length(A.maps)
        Base.Ac_mul_B!(z,A.maps[n],x)
        Base.axpy!(conj(A.coeffs[n]),z,y)
    end
    return y
end
