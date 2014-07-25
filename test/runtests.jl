using LinearMaps
using Base.Test

A=2*rand(Complex128,(20,10)).-1
v=rand(Complex128,10)
w=rand(Complex128,20)

# test wrapped map for matrix
M=LinearMap(A)
@test M*v==A*v

# test transposition and full
@test M'*w==A'*w
@test M.'*w==A.'*w

@test full(M)==A
@test full(M')==A'
@test full(M.')==A.'

# test function map
F=LinearMap(cumsum,2)
@test full(F)==[1. 0.;1. 1.]

N=100
F=LinearMap(fft,N;isreal=false)/sqrt(N)
U=full(F) # will be a unitary matrix
@test_approx_eq U'*U eye(N)

F=LinearMap(cumsum,10)
@test F*v==cumsum(v)
@test_throws ErrorException F'*v

# test linear combinations
A=2*rand(Complex128,(10,10)).-1
M=LinearMap(A)
v=rand(Complex128,10)

@test full(3*M)==3*A
@test full(M+A)==2*A
@test full(-M)==-A
@test full(3*M'-F)==3*A'-full(F)
@test (3*M-1im*F)'==3*M'+1im*F'

@test_approx_eq (2*M'+3*I)*v (2*A'+3*I)*v

# test composition
@test (F*F)*v==F*(F*v)
@test (F*A)*v==F*(A*v)
@test_approx_eq full(M*M.') A*A.'
@test !isposdef(M*M.')
@test isposdef(M*M')
@test isposdef(F.'*F)
@test isposdef((M*F)'*M*F)
@test (M*F).'==F.'*M.'

L=3*F+1im*A+F*M'*F
LF=3*full(F)+1im*A+full(F)*full(M)'*full(F)
@test_approx_eq full(L) LF

# test inplace operations
w=similar(v)
Base.A_mul_B!(w,L,v)
@test_approx_eq w LF*v

# test new type
type SimpleFunctionMap <: AbstractLinearMap{Float64}
    f::Function
    N::Int
end

Base.size(A::SimpleFunctionMap)=(A.N,A.N)
Base.issym(A::SimpleFunctionMap)=false
*(A::SimpleFunctionMap,v::Vector)=A.f(v)

F=SimpleFunctionMap(cumsum,10)
@test size(F,1)==10
w=similar(v)
Base.A_mul_B!(w,F,v)
@test w==F*v
@test_throws MethodError F'*v
@test_throws MethodError F.'*v
@test_throws MethodError Ac_mul_B!(w,F,v)
@test_throws MethodError At_mul_B!(w,F,v)