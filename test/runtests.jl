using Test
using LinearMaps
using SparseArrays
using LinearAlgebra

import Base: *
import LinearAlgebra: issymmetric, mul!

function myft(v::AbstractVector)
    # not so fast fourier transform
    N = length(v)
    w = zeros(complex(eltype(v)), N)
    for k = 1:N
        kappa = (2*(k-1)/N)*pi
        for n = 1:N
            w[k] += v[n]*exp(kappa*(n-1)*im)
        end
    end
    return w
end

A = 2 * rand(ComplexF64, (20, 10)) .- 1
v = rand(ComplexF64, 10)
w = rand(ComplexF64, 20)
wdest = copy(w)
V = rand(ComplexF64, 10, 3)
W = rand(ComplexF64, 20, 3)
Wdest = copy(W)
α = rand()
β = rand()

# test wrapped map for matrix
M = LinearMap(A)
@test M * v == A * v
mul!(Wdest, M, V)
@test Wdest ≈ A * V
@test typeof(M * V) <: LinearMap

# test of mul!
mul!(wdest, M, v, 0, 0)
@test wdest == zero(w)
wdest = copy(w)
mul!(wdest, M, v, 0, 1)
@test wdest == w
wdest = copy(w)
mul!(wdest, M, v, 0, β)
@test wdest == β * w
wdest = copy(w)
mul!(wdest, M, v, 1, 1)
@test wdest ≈ A * v + w
wdest = copy(w)
mul!(wdest, M, v, 1, β)
@test wdest ≈ A * v + β * w
wdest = copy(w)
mul!(wdest, M, v, α, 1)
@test wdest ≈ α * A * v + w
wdest = copy(w)
mul!(wdest, M, v, α, β)
@test wdest ≈ α * A * v + β * w
wdest = copy(w)
mul!(wdest, M, v, α)
@test wdest ≈ α * A * v

# test matrix-mul!
Wdest = copy(W)
mul!(Wdest, M, V, α, β)
@test Wdest ≈ α * A * V + β * W
Wdest = copy(W)
mul!(Wdest, M, V, α)
@test Wdest ≈ α * A * V

# test transposition and Matrix
@test M' * w == A' * w
mul!(V, adjoint(M), W)
@test V ≈ A' * W

@test transpose(M) * w == transpose(A) * w
mul!(V, transpose(M), W)
@test V ≈ transpose(A) * W

@test Matrix(M) == A
@test Array(M) == A
@test convert(Matrix, M) == A
@test convert(Array, M) == A

@test Matrix(M') == A'
@test Matrix(transpose(M)) == copy(transpose(A))

B = LinearMap(Symmetric(rand(10, 10)))
@test transpose(B) == B
@test B == transpose(B)

B = LinearMap(Hermitian(rand(ComplexF64, 10, 10)))
@test adjoint(B) == B
@test B == B'

# test sparse conversions
@test sparse(M) == sparse(Array(M))
@test convert(SparseMatrixCSC, M) == sparse(Array(M))

B = copy(A)
B[rand(1:length(A), 30)] .= 0.
MS = LinearMap(B)
@test sparse(MS) == sparse(Array(MS))

# test function map
F = LinearMap(cumsum, 2)
@test Matrix(F) == [1. 0.; 1. 1.]
@test Array(F) == [1. 0.; 1. 1.]

N = 100
F = LinearMap{ComplexF64}(myft, N) / sqrt(N)
U = Matrix(F) # will be a unitary matrix
@test U'U ≈ Matrix{eltype(U)}(I, N, N)

F = LinearMap(cumsum, 10; ismutating=false)
@test F * v == cumsum(v)
@test *(F, v) == cumsum(v)
@test_throws ErrorException F' * v

F = LinearMap((y,x) -> y .= cumsum(x), 10; ismutating=true)
@test F * v == cumsum(v)
@test *(F, v) == cumsum(v)
@test_throws ErrorException F'v

# Test fallback methods:
L = LinearMap(x -> x, x-> x, 10)
v = randn(10);
@test (2*L)' * v ≈ 2 * v

# test linear combinations
A = 2 * rand(ComplexF64, (10, 10)) .- 1
M = LinearMap(A)
v = rand(ComplexF64, 10)

@test Matrix(3 * M) == 3 * A
@test Array(M + A) == 2 * A
@test Matrix(-M) == -A
@test Array(3 * M' - F) == 3 * A' - Array(F)
@test (3 * M - 1im * F)' == 3 * M' + 1im * F'

@test (2 * M' + 3 * I) * v ≈ (2 * A' + 3 * I) * v

# test composition
@test (F * F) * v == F * (F * v)
@test (F * A) * v == F * (A * v)
@test Matrix(M * transpose(M)) ≈ A * transpose(A)
@test !isposdef(M * transpose(M))
@test isposdef(M * M')
@test isposdef(transpose(F) * F)
@test isposdef((M * F)' * M * F)
@test transpose(M * F) == transpose(F) * transpose(M)

L = 3 * F + 1im * A + F * M' * F
LF = 3 * Matrix(F) + 1im * A + Matrix(F) * Matrix(M)' * Matrix(F)
@test Array(L) ≈ LF

# test inplace operations
w = similar(v)
mul!(w, L, v)
@test w ≈ LF * v

# test new type
struct SimpleFunctionMap <: LinearMap{Float64}
    f::Function
    N::Int
end

Base.size(A::SimpleFunctionMap) = (A.N, A.N)
LinearAlgebra.issymmetric(A::SimpleFunctionMap) = false
*(A::SimpleFunctionMap, v::Vector) = A.f(v)
mul!(y::Vector, A::SimpleFunctionMap, x::Vector) = copyto!(y, *(A, x))

F = SimpleFunctionMap(cumsum, 10)
@test ndims(F) == 2
@test size(F, 1) == 10
@test length(F) == 100
w = similar(v)
mul!(w, F, v)
@test w == F * v
@test_throws MethodError F' * v
@test_throws MethodError transpose(F) * v
@test_throws MethodError mul!(w, adjoint(F), v)
@test_throws MethodError mul!(w, transpose(F), v)

# test composition of several maps
sizes = ( (5, 2), (3, 3), (3, 2), (2, 2), (9, 2), (7, 1) )
N = length(sizes)-1
Lf = []
Lt = []
Lc = []

# build list of operators [LN, ..., L2, L1] for each mode
for (fi, i) in [ (Symbol("f$i"), i) for i in 1:N]
    @eval begin
        function ($fi)(source)
            dest = ones(prod(sizes[$i+1]))
            tmp = reshape(source, sizes[$i])
            return $i*dest
        end
        insert!(Lf, 1, LinearMap($fi, prod(sizes[$i+1]), prod(sizes[$i])))
        insert!(Lt, 1, transpose(LinearMap(x -> x, $fi, prod(sizes[$i]), prod(sizes[$i+1]))))
        insert!(Lc, 1, adjoint(LinearMap{ComplexF64}(x -> x, $fi, prod(sizes[$i]), prod(sizes[$i+1]))))
    end
end

# multiply as composition and as recursion
v1 = ones(prod(sizes[1]))
u1 = ones(prod(sizes[1]))
w1 = ones(ComplexF64, prod(sizes[1]))
for i = N:-1:1
    v2 = prod(Lf[i:N])*ones(prod(sizes[1]))
    u2 = prod(Lt[i:N])*ones(prod(sizes[1]))
    w2 = prod(Lc[i:N])*ones(prod(sizes[1]))

    global v1 = Lf[i]*v1
    global u1 = Lt[i]*u1
    global w1 = Lc[i]*w1

    @test v1 == v2
    @test u1 == u2
    @test w1 == w2
end
