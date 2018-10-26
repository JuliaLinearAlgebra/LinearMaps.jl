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
V = rand(ComplexF64, 10, 3)
W = rand(ComplexF64, 20, 3)
α = rand()
β = rand()

# test wrapped map for matrix
M = LinearMap(A)
N = LinearMap(M)
Av = A * v
AV = A * V
@test M * v == Av
@test N * v == Av
@test mul!(copy(w), M, v) == Av
@test mul!(copy(w), N, v) == Av
@test mul!(copy(W), M, V) ≈ AV
@test typeof(M * V) <: LinearMap

# test of mul!
@test mul!(copy(w), M, v, 0, 0) == zero(w)
@test mul!(copy(w), M, v, 0, 1) == w
@test mul!(copy(w), M, v, 0, β) == β * w
@test mul!(copy(w), M, v, 1, 1) ≈ Av + w
@test mul!(copy(w), M, v, 1, β) ≈ Av + β * w
@test mul!(copy(w), M, v, α, 1) ≈ α * Av + w
@test mul!(copy(w), M, v, α, β) ≈ α * Av + β * w
@test mul!(copy(w), M, v, α)    ≈ α * Av

# test matrix-mul!
@test mul!(copy(W), M, V, α, β) ≈ α * AV + β * W
@test mul!(copy(W), M, V, α) ≈ α * AV

# test transposition and Matrix
@test M' * w == A' * w
@test mul!(copy(V), adjoint(M), W) ≈ A' * W
@test transpose(M) * w == transpose(A) * w
@test transpose(M') * v ≈ transpose(A') * v
@test mul!(copy(V), transpose(M), W) ≈ transpose(A) * W

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
B[rand(1:length(A), 30)] .= 0
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
B = rand(size(A)...)
M = LinearMap(A)
N = LinearMap(B)
v = rand(ComplexF64, 10)

@test Matrix(3 * M) == 3 * A
@test Array(M + A) == 2 * A
@test Matrix(-M) == -A
@test Array(3 * M' - F) == 3 * A' - Array(F)
@test (3 * M - 1im * F)' == 3 * M' + 1im * F'

@test (2 * M' + 3 * I) * v ≈ (2 * A' + 3 * I) * v
@test transpose(LinearMap(2 * M' + 3 * I)) * v ≈ transpose(2 * A' + 3 * I) * v
@test LinearMap(2 * M' + 3 * I)' * v ≈ (2 * A' + 3 * I)' * v

# test composition
@test (F * F) * v == F * (F * v)
@test (F * A) * v == F * (A * v)
@test Matrix(M * transpose(M)) ≈ A * transpose(A)
@test !isposdef(M * transpose(M))
@test isposdef(M * M')
@test issymmetric(N * N')
@test ishermitian(N * N')
@test !issymmetric(M' * M)
@test ishermitian(M' * M)
@test isposdef(transpose(F) * F)
@test isposdef((M * F)' * M * F)
@test transpose(M * F) == transpose(F) * transpose(M)
L = 3 * F + 1im * A + F * M' * F
LF = 3 * Matrix(F) + 1im * A + Matrix(F) * Matrix(M)' * Matrix(F)
@test Array(L) ≈ LF
R1 = rand(ComplexF64, 10, 10)
R2 = rand(ComplexF64, 10, 10)
R3 = rand(ComplexF64, 10, 10)
CompositeR = prod(R -> LinearMap(R), [R1, R2, R3])
Lt = transpose(LinearMap(CompositeR))
@test Lt * v ≈ transpose(R3) * transpose(R2) * transpose(R1) * v
Lc = adjoint(LinearMap(CompositeR))
@test Lc * v ≈ R3' * R2' * R1' * v

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
            return conj.($i*dest)
        end
        insert!(Lf, 1, LinearMap($fi, prod(sizes[$i+1]), prod(sizes[$i])))
        insert!(Lt, 1, LinearMap(x -> x, $fi, prod(sizes[$i]), prod(sizes[$i+1])))
        insert!(Lc, 1, LinearMap{ComplexF64}(x -> x, $fi, prod(sizes[$i]), prod(sizes[$i+1])))
    end
end

# multiply as composition and as recursion
v1 = ones(prod(sizes[1]))
u1 = ones(prod(sizes[1]))
w1 = im.*ones(ComplexF64, prod(sizes[1]))
for i = N:-1:1
    v2 = prod(Lf[i:N])*ones(prod(sizes[1]))
    u2 = transpose(prod(Lt[N:-1:i]))*ones(prod(sizes[1]))
    w2 = adjoint(prod(Lc[N:-1:i]))*ones(prod(sizes[1]))

    global v1 = Lf[i]*v1
    global u1 = transpose(Lt[i])*u1
    global w1 = adjoint(Lc[i])*w1

    @test v1 == v2
    @test u1 == u2
    @test w1 == w2
end
