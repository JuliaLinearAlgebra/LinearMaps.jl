using Test
using LinearMaps
using SparseArrays
using LinearAlgebra

# adopted from: https://discourse.julialang.org/t/way-to-return-the-number-of-allocations/5167/10
macro numalloc(expr)
    return quote
        let
            local f
            function f()
                n1 = Base.gc_num()
                $(expr)
                n2 = Base.gc_num()
                diff = Base.GC_Diff(n2, n1)
                Base.gc_alloc_count(diff)
            end
            f()
        end
    end
end

import Base: *
import LinearAlgebra: issymmetric, mul!

A = 2 * rand(ComplexF64, (20, 10)) .- 1
v = rand(ComplexF64, 10)
w = rand(ComplexF64, 20)
V = rand(ComplexF64, 10, 3)
W = rand(ComplexF64, 20, 3)
α = rand()
β = rand()
M = LinearMap(A)
N = LinearMap(M)

@testset "LinearMaps.jl" begin
    @test eltype(M) == eltype(A)
    @test size(M) == size(A)
    @test size(N) == size(A)
    @test !isreal(M)
    @test ndims(M) == 2
    @test_throws ErrorException size(M, 3)
    @test length(M) == length(A)
    # matrix generation/conversion
    @test Matrix(M) == A
    @test Array(M) == A
    @test convert(Matrix, M) == A
    @test convert(Array, M) == A
    @test Matrix(M') == A'
    @test Matrix(transpose(M)) == copy(transpose(A))
    # sparse matrix generation/conversion
    @test sparse(M) == sparse(Array(M))
    @test convert(SparseMatrixCSC, M) == sparse(Array(M))

    B = copy(A)
    B[rand(1:length(A), 30)] .= 0
    MS = LinearMap(B)
    @test sparse(MS) == sparse(Array(MS)) == sparse(B)
end

Av = A * v
AV = A * V
@testset "mul! and *" begin
    @test M * v == Av
    @test N * v == Av
    @test mul!(copy(w), M, v) == mul!(copy(w), A, v)
    @test ((@allocated mul!(w, M, v)) == 0)
    @test mul!(copy(w), N, v) == Av

    # mat-vec-mul
    @test mul!(copy(w), M, v, 0, 0) == zero(w)
    @test mul!(copy(w), M, v, 0, 1) == w
    @test mul!(copy(w), M, v, 0, β) == β * w
    @test mul!(copy(w), M, v, 1, 1) ≈ Av + w
    @test mul!(copy(w), M, v, 1, β) ≈ Av + β * w
    @test mul!(copy(w), M, v, α, 1) ≈ α * Av + w
    @test mul!(copy(w), M, v, α, β) ≈ α * Av + β * w
    @test mul!(copy(w), M, v, α)    ≈ α * Av

    # test mat-mat-mul!
    @test mul!(copy(W), M, V, α, β) ≈ α * AV + β * W
    @test mul!(copy(W), M, V, α) ≈ α * AV
    @test mul!(copy(W), M, V) ≈ AV
    @test typeof(M * V) <: LinearMap
end

@testset "transpose/adjoint" begin
    @test M' * w == A' * w
    @test mul!(copy(V), adjoint(M), W) ≈ A' * W
    @test transpose(M) * w == transpose(A) * w
    @test transpose(LinearMap(transpose(M))) * v == A * v
    @test LinearMap(M')' * v == A * v
    @test transpose(transpose(M)) == M
    @test (M')' == M
    Mherm = LinearMap(A'A)
    @test ishermitian(Mherm)
    @test !issymmetric(Mherm)
    @test !issymmetric(transpose(Mherm))
    @test ishermitian(transpose(Mherm))
    @test ishermitian(Mherm')
    @test isposdef(Mherm)
    @test isposdef(transpose(Mherm))
    @test isposdef(adjoint(Mherm))
    @test !(transpose(M) == adjoint(M))
    @test !(adjoint(M) == transpose(M))
    @test transpose(M') * v ≈ transpose(A') * v
    @test transpose(LinearMap(M')) * v ≈ transpose(A') * v
    @test LinearMap(transpose(M))' * v ≈ transpose(A)' * v
    @test transpose(LinearMap(transpose(M))) * v ≈ Av
    @test adjoint(LinearMap(adjoint(M))) * v ≈ Av

    @test mul!(copy(w), transpose(LinearMap(M')), v) ≈ transpose(A') * v
    @test mul!(copy(w), LinearMap(transpose(M))', v) ≈ transpose(A)' * v
    @test mul!(copy(w), transpose(LinearMap(transpose(M))), v) ≈ Av
    @test mul!(copy(w), adjoint(LinearMap(adjoint(M))), v) ≈ Av
    @test mul!(copy(V), transpose(M), W) ≈ transpose(A) * W
    @test mul!(copy(V), adjoint(M), W) ≈ A' * W

    B = LinearMap(Symmetric(rand(10, 10)))
    @test transpose(B) == B
    @test B == transpose(B)

    B = LinearMap(Hermitian(rand(ComplexF64, 10, 10)))
    @test adjoint(B) == B
    @test B == B'
end

@testset "function maps" begin
    N = 100
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
    MyFT = LinearMap{ComplexF64}(myft, N) / sqrt(N)
    U = Matrix(MyFT) # will be a unitary matrix
    @test U'U ≈ Matrix{eltype(U)}(I, N, N)

    CS = LinearMap(cumsum, 2)
    @test size(CS) == (2, 2)
    @test !issymmetric(CS)
    @test !ishermitian(CS)
    @test !isposdef(CS)
    @test !(LinearMaps.ismutating(CS))
    @test Matrix(CS) == [1. 0.; 1. 1.]
    @test Array(CS) == [1. 0.; 1. 1.]
    CS = LinearMap(cumsum, 10; ismutating=false)
    v = rand(10)
    cv = cumsum(v)
    @test CS * v == cv
    @test *(CS, v) == cv
    @test_throws ErrorException CS' * v
    CS = LinearMap(cumsum, x -> cumsum(reverse(x)), 10; ismutating=false)
    cv = cumsum(v)
    @test CS * v == cv
    @test *(CS, v) == cv
    @test CS' * v == cumsum(reverse(v))
    @test mul!(similar(v), transpose(CS), v) == cumsum(reverse(v))

    CS! = LinearMap(cumsum!, 10; ismutating=true)
    @test LinearMaps.ismutating(CS!)
    @test CS! * v == cv
    @test *(CS!, v) == cv
    @test mul!(similar(v), CS!, v) == cv
    @test_throws ErrorException CS!'v
    @test_throws ErrorException transpose(CS!) * v

    CS! = LinearMap{ComplexF64}(cumsum!, 10; ismutating=true)
    v = rand(ComplexF64, 10)
    cv = cumsum(v)
    @test LinearMaps.ismutating(CS!)
    @test CS! * v == cv
    @test *(CS!, v) == cv
    @test mul!(similar(v), CS!, v) == cv
    @test_throws ErrorException CS!'v
    @test_throws ErrorException adjoint(CS!) * v
    CS! = LinearMap{ComplexF64}(cumsum!, x -> cumsum!(reverse!(x)), 10; ismutating=true)
    @test LinearMaps.ismutating(CS!)
    @test CS! * v == cv
    @test *(CS!, v) == cv
    @test mul!(similar(v), CS!, v) == cv
    @test CS' * v == cumsum(reverse(v))
    @test mul!(similar(v), transpose(CS), v) == cumsum(reverse(v))
    @test mul!(similar(v), adjoint(CS), v) == cumsum(reverse(v))

    # Test fallback methods:
    L = LinearMap(x -> x, x -> x, 10)
    v = randn(10)
    @test (2 * L)' * v ≈ 2 * v
    @test transpose(2 * L) * v ≈ 2 * v
    L = LinearMap{ComplexF64}(x -> x, x -> x, 10)
    v = rand(ComplexF64, 10)
    @test (2 * L)' * v ≈ 2 * v
    @test transpose(2 * L) * v ≈ 2 * v
end

CS! = LinearMap(cumsum!, 10; ismutating=true)
v = rand(10)
u = similar(v)
mul!(u, CS!, v)
@test ((@allocated mul!(u, CS!, v)) == 0)
n = 10
L = sum(fill(CS!, n))
@test mul!(u, L, v) ≈ n * cumsum(v)
@test ((@numalloc mul!(u, L, v)) <= 1)

A = 2 * rand(ComplexF64, (10, 10)) .- 1
B = rand(size(A)...)
M = LinearMap(A)
N = LinearMap(B)
LC = M + N
v = rand(ComplexF64, 10)
w = similar(v)
mul!(w, M, v)
@test ((@allocated mul!(w, M, v)) == 0)
@testset "linear combinations" begin
    # @test_throws ErrorException LinearMaps.LinearCombination{ComplexF64}((M, N), (1, 2, 3))
    @test size(3M + 2.0N) == size(A)
    # addition
    @test convert(Matrix, LC) == A + B
    @test convert(Matrix, LC + LC) ≈ 2A + 2B
    @test convert(Matrix, M + LC) ≈ 2A + B
    @test convert(Matrix, M + M) ≈ 2A
    # subtraction
    @test Matrix(LC - LC) == zeros(size(LC))
    @test Matrix(LC - M) == B
    @test Matrix(N - LC) == -A
    @test Matrix(M - M) == zeros(size(M))
    # scalar multiplication
    @test Matrix(-M) == -A
    @test Matrix(-LC) == -A - B
    @test Matrix(3 * M) == 3 * A
    @test Matrix(M * 3) == 3 * A
    @test Matrix(3.0 * LC) ≈ Matrix(LC * 3) == 3A + 3B
    @test Matrix(3 \ M) ≈ A/3
    @test Matrix(M / 3) ≈ A/3
    @test Matrix(3 \ LC) ≈ (A + B) / 3
    @test Matrix(LC / 3) ≈ (A + B) / 3
    @test Array(3 * M' - CS!) == 3 * A' - Array(CS!)
    # @test (3 * M + (-1im) * CS!)' == 3 * M' + 1im * CS!'
    @test Array(M + A) == 2 * A
    @test Array(A + M) == 2 * A
    @test Array(M - A) == 0 * A
    @test Array(A - M) == 0 * A
end

# new type
struct SimpleFunctionMap <: LinearMap{Float64}
    f::Function
    N::Int
end
struct SimpleComplexFunctionMap <: LinearMap{Complex{Float64}}
    f::Function
    N::Int
end
Base.size(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}) = (A.N, A.N)
*(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, v::Vector) = A.f(v)
mul!(y::Vector, A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, x::Vector) = copyto!(y, *(A, x))

@testset "composition" begin
    F = LinearMap(cumsum, 10; ismutating=false)
    A = 2 * rand(ComplexF64, (10, 10)) .- 1
    B = rand(size(A)...)
    M = 1 * LinearMap(A)
    N = LinearMap(B)
    @test (F * F) * v == F * (F * v)
    @test (F * A) * v == F * (A * v)
    @test (A * F) * v == A * (F * v)
    @test A * (F * F) * v == A * (F * (F * v))
    @test (F * F) * (F * F) * v == F * (F * (F * (F * v)))
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
    F = SimpleFunctionMap(cumsum, 10)
    FC = SimpleComplexFunctionMap(cumsum, 10)
    @test ndims(F) == 2
    @test size(F, 1) == 10
    @test length(F) == 100
    @test !issymmetric(F)
    @test !ishermitian(F)
    @test !ishermitian(FC)
    @test !isposdef(F)
    w = similar(v)
    mul!(w, F, v)
    @test w == F * v
    @test_throws MethodError F' * v
    @test_throws MethodError transpose(F) * v
    @test_throws MethodError mul!(w, adjoint(F), v)
    @test_throws MethodError mul!(w, transpose(F), v)

    # test composition of several maps with shared data #31
    global sizes = ( (5, 2), (3, 3), (3, 2), (2, 2), (9, 2), (7, 1) )
    N = length(sizes) - 1
    global Lf = []
    global Lt = []
    global Lc = []

    # build list of operators [LN, ..., L2, L1] for each mode
    for (fi, i) in [ (Symbol("f$i"), i) for i in 1:N]
        @eval begin
            function ($fi)(source)
                dest = ones(prod(sizes[$i + 1]))
                tmp = reshape(source, sizes[$i])
                return conj.($i * dest)
            end
            insert!(Lf, 1, LinearMap($fi, prod(sizes[$i + 1]), prod(sizes[$i])))
            insert!(Lt, 1, LinearMap(x -> x, $fi, prod(sizes[$i]), prod(sizes[$i + 1])))
            insert!(Lc, 1, LinearMap{ComplexF64}(x -> x, $fi, prod(sizes[$i]), prod(sizes[$i + 1])))
        end
    end
    @test size(prod(Lf[1:N])) == (prod(sizes[end]), prod(sizes[1]))
    @test isreal(prod(Lf[1:N]))
    # multiply as composition and as recursion
    v1 = ones(prod(sizes[1]))
    u1 = ones(prod(sizes[1]))
    w1 = im.*ones(ComplexF64, prod(sizes[1]))
    for i = N:-1:1
        v2 = prod(Lf[i:N]) * ones(prod(sizes[1]))
        u2 = transpose(LinearMap(prod(Lt[N:-1:i]))) * ones(prod(sizes[1]))
        w2 = adjoint(LinearMap(prod(Lc[N:-1:i]))) * ones(prod(sizes[1]))

        v1 = Lf[i] * v1
        u1 = transpose(Lt[i]) * u1
        w1 = adjoint(Lc[i]) * w1

        @test v1 == v2
        @test u1 == u2
        @test w1 == w2
    end
end

A = rand(10, 20)
B = rand(ComplexF64, 10, 20)
SA = A'A + I
SB = B'B + I
L = LinearMap{Float64}(A)
MA = LinearMap(SA)
MB = LinearMap(SB)
@testset "wrapped maps" begin
    @test size(L) == size(A)
    @test !issymmetric(L)
    @test issymmetric(MA)
    @test !issymmetric(MB)
    @test isposdef(MA)
    @test isposdef(MB)
end

A = 2 * rand(ComplexF64, (10, 10)) .- 1
B = rand(size(A)...)
M = 1 * LinearMap(A)
N = LinearMap(B)
LC = M + N
v = rand(ComplexF64, 10)
w = similar(v)
@testset "identity/scaling map" begin
    Id = LinearMaps.UniformScalingMap(1, 10)
    @test_throws ErrorException LinearMaps.UniformScalingMap(1, 10, 20)
    @test_throws ErrorException LinearMaps.UniformScalingMap(1, (10, 20))
    @test size(Id) == (10, 10)
    @test isreal(Id)
    @test issymmetric(Id)
    @test ishermitian(Id)
    @test isposdef(Id)
    @test Id * v == v
    @test (2 * M' + 3 * I) * v == 2 * A'v + 3v
    @test (3 * I + 2 * M') * v == 2 * A'v + 3v
    @test (2 * M' - 3 * I) * v == 2 * A'v - 3v
    @test (3 * I - 2 * M') * v == -2 * A'v + 3v
    @test transpose(LinearMap(2 * M' + 3 * I)) * v ≈ transpose(2 * A' + 3 * I) * v
    @test LinearMap(2 * M' + 3 * I)' * v ≈ (2 * A' + 3 * I)' * v
end
