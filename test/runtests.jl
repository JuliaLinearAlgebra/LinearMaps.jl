using Test
using LinearMaps
using SparseArrays
using LinearAlgebra
using BenchmarkTools

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
    @test @inferred mul!(copy(w), M, v) == mul!(copy(w), A, v)
    b = @benchmarkable mul!(w, M, v)
    @test run(b, samples=3).allocs == 0
    @test @inferred mul!(copy(w), N, v) == Av

    # mat-vec-mul
    @test @inferred mul!(copy(w), M, v, 0, 0) == zero(w)
    @test @inferred mul!(copy(w), M, v, 0, 1) == w
    @test @inferred mul!(copy(w), M, v, 0, β) == β * w
    @test @inferred mul!(copy(w), M, v, 1, 1) ≈ Av + w
    @test @inferred mul!(copy(w), M, v, 1, β) ≈ Av + β * w
    @test @inferred mul!(copy(w), M, v, α, 1) ≈ α * Av + w
    @test @inferred mul!(copy(w), M, v, α, β) ≈ α * Av + β * w
    @test @inferred mul!(copy(w), M, v, α)    ≈ α * Av

    # test mat-mat-mul!
    @test @inferred mul!(copy(W), M, V, α, β) ≈ α * AV + β * W
    @test @inferred mul!(copy(W), M, V, α) ≈ α * AV
    @test @inferred mul!(copy(W), M, V) ≈ AV
    @test typeof(M * V) <: LinearMap
end

@testset "transpose/adjoint" begin
    @test @inferred M' * w == A' * w
    @test @inferred mul!(copy(V), adjoint(M), W) ≈ A' * W
    @test @inferred transpose(M) * w == transpose(A) * w
    @test @inferred transpose(LinearMap(transpose(M))) * v == A * v
    @test @inferred LinearMap(M')' * v == A * v
    @test @inferred transpose(transpose(M)) == M
    @test (M')' == M
    Mherm = LinearMap(A'A)
    @test @inferred ishermitian(Mherm)
    @test @inferred !issymmetric(Mherm)
    @test @inferred !issymmetric(transpose(Mherm))
    @test @inferred ishermitian(transpose(Mherm))
    @test @inferred ishermitian(Mherm')
    @test @inferred isposdef(Mherm)
    @test @inferred isposdef(transpose(Mherm))
    @test @inferred isposdef(adjoint(Mherm))
    @test @inferred !(transpose(M) == adjoint(M))
    @test @inferred !(adjoint(M) == transpose(M))
    @test @inferred transpose(M') * v ≈ transpose(A') * v
    @test @inferred transpose(LinearMap(M')) * v ≈ transpose(A') * v
    @test @inferred LinearMap(transpose(M))' * v ≈ transpose(A)' * v
    @test @inferred transpose(LinearMap(transpose(M))) * v ≈ Av
    @test @inferred adjoint(LinearMap(adjoint(M))) * v ≈ Av

    @test @inferred mul!(copy(w), transpose(LinearMap(M')), v) ≈ transpose(A') * v
    @test @inferred mul!(copy(w), LinearMap(transpose(M))', v) ≈ transpose(A)' * v
    @test @inferred mul!(copy(w), transpose(LinearMap(transpose(M))), v) ≈ Av
    @test @inferred mul!(copy(w), adjoint(LinearMap(adjoint(M))), v) ≈ Av
    @test @inferred mul!(copy(V), transpose(M), W) ≈ transpose(A) * W
    @test @inferred mul!(copy(V), adjoint(M), W) ≈ A' * W

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
    @test @inferred U'U ≈ Matrix{eltype(U)}(I, N, N)

    CS = LinearMap(cumsum, 2)
    @test size(CS) == (2, 2)
    @test @inferred !issymmetric(CS)
    @test @inferred !ishermitian(CS)
    @test @inferred !isposdef(CS)
    @test @inferred !(LinearMaps.ismutating(CS))
    @test @inferred Matrix(CS) == [1. 0.; 1. 1.]
    @test @inferred Array(CS) == [1. 0.; 1. 1.]
    CS = LinearMap(cumsum, 10; ismutating=false)
    v = rand(10)
    cv = cumsum(v)
    @test CS * v == cv
    @test *(CS, v) == cv
    @test_throws ErrorException CS' * v
    CS = LinearMap(cumsum, x -> cumsum(reverse(x)), 10; ismutating=false)
    cv = cumsum(v)
    @test @inferred CS * v == cv
    @test @inferred *(CS, v) == cv
    @test @inferred CS' * v == cumsum(reverse(v))
    @test @inferred mul!(similar(v), transpose(CS), v) == cumsum(reverse(v))

    CS! = LinearMap(cumsum!, 10; ismutating=true)
    @test @inferred LinearMaps.ismutating(CS!)
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test_throws ErrorException CS!'v
    @test_throws ErrorException transpose(CS!) * v

    CS! = LinearMap{ComplexF64}(cumsum!, 10; ismutating=true)
    v = rand(ComplexF64, 10)
    cv = cumsum(v)
    @test @inferred LinearMaps.ismutating(CS!)
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test_throws ErrorException CS!'v
    @test_throws ErrorException adjoint(CS!) * v
    CS! = LinearMap{ComplexF64}(cumsum!, x -> cumsum!(reverse!(x)), 10; ismutating=true)
    @test @inferred LinearMaps.ismutating(CS!)
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test @inferred CS' * v == cumsum(reverse(v))
    @test @inferred mul!(similar(v), transpose(CS), v) == cumsum(reverse(v))
    @test @inferred mul!(similar(v), adjoint(CS), v) == cumsum(reverse(v))

    # Test fallback methods:
    L = LinearMap(x -> x, x -> x, 10)
    v = randn(10)
    @test @inferred (2 * L)' * v ≈ 2 * v
    @test @inferred transpose(2 * L) * v ≈ 2 * v
    L = LinearMap{ComplexF64}(x -> x, x -> x, 10)
    v = rand(ComplexF64, 10)
    @test @inferred (2 * L)' * v ≈ 2 * v
    @test @inferred transpose(2 * L) * v ≈ 2 * v
end

CS! = LinearMap(cumsum!, 10; ismutating=true)
v = rand(10)
u = similar(v)
b = @benchmarkable mul!(u, CS!, v)
@test run(b, samples=3).allocs == 0
n = 10
L = sum(fill(CS!, n))
@test mul!(u, L, v) ≈ n * cumsum(v)
b = @benchmarkable mul!(u, L, v)
@test run(b, samples=5).allocs <= 1

A = 2 * rand(ComplexF64, (10, 10)) .- 1
B = rand(size(A)...)
M = LinearMap(A)
N = LinearMap(B)
LC = M + N
v = rand(ComplexF64, 10)
w = similar(v)
b = @benchmarkable mul!(w, M, v)
@test run(b, samples=3).allocs == 0
@testset "linear combinations" begin
    # @test_throws ErrorException LinearMaps.LinearCombination{ComplexF64}((M, N), (1, 2, 3))
    @test @inferred size(3M + 2.0N) == size(A)
    # addition
    @test @inferred convert(Matrix, LC) == A + B
    @test @inferred convert(Matrix, LC + LC) ≈ 2A + 2B
    @test @inferred convert(Matrix, M + LC) ≈ 2A + B
    @test @inferred convert(Matrix, M + M) ≈ 2A
    # subtraction
    @test @inferred Matrix(LC - LC) == zeros(size(LC))
    @test @inferred Matrix(LC - M) == B
    @test @inferred Matrix(N - LC) == -A
    @test @inferred Matrix(M - M) == zeros(size(M))
    # scalar multiplication
    @test @inferred Matrix(-M) == -A
    @test @inferred Matrix(-LC) == -A - B
    @test @inferred Matrix(3 * M) == 3 * A
    @test @inferred Matrix(M * 3) == 3 * A
    @test Matrix(3.0 * LC) ≈ Matrix(LC * 3) == 3A + 3B
    @test @inferred Matrix(3 \ M) ≈ A/3
    @test @inferred Matrix(M / 3) ≈ A/3
    @test @inferred Matrix(3 \ LC) ≈ (A + B) / 3
    @test @inferred Matrix(LC / 3) ≈ (A + B) / 3
    @test @inferred Array(3 * M' - CS!) == 3 * A' - Array(CS!)
    @test (3M - 1im*CS!)' == (3M + ((-1im)*CS!))' == M'*3 + CS!'*1im
    @test @inferred Array(M + A) == 2 * A
    @test @inferred Array(A + M) == 2 * A
    @test @inferred Array(M - A) == 0 * A
    @test Array(A - M) == 0 * A
end

# new type
import Base: *, size
import LinearAlgebra: mul!

struct SimpleFunctionMap <: LinearMap{Float64}
    f::Function
    N::Int
end
struct SimpleComplexFunctionMap <: LinearMap{Complex{Float64}}
    f::Function
    N::Int
end
Base.size(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}) = (A.N, A.N)
Base.:(*)(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, v::Vector) = A.f(v)
mul!(y::Vector, A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, x::Vector) = copyto!(y, *(A, x))

@testset "composition" begin
    F = LinearMap(cumsum, 10; ismutating=false)
    A = 2 * rand(ComplexF64, (10, 10)) .- 1
    B = rand(size(A)...)
    M = 1 * LinearMap(A)
    N = LinearMap(B)
    @test @inferred (F * F) * v == @inferred F * (F * v)
    @test @inferred (F * A) * v == @inferred F * (A * v)
    @test @inferred (A * F) * v == @inferred A * (F * v)
    @test @inferred A * (F * F) * v == @inferred A * (F * (F * v))
    @test @inferred (F * F) * (F * F) * v == @inferred F * (F * (F * (F * v)))
    @test @inferred Matrix(M * transpose(M)) ≈ A * transpose(A)
    @test @inferred !isposdef(M * transpose(M))
    @test @inferred isposdef(M * M')
    @test @inferred issymmetric(N * N')
    @test @inferred ishermitian(N * N')
    @test @inferred !issymmetric(M' * M)
    @test @inferred ishermitian(M' * M)
    @test @inferred isposdef(transpose(F) * F)
    @test @inferred isposdef(transpose(F) * F * 3)
    @test @inferred isposdef(transpose(F) * 3 * F)
    @test @inferred !isposdef(-5*transpose(F) * F)
    @test @inferred isposdef((M * F)' * M * 4 * F)
    @test @inferred transpose(M * F) == @inferred transpose(F) * transpose(M)
    @test @inferred (4*((-3*M)*2)) == @inferred -12M*2
    @test @inferred (4*((3*(-M))*2)*(-5)) == @inferred -12M*(-10)
    L = 3 * F + 1im * A + F * M' * F
    LF = 3 * Matrix(F) + 1im * A + Matrix(F) * Matrix(M)' * Matrix(F)
    @test Array(L) ≈ LF
    R1 = rand(ComplexF64, 10, 10); L1 = LinearMap(R1)
    R2 = rand(ComplexF64, 10, 10); L2 = LinearMap(R2)
    R3 = rand(ComplexF64, 10, 10); L3 = LinearMap(R3)
    CompositeR = prod(R -> LinearMap(R), [R1, R2, R3])
    @test transpose(CompositeR) == transpose(L3) * transpose(L2) * transpose(L1)
    @test adjoint(CompositeR) == L3' * L2' * L1'
    @test adjoint(adjoint((CompositeR))) == CompositeR
    @test transpose(transpose((CompositeR))) == CompositeR
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
    @test @inferred ndims(F) == 2
    @test @inferred size(F, 1) == 10
    @test @inferred length(F) == 100
    @test @inferred !issymmetric(F)
    @test @inferred !ishermitian(F)
    @test @inferred !ishermitian(FC)
    @test @inferred !isposdef(F)
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
    @test @inferred !issymmetric(L)
    @test @inferred issymmetric(MA)
    @test @inferred !issymmetric(MB)
    @test @inferred isposdef(MA)
    @test @inferred isposdef(MB)
end

A = 2 * rand(ComplexF64, (10, 10)) .- 1
B = rand(size(A)...)
M = @inferred 1 * LinearMap(A)
N = LinearMap(B)
LC = M + N
v = rand(ComplexF64, 10)
w = similar(v)
@testset "identity/scaling map" begin
    Id = LinearMaps.UniformScalingMap(1, 10)
    @test_throws ErrorException LinearMaps.UniformScalingMap(1, 10, 20)
    @test_throws ErrorException LinearMaps.UniformScalingMap(1, (10, 20))
    @test size(Id) == (10, 10)
    @test @inferred isreal(Id)
    @test @inferred issymmetric(Id)
    @test @inferred ishermitian(Id)
    @test @inferred isposdef(Id)
    @test Id * v == v
    @test (2 * M' + 3 * I) * v == 2 * A'v + 3v
    @test (3 * I + 2 * M') * v == 2 * A'v + 3v
    @test (2 * M' - 3 * I) * v == 2 * A'v - 3v
    @test (3 * I - 2 * M') * v == -2 * A'v + 3v
    @test transpose(LinearMap(2 * M' + 3 * I)) * v ≈ transpose(2 * A' + 3 * I) * v
    @test LinearMap(2 * M' + 3 * I)' * v ≈ (2 * A' + 3 * I)' * v
end

@testset "noncommutative number type" begin
    using Quaternions
    x = Quaternion.(rand(10), rand(10), rand(10), rand(10))
    v = rand(10)
    A = Quaternion.(rand(10,10), rand(10,10), rand(10,10), rand(10,10))
    α = UniformScaling(Quaternion.(rand(4)...))
    β = UniformScaling(Quaternion.(rand(4)...))
    L = LinearMap(A)
    @test Array(L) == A
    @test Array(L') == A'
    @test Array(transpose(L)) == transpose(A)
    @test Array(α * L) == α * A
    @test Array(L * α) == A * α
    @test Array(α * L) == α * A
    @test Array(L * α ) == A * α
    @test Array(α * L') == α * A'
    @test Array((α * L')') ≈ (α * A')' ≈ A * conj(α)
    @test L * x ≈ A * x
    @test L' * x ≈ A' * x
    @test α * (L * x) ≈ α * (A * x)
    @test α * L * x ≈ α * A * x
    @test (α * L') * x ≈ (α * A') * x
    @test (α * L')' * x ≈ (α * A')' * x
    @test (α * L')' * v ≈ (α * A')' * v
    @test Array(@inferred adjoint(α * L * β)) ≈ conj(β) * A' * conj(α)
    @test Array(@inferred transpose(α * L * β)) ≈ β * transpose(A) * α
end

@testset "nonassociative number type" begin
    using Quaternions
    x = Octonion.(rand(10), rand(10), rand(10), rand(10),rand(10), rand(10), rand(10), rand(10))
    v = rand(10)
    A = Octonion.(rand(10,10), rand(10,10), rand(10,10), rand(10,10),rand(10,10), rand(10,10), rand(10,10), rand(10,10))
    α = UniformScaling(Octonion.(rand(8)...))
    β = UniformScaling(Octonion.(rand(8)...))
    L = LinearMap(A)
    @test Array(L) == A
    @test Array(α * L) == α * A
    @test Array(L * α) == A * α
    @test Array(α * L) == α * A
    @test Array(L * α) == A * α
    @test (α * L')' * v ≈ (α * A')' * v
end

@testset "block maps" begin
    @testset "hcat" begin
        for elty in (Float32, Float64, ComplexF64)
            A11 = rand(elty, 10, 10)
            A12 = rand(elty, 10, 20)
            L = @inferred hcat(LinearMap(A11), LinearMap(A12))
            @test L isa LinearMaps.BlockMap{elty}
            A = [A11 A12]
            x = rand(30)
            @test size(L) == size(A)
            @test Matrix(L) ≈ A
            @test L * x ≈ A * x
            A = [I I I A11 A11 A11]
            L = [I I I LinearMap(A11) LinearMap(A11) LinearMap(A11)]
            x = rand(elty, 60)
            @test L isa LinearMaps.BlockMap{elty}
            @test L * x ≈ A * x
            A11 = rand(elty, 11, 10)
            A12 = rand(elty, 10, 20)
            @test_throws DimensionMismatch hcat(LinearMap(A11), LinearMap(A12))
        end
    end
    @testset "vcat" begin
        for elty in (Float32, Float64, ComplexF64)
            A11 = rand(elty, 10, 10)
            A21 = rand(elty, 20, 10)
            L = @inferred vcat(LinearMap(A11), LinearMap(A21))
            @test L isa LinearMaps.BlockMap{elty}
            A = [A11; A21]
            x = rand(10)
            @test size(L) == size(A)
            @test Matrix(L) ≈ A
            @test L * x ≈ A * x
            A = [I; I; I; A11; A11; A11]
            L = [I; I; I; LinearMap(A11); LinearMap(A11); LinearMap(A11)]
            x = rand(elty, 10)
            @test L isa LinearMaps.BlockMap{elty}
            @test L * x ≈ A * x
            A11 = rand(elty, 10, 11)
            A21 = rand(elty, 20, 10)
            @test_throws DimensionMismatch vcat(LinearMap(A11), LinearMap(A21))
        end
    end
    @testset "hvcat" begin
        for elty in (Float32, Float64, ComplexF64)
            A11 = rand(elty, 10, 10)
            A12 = rand(elty, 10, 20)
            A21 = rand(elty, 20, 10)
            A22 = rand(elty, 20, 20)
            A = [A11 A12; A21 A22]
            @inferred hvcat((2,2), LinearMap(A11), LinearMap(A12), LinearMap(A21), LinearMap(A22))
            L = [LinearMap(A11) LinearMap(A12); LinearMap(A21) LinearMap(A22)]
            x = rand(30)
            @test L isa LinearMaps.BlockMap{elty}
            @test size(L) == size(A)
            @test L * x ≈ A * x
            @test Matrix(L) ≈ A
            A = [I A12; A21 I]
            @inferred hvcat((2,2), I, LinearMap(A12), LinearMap(A21), I)
            L = @inferred hvcat((2,2), I, LinearMap(A12), LinearMap(A21), I)
            @test L isa LinearMaps.BlockMap{elty}
            @test size(L) == (30, 30)
            @test Matrix(L) ≈ A
            @test L * x ≈ A * x
            A = rand(elty, 10,10); LA = LinearMap(A)
            B = rand(elty, 20,30); LB = LinearMap(B)
            @test [LA LA LA; LB] isa LinearMaps.BlockMap{elty}
            @test Matrix([LA LA LA; LB]) ≈ [A A A; B]
            @test [LB; LA LA LA] isa LinearMaps.BlockMap{elty}
            @test Matrix([LB; LA LA LA]) ≈ [B; A A A]
            @test [I; LA LA LA] isa LinearMaps.BlockMap{elty}
            @test Matrix([I; LA LA LA]) ≈ [I; A A A]
            A12 = rand(elty, 10, 21)
            A21 = rand(elty, 20, 10)
            @test_throws ArgumentError A = [I A12; A21 I]
            @test_throws ArgumentError A = [A12 A12; A21 A21]
        end
    end
    @testset "adjoint/transpose" begin
        for elty in (Float32, Float64, ComplexF64), transform in (transpose, adjoint)
            A12 = rand(elty, 10, 10)
            A = [I A12; transform(A12) I]
            L = [I LinearMap(A12); transform(LinearMap(A12)) I]
            @test_broken ishermitian(L)
            x = rand(elty, 20)
            @test L isa LinearMaps.LinearMap{elty}
            @test size(L) == size(A)
            @test L * x ≈ A * x
            @test Matrix(L) ≈ A
            Lt = transform(L)
            @test Lt isa LinearMaps.LinearMap{elty}
            @test Lt * x ≈ transform(A) * x
            Lt = transform(LinearMap(L))
            @test Lt * x ≈ transform(A) * x
            @test Matrix(Lt) ≈ Matrix(transform(A))
            A21 = rand(elty, 10, 10)
            A = [I A12; A21 I]
            L = [I LinearMap(A12); LinearMap(A21) I]
            Lt = transform(L)
            @test Lt isa LinearMaps.LinearMap{elty}
            @test Lt * x ≈ transform(A) * x
            @test Matrix(Lt) ≈ Matrix(transform(A))
        end
    end
end
