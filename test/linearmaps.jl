using Test, LinearMaps, LinearAlgebra, SparseArrays, BenchmarkTools

@testset "basic functionality" begin
    A = 2 * rand(ComplexF64, (20, 10)) .- 1
    v = rand(ComplexF64, 10)
    u = rand(ComplexF64, 20, 1)
    V = rand(ComplexF64, 10, 3)
    W = rand(ComplexF64, 20, 3)
    α = rand()
    β = rand()
    M = @inferred LinearMap(A)
    N = @inferred LinearMap(M)

    @testset "LinearMaps.jl" begin
        @test eltype(M) == eltype(A)
        @test size(M) == size(A)
        @test size(N) == size(A)
        @test !isreal(M)
        @test ndims(M) == 2
        @test_throws ErrorException size(M, 3)
        @test length(M) == length(A)
        @test occursin(Base.dims2string(size(M)) * " LinearMaps.WrappedMap", summary(M))
    end

    @testset "dimension checking" begin
        w = vec(u)
        @test_throws DimensionMismatch M * similar(v, length(v) + 1)
        @test_throws DimensionMismatch mul!(similar(w, length(w) + 1), M, v)
        @test_throws DimensionMismatch similar(w, length(w) + 1)' * M
        @test_throws DimensionMismatch mul!(copy(v)', similar(w, length(w) + 1)', M)
        @test_throws DimensionMismatch mul!(similar(W, size(W).+(0,1)), M, V)
        @test_throws DimensionMismatch mul!(copy(W), M, similar(V, size(V).+(0,1)))
    end
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
Base.:(*)(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, v::AbstractVector) = A.f(v)
LinearAlgebra.mul!(y::AbstractVector, A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, x::AbstractVector) = copyto!(y, *(A, x))

@testset "new LinearMap type" begin
    F = SimpleFunctionMap(cumsum, 10)
    @test parent(F) === F
    FC = SimpleComplexFunctionMap(cumsum, 10)
    @test @inferred ndims(F) == 2
    @test @inferred size(F, 1) == 10
    @test @inferred length(F) == 100
    @test @inferred !issymmetric(F)
    @test @inferred !ishermitian(F)
    @test @inferred !ishermitian(FC)
    @test @inferred !isposdef(F)
    @test occursin("10×10 SimpleFunctionMap{$(eltype(F))}", sprint((t, s) -> show(t, "text/plain", s), F))
    @test occursin("10×10 SimpleComplexFunctionMap{$(eltype(FC))}", sprint((t, s) -> show(t, "text/plain", s), FC))
    α = rand(ComplexF64); β = rand(ComplexF64)
    v = rand(ComplexF64, 10); V = rand(ComplexF64, 10, 3)
    w = rand(ComplexF64, 10); W = rand(ComplexF64, 10, 3)
    @test mul!(w, F, v) === w == F * v
    @test_throws ErrorException F' * v
    @test_throws ErrorException transpose(F) * v
    @test_throws ErrorException mul!(w, adjoint(FC), v)
    @test_throws ErrorException mul!(w, transpose(F), v)
    FM = convert(AbstractMatrix, F)
    L = LowerTriangular(ones(10, 10))
    @test FM == L
    @test F * v ≈ L * v
    # generic 5-arg mul! and matrix-mul!
    @test mul!(copy(w), F, v, α, β) ≈ L*v*α + w*β
    @test mul!(copy(w), F, v, 0, β) ≈ w*β
    @test mul!(copy(W), F, V) ≈ L*V
    @test mul!(copy(W), F, V, α, β) ≈ L*V*α + W*β
    @test mul!(copy(W), F, V, 0, β) ≈ W*β

    Fs = sparse(F)
    @test SparseMatrixCSC(F) == Fs == L
    @test Fs isa SparseMatrixCSC
end

struct MyFillMap{T} <: LinearMaps.LinearMap{T}
    λ::T
    size::Dims{2}
    function MyFillMap(λ::T, dims::Dims{2}) where {T}
        all(d -> d >= 0, dims) || throw(ArgumentError("dims of MyFillMap must be non-negative"))
        promote_type(T, typeof(λ)) == T || throw(InexactError())
        return new{T}(λ, dims)
    end
end
Base.size(A::MyFillMap) = A.size
function LinearAlgebra.mul!(y::AbstractVecOrMat, A::MyFillMap, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)
    return fill!(y, iszero(A.λ) ? zero(eltype(y)) : A.λ*sum(x))
end
function LinearAlgebra.mul!(y::AbstractVecOrMat, transA::LinearMaps.TransposeMap{<:Any,<:MyFillMap}, x::AbstractVector)
    LinearMaps.check_dim_mul(y, transA, x)
    λ = transA.lmap.λ
    return fill!(y, iszero(λ) ? zero(eltype(y)) : transpose(λ)*sum(x))
end

@testset "transpose of new LinearMap type" begin
    A = MyFillMap(5.0, (3, 3))
    x = ones(3)
    @test A * x == fill(15.0, 3)
    @test mul!(zeros(3), A, x) == mul!(zeros(3), A, x, 1, 0) == fill(15.0, 3)
    @test mul!(ones(3), A, x, 2, 2) == fill(32.0, 3)
    @test mul!(ones(3,3), A, reshape(collect(1:9), 3, 3), 2, 2)[:,1] == fill(62.0, 3)
    @test A'x == mul!(zeros(3), A', x)
    for α in (true, false, 0, 1, randn()), β in (true, false, 0, 1, randn())
        @test mul!(ones(3), A', x, α, β) == fill(β, 3) + fill(15α, 3)
        @test mul!(ones(3, 2), A', [x x], α, β) == fill(β, 3, 2) + fill(15α, 3, 2)
    end
end
