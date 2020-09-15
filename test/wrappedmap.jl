using Test, LinearMaps, LinearAlgebra

@testset "wrapped maps" begin
    A = rand(10, 20)
    B = rand(ComplexF64, 10, 20)
    SA = A'A + I
    SB = B'B + I
    L = @inferred LinearMap{Float64}(A)
    MA = @inferred LinearMap(SA)
    MB = @inferred LinearMap(SB)
    @test eltype(Matrix{Complex{Float32}}(LinearMap(A))) <: Complex
    @test size(L) == size(A)
    @test @inferred !issymmetric(L)
    @test @inferred issymmetric(MA)
    @test @inferred !issymmetric(MB)
    @test @inferred isposdef(MA)
    @test @inferred isposdef(MB)
end
