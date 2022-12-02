# test/units

using Test: @test, @testset, @inferred, @test_throws
using LinearMaps: LinearMap
using Unitful: m, s, g

@testset "units" begin
    A = rand(4,3) * 1m
    B = rand(3,2) * 1s
    C = A * B
    D = 1f0g * C

    Ma = @inferred LinearMap(A)
    Mb = @inferred LinearMap(B)
    Mc = Ma * Mb
    Md = 1f0g * Mc # todo - UniformScalingMap

    @test Matrix(Ma) == A
    @test Matrix(Mc) == C
    @test_throws ArgumentError Matrix(Md) == D

    x = randn(2)
    @test B * x == Mb * x
    @test_broken C * x == Mc * x
    @test_broken D * x == Md * x
end
