using Test, LinearMaps, LinearAlgebra, BenchmarkTools

@testset "wrapped maps" begin
    A = rand(10, 20)
    B = rand(ComplexF64, 10, 20)
    SA = A'A + I
    SB = B'B + I
    L = @inferred LinearMap{Float64}(A)
    MA = @inferred LinearMap(SA)
    MB = @inferred LinearMap(SB)
    @test size(L) == size(A)
    @test @inferred !issymmetric(L)
    @test @inferred issymmetric(MA)
    @test @inferred !issymmetric(MB)
    @test @inferred isposdef(MA)
    @test @inferred isposdef(MB)
    B = rand(ComplexF64, 20, 20)
    v = rand(ComplexF64, 20)
    u = rand(ComplexF64, 20)
    LB = LinearMap(B)
    for α in (false, true, rand(ComplexF64)), β in (false, true, rand(ComplexF64))
        for transform in (identity, adjoint, transpose)
            @test mul!(copy(u), transform(LB), v, α, β) ≈ transform(B)*v*α + u*β
            @test mul!(copy(u), transform(LinearMap(LB)), v, α, β) ≈ transform(B)*v*α + u*β
            @test mul!(copy(u), transform(LinearMap(LinearMap(LB))), v, α, β) ≈ transform(B)*v*α + u*β
            @test mul!(copy(u), LinearMap(transform(LinearMap(LB))), v, α, β) ≈ transform(B)*v*α + u*β
            if testallocs
                blmap = @benchmarkable mul!($(copy(u)), $(transform(LB)), $v, $α, $β)
                @test run(blmap, samples=3).allocs == 0
            end
        end
    end

    CS! = LinearMap{ComplexF64}(cumsum!,
                                (y, x) -> (copyto!(y, x); reverse!(y); cumsum!(y, y); reverse!(y)), 10;
                                ismutating=true)
    v = rand(ComplexF64, 10)
    u = rand(ComplexF64, 10)
    M = Matrix(CS!)
    for α in (false, true, rand(ComplexF64)), β in (false, true, rand(ComplexF64))
        for transform in (identity, adjoint, transpose)
            @test mul!(copy(u), transform(CS!), v, α, β) ≈ transform(M)*v*α + u*β
            @test mul!(copy(u), transform(LinearMap(CS!)), v, α, β) ≈ transform(M)*v*α + u*β
            @test mul!(copy(u), transform(LinearMap(LinearMap(CS!))), v, α, β) ≈ transform(M)*v*α + u*β
            @test mul!(copy(u), LinearMap(transform(LinearMap(CS!))), v, α, β) ≈ transform(M)*v*α + u*β
        end
    end
end
