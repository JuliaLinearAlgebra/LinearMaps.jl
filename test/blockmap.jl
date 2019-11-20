using Test, LinearMaps, LinearAlgebra

@testset "block maps" begin
    @testset "hcat" begin
        for elty in (Float32, Float64, ComplexF64), n2 = (0, 20)
            A11 = rand(elty, 10, 10)
            A12 = rand(elty, 10, n2)
            L = @inferred hcat(LinearMap(A11), LinearMap(A12))
            @test @inferred(LinearMaps.mulstyle(L)) == matrixstyle
            @test L isa LinearMaps.BlockMap{elty}
            A = [A11 A12]
            x = rand(10+n2)
            @test size(L) == size(A)
            @test Matrix(L) ≈ A
            @test L * x ≈ A * x
            L = @inferred hcat(LinearMap(A11), LinearMap(A12), LinearMap(A11))
            A = [A11 A12 A11]
            @test Matrix(L) ≈ A
            A = [I I I A11 A11 A11]
            L = @inferred hcat(I, I, I, LinearMap(A11), LinearMap(A11), LinearMap(A11))
            @test L == [I I I LinearMap(A11) LinearMap(A11) LinearMap(A11)]
            x = rand(elty, 60)
            @test L isa LinearMaps.BlockMap{elty}
            @test L * x ≈ A * x
            A11 = rand(elty, 11, 10)
            A12 = rand(elty, 10, n2)
            @test_throws DimensionMismatch hcat(LinearMap(A11), LinearMap(A12))
        end
    end

    @testset "vcat" begin
        for elty in (Float32, Float64, ComplexF64)
            A11 = rand(elty, 10, 10)
            L = @inferred vcat(LinearMap(A11))
            @test L == [LinearMap(A11);]
            @test Matrix(L) ≈ A11
            A21 = rand(elty, 20, 10)
            L = @inferred vcat(LinearMap(A11), LinearMap(A21))
            @test L isa LinearMaps.BlockMap{elty}
            @test @inferred(LinearMaps.mulstyle(L)) == matrixstyle
            A = [A11; A21]
            x = rand(10)
            @test size(L) == size(A)
            @test Matrix(L) ≈ A
            @test L * x ≈ A * x
            A = [I; I; I; A11; A11; A11]
            L = @inferred vcat(I, I, I, LinearMap(A11), LinearMap(A11), LinearMap(A11))
            @test L == [I; I; I; LinearMap(A11); LinearMap(A11); LinearMap(A11)]
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
            @test @inferred(LinearMaps.mulstyle(L)) == matrixstyle
            @test @inferred !issymmetric(L)
            @test @inferred !ishermitian(L)
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
            A12 = LinearMap(rand(elty, 10, 21))
            A21 = LinearMap(rand(elty, 20, 10))
            @test_throws DimensionMismatch A = [I A12; A21 I]
            @test_throws DimensionMismatch A = [I A21; A12 I]
            @test_throws DimensionMismatch A = [A12 A12; A21 A21]
            @test_throws DimensionMismatch A = [A12 A21; A12 A21]

            # basic test of "misaligned" blocks
            M = ones(elty, 3, 2) # non-square
            A = LinearMap(M)
            B = [I A; A I]
            C = [I M; M I]
            @test B isa LinearMaps.BlockMap{elty}
            @test Matrix(B) == C
            @test Matrix(transpose(B)) == transpose(C)
            @test Matrix(adjoint(B)) == C'
        end
    end

    @testset "adjoint/transpose" begin
        for elty in (Float32, Float64, ComplexF64), transform in (transpose, adjoint)
            A12 = rand(elty, 10, 10)
            A = [I A12; transform(A12) I]
            L = [I LinearMap(A12); transform(LinearMap(A12)) I]
            @test @inferred(LinearMaps.mulstyle(L)) == matrixstyle
            if elty <: Complex
                if transform == transpose
                    @test @inferred issymmetric(L)
                else
                    @test @inferred ishermitian(L)
                end
            end
            if elty <: Real
                @test @inferred ishermitian(L)
                @test @inferred issymmetric(L)
            end
            x = rand(elty, 20)
            @test L isa LinearMaps.LinearMap{elty}
            @test size(L) == size(A)
            @test L * x ≈ A * x
            @test Matrix(L) ≈ A
            Lt = @inferred transform(L)
            @test Lt isa LinearMaps.LinearMap{elty}
            @test Lt * x ≈ transform(A) * x
            Lt = @inferred transform(LinearMap(L))
            @test Lt * x ≈ transform(A) * x
            @test Matrix(Lt) ≈ Matrix(transform(A))
            A21 = rand(elty, 10, 10)
            A = [I A12; A21 I]
            L = [I LinearMap(A12); LinearMap(A21) I]
            Lt = @inferred transform(L)
            @test Lt isa LinearMaps.LinearMap{elty}
            @test Lt * x ≈ transform(A) * x
            @test Matrix(Lt) ≈ Matrix(transform(A))
        end
    end
end
