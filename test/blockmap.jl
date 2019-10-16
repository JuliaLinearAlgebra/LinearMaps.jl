using Test, LinearMaps, LinearAlgebra, SparseArrays

@testset "block maps" begin
    @testset "hcat" begin
        for elty in (Float32, Float64, ComplexF64), n2 = (0, 20)
            A11 = rand(elty, 10, 10)
            A12 = rand(elty, 10, n2)
            L = @inferred hcat(LinearMap(A11), LinearMap(A12))
            @test @inferred(LinearMaps.MulStyle(L)) === matrixstyle
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
            @test @inferred(LinearMaps.MulStyle(L)) === matrixstyle
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
            @test @inferred(LinearMaps.MulStyle(L)) === matrixstyle
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
            y = randn(elty, size(L, 1))
            for α in (0, 1, rand(elty)), β in (0, 1, rand(elty))
                @test mul!(copy(y), L, x, α, β) ≈ y*β .+ A*x*α
            end
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
            @test @inferred(LinearMaps.MulStyle(L)) === matrixstyle
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

    @testset "block diagonal maps" begin
        for elty in (Float64, ComplexF64)
            m = 5; n = 6
            M1 = 10*(1:m) .+ (1:(n+1))'; L1 = LinearMap(M1)
            M2 = randn(elty, m, n+2); L2 = LinearMap(M2)
            M3 = randn(elty, m, n+3); L3 = LinearMap(M3)

            # Md = diag(M1, M2, M3, M2, M1) # unsupported so use sparse:
            Md = Matrix(blockdiag(sparse.((M1, M2, M3, M2, M1))...))
            x = randn(elty, size(Md, 2))
            Bd = @inferred blockdiag(L1, L2, L3, L2, L1)
            @test Matrix(@inferred blockdiag(L1)) == M1
            @test Matrix(@inferred blockdiag(L1, L2)) == blockdiag(sparse.((M1, M2))...)
            Bd2 = @inferred cat(L1, L2, L3, L2, L1; dims=(1,2))
            @test_throws ArgumentError cat(L1, L2, L3, L2, L1; dims=(2,2))
            @test Bd == Bd2
            @test Bd == blockdiag(L1, M2, M3, M2, M1)
            @test size(Bd) == (25, 39)
            @test !issymmetric(Bd)
            @test !ishermitian(Bd)
            @test @inferred Bd * x ≈ Md * x
            for transform in (identity, adjoint, transpose)
                @test Matrix(@inferred transform(Bd)) == transform(Md)
                @test Matrix(@inferred transform(LinearMap(Bd))) == transform(Md)
            end
            y = randn(elty, size(Md, 1))
            for α in (0, 1, rand(elty)), β in (0, 1, rand(elty))
                @test mul!(copy(y), Bd, x, α, β) ≈ y*β .+ Md*x*α
            end
        end
    end
end
