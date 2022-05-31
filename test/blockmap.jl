using Test, LinearMaps, LinearAlgebra, SparseArrays, InteractiveUtils
using LinearMaps: FiveArg

@testset "block maps" begin
    @testset "hcat" begin
        m = 3
        n = 4
        for elty in (Float32, ComplexF64), n2 in (0, 2)
            A11 = rand(elty, m, n)
            A12 = rand(elty, m, n2)
            a = rand(elty, m)
            L = @inferred hcat(LinearMap(A11), LinearMap(A12))
            @test L.maps isa Tuple
            Lv = @inferred LinearMaps.BlockMap{elty}([LinearMap(A11), LinearMap(A12)], (2,))
            @test Lv.maps isa Vector
            @test L == Lv == LinearMaps.BlockMap([LinearMap(A11), LinearMap(A12)], (2,))
            @test occursin("$m×$(n+n2) LinearMaps.BlockMap{$elty}", sprint((t, s) -> show(t, "text/plain", s), L))
            @test @inferred(LinearMaps.MulStyle(L)) === FiveArg()
            @test L isa LinearMaps.BlockMap{elty}
            if elty <: Complex
                @test_throws ErrorException LinearMaps.BlockMap{Float64}((LinearMap(A11), LinearMap(A12)), (2,))
            end
            A = [A11 A12]
            x = rand(n+n2)
            @test size(L) == size(A) == size(Lv)
            @test Matrix(L) == A == Matrix(Lv) == mul!(copy(A), L, 1, true, false)
            @test L * x ≈ A * x ≈ Lv * x
            L = @inferred hcat(LinearMap(A11), LinearMap(A12), LinearMap(A11))
            A = [A11 A12 A11]
            @test Matrix(L) == A
            A = [I I I A11 A11 A11 a]
            @test (@which [A11 A11 A11]).module != LinearMaps
            @test (@which [I I I A11 A11 A11]).module != LinearMaps
            @test (@which hcat(I, I, I)).module != LinearMaps
            @test (@which hcat(I, I, I, LinearMap(A11), A11, A11)).module == LinearMaps
            maps = @inferred LinearMaps.promote_to_lmaps(ntuple(i->m, 7), 1, 1, I, I, I, LinearMap(A11), A11, A11, a)
            @inferred LinearMaps.rowcolranges(maps, (7,))
            L = @inferred hcat(I, I, I, LinearMap(A11), A11, A11, a)
            @test L == [I I I LinearMap(A11) LinearMap(A11) LinearMap(A11) LinearMap(a)]
            x = ones(elty, size(L, 2))
            @test L isa LinearMaps.BlockMap{elty}
            @test L * x ≈ A * x
            L = @inferred hcat(I, I, I, LinearMap(A11), A11, A11, a, a, a, a)
            @test occursin("$m×$(3m+3n+4) LinearMaps.BlockMap{$elty}", sprint((t, s) -> show(t, "text/plain", s), L))
            L = @inferred hcat(I, I, I, LinearMap(A11), A11, A11, a, a, a, a, a, a, a)
            @test occursin("$m×$(3m+3n+7) LinearMaps.BlockMap{$elty}", sprint((t, s) -> show(t, "text/plain", s), L))
        end
        A11 = zeros(m+1, n)
        A12 = zeros(m, n)
        @test_throws DimensionMismatch hcat(LinearMap(A11), LinearMap(A12))
    end

    @testset "vcat" begin
        m = 2
        n = 3
        for elty in (Float32, ComplexF64)
            A11 = rand(elty, m, n)
            v = rand(elty, n)
            L = @inferred vcat(LinearMap(A11))
            @test L == [LinearMap(A11);]
            @test Matrix(L) ≈ A11
            A21 = rand(elty, 2m, n)
            L = @inferred vcat(LinearMap(A11), LinearMap(A21))
            @test L.maps isa Tuple
            @test L isa LinearMaps.BlockMap{elty}
            @test occursin("$(3m)×$n LinearMaps.BlockMap{$elty}", sprint((t, s) -> show(t, "text/plain", s), L))
            @test @inferred(LinearMaps.MulStyle(L)) === FiveArg()
            Lv = LinearMaps.BlockMap{elty}([LinearMap(A11), LinearMap(A21)], (1,1))
            @test Lv.maps isa Vector
            @test L == Lv
            @test (@which [A11; A21]).module != LinearMaps
            A = [A11; A21]
            x = rand(elty, n)
            @test size(L) == size(A)
            @test Matrix(L) == Matrix(Lv) ==  A
            @test L * x ≈ Lv * x ≈ A * x
            A = [I; I; I; A11; A11; A11; reduce(hcat, fill(v, n))]
            @test (@which [I; I; I; A11; A11; A11; v v v v v v v v v v]).module != LinearMaps
            L = @inferred vcat(I, I, I, LinearMap(A11), LinearMap(A11), LinearMap(A11), reduce(hcat, fill(v, n)))
            @test L == [I; I; I; LinearMap(A11); LinearMap(A11); LinearMap(A11); reduce(hcat, fill(v, n))]
            @test L isa LinearMaps.BlockMap{elty}
            @test L * x ≈ A * x
        end
        A11 = zeros(m, n+1)
        A21 = zeros(2m, n)
        @test_throws DimensionMismatch vcat(LinearMap(A11), LinearMap(A21))
    end

    @testset "hvcat" begin
        m1 = 2
        m2 = 3
        n = 3
        for elty in (Float32, ComplexF64)
            A11 = rand(elty, m1, m1)
            A12 = ones(elty, m1, m2)
            A21 = rand(elty, m2, m1)
            A22 = ones(elty, m2, m2)
            A = [A11 A12; A21 A22]
            @test (@which [A11 A12; A21 A22]).module != LinearMaps
            @inferred hvcat((2,2), LinearMap(A11), LinearMap(A12), LinearMap(A21), LinearMap(A22))
            L = [LinearMap(A11) LinearMap(A12); LinearMap(A21) LinearMap(A22)]
            @test L.maps isa Tuple
            Lv = @inferred LinearMaps.BlockMap{elty}([LinearMap(A11), LinearMap(A12), LinearMap(A21), LinearMap(A22)], (2,2))
            @test Lv.maps isa Vector
            @test @inferred(LinearMaps.MulStyle(L)) === FiveArg()
            @test @inferred !issymmetric(L)
            @test @inferred !ishermitian(L)
            x = rand(m1+m2)
            @test L isa LinearMaps.BlockMap{elty}
            @test size(L) == size(A)
            @test L * x ≈ Lv * x ≈ A * x
            @test Matrix(L) == Matrix(Lv) == A
            @test convert(AbstractMatrix, L) == A
            A = [I A12; A21 I]
            @test (@which [I A12; A21 I]).module != LinearMaps
            @inferred hvcat((2,2), I, LinearMap(A12), LinearMap(A21), I)
            L = @inferred hvcat((2,2), I, LinearMap(A12), LinearMap(A21), I)
            @test L isa LinearMaps.BlockMap{elty}
            @test size(L) == (m1+m2, m1+m2)
            @test Matrix(L) ≈ A
            @test L * x ≈ A * x
            y = randn(elty, size(L, 1))
            for α in (0, 1, rand(elty)), β in (0, 1, rand(elty))
                @test mul!(copy(y), L, x, α, β) ≈ y*β .+ A*x*α
            end
            X = rand(elty, m1+m2, n)
            Y = randn(elty, size(L, 1), n)
            for α in (0, 1, rand(elty)), β in (0, 1, rand(elty))
                @test mul!(copy(Y), L, X, α, β) ≈ Y*β .+ A*X*α
            end
            A = ones(elty, m1, m1); LA = LinearMap(A)
            B = zeros(elty, m2, 3m1); LB = LinearMap(B)
            @test [LA LA LA; LB] isa LinearMaps.BlockMap{elty}
            @test Matrix([LA LA LA; LB]) ≈ [A A A; B]
            @test [LB; LA LA LA] isa LinearMaps.BlockMap{elty}
            @test Matrix([LB; LA LA LA]) ≈ [B; A A A]
            @test [I; LA LA LA] isa LinearMaps.BlockMap{elty}
            @test Matrix([I; LA LA LA]) ≈ [I; A A A]
            A12 = LinearMap(zeros(elty, m1, m2+1))
            A21 = LinearMap(zeros(elty, m2, m1))
            @test_throws DimensionMismatch [I A12; A21 I]
            @test_throws DimensionMismatch [I A21; A12 I]
            @test_throws DimensionMismatch [A12 A12; A21 A21]
            @test_throws DimensionMismatch [A12 A21; A12 A21]

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
        for elty in (Float32, ComplexF64), transform in (transpose, adjoint)
            A12 = rand(elty, 10, 10)
            A = [I A12; transform(A12) I]
            L = [I LinearMap(A12); transform(LinearMap(A12)) I]
            @test @inferred(LinearMaps.MulStyle(L)) === FiveArg()
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
            @test Matrix(L) == A
            @test convert(AbstractMatrix, L) == A
            @test sparse(L) == sparse(A)
            Lt = @inferred transform(L)
            @test Lt isa LinearMaps.LinearMap{elty}
            @test Lt * x ≈ transform(A) * x
            @test convert(AbstractMatrix, Lt) == transform(A)
            @test sparse(transform(L)) == transform(A)
            Lt = @inferred transform(LinearMap(L))
            @test Lt * x ≈ transform(A) * x
            @test Matrix(Lt) == Matrix(transform(A))
            @test mul!(copy(transform(A)), Lt, 1, true, true) == 2transform(A)
            A21 = rand(elty, 10, 10)
            A = [I A12; A21 I]
            L = [I LinearMap(A12); LinearMap(A21) I]
            Lt = @inferred transform(L)
            @test Lt isa LinearMaps.LinearMap{elty}
            @test Lt * x ≈ transform(A) * x
            @test Matrix(Lt) ≈ Matrix(transform(LinearMap(L))) ≈ Matrix(transform(A))
            @test Matrix(transform(LinearMap(L))+transform(LinearMap(L))) ≈ 2Matrix(transform(A))
            X = rand(elty, size(L, 1), 10)
            Y = randn(elty, size(L, 2), 10)
            for α in (0, 1, rand(elty)), β in (0, 1, rand(elty))
                @test mul!(copy(Y), Lt, X, α, β) ≈ Y*β .+ transform(A)*X*α
            end
        end
    end

    @testset "block diagonal maps" begin
        for elty in (Float32, ComplexF64)
            m = 2; n = 3
            M1 = 10*(1:m) .+ (1:(n+1))'; L1 = LinearMap(M1)
            M2 = randn(elty, m, n+2); L2 = LinearMap(M2)
            M3 = randn(elty, m, n+3); L3 = LinearMap(M3)

            # Md = diag(M1, M2, M3, M2, M1) # unsupported so use sparse:
            if elty <: Complex
                @test_throws ErrorException LinearMaps.BlockDiagonalMap{Float64}((L1, L2, L3, L2, L1))
            end
            Md = Matrix(blockdiag(sparse.((M1, M2, M3, M2, M1))...))
            @test (@which blockdiag(sparse.((M1, M2, M3, M2, M1))...)).module != LinearMaps
            @test (@which cat(M1, M2, M3, M2, M1; dims=(1,2))).module != LinearMaps
            x = randn(elty, size(Md, 2))
            Bd = @inferred blockdiag(L1, L2, L3, L2, L1)
            @test Bd.maps isa Tuple
            Bdv = @inferred LinearMaps.BlockDiagonalMap{elty}([L1, L2, L3, L2, L1])
            @test Bdv.maps isa Vector
            @test @inferred(LinearMaps.MulStyle(Bd)) === FiveArg()
            @test occursin("$(5m)×$(5n+9) LinearMaps.BlockDiagonalMap{$elty}", sprint((t, s) -> show(t, "text/plain", s), Bd))
            @test Matrix(Bd) == Md
            @test convert(AbstractMatrix, Bd) isa SparseMatrixCSC
            @test sparse(Bd) == Md
            @test Matrix(@inferred blockdiag(L1)) == M1
            @test Matrix(@inferred blockdiag(L1, L2)) == blockdiag(sparse.((M1, M2))...)
            Bd2 = @inferred cat(L1, L2, L3, L2, L1; dims=(1,2))
            @test_throws ArgumentError cat(L1, L2, L3, L2, L1; dims=(2,2))
            @test Bd == Bdv == Bd2
            @test Bd == blockdiag(L1, M2, M3, M2, M1)
            @test size(Bd) == (5m, 5n+9)
            @test !issymmetric(Bd)
            @test !ishermitian(Bd)
            @test (@inferred Bd * x) ≈ Bdv * x ≈ Md * x
            for transform in (identity, adjoint, transpose)
                @test Matrix(@inferred transform(Bd)) == transform(Md)
                @test Matrix(@inferred transform(LinearMap(Bd))) == transform(Md)
            end
            y = randn(elty, size(Md, 1))
            for α in (0, 1, rand(elty)), β in (0, 1, rand(elty))
                @test mul!(copy(y), Bd, x, α, β) ≈ y*β .+ Md*x*α
                @test mul!(copy(y), Bdv, x, α, β) ≈ y*β .+ Md*x*α
            end
            X = randn(elty, size(Md, 2), 3)
            Y = randn(elty, size(Md, 1), 3)
            for α in (0, 1, rand(elty)), β in (0, 1, rand(elty))
                @test mul!(copy(Y), Bd, X, α, β) ≈ Y*β .+ Md*X*α
                @test mul!(copy(Y), Bdv, X, α, β) ≈ Y*β .+ Md*X*α
            end
        end
    end

    @testset "function block map" begin
        N = 5
        T = ComplexF64
        CS! = LinearMap{T}(cumsum!,
            (y, x) -> (copyto!(y, x); reverse!(cumsum!(y, reverse!(y)))), N;
            ismutating=true)
        A = rand(T, N, N)
        B = rand(T, N, N)
        LT = LowerTriangular(ones(T, N, N))
        L1 = [CS! CS! CS!; CS! CS! CS!; CS! CS! CS!]
        M1 = [LT LT LT; LT LT LT; LT LT LT]
        L2 = [CS! LinearMap(A) CS!; LinearMap(B) CS! CS!; CS! CS! CS!]
        M2 = [LT A LT
              B LT LT
              LT LT LT]
        u = rand(T, 3N)
        v = rand(T, 3N)
        for α in (false, true, rand(T)), β in (false, true, rand(T))
            for transform in (identity, adjoint), (L, M) in ((L1, M1), (L2, M2))
                # @show α, β, transform
                @test mul!(copy(v), transform(L), u, α, β) ≈ transform(M)*u*α + v*β
                @test mul!(copy(v), transform(LinearMap(L)), u, α, β) ≈ transform(M)*u*α + v*β
                @test mul!(copy(v), LinearMap(transform(L)), u, α, β) ≈ transform(M)*u*α + v*β
                if transform != adjoint
                    transL = transform(L)
                    alloc = @allocated similar(v)
                    if L == L2 && α != false
                        @test_broken (@allocated mul!(v, transL, u, α, β)) <= alloc
                    else
                        @test (@allocated mul!(v, transL, u, α, β)) <= alloc
                    end
                end
            end
        end
    end
end
