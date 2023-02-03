using Test, LinearMaps, LinearAlgebra, SparseArrays, Statistics
using LinearMaps: FiveArg, LinearMapTuple, LinearMapVector, FunctionMap

@testset "linear combinations" begin
    CR  = FunctionMap{Float64}(cumsum, reverse∘cumsum∘reverse, 10, 10)
    CS  = FunctionMap{ComplexF64}(cumsum, reverse∘cumsum∘reverse, 10, 10)
    CS! = LinearMap{ComplexF64}(cumsum!,
                                (y, x) -> (copyto!(y, x); reverse!(cumsum!(y, reverse!(y)))), 10;
                                ismutating=true)
    v = rand(ComplexF64, 10)
    u = similar(v)
    mul!(u, CS!, v)
    @test (@allocated mul!(u, CS!, v)) == 0
    n = 10
    alloc = @allocated similar(v)
    Loop = @inferred CS + CS + CS
    @test Loop * v ≈ 3cumsum(v)
    @test (CS + CR + CS) * v ≈ 3cumsum(v)
    @test (@allocated Loop * v) <= 3alloc
    Loop = @inferred CS + CS; Loop * v
    @test (@allocated Loop * v) <= 2alloc
    Lmix = @inferred CS + CS + CS!; Lmix * v
    @test (@allocated Lmix * v) <= 3alloc
    Lmix = @inferred CS + CS + CS!; Lmix * v
    @test (@allocated Lmix * v) <= 3alloc
    Lmix = @inferred CS! + (CS + CS); Lmix * v
    @test (@allocated Lmix * v) <= 3alloc
    L = @inferred sum(ntuple(_ -> CS!, n))
    @test (@inferred sum(L.maps::LinearMapTuple)) == L
    Lv = @inferred LinearMaps.LinearCombination{ComplexF64}(fill(CS!, n))
    @test (@inferred sum(Lv.maps::LinearMapVector)) == Lv
    @test isa((@inferred mean(Lv.maps)),
        LinearMaps.ScaledMap{ComplexF64,Float64,<:LinearMaps.LinearCombination{ComplexF64,<:LinearMapVector}})
    @test (@inferred mean(L.maps)) == (@inferred mean(Lv.maps)) == (@inferred sum(Lv.maps))/n
    @test (@inferred mean(L)) == (@inferred mean(Lv))
    @test (@inferred mean(x -> x*x, L.maps)) == (@inferred sum(x -> x*x, L.maps))/n
    @test mean(x -> x*x, Lv.maps) == (sum(x -> x*x, Lv.maps))/n
    @test L == Lv
    A = LinearMap(randn(eltype(CS!), size(CS!)))
    Ar = LinearMap(real(A.lmap))
    @test isa((@inferred sum([CS!, A])),
        LinearMaps.LinearCombination{<:ComplexF64,<:LinearMapVector})
    @test (@inferred mean([CS!, A])) == (@inferred sum([CS!, A]))/2
    @test (@inferred mean([CS!, A])) == (@inferred mean(identity, [CS!, A])) == (@inferred sum([CS!, A]))/2
    @test isa(sum([CS!, Ar]), LinearMaps.LinearCombination{<:ComplexF64,<:LinearMapVector})
    @test sum([CS!, Ar])/2 == mean([CS!, Ar])
    @test sum([CS!, Ar]) == sum(identity, [CS!, Ar])
    for sum1 in (CS!, L, Lv), sum2 in (CS!, L, Lv)
        m1 = sum1 == CS! ? 1 : 10
        m2 = sum2 == CS! ? 1 : 10
        vect = any(x -> isa(x, LinearMaps.LinearCombination{ComplexF64,<:LinearMapVector}), (sum1, sum2))
        maptyp = vect ? LinearMapVector : LinearMapTuple
        @test (sum1+sum2) isa LinearMaps.LinearCombination{ComplexF64,<:maptyp}
        @test (sum1+sum2) * v ≈ (m1+m2)*cumsum(v)
    end
    M, Mv = Matrix.((L, Lv))
    @test M == Mv == LowerTriangular(fill(n, size(L)))
    @test_throws AssertionError LinearMaps.LinearCombination{Float64}((CS!, CS!))
    @test occursin("10×10 $LinearMaps.LinearCombination{$(eltype(L))}", sprint((t, s) -> show(t, "text/plain", s), L))
    @test occursin("10×10 $LinearMaps.LinearCombination{$(eltype(L))}", sprint((t, s) -> show(t, "text/plain", s), L+CS!))
    @test mul!(u, L, v) ≈ n * cumsum(v)
    @test mul!(u, Lv, v) ≈ n * cumsum(v)
    alloc = @allocated similar(u)
    mul!(u, L, v, 2, 2)
    @test (@allocated mul!(u, L, v, 2, 2)) <= alloc
    mul!(u, Lv, v, 2, 2)
    @test (@allocated mul!(u, Lv, v, 2, 2)) <= alloc
    for α in (false, true, rand(ComplexF64)), β in (false, true, rand(ComplexF64))
        for transform in (identity, adjoint, transpose)
            @test mul!(copy(u), transform(L), v, α, β) ≈ transform(M)*v*α + u*β
        end
    end
    V = rand(ComplexF64, 10, 3)
    U = similar(V)
    @test mul!(U, L, V) ≈ n*cumsum(V, dims=1)
    @test mul!(U, LinearMap(L), V) ≈ n*cumsum(V, dims=1)

    A = randn(3,3)
    B = LinearMap(A) + LinearMap(A)'
    C = LinearMap(copy(A)) + LinearMap(copy(A))'
    @test_throws DimensionMismatch A + CS!
    @test B == C

    A = 2 * rand(ComplexF64, (10, 10)) .- 1
    B = rand(ComplexF64, size(A)...)
    M = @inferred LinearMap(A)
    N = @inferred LinearMap(B)
    @test @inferred(LinearMaps.MulStyle(M)) === FiveArg()
    @test @inferred(LinearMaps.MulStyle(N)) === FiveArg()
    LC = @inferred M + N
    @test @inferred(LinearMaps.MulStyle(LC)) === FiveArg()
    @test @inferred(LinearMaps.MulStyle(LC + I)) === FiveArg()
    @test @inferred(LinearMaps.MulStyle(LC + 2.0*I)) === FiveArg()
    @test sparse(LC) == Matrix(LC) == A+B
    v = rand(ComplexF64, 10)
    w = similar(v)
    mul!(w, M, v)
    @test (@allocated mul!(w, M, v)) == 0
    mul!(w, LC, v)
    @test (@allocated mul!(w, LC, v)) == 0
    for α in (false, true, rand(ComplexF64)), β in (false, true, rand(ComplexF64))
        y = rand(ComplexF64, size(v))
        MC = Matrix(LC)
        @test mul!(copy(y), LC, v, α, β) ≈ MC*v*α + y*β
        @test mul!(copy(y), LC+I, v, α, β) ≈ (MC+I)*v*α + y*β
        @test mul!(copy(y), I+LC, v, α, β) ≈ (I+MC)*v*α + y*β
        @test (@allocated mul!(w, LC, v, α, β)) == 0
        ILC = I + LC
        @test (@allocated mul!(w, ILC, v, α, β)) == 0
        LCI = LC + I
        @test (@allocated mul!(w, LCI, v, α, β)) == 0
    end
    # @test_throws ErrorException LinearMaps.LinearCombination{ComplexF64}((M, N), (1, 2, 3))
    @test @inferred size(3M + 2.0N) == size(A)
    # addition
    @test @inferred convert(Matrix, LC) == A + B
    @test @inferred convert(Matrix, LC + LC) ≈ 2A + 2B
    @test @inferred convert(Matrix, M + LC) ≈ 2A + B
    @test @inferred convert(Matrix, M + M) ≈ 2A
    # subtraction
    @test (@inferred Matrix(LC - LC)) ≈ zeros(eltype(LC), size(LC)) atol=10eps()
    @test (@inferred Matrix(LC - M)) ≈ B
    @test (@inferred Matrix(N - LC)) ≈ -A
    @test (@inferred Matrix(M - M)) ≈ zeros(size(M)) atol=10eps()
    # scalar multiplication
    @test @inferred Matrix(-M) == -A
    @test @inferred Matrix(-LC) == -A - B
    @test @inferred Matrix(3 * M) == 3 * A
    @test @inferred Matrix(M * 3) == 3 * A
    @test Matrix(3.0 * LC) ≈ Matrix(LC * 3) ≈ 3A + 3B
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
