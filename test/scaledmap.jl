#=
scaledmap.jl
Tests for ScaledMap objects
=#

using Test, LinearMaps, LinearAlgebra

@testset "scaledmap" begin
    N = 7
    A = LinearMap(cumsum, reverse ∘ cumsum ∘ reverse, N)

    # real case
    α = 10
    B = α*A
    x = rand(N)

    @test B isa LinearMaps.ScaledMap
    @test @inferred size(B) == size(A)
    @test @inferred isreal(B)
    @test @inferred issymmetric(B) == issymmetric(A)
    @test @inferred ishermitian(B) == ishermitian(A)
    @test @inferred isposdef(B) == isposdef(A)
    @test @inferred transpose(B) == α * transpose(A)

    @test B == A * α
    @test B * x == α * (A * x)
    @test @inferred Matrix(B) == α * Matrix(A)
    @test @inferred Matrix(B') == Matrix(B)'

    @test A / (1/α) == B
    @test -A == (-1) * A
    @test -A isa LinearMaps.ScaledMap

    @test A * (α*I) == B
    @test (α*I) * A == B

    # complex case
    β = 10im
    C = β * A
    T = ComplexF32
    xc = rand(T, N)

    @test C isa LinearMaps.ScaledMap
    @test @inferred !isreal(C)
    @test @inferred !issymmetric(C)
    @test @inferred ishermitian(C) == ishermitian(A)
    @test @inferred !isposdef(C)
    @test @inferred transpose(C) == β * transpose(A)
    @test @inferred adjoint(C) == conj(β) * adjoint(A)

    @test C == A * β
    @test convert(Array{T}, C * xc) ≈ β * (A * xc)
    @test @inferred Matrix(C) == β * Matrix(A)
    @test @inferred Matrix(adjoint(C)) == conj(β) * Matrix(adjoint(A))
    @test @inferred Matrix(C') == Matrix(C)'

    # composition
    BC = B * C
    @test BC isa LinearMaps.ScaledMap
    @test Matrix(BC) == α*β*Matrix(A)*Matrix(A)

    # in-place
    y1 = β * (A * xc)
    y2 = similar(y1)
    mul!(y2, C, xc)
    @test y2 == y1

    y1 = β * (A * xc)
    y2 = similar(y1)
    mul!(y2, C, xc)
    @test y2 == y1

    x1 = conj(β) * A'*y1
    x2 = similar(x1)
    mul!(x2, C', y1)
    @test x2 == x1

    # allocation (WIP)
#=
    function cumsum_adj!(x, y, work)
        return reverse!(cumsum!(x, reverse!(copyto!(work,y))))
    end
    function cumsum_adj_make(x)
        work = similar(x)
        return (x,y) -> cumsum_adj!(x, y, work)
    end
    adj! = cumsum_adj_make(xc)
    Ai = LinearMap(cumsum!, adj!, N)
    @test Matrix(Ai) == Matrix(A)

    function alloc_test_forw!(y, A, x)
        mul!(y, A, x)
    end
    function alloc_test_back(x, A, y)
        mul!(x, A', y)
    end
#   @show (@allocated alloc_test_forw!(y1, A, xc))
#   @show (@allocated alloc_test_forw!(y1, Ai, xc)) # why nonzero?

    using BenchmarkTools
#   @btime mul!($y1, $A, $xc)
#   @btime mul!($y1, $Ai, $xc) # 0

#   @allocated mul!($y1, $A, $x)
#   @allocated mul!($y1, $Ai, $x)

#   @btime mul!($x1, $A', $y1)
#   @btime mul!($x1, $Ai', $y1) # ~0

    @btime mul!($y1, $C, $xc)
    C = β * Ai
    @btime mul!($y1, $C, $xc)
=#
end
