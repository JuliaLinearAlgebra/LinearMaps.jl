using Test, LinearMaps
using LinearAlgebra: mul!

# y'*L is an exception to the left multiplication rule that makes a WrappedMap

function left_tester(L::LinearMap{T}) where {T}
	M,N = size(L)
    A = Matrix(L)

    x = rand(T, N)
    y = rand(T, M)

	# *
    @test y' * A ≈ y' * L
    @test (y' * A) * x ≈ y' * (L * x) # associative
    @test transpose(y) * A ≈ transpose(y) * L

    # mul!
    b1 = y' * A
    b2 = similar(b1)
    mul!(b2, y', L) # 3-arg
    @test b1 ≈ b2
    mul!(b2, y', L, true, false) # 5-arg
    @test b1 ≈ b2

    b1 = transpose(y) * A
    b2 = similar(b1)
    mul!(b2, transpose(y), L) # 3-arg
    @test b1 ≈ b2
    mul!(b2, transpose(y), L, true, false) # 5-arg
    @test b1 ≈ b2

	true
end


@testset "left mul vec" begin
    T = ComplexF32
	M,N = 5,5
	L = LinearMap{T}(cumsum, reverse ∘ cumsum ∘ reverse, N)

	@test left_tester(L) # FunctionMap
	@test left_tester(L'*L) # CompositeMap
	@test left_tester(2L) # ScaledMap
	@test left_tester(kron(L,L')) # KroneckerMap

#=
todo: fails with DimensionMismatch
	W = LinearMap(randn(T,5,4)) #
	@test left_tester(W) # WrappedMap
=#

#=
todo: fails with stack overflow
	@test left_tester([L L]) # BlockMap
	@test left_tester(2L+3L') # LinearCombination
=#
end
