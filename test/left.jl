using Test, LinearMaps, LinearAlgebra

function left_tester(L::LinearMap{T}) where {T}
    M, N = size(L)
    A = Matrix(L)

    x = rand(T, N)
    y = rand(T, M)

    # *
    @test y' * A ≈ y' * L
    @test (y' * A) * x ≈ y' * (L * x) # associative
    @test transpose(y) * A ≈ transpose(y) * L

    # mul!
    α = rand(T); β = rand(T)
    b1 = y' * A
    b2 = similar(b1)
    bt = copy(b1')'
    # bm = Matrix(bt) # TODO: this requires a generalization of the output to AbstractVecOrMat
    @test mul!(b2, y', L) ≈ mul!(bt, y', L)# ≈ mul!(bm, y', L)# 3-arg
    @test mul!(b2, y', L) === b2
    @test b1 ≈ b2 ≈ bt
    b3 = copy(b2)
    mul!(b3, y', L, α, β)
    @test b3 ≈ b2*β + b1*α

    b1 = transpose(y) * A
    b2 = similar(b1)
    bt = transpose(copy(transpose(b1)))
    @test mul!(b2, transpose(y), L) ≈ mul!(bt, transpose(y), L)
    @test b1 ≈ b2 ≈ bt
    b3 = copy(b2)
    mul!(b3, transpose(y), L, α, β)
    @test b3 ≈ b2*β + b1*α

    Y = rand(T, M, 3)
    X = similar(Y, 3, N)
    Xt = copy(X')'
    @test Y'L isa LinearMap
    @test Matrix(Y'L) ≈ Y'A
    @test mul!(X, Y', L) ≈ mul!(Xt, Y', L) ≈ Y'A
    @test mul!(Xt, Y', L) === Xt
    @test mul!(copy(X), Y', L, α, β) ≈ X*β + Y'A*α
    @test mul!(X, Y', L) === X
    @test mul!(X, Y', L, α, β) === X

    true
end

@testset "left multiplication" begin
    T = ComplexF32
    N = 5
    L = LinearMap{T}(cumsum, reverse ∘ cumsum ∘ reverse, N)

    @test left_tester(L) # FunctionMap
    @test left_tester(L'*L) # CompositeMap
    @test left_tester(2L) # ScaledMap
    @test left_tester(kron(L, L')) # KroneckerMap
    @test left_tester(2L + 3L') # LinearCombination
    @test left_tester([L L]) # BlockMap

    W = LinearMap(randn(T,5,4))
    @test left_tester(W) # WrappedMap
end
