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
    bm = Matrix(bt)
    @test mul!(b2, y', L) ≈ mul!(bt, y', L) ≈ mul!(bm, y', L)
    @test mul!(b2, y', L) === b2
    @test b1 ≈ b2 ≈ bt
    @test mul!(copy(b2), y', L, α, β) ≈ b2*β + y'A*α

    b1 = transpose(y) * A
    b2 = similar(b1)
    bt = transpose(copy(transpose(b1)))
    @test mul!(b2, transpose(y), L) ≈ mul!(bt, transpose(y), L)
    @test b1 ≈ b2 ≈ bt
    @test mul!(copy(b2), transpose(y), L, α, β) ≈ b2*β + transpose(y)*A*α

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

    @test transpose(Y)*L isa LinearMap
    @test Matrix(transpose(Y)*L) ≈ transpose(Y)*A
    @test mul!(X, transpose(Y), L) ≈ mul!(Xt, transpose(Y), L) ≈ transpose(Y)*A
    @test mul!(Xt, transpose(Y), L) === Xt
    @test mul!(copy(X), transpose(Y), L, α, β) ≈ X*β + transpose(Y)*A*α
    @test mul!(X, transpose(Y), L) === X
    @test mul!(X, transpose(Y), L, α, β) === X

    return true
end

@testset "left multiplication" begin
    T = ComplexF32
    N = 5
    L = LinearMap{T}(cumsum, reverse ∘ cumsum ∘ reverse, N)

    @test left_tester(L) # FunctionMap
    @test left_tester(adjoint(L)) # AdjointMap
    @test left_tester(transpose(L)) # TransposeMap
    @test left_tester(L'*L) # CompositeMap
    @test left_tester(2rand(T)*L) # ScaledMap
    @test left_tester(kron(L, L')) # KroneckerMap
    @test left_tester(2rand(T)*L + 3rand(T)*L') # LinearCombination
    @test left_tester([L L]) # BlockMap

    W = LinearMap(randn(T,5,4)) # WrappedMap
    @test left_tester(W)
    @test left_tester(W')
    @test left_tester(transpose(W))
    
    J = LinearMap(I, 5) # UniformScalingMap
    @test left_tester(J)
end
