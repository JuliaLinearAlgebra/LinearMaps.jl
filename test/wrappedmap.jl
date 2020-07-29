using Test, LinearMaps, LinearAlgebra

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
end

# y'*L is an exception to the left multiplication rule that makes a WrappedMap
@testset "left mul vec" begin
    T = ComplexF32
    A = rand(T, 4, 5)
    x = rand(T, 5)
    y = rand(T, 4)
    L = LinearMap{T}(A)
    @test y'A ≈ y'L
    @test (y' * A) * x ≈ y' * (L * x) # associative
    @test transpose(y) * A ≈ transpose(y) * L

    LL = L * [L' L'] # stress test with CompositeMap & BlockMap
    AA = A * [A' A']
    x = rand(T, 8)
    @test y'AA ≈ y'LL
    @test (y' * AA) * x ≈ y' * (LL * x) # associative
    @test transpose(y)*AA ≈ transpose(y)*LL

    # mul! versions
    b1 = y'*L
    b2 = similar(b1)
    mul!(b2, y', A)
    @test b1 ≈ b2

    b1 = transpose(y)*L
    b2 = similar(b1)
    mul!(b2, transpose(y), A)
    @test b1 ≈ b2
end
