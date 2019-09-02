using Test, LinearMaps, LinearAlgebra, Quaternions

@testset "noncommutative number type" begin
    x = Quaternion.(rand(10), rand(10), rand(10), rand(10))
    v = rand(10)
    A = Quaternion.(rand(10,10), rand(10,10), rand(10,10), rand(10,10))
    α = UniformScaling(Quaternion.(rand(4)...))
    β = UniformScaling(Quaternion.(rand(4)...))
    L = LinearMap(A)
    @test Array(L) == A
    @test Array(L') == A'
    @test Array(transpose(L)) == transpose(A)
    @test Array(α * L) == α * A
    @test Array(L * α) == A * α
    @test Array(α * L) == α * A
    @test Array(L * α ) == A * α
    @test Array(α * L') == α * A'
    @test Array((α * L')') ≈ (α * A')' ≈ A * conj(α)
    @test L * x ≈ A * x
    @test L' * x ≈ A' * x
    @test α * (L * x) ≈ α * (A * x)
    @test α * L * x ≈ α * A * x
    @test (α * L') * x ≈ (α * A') * x
    @test (α * L')' * x ≈ (α * A')' * x
    @test (α * L')' * v ≈ (α * A')' * v
    @test Array(@inferred adjoint(α * L * β)) ≈ conj(β) * A' * conj(α)
    @test Array(@inferred transpose(α * L * β)) ≈ β * transpose(A) * α
end

@testset "nonassociative number type" begin
    x = Octonion.(rand(10), rand(10), rand(10), rand(10),rand(10), rand(10), rand(10), rand(10))
    v = rand(10)
    A = Octonion.(rand(10,10), rand(10,10), rand(10,10), rand(10,10),rand(10,10), rand(10,10), rand(10,10), rand(10,10))
    α = UniformScaling(Octonion.(rand(8)...))
    β = UniformScaling(Octonion.(rand(8)...))
    L = LinearMap(A)
    @test Array(L) == A
    @test Array(α * L) == α * A
    @test Array(L * α) == A * α
    @test Array(α * L) == α * A
    @test Array(L * α) == A * α
    @test (α * L')' * v ≈ (α * A')' * v
end
