using Flux, LinearAlgebra

@testset "AD rules" begin
    A = LinearMap(rand(10, 10))
    x = randn(10)
    g1 = A'*A*x
    # Multiplication rule
    g2 = gradient(x -> .5*norm(A*x)^2, x)
    @test g1 ≈ g2[1]
    # Call rule
    g3 = gradient(x -> .5*norm(A(x))^2, x)
    @test g1 ≈ g3[1]

    g1 = A*A'*x
    # Multiplication rule
    g2 = gradient(x -> .5*norm(A'*x)^2, x)
    @test g1 ≈ g2[1]
    # Call rule
    g3 = gradient(x -> .5*norm(A'(x))^2, x)
    @test g1 ≈ g3[1]
end
