using Test, LinearMaps, LinearAlgebra

@testset "function maps" begin
    N = 100
    function myft(v::AbstractVector)
        # not so fast fourier transform
        N = length(v)
        w = zeros(complex(eltype(v)), N)
        for k = 1:N
            kappa = (2*(k-1)/N)*pi
            for n = 1:N
                w[k] += v[n]*exp(kappa*(n-1)*im)
            end
        end
        return w
    end
    MyFT = @inferred LinearMap{ComplexF64}(myft, N) / sqrt(N)
    U = Matrix(MyFT) # will be a unitary matrix
    @test @inferred U'U ≈ Matrix{eltype(U)}(I, N, N)

    CS = @inferred LinearMap(cumsum, 2)
    @test size(CS) == (2, 2)
    @test @inferred !issymmetric(CS)
    @test @inferred !ishermitian(CS)
    @test @inferred !isposdef(CS)
    @test @inferred !(LinearMaps.ismutating(CS))
    @test @inferred Matrix(CS) == [1. 0.; 1. 1.]
    @test @inferred Array(CS) == [1. 0.; 1. 1.]
    CS = @inferred LinearMap(cumsum, 10; ismutating=false)
    v = rand(10)
    cv = cumsum(v)
    @test CS * v == cv
    @test *(CS, v) == cv
    @test_throws ErrorException CS' * v
    CS = @inferred LinearMap(cumsum, x -> reverse(cumsum(reverse(x))), 10; ismutating=false)
    cv = cumsum(v)
    @test @inferred CS * v == cv
    @test @inferred *(CS, v) == cv
    @test @inferred CS' * v == reverse!(cumsum(reverse(v)))
    @test @inferred mul!(similar(v), transpose(CS), v) == reverse!(cumsum(reverse(v)))

    CS! = @inferred LinearMap(cumsum!, 10; ismutating=true)
    @test @inferred LinearMaps.ismutating(CS!)
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test_throws ErrorException CS!'v
    @test_throws ErrorException transpose(CS!) * v

    CS! = @inferred LinearMap{ComplexF64}(cumsum!, 10; ismutating=true)
    v = rand(ComplexF64, 10)
    cv = cumsum(v)
    @test @inferred LinearMaps.ismutating(CS!)
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test_throws ErrorException CS!'v
    @test_throws ErrorException adjoint(CS!) * v
    CS! = LinearMap{ComplexF64}(cumsum!, (y, x) -> (copyto!(y, x); reverse!(y); cumsum!(y, y)), 10; ismutating=true)
    @inferred adjoint(CS!)
    @test @inferred LinearMaps.ismutating(CS!)
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test @inferred CS' * v == reverse!(cumsum(reverse(v)))
    @test @inferred mul!(similar(v), transpose(CS), v) == reverse!(cumsum(reverse(v)))
    @test @inferred mul!(similar(v), adjoint(CS), v) == reverse!(cumsum(reverse(v)))

    # Test fallback methods:
    L = @inferred LinearMap(x -> x, x -> x, 10)
    v = randn(10)
    @test @inferred (2 * L)' * v ≈ 2 * v
    @test @inferred transpose(2 * L) * v ≈ 2 * v
    L = @inferred LinearMap{ComplexF64}(x -> x, x -> x, 10)
    v = rand(ComplexF64, 10)
    @test @inferred (2 * L)' * v ≈ 2 * v
    @test @inferred transpose(2 * L) * v ≈ 2 * v
end
