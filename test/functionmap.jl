using Test, LinearMaps, LinearAlgebra, BenchmarkTools

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
    @test occursin("$N×$N LinearMaps.FunctionMap{$(eltype(MyFT))}", sprint((t, s) -> show(t, "text/plain", s), MyFT))
    @test parent(LinearMap{ComplexF64}(myft, N)) === (myft, nothing)

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
    @test occursin("10×10 LinearMaps.FunctionMap{Float64}", sprint((t, s) -> show(t, "text/plain", s), CS))
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
    @test_throws ErrorException mul!(similar(v), CS!', v)
    @test_throws ErrorException mul!(similar(v), transpose(CS!), v)
    CS! = LinearMap{ComplexF64}(cumsum!, (y, x) -> (copyto!(y, x); reverse!(y); cumsum!(y, y); reverse!(y)), 10; ismutating=true)
    M = Matrix(CS!)
    @inferred adjoint(CS!)
    @test @inferred LinearMaps.ismutating(CS!)
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test @inferred CS' * v == reverse!(cumsum(reverse(v)))
    @test @inferred mul!(similar(v), transpose(CS), v) == reverse!(cumsum(reverse(v)))
    @test @inferred mul!(similar(v), adjoint(CS), v) == reverse!(cumsum(reverse(v)))
    u = similar(v)
    b = @benchmarkable mul!($u, $(3*CS!), $v)
    @test run(b, samples=3).allocs == 0
    b = @benchmarkable mul!($u, $(3*CS!'), $v)
    @test run(b, samples=3).allocs == 0
    u = rand(ComplexF64, 10)
    v = rand(ComplexF64, 10)
    for α in (false, true, rand(ComplexF64)), β in (false, true, rand(ComplexF64))
        for transform in (identity, adjoint, transpose)
            @test mul!(copy(v), transform(CS!), u, α, β) ≈ transform(M)*u*α + v*β
            @test mul!(copy(v), transform(LinearMap(CS!)), u, α, β) ≈ transform(M)*u*α + v*β
            @test mul!(copy(v), LinearMap(transform(CS!)), u, α, β) ≈ transform(M)*u*α + v*β
            if transform != transpose
                bm = @benchmarkable mul!($(copy(v)), $(transform(CS!)), $u, $α, $β)
                @test run(bm, samples=3).allocs <= 1
            end
        end
    end

    # Test fallback methods:
    L = @inferred LinearMap(x -> x, x -> x, 10)
    v = randn(10)
    @test @inferred (2 * L)' * v ≈ 2 * v
    @test @inferred transpose(2 * L) * v ≈ 2 * v
    L = @inferred LinearMap{ComplexF64}(x -> x, x -> x, 10)
    v = rand(ComplexF64, 10)
    w = similar(v)
    @test @inferred (2 * L)' * v ≈ 2 * v
    @test @inferred transpose(2 * L) * v ≈ 2 * v

    A = rand(ComplexF64, 10, 10)
    L = LinearMap{ComplexF64}(x -> A*x, 10)
    @test L * v == A * v == mul!(w, L, v)
    L = LinearMap{ComplexF64}((y, x) -> mul!(y, A, x), 10)
    @test L * v == A * v == mul!(w, L, v)
    L = LinearMap{ComplexF64}((y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x), 10)
    @test L * v == A * v == mul!(w, L, v)
    @test adjoint(L) * v ≈ A'v ≈ mul!(w, L', v)
    @test transpose(L) * v ≈ transpose(A)*v ≈ mul!(w, transpose(L), v)

    A = Symmetric(rand(ComplexF64, 10, 10))
    L = LinearMap{ComplexF64}(x -> A*x, 10; issymmetric=true)
    @test L * v == A * v == mul!(w, L, v)
    @test adjoint(L) * v ≈ A'v ≈ mul!(w, L', v)
    @test transpose(L) * v ≈ transpose(A)*v ≈ mul!(w, transpose(L), v)
    L = LinearMap{ComplexF64}((y, x) -> mul!(y, A, x), 10; issymmetric=true)
    @test L * v == A * v == mul!(w, L, v)
    @test adjoint(L) * v ≈ A'v ≈ mul!(w, L', v)
    @test transpose(L) * v ≈ transpose(A)*v ≈ mul!(w, transpose(L), v)

    A = Hermitian(rand(ComplexF64, 10, 10))
    L = LinearMap{ComplexF64}(x -> A*x, 10; ishermitian=true)
    @test L * v == A * v == mul!(w, L, v)
    @test adjoint(L) * v ≈ A'v ≈ mul!(w, L', v)
    @test transpose(L) * v ≈ transpose(A)*v ≈ mul!(w, transpose(L), v)
    L = LinearMap{ComplexF64}((y, x) -> mul!(y, A, x), 10; ishermitian=true)
    @test L * v == A * v == mul!(w, L, v)
    @test adjoint(L) * v ≈ A'v ≈ mul!(w, L', v)
    @test transpose(L) * v ≈ transpose(A)*v ≈ mul!(w, transpose(L), v)
end
