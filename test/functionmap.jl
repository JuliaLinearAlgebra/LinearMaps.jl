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
    @test occursin("$N×$N FunctionMap{$(eltype(MyFT)),false}", sprint((t, s) -> show(t, "text/plain", s), MyFT))

    CS = LinearMap(cumsum, 2)
    @test size(CS) == (2, 2)
    @test @inferred !issymmetric(CS)
    @test @inferred !ishermitian(CS)
    @test @inferred !isposdef(CS)
    @test (@test_deprecated (@inferred LinearMaps.ismutating(CS))) == false
    @test @inferred Matrix(CS) == [1. 0.; 1. 1.]
    @test @inferred Array(CS) == [1. 0.; 1. 1.]
    CS = @inferred (() -> LinearMap(cumsum, 10; ismutating=false))()
    v = rand(10)
    cv = cumsum(v)
    @test CS * v == cv
    @test *(CS, v) == cv
    @test_throws ErrorException CS' * v
    CS = LinearMap(cumsum, x -> reverse(cumsum(reverse(x))), 10; ismutating=false)
    @test occursin("10×10 FunctionMap{Float64,false}", sprint((t, s) -> show(t, "text/plain", s), CS))
    cv = cumsum(v)
    @test @inferred CS * v == cv
    @test @inferred *(CS, v) == cv
    @test @inferred CS' * v == reverse!(cumsum(reverse(v)))
    @test @inferred mul!(similar(v), transpose(CS), v) == reverse!(cumsum(reverse(v)))

    CS! = LinearMap(cumsum!, 10; ismutating=true)
    @test (@test_deprecated (@inferred LinearMaps.ismutating(CS!)))
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test_throws ErrorException CS!'v
    @test_throws ErrorException transpose(CS!) * v

    CS! = LinearMap{ComplexF64}(cumsum!, 10; ismutating=true)
    @test CS! == FunctionMap{ComplexF64, true}(cumsum!, 10, 10)
    v = rand(ComplexF64, 10)
    cv = cumsum(v)
    @test (@test_deprecated (@inferred LinearMaps.ismutating(CS!)))
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
    @test (@test_deprecated (@inferred LinearMaps.ismutating(CS!)))
    @test @inferred CS! * v == cv
    @test @inferred *(CS!, v) == cv
    @test @inferred mul!(similar(v), CS!, v) == cv
    @test @inferred CS' * v == reverse!(cumsum(reverse(v)))
    @test @inferred mul!(similar(v), transpose(CS), v) == reverse!(cumsum(reverse(v)))
    @test @inferred mul!(similar(v), adjoint(CS), v) == reverse!(cumsum(reverse(v)))
    u = similar(v)
    CS!3 = 3*CS!
    mul!(u, CS!3, v)
    @test (@allocated mul!(u, CS!3, v)) == 0
    CS!3t = 3*CS!'
    mul!(u, CS!3t, v)
    @test (@allocated mul!(u, CS!3t, v)) == 0
    u = rand(ComplexF64, 10)
    v = rand(ComplexF64, 10)
    for α in (false, true, rand(ComplexF64)), β in (false, true, rand(ComplexF64))
        for transform in (identity, adjoint, transpose)
            @test mul!(copy(v), transform(CS!), u, α, β) ≈ transform(M)*u*α + v*β
            @test mul!(copy(v), transform(LinearMap(CS!)), u, α, β) ≈ transform(M)*u*α + v*β
            @test mul!(copy(v), LinearMap(transform(CS!)), u, α, β) ≈ transform(M)*u*α + v*β
            if transform != transpose
                transCS! = transform(CS!)
                alloc = @allocated similar(v)
                @test (@allocated mul!(v, transCS!, u, α, β)) <= alloc
            end
        end
    end

    # Test fallback methods:
    L = LinearMap(x -> x, x -> x, 10)
    v = randn(10)
    @test @inferred (2 * L)' * v ≈ 2 * v
    @test @inferred transpose(2 * L) * v ≈ 2 * v
    L = @inferred FunctionMap{ComplexF64,false}(x -> x, x -> x, 10)
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
