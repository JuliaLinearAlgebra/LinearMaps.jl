using Test, LinearMaps, LinearAlgebra, SparseArrays

@testset "transpose/adjoint" begin
    CS = @inferred LinearMap{ComplexF64}(cumsum, x -> reverse(cumsum(reverse(x))), 10; ismutating=false)
    for transform in (adjoint, transpose)
        @test transform(CS) != CS
        @test CS != transform(CS)
        @test transform(transform(CS)) == CS
        @test LinearMaps.MulStyle(transform(CS)) === LinearMaps.MulStyle(CS)
    end
    @test occursin("10×10 LinearMaps.TransposeMap{$(eltype(CS))}", sprint((t, s) -> show(t, "text/plain", s), transpose(CS)))
    @test occursin("10×10 LinearMaps.AdjointMap{$(eltype(CS))}", sprint((t, s) -> show(t, "text/plain", s), adjoint(CS)))
    @test !(transpose(CS) == adjoint(CS))
    @test !(adjoint(CS) == transpose(CS))
    M = Matrix(CS)
    @test M == LowerTriangular(ones(10,10)) == mul!(copy(M), CS, 1, true, false)
    x = rand(ComplexF64, 10); w = rand(ComplexF64, 10)
    X = rand(ComplexF64, 10, 3); W = rand(ComplexF64, 10, 3)
    α, β = rand(ComplexF64, 2)
    for transform in (adjoint, transpose)
        @test convert(AbstractMatrix, transform(CS)) == transform(M)
        @test sparse(transform(CS)) == transform(M)
        @test transform(CS) * x ≈ transform(M)*x
        @test mul!(copy(w), transform(CS), x) ≈ transform(M)*x
        @test mul!(copy(W), transform(CS), X) ≈ transform(M)*X
        @test mul!(copy(w), transform(CS), x, α, β) ≈ transform(M)*x*α + w*β
        @test mul!(copy(W), transform(CS), X, α, β) ≈ transform(M)*X*α + W*β
    end
    for transform1 in (adjoint, transpose), transform2 in (adjoint, transpose)
        @test transform1(transform1(CS)) === CS
        @test transform2(transform1(transform2(CS))) === transform1(CS)
        @test transform2(transform1(CS)) * x ≈ transform2(transform1(M))*x
        @test mul!(copy(w), transform2(transform1(CS)), x) ≈ transform2(transform1(M))*x
        @test mul!(copy(W), transform2(transform1(CS)), X) ≈ transform2(transform1(M))*X
        @test mul!(copy(M), transform2(transform1(CS)), 2) ≈ transform2(transform1(M))*2
        @test mul!(copy(w), transform2(transform1(CS)), x, α, β) ≈ transform2(transform1(M))*x*α + w*β
        @test mul!(copy(W), transform2(transform1(CS)), X, α, β) ≈ transform2(transform1(M))*X*α + W*β
        @test mul!(copy(M), transform2(transform1(CS)), 2, α, β) ≈ transform2(transform1(M))*2*α + M*β
    end

    id = @inferred LinearMap(identity, identity, 10; issymmetric=true, ishermitian=true, isposdef=true)
    for transform in (adjoint, transpose)
        @test transform(id) == id
        @test id == transform(id)
        for prop in (issymmetric, ishermitian, isposdef)
            @test prop(transform(id))
        end
    end
end
