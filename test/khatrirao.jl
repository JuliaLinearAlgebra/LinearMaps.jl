using Test, LinearMaps, LinearAlgebra

@testset "KhatriRaoMap & FaceSplittingMap" begin
    for trans in (identity, complex)
        A = collect(reshape(trans(1:6), 3, 2))
        B = collect(reshape(trans(1:8), 4, 2))
        K = @inferred khatrirao(A, B)
        @test facesplitting(A', B')' === K
        M = mapreduce(kron, hcat, eachcol(A), eachcol(B))
        Mx = mapreduce((a, b) -> kron(permutedims(a), permutedims(b)), vcat, eachrow(A'), eachrow(B'))
        @test size(K) == size(M)
        @test size(@inferred adjoint(K)) == reverse(size(K))
        @test size(@inferred transpose(K)) == reverse(size(K))
        @test Matrix(K) == M
        @test Matrix(K') == Mx
        @test (K')' === K
        @test transpose(transpose(K)) === K
        x = trans(rand(-10:10, size(K, 2)))
        y = trans(rand(-10:10, size(K, 1)))
        for α in (false, true, trans(rand(2:5))), β in (false, true, trans(rand(2:5)))
            @test mul!(copy(y), K, x, α, β) == y * β + K * x * α
            @test mul!(copy(x), K', y, α, β) == x * β + K' * y * α
        end
    end
end
