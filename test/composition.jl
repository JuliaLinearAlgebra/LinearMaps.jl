using Test, LinearMaps, LinearAlgebra

# new type
struct SimpleFunctionMap <: LinearMap{Float64}
    f::Function
    N::Int
end
struct SimpleComplexFunctionMap <: LinearMap{Complex{Float64}}
    f::Function
    N::Int
end
Base.size(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}) = (A.N, A.N)
Base.:(*)(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, v::Vector) = A.f(v)
LinearAlgebra.mul!(y::Vector, A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, x::Vector) = copyto!(y, *(A, x))

@testset "composition" begin
    F = @inferred LinearMap(cumsum, y -> reverse(cumsum(reverse(x))), 10; ismutating=false)
    FC = @inferred LinearMap{ComplexF64}(cumsum, y -> reverse(cumsum(reverse(x))), 10; ismutating=false)
    A = 2 * rand(ComplexF64, (10, 10)) .- 1
    B = rand(size(A)...)
    M = @inferred 1 * LinearMap(A)
    N = @inferred LinearMap(B)
    v = rand(ComplexF64, 10)
    @test @inferred (F * F) * v == @inferred F * (F * v)
    @test @inferred (F * A) * v == @inferred F * (A * v)
    @test @inferred (A * F) * v == @inferred A * (F * v)
    @test @inferred A * (F * F) * v == @inferred A * (F * (F * v))
    @test @inferred (F * F) * (F * F) * v == @inferred F * (F * (F * (F * v)))
    @test @inferred Matrix(M * transpose(M)) ≈ A * transpose(A)
    @test @inferred !isposdef(M * transpose(M))
    @test @inferred isposdef(M * M')
    @test @inferred issymmetric(N * N')
    @test @inferred ishermitian(N * N')
    @test @inferred !issymmetric(M' * M)
    @test @inferred ishermitian(M' * M)
    @test @inferred issymmetric(F'F)
    @test @inferred ishermitian(F'F)
    @test @inferred !issymmetric(FC'FC)
    @test @inferred ishermitian(FC'FC)
    @test @inferred isposdef(transpose(F) * F * 3)
    @test @inferred isposdef(transpose(F) * 3 * F)
    @test @inferred !isposdef(-5*transpose(F) * F)
    @test @inferred isposdef((M * F)' * M * 4 * F)
    @test @inferred transpose(M * F) == @inferred transpose(F) * transpose(M)
    @test @inferred (4*((-3*M)*2)) == @inferred -12M*2
    @test @inferred (4*((3*(-M))*2)*(-5)) == @inferred -12M*(-10)
    L = @inferred 3 * F + 1im * A + F * M' * F
    LF = 3 * Matrix(F) + 1im * A + Matrix(F) * Matrix(M)' * Matrix(F)
    @test Array(L) ≈ LF
    R1 = rand(ComplexF64, 10, 10); L1 = LinearMap(R1)
    R2 = rand(ComplexF64, 10, 10); L2 = LinearMap(R2)
    R3 = rand(ComplexF64, 10, 10); L3 = LinearMap(R3)
    CompositeR = prod(R -> LinearMap(R), [R1, R2, R3])
    @test @inferred L1 * L2 * L3 == CompositeR
    @test @inferred transpose(CompositeR) == transpose(L3) * transpose(L2) * transpose(L1)
    @test @inferred adjoint(CompositeR) == L3' * L2' * L1'
    @test @inferred adjoint(adjoint((CompositeR))) == CompositeR
    @test transpose(transpose((CompositeR))) == CompositeR
    Lt = @inferred transpose(LinearMap(CompositeR))
    @test Lt * v ≈ transpose(R3) * transpose(R2) * transpose(R1) * v
    Lc = @inferred adjoint(LinearMap(CompositeR))
    @test Lc * v ≈ R3' * R2' * R1' * v

    # test inplace operations
    w = similar(v)
    mul!(w, L, v)
    @test w ≈ LF * v

    # test new type
    F = SimpleFunctionMap(cumsum, 10)
    FC = SimpleComplexFunctionMap(cumsum, 10)
    @test @inferred ndims(F) == 2
    @test @inferred size(F, 1) == 10
    @test @inferred length(F) == 100
    @test @inferred !issymmetric(F)
    @test @inferred !ishermitian(F)
    @test @inferred !ishermitian(FC)
    @test @inferred !isposdef(F)
    w = similar(v)
    mul!(w, F, v)
    @test w == F * v
    @test_throws MethodError F' * v
    @test_throws MethodError transpose(F) * v
    @test_throws MethodError mul!(w, adjoint(F), v)
    @test_throws MethodError mul!(w, transpose(F), v)

    # test composition of several maps with shared data #31
    global sizes = ( (5, 2), (3, 3), (3, 2), (2, 2), (9, 2), (7, 1) )
    N = length(sizes) - 1
    global Lf = []
    global Lt = []
    global Lc = []

    # build list of operators [LN, ..., L2, L1] for each mode
    for (fi, i) in [ (Symbol("f$i"), i) for i in 1:N]
        @eval begin
            function ($fi)(source)
                dest = ones(prod(sizes[$i + 1]))
                tmp = reshape(source, sizes[$i])
                return conj.($i * dest)
            end
            insert!(Lf, 1, LinearMap($fi, prod(sizes[$i + 1]), prod(sizes[$i])))
            insert!(Lt, 1, LinearMap(x -> x, $fi, prod(sizes[$i]), prod(sizes[$i + 1])))
            insert!(Lc, 1, LinearMap{ComplexF64}(x -> x, $fi, prod(sizes[$i]), prod(sizes[$i + 1])))
        end
    end
    @test size(prod(Lf[1:N])) == (prod(sizes[end]), prod(sizes[1]))
    @test isreal(prod(Lf[1:N]))
    # multiply as composition and as recursion
    v1 = ones(prod(sizes[1]))
    u1 = ones(prod(sizes[1]))
    w1 = im.*ones(ComplexF64, prod(sizes[1]))
    for i = N:-1:1
        v2 = prod(Lf[i:N]) * ones(prod(sizes[1]))
        u2 = transpose(LinearMap(prod(Lt[N:-1:i]))) * ones(prod(sizes[1]))
        w2 = adjoint(LinearMap(prod(Lc[N:-1:i]))) * ones(prod(sizes[1]))

        v1 = Lf[i] * v1
        u1 = transpose(Lt[i]) * u1
        w1 = adjoint(Lc[i]) * w1

        @test v1 == v2
        @test u1 == u2
        @test w1 == w2
    end
end
