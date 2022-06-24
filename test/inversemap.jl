using Test, LinearMaps, LinearAlgebra, SparseArrays, IterativeSolvers

@testset "inversemap" begin
    # First argument to cg!/gmres! doubles as the initial guess so we make sure
    # it is 0 instead of potential garbage since we don't control the
    # allocation of the output vector.
    cgz! = (x, A, b) -> IterativeSolvers.cg!(fill!(x, 0), A, b)
    gmresz! = (x, A, b) -> IterativeSolvers.gmres!(fill!(x, 0), A, b)

    # Dense test data
    A = rand(10, 10) + 5I; A = A'A
    B = rand(10, 10)
    b = rand(10)
    ## LU
    @test A \ b ≈ InverseMap(lu(A)) * b
    ## Cholesky
    @test A \ b ≈ InverseMap(cholesky(A)) * b
    ## Specify solver
    @test A \ b ≈ InverseMap(A; solver=cgz!) * b atol=1e-8
    ## Composition with other maps
    @test A \ B * b ≈ InverseMap(lu(A)) * B * b
    ## Composition: make sure solvers called with vector B * b and not matrix B
    my_ldiv! = (y, A, x) -> begin
        @test x isa AbstractVector
        @test x ≈ B * b
        return ldiv!(y, lu(A), x)
    end
    @test A \ B * b ≈ InverseMap(A; solver=my_ldiv!) * B * b
    ## 3- and 5-arg mul!
    iA = InverseMap(lu(A))
    y = zeros(size(A, 1))
    mul!(y, iA, b)
    @test A \ b ≈ y
    Y = zeros(size(A))
    mul!(Y, iA, B)
    @test A \ B ≈ Y
    y = rand(size(A, 1))
    yc = copy(y)
    α, β = 1.2, 3.4
    mul!(y, iA, b, α, β)
    @test A \ b * α + yc * β ≈ y
    Y = rand(size(A)...)
    Yc = copy(Y)
    mul!(Y, iA, B, α, β)
    @test A \ B * α + Yc * β ≈ Y

    # Sparse test data
    A = sprand(10, 10, 0.2) + 5I; A = A'A
    B = sprand(10, 10, 0.5)
    @test A \ b ≈ InverseMap(lu(A)) * b
    ## Cholesky (CHOLMOD doesn't support inplace ldiv!)
    my_ldiv! = (y, A, x) -> copy!(y, A \ x)
    @test A \ b ≈ InverseMap(cholesky(A); solver=my_ldiv!) * b
    ## Specify solver
    @test A \ b ≈ InverseMap(A; solver=cgz!) * b atol=1e-8
    ## Composition with other maps
    @test A \ (B * b) ≈ InverseMap(lu(A)) * B * b
    ## Composition: make sure solver is called with vector B * b and not matrix B
    my_ldiv! = (y, A, x) -> begin
        @test x isa AbstractVector
        @test x ≈ B * b
        return ldiv!(y, lu(A), x)
    end
    @test A \ (B * b) ≈ InverseMap(A; solver=my_ldiv!) * B * b
    ## 3- and 5-arg mul!
    iA = InverseMap(lu(A))
    y = zeros(size(A, 1))
    mul!(y, iA, b)
    @test A \ b ≈ y
    y = rand(size(A, 1))
    yc = copy(y)
    α, β = 1.2, 3.4
    mul!(y, iA, b, α, β)
    @test A \ b * α + yc * β ≈ y

    # Combine with another LinearMap
    A = LinearMap(cumsum, 10, 10)
    iA = InverseMap(A; solver=gmresz!)
    y = zeros(size(A, 1))
    mul!(y, iA, b)
    @test IterativeSolvers.gmres(A, b) ≈ iA * b ≈ y
    y = rand(size(A, 1))
    yc = copy(y)
    α, β = 1.2, 3.4
    mul!(y, iA, b, α, β)
    @test IterativeSolvers.gmres(A, b * α) + β * yc ≈ iA * b * α + yc * β ≈ y

    # Interface testing: note that not all combinations of factorization and
    # is(symmetric|hermitian|posdef) and transpose/adjoint are supported by LinearAlgebra,
    # so we test this for a custom type just to make sure the call is forwarded correctly
    # and then run some tests for supported combinations.
    struct TestMap{T} <: LinearMap{T}
        A::Matrix{T}
    end
    Base.size(tm::TestMap) = size(tm.A)
    Base.transpose(tm::TestMap) = transpose(tm.A)
    Base.adjoint(tm::TestMap) = adjoint(tm.A)
    LinearAlgebra.issymmetric(tm::TestMap) = issymmetric(tm.A)
    LinearAlgebra.ishermitian(tm::TestMap) = ishermitian(tm.A)
    LinearAlgebra.isposdef(tm::TestMap) = isposdef(tm.A)
    A = [5.0 2.0; 2.0 4.0]
    itm = InverseMap(TestMap(A))
    @test size(itm) == size(A)
    @test transpose(itm).A === transpose(A)
    @test adjoint(itm).A === adjoint(A)
    @test issymmetric(itm)
    @test ishermitian(itm)
    @test isposdef(itm)
    ## Real symmetric (and Hermitian)
    A = Float64[3 2; 2 5]; x = rand(2)
    ### Wrapping a matrix and factorize in solver
    iA = InverseMap(A; solver=(y, A, x)->ldiv!(y, cholesky(A), x))
    @test ishermitian(A) == ishermitian(iA) == true
    @test issymmetric(A) == issymmetric(iA) == true
    @test isposdef(A) == isposdef(iA) == true
    @test A \ x ≈ iA * x
    if VERSION >= v"1.8.0-"
        @test transpose(A) \ x ≈ transpose(iA) * x
        @test adjoint(A) \ x ≈ adjoint(iA) * x
    end
    ### Wrapping a factorization
    iA = InverseMap(cholesky(A))
    # @test ishermitian(A) == ishermitian(iA) == true
    # @test issymmetric(A) == issymmetric(iA) == true
    @test isposdef(A) == isposdef(iA) == true
    @test A \ x ≈ iA * x
    # @test transpose(A) \ x ≈ transpose(iA) * x
    if VERSION >= v"1.7.0"
        @test adjoint(A) \ x ≈ adjoint(iA) * x
    end
    ## Real non-symmetric
    A = Float64[3 2; -2 5]; x = rand(2)
    ### Wrapping a matrix and factorize in solver
    iA = InverseMap(A; solver=(y, A, x)->ldiv!(y, lu(A), x))
    @test ishermitian(A) == ishermitian(iA) == false
    @test issymmetric(A) == issymmetric(iA) == false
    @test isposdef(A) == isposdef(iA) == false
    @test A \ x ≈ iA * x
    @test transpose(A) \ x ≈ transpose(iA) * x
    @test adjoint(A) \ x ≈ adjoint(iA) * x
    ### Wrapping a factorization
    iA = InverseMap(lu(A))
    # @test ishermitian(A) == ishermitian(iA) == true
    # @test issymmetric(A) == issymmetric(iA) == true
    # @test isposdef(A) == isposdef(iA) == true
    @test A \ x ≈ iA * x
    @test transpose(A) \ x ≈ transpose(iA) * x
    @test adjoint(A) \ x ≈ adjoint(iA) * x
    ## Complex Hermitian
    A = ComplexF64[3 2im; -2im 5]; x = rand(ComplexF64, 2)
    ### Wrapping a matrix and factorize in solver
    iA = InverseMap(A; solver=(y, A, x)->ldiv!(y, cholesky(A), x))
    @test ishermitian(A) == ishermitian(iA) == true
    @test issymmetric(A) == issymmetric(iA) == false
    @test isposdef(A) == isposdef(iA) == true
    @test A \ x ≈ iA * x
    if VERSION >= v"1.8.0-"
        @test transpose(A) \ x ≈ transpose(iA) * x
        @test adjoint(A) \ x ≈ adjoint(iA) * x
    end
    ### Wrapping a factorization
    iA = InverseMap(cholesky(A))
    # @test ishermitian(A) == ishermitian(iA) == true
    # @test issymmetric(A) == issymmetric(iA) == true
    @test isposdef(A) == isposdef(iA) == true
    @test A \ x ≈ iA * x
    # @test transpose(A) \ x ≈ transpose(iA) * x
    if VERSION >= v"1.7.0"
        @test adjoint(A) \ x ≈ adjoint(iA) * x
    end
    ## Complex non-Hermitian
    A = ComplexF64[3 2im; 3im 5]; x = rand(ComplexF64, 2)
    ### Wrapping a matrix and factorize in solver
    iA = InverseMap(A; solver=(y, A, x)->ldiv!(y, lu(A), x))
    @test ishermitian(A) == ishermitian(iA) == false
    @test issymmetric(A) == issymmetric(iA) == false
    @test isposdef(A) == isposdef(iA) == false
    @test A \ x ≈ iA * x
    @test transpose(A) \ x ≈ transpose(iA) * x
    @test adjoint(A) \ x ≈ adjoint(iA) * x
    ### Wrapping a factorization
    iA = InverseMap(lu(A))
    # @test ishermitian(A) == ishermitian(iA) == true
    # @test issymmetric(A) == issymmetric(iA) == true
    # @test isposdef(A) == isposdef(iA) == true
    @test A \ x ≈ iA * x
    @test transpose(A) \ x ≈ transpose(iA) * x
    @test adjoint(A) \ x ≈ adjoint(iA) * x

    # Example from https://www.dealii.org/current/doxygen/deal.II/step_20.html#SolvingusingtheSchurcomplement
    M = sparse(2.0I, 10, 10) + sprand(10, 10, 0.1); M = M'M
    iM = InverseMap(M; solver=cgz!)
    B = sparse(5.0I, 10, 5)
    F = rand(10)
    G = rand(5)
    ## Solve using Schur complement
    G′ = B' * iM * F - G
    S = B' * iM * B
    P = IterativeSolvers.cg(S, G′)
    U = IterativeSolvers.cg(M, F - B * P)
    ## Solve using standard method and compare
    @test [M B; B' 0I] \ [F; G] ≈ [U; P] atol=1e-8
end
