immutable MatrixMap{T}<:LinearMap{T}
    matrix::AbstractMatrix{T}
end

# properties
Base.size(A::MatrixMap,n)=(n==1 || n==2 ? size(A.matrix,n) : error("LinearMap objects have only 2 dimensions"))
Base.size(A::MatrixMap)=size(A.matrix)
Base.isreal(A::MatrixMap)=isreal(A.matrix)
Base.issym(A::MatrixMap)=issym(A.matrix)
Base.ishermitian(A::MatrixMap)=ishermitian(A.matrix)
Base.isposdef(A::MatrixMap)=isposdef(A.matrix)

# comparison
==(A::MatrixMap,B::MatrixMap)=A.matrix==B.matrix

# multiplication with vector
Base.A_mul_B!(y::AbstractVector,A::MatrixMap,x::AbstractVector)=Base.A_mul_B!(y,A.matrix,x)
*(A::MatrixMap,x::AbstractVector)=*(A.matrix,x)

Base.At_mul_B!(y::AbstractVector,A::MatrixMap,x::AbstractVector)=Base.At_mul_B!(y,A.matrix,x)
Base.At_mul_B(A::MatrixMap,x::AbstractVector)=Base.At_mul_B(A.matrix,x)

Base.Ac_mul_B!(y::AbstractVector,A::MatrixMap,x::AbstractVector)=Base.Ac_mul_B!(y,A.matrix,x)
Base.Ac_mul_B(A::MatrixMap,x::AbstractVector)=Base.Ac_mul_B(A.matrix,x)

# combine LinearMap and Matrix objects: linear combinations and map composition
+(A1::LinearMap,A2::AbstractMatrix)=+(A1,MatrixMap(A2))
+(A1::AbstractMatrix,A2::LinearMap)=+(MatrixMap(A1),A2)

*(A1::LinearMap,A2::AbstractMatrix)=*(A1,MatrixMap(A2))
*(A1::AbstractMatrix,A2::LinearMap)=*(MatrixMap(A1),A2)
