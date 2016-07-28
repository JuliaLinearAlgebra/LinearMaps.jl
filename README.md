# LinearMaps

[![Build Status](https://travis-ci.org/Jutho/LinearMaps.jl.svg)](https://travis-ci.org/Jutho/LinearMaps.jl) [![Coverage Status](https://img.shields.io/coveralls/Jutho/LinearMaps.jl.svg)](https://coveralls.io/r/Jutho/LinearMaps.jl)

A Julia package for defining and working with linear maps, also known as linear transformations or linear operators acting on vectors. The only requirement for a LinearMap is that it can act on a vector (by multiplication) efficiently.

##What's new

Simplified interface. Only the names `AbstractLinearMap` and `LinearMap` are exported. `AbstractLinearMap` is the root type that newly defined linear maps should be subtypes of. `LinearMap` acts as a general purpose constructor for constructing linear maps from matrices, functions or altering the properties of existing `AbstractLinearMap` objects, even though there is no actual type called `LinearMap`.

##Installation

Install with the package manager, i.e. `Pkg.add("LinearMaps")`.

## Philosophy

Several iterative linear algebra methods such as linear solvers or eigensolvers only require an efficient evaluation of the matrix vector product, where the concept of a matrix can be formalized / generalized to a linear map (or linear operator in the special case of a square matrix).

The LinearMaps package provides the following functionality:

1. An `AbstractLinearMap` type that shares with the `AbstractMatrix` type that it responds to the functions `size`, `eltype`, `isreal`, `issym`, `ishermitian` and `isposdef`, `transpose` and `ctranspose` and multiplication with a vector using both `*` or the in-place version `A_mul_B!`. Depending on the subtype, also `At_mul_B`, `At_mul_B!`, `Ac_mul_B` and `Ac_mul_B!` are supported. Linear algebra functions that uses duck-typing for its arguments can handle `AbstractLinearMap` objects similar to `AbstractMatrix` objects, provided that they can be written using the above methods. Unlike `AbstractMatrix` types, `AbstractLinearMap` objects cannot be indexed, neither using `getindex` or `setindex!`.
2. A single method `LinearMap` that allows to construct `AbstractLinearMap` objects from objects of type `Function`, `AbstractMatrix` or `AbstractLinearMap`. This method allows to (re)define the properties (`isreal`, `issym`, `ishermitian`, `isposdef`) of the corresponding linear map.
3. A framework for combining objects of type `AbstractLinearMap` and of type `AbstractMatrix` using linear combinations, transposition and composition, where the  linear map resulting from these operations is never explicitly evaluated but only its matrix vector product is defined (i.e. lazy evaluation). The matrix vector product is written to minimize memory allocation by using a minimal number of temporary vectors. There is full support for the in-place version `A_mul_B!`, which should be preferred for higher efficiency in critical algorithms. In
 addition, it tries to recognize the properties of combinations of linear maps. In particular, compositions such as `A'*A` for arbitrary `A` or even `A'*B*C*B'*A` with arbitrary `A` and `B` and positive definite `C` are recognized as being positive definite and hermitian. In case a certain property of the resulting `AbstractLinearMap` object is not correctly inferred, the `LinearMap` method can be called to redefine the properties.

##Methods

* `LinearMap`

  General purpose method to construct AbstractLinearMap objects of specific types, as described in the Types section below
  
  ```julia
  LinearMap(A::AbstractMatrix;isreal::Bool,issym::Bool,ishermitian::Bool,isposdef::Bool)
  LinearMap(A::AbstractLinearMap;isreal::Bool,issym::Bool,ishermitian::Bool,isposdef::Bool)
  ```
  
  Create a `WrappedMap` object that will respond to the methods `isreal`, `issym`, `ishermitian`, `isposdef` with the values provided by the keyword arguments. The default values correspond to the result of calling these methods on the argument `A`. This allows to use an `AbstractMatrix` within the `AbstractLinearMap` framework and to redefine the properties of an existing `AbstractLinearMap`.
  
  ```julia
  LinearMap(f::Function,M::Int,N::Int=M;ismutating::Bool,issym::Bool,ishermitian::Bool,isposdef::Bool,ftranspose,fctranspose)
  LinearMap(f::Function,eltype::Type,M::Int,N::Int=M;ismutating::Bool,issym::Bool,ishermitian::Bool,isposdef::Bool,ftranspose,fctranspose)
  ```

  Create `FunctionMap` object that wraps a function describing the action of the linear map on a vector. The corresponding properties of the linear map can also be specified. Here, `f` represents the function implementing the action of the linear map on a vector, either as returning the result (i.e. `f(src::AbstractVector) -> dest::AbstractVector`) when `ismutating=false` (default) or as a mutating function that accepts a vector for the destination (i.e. `f(dest::AbstractVector,src::AbstractVector) -> dest`). `M` is the number of rows (length of the output vectors) and `N` the number of columns (length of the input vectors). When the latter is not specified, `N=M`. Using the second calling convention, the `eltype` of the resulting linear map can explicitly be specified. The keyword arguments and their default values are:
  * `ismutating [=false]`: `false` if the function `f` (and if provided `ftranspose` and or `fctranspose`) accepts a single vector argument corresponding to the input, and `true` if they accept two vector arguments where the first will be mutated so as to contain the result. In both cases, the resulting `A::FunctionMap` will support both the mutating as nonmutating matrix vector multiplication.
  * `isreal [=true]` (only in the first calling convention): if `true`, it will create `A::FunctionMap{Float64}`, otherwise `A::FunctionMap{Complex128}`. If the matrix representation of the function could be represented using a different `eltype`, then the second calling scheme is recommended.
  * `issym [=false]`: whether the function represents the multiplication with a symmetric matrix. If `true`, this will automatically enable `A'*x` and `A.'*x`.
  * `ishermitian [=false]`: whether the function represents the multiplication with a hermitian matrix. If `true`, this will automatically enable `A'*x` and `A.'*x`.
  * `isposdef [=false]`: whether the function represents the multiplication with a positive definite matrix.
  * `ftranspose [=nothing]`: an optional argument that can be used to pass a function that implements the multiplication with the transposed matrix
  * `fctranspose [=nothing]`: an optional argument that can be used to pass a function that implements the multiplication with the hermitian conjugated matrix
  
* `Base.full(linearmap)`
  
  Creates a full matrix representation of the linearmap object, by multiplying it with the successive basis vectors.
  
* All matrix multiplication methods and the corresponding mutating versions.
  
##Types

None of the types below need to be constructed directly; they arise from performing operations between `AbstractLinearMap` objects or by calling the `LinearMap` method described above.

* `AbstractLinearMap`
  
  Abstract supertype

* `FunctionMap`

  Type for wrapping an arbitrary function that is supposed to implement the matrix vector product as an `AbstractLinearMap`.
    
* `WrappedMap`
  
  Type for wrapping an `AbstractMatrix` or `AbstractLinearMap` and to possible redefine the properties `isreal`, `issym`, `ishermitian` and `isposdef`. An `AbstractMatrix` will automatically be converted to a `WrappedMap` when it is combined with other `AbstractLinearMap` objects via linear combination or composition (multiplication). Note that `WrappedMap(mat1)*WrappedMap(mat2)` will never evaluate `mat1*mat2`, since this is more costly then evaluating `mat1*(mat2*x)` and the latter is the only operation that needs to be performed by `AbstractLinearMap` objects anyway. While the cost of matrix addition is comparible to matrix vector multiplication, this too is not performed explicitly since this would require new storage of the same amount as of the original matrices.
  
* `IdentityMap`
  
  Type for representing the identity map of a certain size `M=N`, obtained simply as `IdentityMap{T}(M)`, `IdentityMap(T,M)=IdentityMap(T,M,N)=IdentityMap(T,(M,N))` or even `IdentityMap(M)=IdentityMap(M,N)=IdentityMap((M,N))`. If `T` is not specified, `Bool` is assumed, since operations between `Bool` and any other `Number` will always be converted to the type of the other `Number`. If `M!=N`, an error is returned. An `IdentityMap` of the correct size and element type will automatically be created if `LinearMap` objects are combined with `I`, Julia's built in identity (`UniformScaling`).
  
* `LinearCombination`, `CompositeMap`, `TransposeMap` and `CTransposeMap`

  Used to add and multiply `LinearMap` objects, don't need to be constructed explicitly. 

## Examples

The `LinearMap` object combines well with the iterative eigensolver `eigs`, which is the Julia wrapper for Arpack.

```julia
using LinearMaps

function leftdiff!(y::Vector,x::Vector) # left difference assuming periodic boundary conditions
    length(y)==length(x) || throw(DimensionMismatch())
    N=length(x)
    @inbounds for i=1:N
         y[i]=x[i]-x[mod1(i-1,N)]
    end
    return y
end

function mrightdiff!(y::Vector,x::Vector) # minus right difference
    length(y)==length(x) || throw(DimensionMismatch())
    N=length(x)
    @inbounds for i=1:N
        y[i]=x[i]-x[mod1(i+1,N)]
    end
    return y
end

D=LinearMap(leftdiff!,100;ismutating=true,fctranspose=mrightdiff!)
eigs(D'*D;nev=3,which=:SR)
```
