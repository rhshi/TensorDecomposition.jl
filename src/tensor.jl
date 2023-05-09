using Random, LinearAlgebra, TensorToolbox, Combinatorics, TensorOperations

function makeRankedTensor(L::Vector, A::Array, d::Int)
    A = kronMat(A, d)
    return reshape(sum(kronMat(A, d) .* L', dims=2), tuple(repeat([n], d)...))
end;

function randomTensor(n::Int, d::Int; real::Bool=false)
    T_ = randn(tuple(repeat([n], d)...)...)
    if !real
        Q = im*randn(tuple(repeat([n], d)...)...)
        T_ = T_+Q
    end;
    T = copy(T_)
    perms = permutations(1:d)
    for perm in perms
        if perm != 1:d 
            T = T + permutedims(T_, perm)
        end;
    end;
    return T/factorial(d)
end;

function randomRankedTensor(n, d, r; real=false)
    if !real
        A = complexGaussian(n, r)
    else 
        A = randn(n, r)
    end;
    L = ones(r)
    A_ = copy(A)
    A1 = A_[1, :]
    L_ = zeros(eltype(A_), r)
    for i=1:r
        A_[:, i] ./= A1[i]
        L_[i] = A1[i]^d
    end;
    return makeRankedTensor(L, A, d), A_, L_
end;

function contract(T::Array, V::Array)
    d1 = length(size(T))
    d2 = length(size(V))
    return ncon((T, V), (vcat(collect(1:d2), -collect(1:d1-d2)), collect(1:d2)))
end;

function kronMat(A::Matrix, d)
    n, r = size(A)
    B = zeros(eltype(A), (n^d, r))
    for i=1:r 
        B[:, i] = kron(ntuple(x->A[:, i], d)...)
    end;
    return B
end;