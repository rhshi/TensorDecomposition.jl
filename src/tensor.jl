using Random, LinearAlgebra, TensorToolbox, Combinatorics, TensorOperations

function tensorDiagonal(L::Vector, d::Int)
    r = length(L)
    D = zeros(tuple(repeat([r], d)...))
    for i = 1:r 
        D[tuple(repeat([i], d)...)...] = L[i]
    end;
    return D
end;

function makeRankedTensor(L::Vector, A::Matrix, d::Int)
    T = tensorDiagonal(L, d)
    for i=1:d 
        T = ttm(T, A, i)
    end;
    return T
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

function complexGaussian(m, n)
    A = randn(m, n)
    B = randn(m, n)
    return A + im*B
end;

function randomRankedTensor(n, d, r; real=false)
    if !real
        A = complexGaussian(n, r)
    else 
        A = randn(n, r)
    end;
    L = ones(r)
    return makeRankedTensor(L, A, d), A
end;

function contract(T::Array, V::Array)
    d1 = length(size(T))
    d2 = length(size(V))
    return ncon((T, V), (vcat(collect(1:d2), -collect(1:d1-d2)), collect(1:d2)))
end;