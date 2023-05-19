export makeRankedTensor, randomRankedTensor, randomTensor, contract, catMat

function makeRankedTensor(L::Vector, A::Array, d::Int)
    n, _ = size(A)
    A_ = kronMat(A, d)
    return reshape(sum(A_ .* L', dims=2), tuple(repeat([n], d)...))
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

function catMat(T, k)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    low = k 
    high = d-k 

    row_inds = collect(with_replacement_combinations(1:n, low))
    row_inds = map(x->from_multiindex(x, n), row_inds)

    col_inds = collect(with_replacement_combinations(1:n, high))
    col_inds = map(x->from_multiindex(x, n), col_inds)

    return (reshape(T, (n^low, n^high)))[row_inds, col_inds]
end;