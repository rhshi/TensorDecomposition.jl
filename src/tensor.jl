export makeRankedTensor, randomRankedTensor, randomTensor, contract, hankMat, catMat

function makeRankedTensor(L::Vector, A::Array, d::Int)
    n, _ = size(A)
    A_ = kronMat(A, d)
    return reshape(sum(transpose(L) .* A_, dims=2), tuple(repeat([n], d)...))
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

function hankMat(T)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    Fdim = binomial(n-1+d, d)
    Fliftdim = binomial(n-1+2*d, n-1)

    F = zeros(eltype(T), Fdim)
    for (i, ind) in enumerate(with_replacement_combinations(0:n-1, d))
        F[i] = T[(ind .+ 1)...]
    end

    Flift = Array{Expression}(undef, Fliftdim)
    for i=1:length(F)
        Flift[i] = F[i]
    end

    @var h[Fdim+1:Fliftdim]
    for i=1:Fliftdim-Fdim
        Flift[Fdim+i] = h[i]
    end

    D = Dict()
    for (i, ind) in enumerate(with_replacement_combinations(0:n-1, 2*d))
        D[Tuple(x for x in ind)] = i
    end
    
    Thank = Array{Expression}(undef, Fdim, Fdim)
    for (i, row_ind) in enumerate(with_replacement_combinations(0:n-1, d))
        for (j, col_ind) in enumerate(with_replacement_combinations(0:n-1, d))
            c = Tuple(x for x in sort(vcat(row_ind, col_ind)))
            Thank[i, j] = Flift[D[c]]
        end
    end

    return Thank

end;

function hankMat2(T)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    Fdim = binomial(n-1+d, d)
    Fliftdim = binomial(n-1+2*d, n-1)

    F = zeros(eltype(T), Fdim)
    alphas = sort(collect(alpha_iterator(Val(n), d)), lt=monomialOrder)
    for (i, ind) in enumerate(convertIndices.(alphas))
        F[i] = T[(ind .+ 1)...]
    end

    if eltype(T) == ComplexF64
        Flift = Array{Complex{Num}}(undef, Fliftdim)
        @variables h[Fdim+1:Fliftdim]::Complex{Real}
        Thank = Array{Complex{Num}}(undef, Fdim, Fdim)
    else 
        Flift = Array{Num}(undef, Fliftdim)
        @variables h[Fdim+1:Fliftdim]
        Thank = Array{Num}(undef, Fdim, Fdim)
    end
    for i=1:length(F)
        Flift[i] = F[i]
    end


    for i=1:Fliftdim-Fdim
        Flift[Fdim+i] = h[i]
    end

    alphas2 = sort(collect(alpha_iterator(Val(n), 2*d)), lt=monomialOrder)
    D = Dict()
    for (i, ind) in enumerate(alphas2)
        D[ind] = i
    end

    for (i, row_ind) in enumerate(alphas)
        for (j, col_ind) in enumerate(alphas)
            row_mon = [x for x in row_ind]
            col_mon = [x for x in col_ind]
            ind = Tuple(x for x in row_mon+col_mon)
            Thank[i, j] = Flift[D[ind]]
        end
    end
    return Thank
    
end;