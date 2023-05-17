export jennrich, jennrich2

function jennrich(T::Array; tol=1e-10)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    # draw random vectors or matrices for contraction
    B = randn(repeat([n], Int(iseven(d))+1)...);
    C = randn(repeat([n], Int(iseven(d))+1)...);

    # contract to tensor with largest even order strictly less than the order of d
    # d odd -> order n^{d-1} tensor
    # d even -> order n^{d-2} tensor 
    T1 = contract(T, B);
    T2 = contract(T, C);

    delt = Int(floor((d-1)/2))

    # flatten to matrices
    T1flat = reshape(T1, (n^delt, n^delt));
    T2flat = reshape(T2, (n^delt, n^delt));

    # obtain the factors
    # T real -> true factors are obtained, possibly up to sign
    # T complex -> true factors are obtained, but up to a complex scalar factor of norm 1 
    w, Ahat_ = eigen(T1flat*pinv(T2flat))
    Ahat_ = Ahat_[:, abs.(w).>tol]
    r = last(size(Ahat_))
    Ahat = zeros(eltype(Ahat_), (n, r))
    for i=1:r
        a = Ahat_[:, i]
        U, s, _ = svd(reshape(a, (n, n^(delt-1))))
        Ahat[:, i] = U[:, abs.(s).>tol]
    end;
    Ahat = dehomogenize!(Ahat)

    # obtain L and deflate
    low = Int(floor(d/2))
    high = Int(ceil(d/2))
    Tflat = reshape(T, (n^low, n^high))
    Alow = kronMat(Ahat, low)
    Ahigh = kronMat(Ahat, high)

    Lhat = diag(pinv(Alow)*Tflat*pinv(transpose(Ahigh)))

    return Ahat, Lhat
end;

function jennrich2(T; tol=1e-10)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    low = Int(floor((d-1)/2))
    high = Int(ceil((d-1)/2))

    e1 = e(1, n)
    T1 = contract(T, e1)
    T1 = reshape(T1, (n^low, n^high))
    r = rank(T1)

    T1inv = pinv(T1)

    Ahat = zeros(eltype(T), n, r)
    Ahat[1, :] = ones(r)

    for j=2:n
        ej = e(j, n)
        Tj = contract(T, ej)
        Tj = reshape(Tj, (n^low, n^high))
        wj_, Vj = eigen(Tj*T1inv)
        wj = wj_[abs.(wj_).>tol];
        Vj = Vj[:, abs.(wj_).>tol];
        argsort = sortperm(Vj[1, :], by=x -> (real(x), imag(x)), rev=true);
        Ahat[j, :] = wj[argsort]
    end


    low = Int(floor(d/2))
    high = Int(ceil(d/2))
    Tflat = reshape(T, (n^low, n^high))
    Alow = kronMat(Ahat, low)
    Ahigh = kronMat(Ahat, high)

    Lhat = diag(pinv(Alow)*Tflat*pinv(transpose(Ahigh)))

    return Ahat, Lhat
end;