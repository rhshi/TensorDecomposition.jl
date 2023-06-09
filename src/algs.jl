export jennrich, jennrich2, catalecticant, hankel_r, hankel, hankel_system

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

    gam = gamma(d)

    # flatten to matrices
    T1flat = reshape(T1, (n^gam, n^gam));
    T2flat = reshape(T2, (n^gam, n^gam));

    # obtain the factors
    # T real -> true factors are obtained, possibly up to sign
    # T complex -> true factors are obtained, but up to a complex scalar factor of norm 1 
    w, Ahat_ = eigen(T1flat*pinv(T2flat))
    Ahat_ = Ahat_[:, abs.(w).>tol]
    r = last(size(Ahat_))
    Ahat = zeros(eltype(Ahat_), (n, r))
    for i=1:r
        a = Ahat_[:, i]
        U, s, _ = svd(reshape(a, (n, n^(gam-1))))
        Ahat[:, i] = U[:, abs.(s).>tol]
    end;
    Ahat = dehomogenize!(Ahat)

    # obtain L and deflate
    low = delta(d)
    high = d-low
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

    low = gamma(d)
    high = d-1-low

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


    low = delta(d)
    high = d-low
    Tflat = reshape(T, (n^low, n^high))
    Alow = kronMat(Ahat, low)
    Ahigh = kronMat(Ahat, high)

    Lhat = diag(pinv(Alow)*Tflat*pinv(transpose(Ahigh)))

    return Ahat, Lhat
end;

function catalecticant(T)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    delt = delta(d);
    Tcat = catMat(T, delt);

    # homotopy continuation to solve
    @var x[1:n]
    Tker = sum(nullspace(Tcat) .* monomials(x, d-delt), dims=1);
    F = System(vec(Tker));
    result = solve(F);
    sols = solutions(result);
    Ahat = dehomogenize(reduce(hcat,sols))

    # get coefficients
    Tflat = reshape(T, (n^(delt), n^(d-delt)))
    Alow = kronMat(Ahat, delt)
    Ahigh = kronMat(Ahat, d-delt)

    Lhat = diag(pinv(Alow)*Tflat*pinv(transpose(Ahigh)))

    return Ahat, Lhat
end;


function hankel(T; tol=1e-10)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    delt = delta(d)
    rmin = LinearAlgebra.rank(catMat(T, delt));
    println(@sprintf "Rank of catalecticant: %d" rmin)
    println("Using the rank of catalecticant as initial guess.")

    for r=rmin:n^d 
        try
            return hankel_r(T, n, d, r, tol=tol)
        catch 
            println("Increasing rank by 1.")
        end
    end
end;

function hankel_r(T, n, d, r; tol=1e-10)
    if iseven(d)
        if n > 3
            if r > binomial(Int(n-1+d/2), n-1) - n 
                println("Outside the regime of the Catalecticant algorithm.")
            end
        else
            if r > binomial(Int(n-1+d/2), n-1) - n + 1
                println("Outside the regime of the Catalecticant algorithm.")
            end
        end
    else 
        if r > binomial(Int(n-1+(d-1)/2), n-1)
            println("Outside the regime of the Catalecticant algorithm.")
        end
    end


    Thank = hankMat(T)
    H0 = Thank[1:r, 1:r];
    if r <= binomial(n-1+delta(d), n-1)
        H0 = convert(Matrix{eltype(T)}, H0)
    end
    H0_inv = inv(H0)

    D = Dict()
    first_r = []
    for (i, ind) in enumerate(with_replacement_combinations(0:n-1, d))
        if i <= r
            push!(first_r, ind)
        end
        D[Tuple(x for x in ind)] = i
    end

    Hs = []
    for i=1:n-1
        hankInds = []
        multMap = map(x -> multMon(x, i), first_r)
        for ind in multMap
            push!(hankInds, D[Tuple(x for x in ind)])
        end
        push!(Hs, Thank[1:r, hankInds])
    end

    Ms = []
    for H in Hs
        push!(Ms, H*H0_inv)
    end;

    gam = gamma(d);
    if r > binomial(n-1+gam, n-1)
        println("High rank tensor decomposition -- must solve quadratic system.")
        eqMats = []
        for i=1:n-1
            for j=i+1:n-1
                push!(eqMats, Ms[i]*Ms[j]-Ms[j]*Ms[i])
            end
        end
        F = System(mapreduce(M -> vec(M), vcat, eqMats))
        result = solve(F)
        sols = solutions(result, only_nonsingular=false)
        println(sols)
        if length(sols) == 0
            error("There are no solutions h such that the multiplication matrices commute.  The input r is likely not the correct rank.")
        end
        sol = sols[1]
        hs = variables(F)
        for i=1:length(Ms)
            Ms[i] = evaluate(Ms[i], hs => sol)
        end
    else
        println("Low rank tensor decomposition -- can use Jennrich's.")
        Ms = map(M -> map(x -> to_number(x), M), Ms) 
        for i=1:n-1
            for j=i+1:n-1
                if norm(Ms[i]*Ms[j]-Ms[j]*Ms[i], Inf) > tol
                    error("Multiplication matrices do not commute.  The input r is likely not the correct rank.")
                end
            end
        end
        F = nothing
    end

    Ahat = zeros(ComplexF64, n, r)
    Ahat[1, :] = ones(r)
    for j=1:length(Ms)
        wj, Vj = eigen(Ms[j])
        argsort = sortperm(Vj[1, :], by=x -> (real(x), imag(x)), rev=true);
        Ahat[j+1, :] = wj[argsort]
    end

    low = Int(floor(d/2))
    high = Int(ceil(d/2))
    Tflat = reshape(T, (n^low, n^high))
    Alow = kronMat(Ahat, low)
    Ahigh = kronMat(Ahat, high)

    Lhat = diag(pinv(Alow)*Tflat*pinv(transpose(Ahigh)))
    
    return Ahat, Lhat, F
end;

function hankel_system(T, r)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    Thank = hankMat(T)
    H0 = Thank[1:r, 1:r];
    if r <= binomial(n-1+delta(d), n-1)
        H0 = convert(Matrix{eltype(T)}, H0)
    end
    H0_inv = inv(H0)

    D = Dict()
    first_r = []
    for (i, ind) in enumerate(with_replacement_combinations(0:n-1, d))
        if i <= r
            push!(first_r, ind)
        end
        D[Tuple(x for x in ind)] = i
    end

    Hs = []
    for i=1:n-1
        hankInds = []
        multMap = map(x -> multMon(x, i), first_r)
        for ind in multMap
            push!(hankInds, D[Tuple(x for x in ind)])
        end
        push!(Hs, Thank[1:r, hankInds])
    end

    Ms = []
    for H in Hs
        push!(Ms, H*H0_inv)
    end;

    eqMats = []
    for i=1:n-1
        for j=i+1:n-1
            push!(eqMats, Ms[i]*Ms[j]-Ms[j]*Ms[i])
        end
    end

    F = System(mapreduce(M -> vec(M), vcat, eqMats));
    return F

end