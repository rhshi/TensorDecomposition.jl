
export findB, varTups, linEqTups, linearSystem, obtainDecomp, multMatrices, efficientMMExtension

function findB(T; atol=1e-10)
    n, d = getDims(T)
    delta = Int(floor(d/2))
    ranks = catRanks(T)
    Tcat = catMat(T, delta)
    _, R = qr(Tcat)
    Rdiag = diag(R)

    basis_inds = []

    for (k, rk) in enumerate(ranks)
        Bk = []
        for i=binomial(n+k-2, n):binomial(n+k-1, n)-1
            if abs(Rdiag[i+1]) > atol 
                push!(Bk, i+1)
            end
            if length(Bk) == rk
                break
            end
        end
        append!(basis_inds, Bk)
    end

    return basis_inds
end

function processEqs2(eqs2)
    function getVertex!(dict, v, i)
        get!(dict, v) do 
            i
        end
    end;

    g = SimpleGraph()

    c = 1
    vertexDict = Dict()
    for (eqL, eqR) in eqs2
        getVertex!(vertexDict, eqL, c)
        c += 1
        getVertex!(vertexDict, eqR, c)
        c += 1
    end
    vertexDictFlip = Dict([v => k for (k, v) in vertexDict])
    add_vertices!(g, length(keys(vertexDict)))
    for (eqL, eqR) in eqs2
        add_edge!(g, vertexDict[eqL], vertexDict[eqR])
    end

    new_g = SimpleGraph(kruskal_mst(g))
    newEqs2 = Set()
    for edge in edges(new_g)
        eL = vertexDictFlip[src(edge)]
        eR = vertexDictFlip[dst(edge)]
        push!(newEqs2, (eL, eR))
    end
    return newEqs2
end

function varTups(basis, n, d)
    vars = Set()
    for i=1:length(basis) 
        alpha1 = basis[i]
        for j=i:length(basis) 
            alpha2 = basis[j]
            if sum(alpha1)+sum(alpha2) >= d 
                for k=1:n 
                    ek = e(k, n)
                    push!(vars, alpha1+alpha2+ek)
                end
            end
        end
    end
    return sort(collect(vars), rev=true)
end

function linEqTups(basisD, n, d)
    # vars_ = Set(vars)
    Ds = collect(keys(basisD))
    eqs1 = Set()
    eqs2 = Set()

    if Int(d/2) in Ds
        alphas = basisD[Int(d/2)]
        alphas_ = Set(alphas)
        betas = basisD[Int(d/2)-1]
        for alpha in alphas 
            for beta in betas 
                for i=1:n 
                    ei = e(i, n)
                    for j=i+1:n
                        ej = e(j, n)
                        if ((beta+ei) in alphas_) && !((beta+ej) in alphas_)
                            push!(eqs1, (alpha+ei, beta+ej))
                        elseif ((beta+ej) in alphas_) && !((beta+ei) in alphas_)
                            push!(eqs1, (alpha+ej, beta+ei))
                        elseif !((beta+ei) in alphas_) && !((beta+ej) in alphas_)
                            push!(eqs2, ((alpha+ei, beta+ej), (alpha+ej, beta+ei)))
                        end
                    end 
                end 
            end 
        end
    end

    for eqTup in eqs2 
        if (eqTup[1] in eqs1) && (eqTup[2] in eqs1)
            delete!(eqs2, eqTup)
        end
    end
    if !isempty(eqs2)
        eqs2 = processEqs2(eqs2)
    end


    eqs1 = sort(collect(eqs1), lt=tupLT, rev=true)
    eqs2 = sort(collect(eqs2), lt=tupLT, rev=true);

    return eqs1, eqs2
end



function linearSystem(T, H0, basis_inds, basisD, D, vars, eqs1, eqs2; type=Complex, sparse=true)
    _, d = getDims(T)

    gamma = Int(floor(d/2))
    delta = Int(floor(d/2))-1;
    
    alphas = basisD[gamma]
    
    TcatGam = catMat(T, gamma)
    TcatDelt = catMat(T, delta)
    
    coeffDict = Dict()
    for (i, var) in enumerate(vars)
        coeffDict[var] = i
    end
    
    b_js = Set()
    for eqTup in eqs1
        b_j = eqTup[2]
        push!(b_js, b_j)
    end 
    for eqTup in eqs2
        b_j1 = eqTup[1][2]
        b_j2 = eqTup[2][2]
        push!(b_js, b_j1)
        push!(b_js, b_j2)
    end

    F = lu(H0)
    bjDict = Dict()
    for b_j in b_js 
        b_ = TcatGam[basis_inds, D[b_j]] 
        y_ = F.L \ b_[F.p]
        bjDict[b_j] = F.U \ y_
    end
    
    if sparse
        A = spzeros(eltype(T), length(eqs1)+length(eqs2), length(vars));
    else 
        A = zeros(eltype(T), length(eqs1)+length(eqs2), length(vars))
    end
    
    b = zeros(type, length(eqs1)+length(eqs2))

    for (k, eq) in enumerate(eqs1)
        a_i = eq[1]
        b_j = eq[2]
        
        inds = [coeffDict[a_i+alpha_prime] for alpha_prime in alphas]
    
        coeffs = bjDict[b_j]
        active = coeffs[(end-length(alphas)+1):end]
        nonactive = coeffs[1:(end-length(alphas))]
    
        A[k, coeffDict[a_i+b_j]] = 1
        A[k, inds] = -active 
        b[k] = dotu(TcatDelt[basis_inds[1:(end-length(alphas))], D[a_i]], nonactive)
    
    end
    for (k, eq) in enumerate(eqs2)
        eqTupL = eq[1]
        eqTupR = eq[2]
    
        a_i = eqTupL[1]
        b_j = eqTupL[2]
        a_j = eqTupR[1]
        b_i = eqTupR[2] 
    
        indsL = [coeffDict[a_i+alpha_prime] for alpha_prime in alphas]
        indsR = [coeffDict[a_j+alpha_prime] for alpha_prime in alphas]
    
        coeffsL = bjDict[b_j]
        activeL = coeffsL[(end-length(alphas)+1):end]
        nonactiveL = coeffsL[1:(end-length(alphas))]
    
        coeffsR = bjDict[b_i]
        activeR = coeffsR[(end-length(alphas)+1):end]
        nonactiveR = coeffsR[1:(end-length(alphas))]
    
        A[k+length(eqs1), indsL] += -activeL
        A[k+length(eqs1), indsR] += activeR 
    
        constantL = dotu(TcatDelt[basis_inds[1:(end-length(alphas))], D[a_i]], nonactiveL)
        constantR = dotu(TcatDelt[basis_inds[1:(end-length(alphas))], D[a_j]], nonactiveR)
        b[k+length(eqs1)] = constantL - constantR
    
    end
    return A, b
end

function obtainDecomp(T, Ms)
    n, d = getDims(T)
    
    M = sum([randn()*M_ for M_ in Ms])
    Zhat_ = eigvecs(M)[1:n+1, :]
    Zhat = Zhat_ ./ permutedims(Zhat_[1, :])
    lhat = khatri_rao(Zhat, d; type=eltype(Zhat)) \ reshape(T, (n+1)^d)
    return lhat, Zhat
end

function multMatrices(T, basis, solDict, D, H0)
    n, d = getDims(T)
    Tzero = catMat(T, d);

    Hs = []
    for i=1:n
        basis_i = [multMon(b, i) for b in basis]
        Hi = Array{eltype(T), 2}(undef, length(basis), length(basis_i))
        for (j, alpha1) in enumerate(basis)
            for (k, alpha2) in enumerate(basis_i)
                gamma = alpha1+alpha2 
                if sum(gamma) <= d 
                    Hi[j, k] = Tzero[D[gamma], 1]
                else 
                    Hi[j, k] = solDict[gamma]
                end
            end 
        end
        push!(Hs, Hi)
    end

    Ms = []
    for H in Hs 
        push!(Ms, H*inv(H0))
    end

    return Ms
end

