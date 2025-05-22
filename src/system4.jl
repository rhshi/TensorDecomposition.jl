export linearSystemA, reducedSystemC


function linearSystemA(T, H0, basis_inds, basisD, D, vars, eqs1, eqs2; type=Complex)
    _, d = getDims(T)
    gamma = Int(floor(d/2))
    alphas = basisD[gamma]
    Tcat = catMat(T, gamma)

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

    

    bjDict = Dict()
    if eltype(T) <: Number 
        F = lu(H0)
        for b_j in b_js 
            b_ = Tcat[basis_inds, D[b_j]] 
            y_ = F.L \ b_[F.p]
            bjDict[b_j] = (F.U \ y_)[(end-length(alphas)+1):end]
        end
    else 
        _, p_, L, U = lu(matrix(H0))
        p = [getindex(p_, k) for k in 1:length(basis_inds)]
        for b_j in b_js 
            b_ = Tcat[basis_inds, D[b_j]] 
            y_ = solve(L, b_[p], side=:right)
            bjDict[b_j] = solve(U, y_, side=:right)[(end-length(alphas)+1):end]
        end
    end


    A = zeros(type, length(eqs1)+length(eqs2), length(vars))
    for (k, eq) in enumerate(eqs1)
        a_i = eq[1]
        b_j = eq[2]

        A[k, coeffDict[a_i+b_j]] = type(1)

        inds = [coeffDict[a_i+alpha_prime] for alpha_prime in alphas]
        A[k, inds] = -bjDict[b_j]
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

        A[k+length(eqs1), indsL] += -bjDict[b_j]
        A[k+length(eqs1), indsR] += bjDict[b_i]
    end

    return A

end

function processVarsEqsC(vars, eqs, c)
    eqs1_ = []
    eqs2_ = []

    lastVarDict = Dict()
    for (i, eqTup) in enumerate(eqs)
        lastVar = eqTup[1] + eqTup[2]
        if !(lastVar in keys(lastVarDict))
            lastVarDict[lastVar] = [i]
        else 
            push!(lastVarDict[lastVar], i)
        end 
    end

    for lastVar in keys(lastVarDict)
        if length(lastVarDict[lastVar]) == 1
            i = lastVarDict[lastVar][1]
            if sum((eqs[i][1])[1:c]) == 3
                push!(eqs1_, eqs[i])
            end
        elseif length(lastVarDict[lastVar]) == 2
            i = lastVarDict[lastVar][1]
            j = lastVarDict[lastVar][2]
            push!(eqs2_, (eqs[i], eqs[j]))
        elseif length(lastVarDict[lastVar]) == 3
            i = lastVarDict[lastVar][1]
            j = lastVarDict[lastVar][2]
            k = lastVarDict[lastVar][3]
            push!(eqs2_, (eqs[i], eqs[j]))
            push!(eqs2_, (eqs[i], eqs[k]))
        end 
    end 

    return filter(x -> sum(x[1:c])>=3, vars), eqs1_, eqs2_

end

function reducedSystemC(T, c, H0_inv, basis_inds, basisD, D, vars, eqs1; type=Complex)
    vars_, eqs1_, eqs2_ = processVarsEqsC(vars, eqs1, c)
    A = linearSystemA(T, H0_inv, basis_inds, basisD, D, vars_, eqs1_, eqs2_; type=type)
    
    Y3 = []
    Y45 = []
    for (i, var) in enumerate(vars_)
        if sum(var[1:c]) == 3 
            push!(Y3, i)
        else 
            push!(Y45, i)
        end
    end

    M11 = A[length(eqs1_)+1:end, Y3]
    M12 = A[1:length(eqs1_), Y45]
    M2 = A[length(eqs1_)+1:end, Y45];

    return M11*M12-M2, length(eqs2_), length(Y45)

end