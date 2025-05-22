export monomialTensor, monomialBasisFn, monomialVarFn, decomposeMonomial


function monomialTensor(degrees)
    d = sum(degrees);
    n = length(degrees);
    inds = []
    for (i, degree) in enumerate(degrees)
        append!(inds, fill(i, degree))
    end
    T = zeros(fill(n, d)...)
    for ind in Combinatorics.permutations(inds)
        T[ind...] = 1
    end
    return T
end;


function basisFnHelper(degs)
    tup = [[0:deg for deg in degs[2:end]]...]
    return Iterators.product(tup...)
end

function monomialBasisFn(degs, D)
    basis_ = collect.(vec(collect(basisFnHelper(degs))))
    basis_inds_ = [D[b] for b in basis_];

    basis = basis_[sortperm(basis_inds_)]
    basis_inds = sort(basis_inds_);
    return basis, basis_inds 
end;

function monomialParams(vars, degrees)
    param = Set()
    for var in vars
        if sum(var .> degrees[2:end]) == 1
            push!(param, var)
        end 
    end 
    return param 
end;

function monomialEq(var, degrees)
    i_ = findfirst(x -> x > 0, var - 2*degrees[2:end])
    if isnothing(i_)
        bigger = findall(x -> x > 0, var - degrees[2:end])
        j = first(bigger)
        i = last(bigger)
    else 
        bigger = setdiff(findall(x -> x > 0, var - degrees[2:end]), i_)
        j = first(bigger)
        i = i_
    end
    alpha_i = zeros(Int, length(var))
    beta_j = zeros(Int, length(var))

    for k=1:length(var)
        d_k = degrees[2:end][k]

        if k == i 
            alpha_i[k] = d_k+1
            beta_j[k] = var[k] - alpha_i[k]
        elseif k == j 
            beta_j[k] = d_k+1
            alpha_i[k] = var[k] - beta_j[k]
        elseif var[k] > d_k
            alpha_i[k] = d_k
            beta_j[k] = var[k] - alpha_i[k]
        else 
            alpha_i[k] = var[k]
            beta_j[k] = 0
        end

    end
    
    return alpha_i, beta_j

end;


function monomialVarFn(basis, n, d, degrees)
    vars = TensorDecomposition.varTups(basis, n, d)
    params = monomialParams(vars, degrees)
    return vars, params
end

function decomposeMonomial(degrees, basis, D, vars, params, paramVals)
    d = sum(degrees)
    m = Int(ceil((maximum(sum.(vars))-d)/(degrees[1]+1)))

    varDict = Dict([k => Set() for k=1:m])
    eqDict = Dict()

    for var in setdiff(vars, params)
        k = Int(ceil((sum(var) .- d)./(degrees[1]+1)))
        push!(varDict[k], var)
        eqDict[var] = monomialEq(var, degrees)
    end

    paramsDict = Dict([p => pv for (p, pv) in zip(params, paramVals)]);

    H0inds = Dict()
    H0 = zeros(eltype(paramVals), length(basis), length(basis))
    for (i, b1) in enumerate(basis)
        for (j, b2) in enumerate(basis)
            if b1+b2 == degrees[2:end]
                H0[i, j] = 1
            elseif sum(b1+b2) > d 
                if b1+b2 in params 
                    H0[i, j] = paramsDict[b1+b2]
                else 
                    if b1+b2 in keys(H0inds)
                        push!(H0inds[b1+b2], (i, j))
                    else 
                        H0inds[b1+b2] = Set()
                        push!(H0inds[b1+b2], (i, j))
                    end
                end
            end
        end 
    end
    
    solDict = copy(paramsDict);

    for k=1:m
        for var in varDict[k]
            if sum(var .> degrees[2:end]) > k 
                val = 0
            else
                alpha_i, beta_j = eqDict[var]

                avec = Array{eltype(paramVals)}(undef, length(basis))
                bvec = Array{eltype(paramVals)}(undef, length(basis))

                for (i, b) in enumerate(basis) 
                    avec[i] = get(solDict, alpha_i+b, 0)
                    bvec[i] = get(solDict, beta_j+b, 0) 
                end

                temp = H0 \ bvec
                val = (permutedims(avec)*temp)[1]
            end
            solDict[var] = val

            if var in keys(H0inds)
                for (i, j) in H0inds[var]
                    H0[i, j] = val 
                end 
            end 
        end
    end

    T = monomialTensor(degrees);
    Ms = TensorDecomposition.multMatrices(T, basis, solDict, D, H0);
    lhat, Zhat = TensorDecomposition.obtainDecomp(T, Ms)
    return lhat, Zhat, solDict;
end