export vandermonde, khatri_rao, makeDicts, dotu, getColSubset, multMon, basisFn


function getDims(T)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)
    return n-1, d
end

function from_multiindex(x, n)
    d = length(x)
    c = 0
    for i=1:d-1
        c += (x[i]-1)*n^(d-i)
    end
    return c + x[d]
end;

function getInds(n, d)
    if d == 0
        return [1]
    else
        inds = collect(with_replacement_combinations(1:n+1, d))
        inds = map(x->from_multiindex(x, n+1), inds)
    end
    return inds 
end

function khatri_rao(A, d; type=Complex)
    if d == 1
        B = A
    else
        n, r = size(A)
        B = zeros(type, n^d, r)
        for i=1:r 
            B[:, i] = kron(ntuple(x->A[:, i], d)...)
        end;
    end;
    return B
end

function tupLT(a, b)
    if a[1][1] isa Number && !(b[1][1] isa Number)
        return false
    elseif !(a[1][1] isa Number) && b[1][1] isa Number
        return true
    else
        return a < b
    end
end;

function vandermonde(A, d; type=Complex)
    n = size(A)[1]-1
    return permutedims(khatri_rao(A, d; type=type)[getInds(n, d), :])
end

function alpha_iterator(::Val{N}, s, t=()) where {N}
    N <= 1 && return ([s, t...],) # Iterator over a single Tuple
    Iterators.flatten(alpha_iterator(Val(N-1), s-i, [i, t...]) for i in 0:s)
end

function e(j, n)
    ej = zeros(Int64, n)
    ej[j] = 1
    return ej
end

function complexGaussian(m, n)
    A = randn(m, n)
    B = randn(m, n)
    return A + im*B
end;

function makeDicts(n, d)
    monomials = multiexponents(n+1, d)

    D = Dict()
    Drev = Dict()
    for (i, mon) in enumerate(monomials)
        D[mon[2:end]] = i 
        Drev[i] = mon[2:end]
    end 
    return D, Drev
end

function dotu(x::Vector, y::Vector)
    return (permutedims(x)*y)[1]
end;

function getRank(A_, tol=1e-10)
    A = A_ ./ vec(mapslices(norm, A_, dims=2))
    svdvalsA = svdvals(A)
    c = 0
    if svdvalsA[1] >= tol
        c += 1
    end
    for i=1:length(svdvalsA)-1
        a1 = svdvalsA[i]
        a2 = svdvalsA[i+1]
        if a2 / a1 < tol
            break 
        end 
        c += 1
    end
    return c
end

function getColSubset(A_, tol=1e-8)
    A = A_ ./ vec(mapslices(norm, A_, dims=2))
    _, R = qr(A)
    colInds = []
    for i=1:minimum(size(R))
        if abs(R[i, i]) > tol
            push!(colInds, i)
        end
    end
    return colInds
end;

function multMon(x, j)
    c = copy([x_ for x_ in x])
    c[j] += 1
    return c
end;

function basisFn(basis_inds, Drev)
    basis = [Drev[ind] for ind in basis_inds];
    
    basisD = Dict()
    for b in basis
        if sum(b) in keys(basisD)
            push!(basisD[sum(b)], b)
        else
            basisD[sum(b)] = [b]
        end
    end
    return basis, basisD
end;