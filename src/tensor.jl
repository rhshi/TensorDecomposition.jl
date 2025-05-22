export rankedTensor, catMat, catRanks

function rankedTensor(l::Array, Z::Array, d::Int; type=Complex)
    n, _ = size(Z)
    Zd = khatri_rao(Z, d; type=type)
    return reshape(sum(permutedims(l) .* Zd, dims=2), tuple(repeat([n], d)...))
end

function catMat(T, k)
    n, d = getDims(T)

    row_inds = getInds(n, k)
    col_inds = getInds(n, d-k)

    return (reshape(T, ((n+1)^k, (n+1)^(d-k))))[row_inds, col_inds]
end;

function catRanks(T)
    _, d = getDims(T)
    ranks = [1]
    for k=1:Int(floor(d/2))
        push!(ranks, rank(catMat(T, k)))
    end
    return ranks
end