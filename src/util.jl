export dehomogenize, normcol

function dehomogenize!(A)
    for col in eachcol(A)
        col ./= col[1]
    end;
    return A
end;
dehomogenize(A) = dehomogenize!(copy(A));

function normcol!(A)
    for col in eachcol(A)
        col ./= norm(col)
    end;
    return A
end;
normcol(A) = normcol!(copy(A));

function complexGaussian(m, n)
    A = randn(m, n)
    B = randn(m, n)
    return A + im*B
end;

function e(j, n)
    e = zeros(n)
    e[j] = 1
    return e
end;

function from_multiindex(x, n)
    d = length(x)
    c = 0
    for i=1:d-1
        c += (x[i]-1)*n^(d-i)
    end
    return c + x[d]
end;

function tdims(T)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)
    return n, d
end;

function kronMat(A::Matrix, d)
    if d == 1
        B = A
    else
        n, r = size(A)
        B = zeros(eltype(A), (n^d, r))
        for i=1:r 
            B[:, i] = kron(ntuple(x->A[:, i], d)...)
        end;
    end;
    return B
end;

function cofactor(A::AbstractMatrix)
    ax = axes(A)
    out = similar(A, eltype(A), ax)
    for col in ax[1]
        for row in ax[2]
            out[col, row] = (-1)^(col + row) * det(A[Not(col), Not(row)])
        end
    end
    return out
end

function inv_(A::Matrix{Expression})
    if length(A) == 1
        return 1 ./ A 
    else 
        return transpose(cofactor(A))/det(A)
    end
end;
Base.inv(A::Matrix{Expression}) = inv_(A)

function multMon(x, j)
    c = copy(reverse(x))
    for i=1:length(c)
        if c[i] == 0
            c[i] = j
            break
        end
    end;
    return sort(reverse(c))
end;

monomials(x, n) = collect((prod(y) for y in with_replacement_combinations(x, n)));
delta(d) = Int(floor(d/2))
gamma(d) = Int(floor((d-1)/2))