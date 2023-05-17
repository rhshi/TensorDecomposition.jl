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