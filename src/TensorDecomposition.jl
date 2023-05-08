module TensorDecomposition

using Random, LinearAlgebra, TensorToolbox, Combinatorics, TensorOperations

function jennrich(T::Array; tol=1e-10)
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
    w, Ahat = eigen(T1flat*pinv(T2flat));
    Ahat = Ahat[:, abs.(w).>tol];
end;

end
