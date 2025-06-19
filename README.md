# TensorDecomposition

[![Build Status](https://github.com/rhshi/TensorDecomposition.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rhshi/TensorDecomposition.jl/actions/workflows/CI.yml?query=branch%3Amain)

This repository is Julia code for the paper "Efficient tensor decomposition via moment matrix extension."  This code includes an implementation of efficient moment matrix extension via linear algebraic operations, requiring no symbolic algebra packages.  Also included is an implementation of monomial decomposition via a paramaterization of the space of decompositions.

## Proofs
As described in the paper, included are computer assisted proofs for efficient formats in `proofs/4-17.jld2` for $n=4, \dots, 17$.  We use the computer algebra package Nemo to perform computations over finite fields; some examples are given in `proofs/example.ipynb`, giving some specializations such that $\mathbf{A}$ is full column rank over finite fields.

## Examples
All examples are included in the folder `examples`:
- `d_4.ipynb`: An example for $d=4$.  Includes an example of efficient decomposition for $n=10, r=51$ where the matrix $\mathbf{A}$ is generically full column rank.  Also includes an example where a decomposition has three colinear points so the decomposition is not unique; decompositions are given via a parameterization of the space of decompositions.
- `d_6.ipynb`: An example for $d=6$.  Includes an example of efficient decomposition for $n=5, r=50$ where the matrix $\mathbf{A}$ is generically full column rank.  Also includes an example of a tensor where a set $B$ cannot be chosen to be the first $r$ monomials yet the tensor is still efficiently decomposable with unique decomposition.
- `monomial.ipynb`: An example of monomial decomposition, giving multiple decompositions via choosing different sets of parameters.
- `binary.ipynb`: An example of binary tensor decomposition, specifically Example 3.20 from the paper.

## Packages
Julia is required with the following packages:
- Combinatorics
- Graphs
- LinearAlgebra
- Random
- LinearSolve
- SparseArrays
- Nemo
- JLD2
- IJulia
