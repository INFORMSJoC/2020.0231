using LinearAlgebra, Random, Gurobi, DataFrames, CSV, Printf;
using Distributed
addprocs(50)
@everywhere using JuMP, BARON
include("Decomposition_with_AdaptiveSampling.jl")
@everywhere root=("//panfs//roc//groups//10//qizh//rishabh//Adaptive//")
@everywhere include("ASP.jl")
@everywhere include("Utilities.jl")

global rng = MersenneTwister(1234)
n = 25
S = 5
dim = n

for instance = 1:10
    noisyp = rand(rng,1:1000,dim)
    noisyp = noisyp/norm(noisyp, 1)
    A = zeros(Float64, (2*dim)+1, dim)
    b = zeros(Float64, (2*dim)+1)
    obj = rand(rng,1:1000,dim)
    obj = obj/ norm(obj, 1)

    # Only generate the first training sample here, rest will be determined through adaptive sampling.
    for i = 1:dim
        A[i, i] = 1
        b[i, 1] = 1
        A[dim+1, i] = rand(rng, 5000:15000)/100
    end
    b[dim+1, 1] = sum(A[dim+1, :])*0.6

    for i = 1:dim
        A[dim+1+i, i] = -1
        b[dim+1+i, 1] = 0
    end

    global n = length(A[1,:])
    global c = length(A[:,1])
    TS = 100
    testx  = zeros(Float32, TS, n)
    btest = zeros(Float64, TS, c)
    Atest = zeros(Float64, c, n, TS)

    for test = 1:TS
        btest[test, :] = b # Use the same b as the training dataset
        Atest[:, :, test] = A
        for i = 1:dim
            Atest[dim+1, i, test] = rand(rng, 5000:15000)/100
        end
        testx[test, 1:n] = SolveModel(Atest[:, :, test], obj, btest[test, :], n)
    end

    t = adaptive_sampling(A,b,obj,noisyp,Atest,btest,testx,instance,S)
end
