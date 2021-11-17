using LinearAlgebra, Random, DataFrames, CSV, Printf, CPLEX, Gurobi;
using Distributed
addprocs(6)
@everywhere using JuMP, BARON
include("DecompositionAlgorithm_AdaptiveSampling.jl")
include("DecompositionAlgorithm_RandomSampling.jl")
include("Utilities.jl")

@everywhere root=("//panfs//roc//groups//10//qizh//rishabh//ProdPlanning1//Adaptive//")
@everywhere include("ASP.jl")

function ModelParameters(II, JJ, TT, Mu, D)
    Pmax = 400*ones(Int16, 38)
    Q0 = zeros(Int16, 28)
    Q0[10:26] = [10 5 0 5 5 7 10 2 3 5 10 7 5 5 3 3 0]
    Qmin = zeros(Int16, 28)
    Qmax = 200*ones(Int16, 28)
    Wmax = zeros(Int16, 28, TT)
    Wmax[1:10, 1] = [200 125 200 100 50 250 600 40 1000 20]
    for t = 1:TT
        Wmax[1:10, t] = 400*ones(Int16, 10)
    end
    IhatJ = zeros(Int16, 28, 38)
    IbarJ = zeros(Int16, 28, 38)

    for j = 1:JJ
        for i = 1:II
            if Mumax[i,j] <= -0.001
                IhatJ[j,i] = 1
            elseif Mumax[i,j] >= 0.001
                IbarJ[j,i] = 1
            end
        end
    end

    A = zeros(Float64, (4*JJ+2*II)*TT, (II+JJ)*TT)
    b = zeros(Float64, (4*JJ+2*II)*TT)

    for i = 1:II
        for j = 1:JJ
            Mu[i,j] = abs(Mu[i,j])
        end
    end

    # Constraint Matrix
    for t = 1:TT
        for j = 1:JJ
            b[JJ*(t-1)+j] = -(Qmin[j] - Q0[j] + sum(D[j, tprime] for tprime = 1:t))
            b[JJ*TT + JJ*(t-1)+j] = Qmax[j] - Q0[j] + sum(D[j, tprime] for tprime = 1:t)
            b[2*JJ*TT+JJ*(t-1)+j] = Wmax[j,t]
            b[3*JJ*TT+JJ*(t-1)+j] = 0
            for tprime = 1:t
                for i = 1:II
                    if IhatJ[j,i] == 1
                        A[JJ*(t-1)+j, II*(tprime-1)+i] = -Mu[i,j]
                        A[JJ*TT+JJ*(t-1)+j, II*(tprime-1)+i] = Mu[i,j]
                    elseif IbarJ[j,i] == 1
                        A[JJ*(t-1)+j, II*(tprime-1)+i] = Mu[i,j]
                        A[JJ*TT+JJ*(t-1)+j, II*(tprime-1)+i] = -Mu[i,j]
                    end
                    A[JJ*(t-1)+j, II*TT+JJ*(tprime-1)+j] = -1
                    A[JJ*TT+JJ*(t-1)+j, II*TT+JJ*(tprime-1)+j] = 1
                    A[4*JJ*TT+II*(t-1)+i, II*(t-1)+i] = 1
                    A[4*JJ*TT+II*TT+II*(t-1)+i, II*(t-1)+i] = -1
                    b[4*JJ*TT+II*(t-1)+i] = Pmax[i]
                    b[4*JJ*TT+II*TT+II*(t-1)+i] = 0
                end
            end
            A[2*JJ*TT+JJ*(t-1)+j, II*TT+JJ*(t-1)+j] = 1
            A[3*JJ*TT+JJ*(t-1)+j, II*TT+JJ*(t-1)+j] = -1
        end
    end
    return A, b
end

global rng = MersenneTwister(1234)

II = 38 # No of Processes
JJ = 28 # No of Chemicals involved
H = 1 # No of time periods
TT = H # This code uses TT isntead of H as the total number of time periods
S = 100 # number of samples for heuristic adative sampling
dim = (II+JJ)*TT
global n = dim

global Nominal_Demand = zeros(Int16, 28)
Nominal_Demand[10:25] = [40 150 30 75 60 30 30 50 150 70 30 75 100 75 250 50]
global Mumax = zeros(Float64, 38, 28)
Mumax[1,1] = 0.58; Mumax[1,6] = 0.63; Mumax[1,11] = -1;
Mumax[2,6] = 0.64; Mumax[2,12] = -1;
Mumax[3, 1] = -.055; Mumax[3,2] = 1.25; Mumax[3,11] = -1;
Mumax[4,2] = 0.40; Mumax[4,3] = 0.69; Mumax[4, 14] = -1;
Mumax[5, 13] = -1; Mumax[5,14] = 2.3; Mumax[5,17] = -1.7;
Mumax[6, 2] = 0.74; Mumax[6,15] = -1;
Mumax[7, 13] = -1; Mumax[7,15] = 1.1;
Mumax[8, 3] = 1; Mumax[8,16] = -1;
Mumax[9,16] = 1.26; Mumax[9,17] = -1;
Mumax[10,13] = 1.57; Mumax[10,27] = -1;
Mumax[11, 3] = 1.01; Mumax[11,17] = -1;
Mumax[12,3] = 0.76; Mumax[12, 4] = 0.28; Mumax[12, 8] = -1;
Mumax[13, 8] = 1.14; Mumax[13, 18] = -1;
Mumax[14,2] = 0.78; Mumax[14,13] = -1;
Mumax[15, 12] = -1; Mumax[15, 19] = 1.34;
Mumax[16, 4] = 0.60; Mumax[16,19] = -1;
Mumax[17, 4] = 0.67; Mumax[17, 12] = -1;
Mumax[18, 12] = 1.1; Mumax[18,20] = -1;
Mumax[19, 19] = 0.98; Mumax[19, 20] = -1;
Mumax[20, 4] = 0.35; Mumax[20,20] = 0.71; Mumax[20,21] = -1;
Mumax[21, 6] = 0.32; Mumax[21,20] = 0.72; Mumax[21,21] = -1;
Mumax[22, 4] = 0.88; Mumax[22, 5] = -1; Mumax[22, 24] = -0.03;
Mumax[23, 1] = 0.56; Mumax[23, 5] = 0.92; Mumax[23, 11] = -1;
Mumax[24, 4] = 0.39; Mumax[24, 28] = -1;
Mumax[25, 5] = -1; Mumax[25, 28] = -1;
Mumax[26, 4] = 0.3; Mumax[26, 23] = -1;
Mumax[27, 20] = 0.65; Mumax[27, 22]= -1; Mumax[27, 27] = 0.46;
Mumax[28, 7] = 0.56; Mumax[28, 10] = 0.56; Mumax[28, 20] = -1;
Mumax[29, 12] = 1.2; Mumax[29, 22] = -1;
Mumax[30, 4] = 1.17; Mumax[30, 25] = -1;
Mumax[31, 5] = 0.75; Mumax[31, 24] = -1;
Mumax[32, 4] = 0.53; Mumax[32, 24] = -1;
Mumax[33, 12] = 0.6; Mumax[33, 20] = 0.82; Mumax[33, 21] = -1;
Mumax[34, 10] = 0.42; Mumax[34, 25] = -1;
Mumax[35, 9] = 0.5; Mumax[35, 10] = -1;
Mumax[36, 7] = 0.53; Mumax[36, 24] = -1; Mumax[36, 25] = 0.57;
Mumax[37, 24] = -1; Mumax[37, 28] = 1.44;
Mumax[38, 2] = -0.38; Mumax[38, 3] = -0.22; Mumax[38, 4] = -1; Mumax[38, 9] = 3.08; Mumax[38, 26] = -1.81;
Mu = deepcopy(Mumax)
D = zeros(Float64, JJ, TT)
for t = 1:TT
    D[:, t] = Nominal_Demand[1:JJ]
end

A, b = ModelParameters(II, JJ, TT, Mu, D)

obj = rand(rng,1:1000,dim)
obj = -1*obj/ norm(obj, 1)
noisy_p = rand(rng,1.0:1000.0, 10, dim)
for i = 1:10
    noisy_p[i, :] = -1*noisy_p[i,:]/norm(noisy_p[i,:], 1)
end

global c = length(A[:,1])
TS = 100
test_x  = zeros(Float16, TS, n)
b_test = zeros(Float64, TS, c)
A_test = zeros(Float64, c, n, TS)

for test = 1:TS
    mutest = zeros(Float64, II, JJ)
    Dtest = zeros(Float64, JJ, TT)
    for i = 1:II
        for j = 1:JJ
            mutest[i,j] = (rand(rng, 50:100)/100)*Mumax[i,j]
            for t = 1:TT
                Dtest[j, t] = (rand(rng, 90:110)/100)*Nominal_Demand[j]
            end
        end
    end
    A_test[:,:,test], b_test[test, :] = ModelParameters(II,JJ,TT,mutest,Dtest)
    test_x[test, :] = SolveModel(A_test[:, :, test], obj, b_test[test, :], n)
end

instance = 1
adaptive_sampling(A,b,obj,noisy_p,A_test,b_test,test_x,instance,S)
random_sampling(A,b,obj,noisy_p,A_test,b_test,test_x,instance)
