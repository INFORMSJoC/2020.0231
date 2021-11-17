using JuMP, LinearAlgebra, Random, Gurobi, DataFrames, CSV, Printf;
root=("Please specify the root directory here")
include("Utilities.jl")
include("FullSpace.jl")
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
function SolveModel(A, obj, b, n)
    # Solves the forward optimization problem

# Step 1: Create the model
    m1 = Model(solver=GurobiSolver(OutputFlag = 0))
    @variable(m1, x[1:n])
    @objective(m1, Max, dot(obj, x))
    for i = 1:length(A[:, 1])
        @constraint(m1, dot(A[i, :], x) <= b[i])
    end

# Step 2: Solve the model
    status = solve(m1; suppress_warnings = true)
# Step 3: return optimal value;
    val = getvalue(x);
end

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

function SolveFP(cone, iteration)
    # Feasibility Problem to be solved at every iteration of the decomposition algorithm
    epsilon = 1
    m2 = Model(solver=GurobiSolver(OutputFlag = 1))
    @variable(m2, y[1:n]>=1e-4)
    @variable(m2, gamma[1:iteration, 1:c] >= 0)
    @variable(m2, w, Bin)

    for i = 1:iteration
        for j = 1:n
            @constraint(m2, y[j] == sum(gamma[i,k]*cone[i][k,j] for k = 1:length(cone[i][:,1])))
        end
    end

    @constraint(m2, sum(y) == epsilon)

    @objective(m2, Min, 0)
    status = solve(m2; suppress_warnings = true)
    return status
end

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

function SolveP2(C_bar, cone, iteration)

    m3 = Model(solver=GurobiSolver(OutputFlag = 0))
    @variable(m3, y[1:n])
    @variable(m3, gamma[1:iteration, 1:c] >= 0)

    for i = 1:iteration
        for j = 1:n
            @constraint(m3, y[j] == sum(gamma[i,k]*cone[i][k,j] for k = 1:length(cone[i][:,1])))
        end
    end

    @objective(m3, Min, sum((y[j]-C_bar[j])^2 for j = 1:n))
    status = solve(m3; suppress_warnings = true)
    return getvalue(y), getobjectivevalue(m3)
end

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

function SolveP1(A, b, x, NoS, J)
    # NoS (number of samples, i.e., I is always 1 here.)
    n = length(A[1, :, 1])
    m4 = Model(solver = GurobiSolver(OutputFlag = 0, IntFeasTol = 1e-9, MIPGapAbs = 1e-5, MIPGap = 0))
    M = 50
    epsilon = 1

    @variable(m4, xbar[1:NoS, 1:n], start = 0)
    @variable(m4, lambda[1:NoS, 1:c] >= 0)
    @variable(m4, s[1:NoS, 1:c] >= 0)
    @variable(m4, z[1:NoS, 1:c], Bin)
    @variable(m4, C[1:n]>=1e-4)
    @variable(m4, t[1:J, 1:n, 1:NoS] >= 0)

    @constraint(m4, S[k = 1:NoS, j = 1:n], C[j] - sum(lambda[k, i]*A[i, j, k] for i = 1:c) == 0)
    @constraint(m4, PF[k = 1:NoS, i = 1:c], sum(A[i, j, k]*xbar[k, j] for j = 1:n) + s[k, i] == b[i])
    @constraint(m4, CS1[k = 1:NoS, i = 1:c], s[k, i] <= M*(1-z[k, i]))
    @constraint(m4, CS2[k = 1:NoS, i = 1:c], lambda[k, i] <= M*z[k, i])
    @constraint(m4, VC[k = 1:NoS] , sum(z[k,:]) >= n)
    @constraint(m4, linearization1[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= xbar[k,i] - x[j,i,k])
    @constraint(m4, linearization2[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= -(xbar[k,i] - x[j,i,k]))
    @constraint(m4, sum(C) == epsilon)

    @objective(m4, Min, sum(t))
    status = solve(m4; suppress_warnings = true)
    loss1 = getobjectivevalue(m4)

    return getvalue(xbar), loss1, getvalue(z)
end

###############################################################################

function SolveP1m(A, b, x, NoS, J, instance, initx, initz)
    # Solves P1 with multiple (NoS > 1) experiments
    n = length(A[1, :, 1])
    initz = round.(initz)
    m5 = Model(solver = GurobiSolver(OutputFlag = 1, TimeLimit = 7200, ConcurrentMIP = 3, 
    IntFeasTol = 1e-9, MIPGapAbs = 1e-5, MIPGap = 0))
    M = 20
    epsilon = 1

    @variable(m5, xbar[j = 1:NoS, k = 1:n])
    @variable(m5, lambda[1:NoS, 1:c] >= 0)
    @variable(m5, s[1:NoS, 1:c] >= 0)
    @variable(m5, z[j = 1:NoS, k = 1:c], Bin)
    for j = 1:NoS-1
        for k = 1:c
            setvalue(z[j,k], initz[j,k])
        end
    end
    @variable(m5, C[1:n]>=1e-4)
    @variable(m5, t[1:J, 1:n, 1:NoS] >= 0)

    @constraint(m5, S[k = 1:NoS, j = 1:n], C[j] - sum(lambda[k, i]*A[i, j, k] for i = 1:c) == 0)
    @constraint(m5, PF[k = 1:NoS, i = 1:c], sum(A[i, j, k]*xbar[k, j] for j = 1:n) + s[k, i] == b[k, i])
    @constraint(m5, CS1[k = 1:NoS, i = 1:c], s[k, i] <= M*(1-z[k, i]))
    @constraint(m5, CS2[k = 1:NoS, i = 1:c], lambda[k, i] <= M*z[k, i])
    @constraint(m5, VC[k = 1:NoS] , sum(z[k,:]) >= n)
    @constraint(m5, epsilon == sum(C))
    @constraint(m5, linearization1[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= xbar[k,i] - x[j,i,k])
    @constraint(m5, linearization2[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= -(xbar[k,i] - x[j,i,k]))

    @objective(m5, Min, sum(t))
    status = solve(m5; suppress_warnings = true)
    loss = getobjectivevalue(m5)
    return getvalue(xbar), loss, getvalue(z), status
end

#------------------------------------------------------------------------------

function accuracy(x, x_hat)
NoS = length(x[:, 1])
n = length(x[1,:])
error = 0
    for i = 1:NoS
        if !isapprox(norm(x[i,:] - x_hat[i,:]), 0, atol = 1e-3)
            error += 1
        end
    end
    return error/NoS
end

#------------------------------------------------------------------------------
#   Main body of the program
#------------------------------------------------------------------------------
rng = MersenneTwister(1234)

# Edit the following parameters
n = 25 # Dimension of the forward problem
J_scheme = 1

# Dummy A and b
A = zeros(Float64, (2*dim)+1, dim) # Initialize A matrix
b = zeros(Float64, (2*dim)+1) # Initialize b vector
obj = rand(rng,1:1000,dim) 
obj = obj/ norm(obj, 1) # Generate random objective coefficients

for i = 1:dim
    A[i, i] = 1
    b[i, 1] = 1
end

for i = 1:dim
    A[dim+1, i] = obj[i] + 100 + rand(rng, -10:10)
end
b[dim+1, 1] = rand(rng, sum(A[dim+1, :])/2:sum(A[dim+1, :]) - 1)

for i = 1:dim
    A[dim+1+i, i] = -1
    b[dim+1+i, 1] = 0
end

# Generate a random reference vector
noisy_p = rand(rng,1:1000,dim)
noisy_p = noisy_p/ norm(noisy_p, 1)


I = 100 # Number of experiments
TS = 100 # Size of test dataset

dim = n # Some part of this code uses dim for the dimension of the forward problem
c = length(A[:,1]) # Total number of constraints in the problem


for w in [10 20 100] # w decides the level of noise in data, (sigma = 1/w)
    global J # Initialize Number of samples per experiment
    if J_scheme == 1
        if w == 10
            J = 5
        elseif w == 20
            J = 10
        else
            J = 20
        end
    elseif J_scheme == 2
        J = 250
    end

    Total_Time = []
    Resolves = []
    
    Pred_Error_Evolution = zeros(Float64, I, 10)

    # Make directories to store results
    mkdir(string(root,"Results_dim_",dim,"_noise_", w))

    for instance = 1:10 # For every dim and w, solve 10 random instances

        t_instance_s = time() # # To store the time taken to solve one instance

        # Arrays to store training data
        b_train = zeros(Float64, I, c)
        A_train = zeros(Float64, c, n, I)
        true_x  = zeros(Float32, I, n)
        noisy_x = zeros(Float64, J, n, I)

        # Arrays to store test data
        test_x  = zeros(Float32, TS, n)
        b_test = zeros(Float64, TS, c)
        A_test = zeros(Float64, c, n, TS)

        # We also store some performance data at every step of the algorithm, initialize required arrays
        runtime = zeros(Float64,I)
        error =  zeros(Float64,I)
        Is_resolve = zeros(Float64,I)
        
        # Generate training data 
        for T = 1:I
            b_train[T, :] = b
            A_train[:, :, T] = A
            for i = 1:dim
                A_train[dim+1, i, T] = 100 + rand(rng, -5000:5000)/100
            end
        	A_train[dim+1, :, T] = A_train[dim+1, :, T]/ 1000
            b_train[T, dim+1] = 0.6*sum(A_train[dim+1, :,1]) # Keep b the same across different experiments
            true_x[T, 1:n] = SolveModel(A_train[:, :, T], obj, b_train[T, :], n)
        end
        
        # Generate test data
        for test = 1:TS
            b_test[test, :] = b
            A_test[:, :, test] = A
            for i = 1:dim
                A_test[dim+1, i, test] = 100 + rand(rng, -5000:5000)/100
            end
            b_test[test, dim+1] = deepcopy(b_train[1, dim+1])*1000 # Same b as the training dataset
            test_x[test, 1:n] = SolveModel(A_test[:, :, test], obj, b_test[test, :], n)
        end

        # Add noise to the training data
        for T = 1:I
            for j = 1:J
                for k = 1:n
                    noisy_x[j, k, T] = true_x[T,k] + randn(rng, Float64)/w
                end
            end
        end
        
        # First solve the full-space (P1)
        batchlearning(obj, noisy_p, n, c, I, J, A_train, b_train, true_x, noisy_x, A_test, b_test, test_x)

        # Next solve the same problem with adaptive sampling
        resolve_counter = 0

        global cone = fill(zeros(n,n), 1) # Store the active constraints that form the cone for every experiment

        # Store estimates for warm-start (though we only use z for warm-starting)
        hat_x = zeros(Float64, I, n) 
        hat_z = zeros(Float64, I, c)

        # Start the solution process
        for T = 1:I
            flag = 0 # Just to check if a resolve was required for this experiment or not
            # push!(sr, T)
            t_experiment_s = time() # To store the time taken to solve for one experiment
            istight = [] # To Store which constraints are active for this experiment

            # Solve (P1)_{\ell}, obtain \hat{x}_{\ell}^{*}
            hat_x[T, :], r, z = SolveP1(A_train[:,:,T], b_train[T,:], noisy_x[:,:,T], 1, J)
            # init_x[T, :] = temp_x
            hat_z[T, :] = z

            # Find the cone for current experiment
            for i = 1:c
                if isapprox(z[1, i], 1, atol = 1e-3) == 1
                    push!(istight, i)
                end
            end

            # To-Do: Check why this does not work in one step.
            if T == 1
                for j = 1:length(istight)
                    cone[T][j,:] = A_train[istight[j], :, T]
                end
            else
                B = zeros(Float64, length(istight), n) # Temporary matrix
                for j = 1:length(istight)
                    B[j,:] = A_train[istight[j], :, T]
                end
                push!(cone, B)
            end

            status = SolveFP(cone, T) # Check if (FP)_{\ell} is feasible 

            if status == :Infeasible # Solve full problem if infeasible
                flag = 1 # Set flag to one because a resolve is required
                resolve_counter += 1 
        
                z = []
                x, r, z, status = SolveP1m(A_train[:,:,1:T], b_train[1:T,:], noisy_x[:,:,1:T], T, J, instance, hat_x, hat_z)

                if status != :Optimal # If problem does not get solved in the cutoff time, stop this instance
                    break
                end

                # Update the estimated solutions with what we find after the resolve
                hat_x[1:T, :] = x
                hat_z[1:T, :] = z

                # Also update the cone
                global cone = []
                global cone = fill(zeros(n,n), 1)
                for k = 1:T
                    istight = []
                    for i = 1:c
                        if isapprox(z[k, i], 1, atol = 1e-3)
                            push!(istight, i)
                        end
                    end
                    if k == 1
                        for j = 1:length(istight)
                            cone[k][j,:] = A_train[istight[j], :, k]
                        end
                    else
                        B = zeros(Float64, length(istight), n)
                        for j = 1:length(istight)
                            B[j,:] = A_train[istight[j], :, k]
                        end
                        push!(cone, B)
                    end
                end

                status = SolveFP(cone, T) # This is not required but still do it.
                if status == :Infeasible
                    println("Potential problem with instance: ", instance)
                end
            end

            # Solve (P2) 
            hat_c, loss = SolveP2(noisy_p, cone, T)

            # Finally compute the prediction error
            x_pred  = zeros(Float32, TS, n)

            for test = 1:TS
                x_pred[test, 1:n] = SolveModel(A_test[:, :, test], hat_c, b_test[test, :], n)
            end
            runtime[T] = time() - t_experiment_s
            error[T] = accuracy(test_x, x_pred)
            Is_resolve[T] = flag
        end
        ##----------------------------------------------

        # Store experiment level data
        cd(string(root,"Results_dim_",dim,"_noise_",w))
        df1 = DataFrame(Computation_Time = runtime, Prediction_Error = error, Resolve_Flag = Is_resolve)
        # Uncomment if required
        # s1 = @sprintf "InnerLoopResultsSummary_dim_%03.d_Noise_%03.d_Instance_%03.d.csv" dim w instance
        # CSV.write(s1, df1)
        
        push!(Total_Time, time() - t_instance_s)
        push!(Resolves, resolve_counter)
        Pred_Error_Evolution[:,instance] = error
    end

    # Store instance level data
    df2 = DataFrame(Instance = 1:1:I, Computation_Time = Total_Time, Resolves_reqd = Resolves)
    s2 = @sprintf "OuterLoopResultsSummary_dim_%03.d_Noise_%03.d.csv" dim w;
    CSV.write(s2, df2)
    df3 = DataFrame(Pred_Error_Evolution)
    s3 = @sprintf "ErrEvolSummary_dim_%03.d_Noise_%03.d.csv" dim w;
    CSV.write(s3, df3)
end
