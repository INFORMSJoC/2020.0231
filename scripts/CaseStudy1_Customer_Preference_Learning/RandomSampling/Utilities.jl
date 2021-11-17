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

