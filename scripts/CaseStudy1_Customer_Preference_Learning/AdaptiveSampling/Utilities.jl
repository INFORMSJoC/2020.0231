#------------------------------------------------------------------------------
function SolveModel(A, obj, b, n)
    # Step 1: Create the model]
        # count += 1
        # m1 = Model(solver=CplexSolver(CPX_PARAM_SCRIND = 0, CPX_PARAM_LPMETHOD = 1))
        m1 = Model(solver=GurobiSolver(OutputFlag = 0))
        @variable(m1, x[1:n])
        @objective(m1, Max, dot(obj, x))
        for i = 1:length(A[:, 1])
            @constraint(m1, dot(A[i, :], x) <= b[i])
        end
    
    # Step 2: Solve the model
        status = solve(m1; suppress_warnings = true)
    # Step 3: output optimal value;
        val = getvalue(x);
    end
    #------------------------------------------------------------------------------
    
    function SolveFP(cone, iteration)
        epsilon = 1
        M2 = 10
        m5 = Model(solver=GurobiSolver(OutputFlag = 1))
        # m5 = Model(solver=CplexSolver(CPX_PARAM_SCRIND = 0))
        @variable(m5, y[1:n]>=1e-4)
        @variable(m5, gamma[1:iteration, 1:c] >= 0)
        @variable(m5, w, Bin)
    
        for i = 1:iteration
            for j = 1:n
                @constraint(m5, y[j] == sum(gamma[i,k]*cone[i][k,j] for k = 1:length(cone[i][:,1])))
            end
        end
        # @constraint(m5, epsilon <= M2*w + sum(y))
        # @constraint(m5, epsilon <= M2*(1-w) - sum(y))
        @constraint(m5, sum(y) == epsilon)
    
        @objective(m5, Min, 0)
        status = solve(m5; suppress_warnings = true)
        return status
    end
    
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    
    function SampleC(cone)
        iteration = length(cone[:])
        # m3 = Model(solver=CplexSolver(CPX_PARAM_SCRIND = 0))
        m3 = Model(solver=GurobiSolver(OutputFlag = 0, Seed = rand(rng, 1:1000)))
        @variable(m3, y[1:n] >= 0)
        @variable(m3, 1 >= gamma[1:iteration, 1:c] >= 0)
        T = rand(rng, iteration, c)
        for i = 1:iteration
            for j = 1:n
                @constraint(m3, y[j] == sum(gamma[i,k]*cone[i][k,j] for k = 1:length(cone[i][:,1])))
            end
        end
        @objective(m3, Max, sum(T[i,k]*gamma[i,k] for i = 1:iteration for k = 1:length(cone[i][:,1])))
        status = solve(m3; suppress_warnings = true)
        return getvalue(y)
    end
    
    #------------------------------------------------------------------------------
    function tightind(A, b, x)
        # Outputs an array with the indices of active constraints
        epsilon = 1e-6
        istight = []
        z = zeros(Int16, c)
        for i = 1:c
            if ((-dot(A[i,:], x) + b[i] <= epsilon) && sum(A[i,j] for j = 1:n) != 0)
                push!(istight, i)
                z[i] = 1
            end
        end
        return istight, z
    end
    
    #------------------------------------------------------------------------------
    function SolveP2(Cprime, cone, iteration)
        # m3 = Model(solver=CplexSolver(CPX_PARAM_SCRIND = 0))
        m3 = Model(solver=GurobiSolver(OutputFlag = 0))
        @variable(m3, y[1:n])
        @variable(m3, gamma[1:iteration, 1:c] >= 0)
    
        for i = 1:iteration
            for j = 1:n
                @constraint(m3, y[j] == sum(gamma[i,k]*cone[i][k,j] for k = 1:length(cone[i][:,1])))
            end
        end
    
        @objective(m3, Min, sum((y[j]-Cprime[j])^2 for j = 1:n))
        status = solve(m3; suppress_warnings = true)
        return getvalue(y), getobjectivevalue(m3)
    end
    
    #------------------------------------------------------------------------------
    
    function SolveP1(A, b, x, NoS, J)
        n = length(A[1, :, 1])
        # m2 = Model(solver=CplexSolver(CPX_PARAM_SCRIND = 0, CPX_PARAM_EPINT = 0, CPX_PARAM_EPAGAP = 0))
        m2 = Model(solver = GurobiSolver(OutputFlag = 0, IntFeasTol = 1e-9, MIPGapAbs = 1e-5, MIPGap = 0))
        M = 50
        epsilon = 1
        # M2 = 1000
    
        @variable(m2, xbar[1:NoS, 1:n], start = 0)
        @variable(m2, lambda[1:NoS, 1:c] >= 0)
        @variable(m2, s[1:NoS, 1:c] >= 0)
        @variable(m2, z[1:NoS, 1:c], Bin)
        @variable(m2, C[1:n]>=1e-4)
        @variable(m2, t[1:J, 1:n, 1:NoS] >= 0)
        # @variable(m2, w[1:n], Bin)
        # @variable(m2, y[1:n] >= 0)
    
        @constraint(m2, DF[k = 1:NoS, j = 1:n], C[j] - sum(lambda[k, i]*A[i, j, k] for i = 1:c) == 0)
        @constraint(m2, PF[k = 1:NoS, i = 1:c], sum(A[i, j, k]*xbar[k, j] for j = 1:n) + s[k, i] == b[i])
        @constraint(m2, CS1[k = 1:NoS, i = 1:c], s[k, i] <= M*(1-z[k, i]))
        @constraint(m2, CS2[k = 1:NoS, i = 1:c], lambda[k, i] <= M*z[k, i])
        @constraint(m2, VC[k = 1:NoS] , sum(z[k,:]) >= n)
        @constraint(m2, linearization1[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= xbar[k,i] - x[j,i,k])
        @constraint(m2, linearization2[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= -(xbar[k,i] - x[j,i,k]))
        # @constraint(m2, epsilon <= M2*w + sum(C))
        # @constraint(m2, epsilon <= M2*(1-w) - sum(C))
        # @constraint(m2, NC1[j = 1:n], y[j] <= M2*(1-w[j]) + C[j])
        # @constraint(m2, NC2[j = 1:n], y[j] <= M2*w[j] - C[j])
        @constraint(m2, sum(C) == epsilon)
        @objective(m2, Min, sum(t))
        status = solve(m2; suppress_warnings = true)
        loss1 = getobjectivevalue(m2)
    
        return getvalue(xbar), loss1, getvalue(z)
    end
    
    #-------------------------------------------------------------------------------
    function SolveP1m(A, b, x, NoS, J, instance, initx, initz)
        n = length(A[1, :, 1])
        initz = round.(initz)
        # m2 = Model(solver=CplexSolver(CPX_PARAM_SCRIND = 1, CPX_PARAM_MIPEMPHASIS = 1, CPX_PARAM_TILIM = 3600, CPX_PARAM_EPINT = 0))
        m2 = Model(solver = GurobiSolver(OutputFlag = 0, TimeLimit = 7200, ConcurrentMIP = 3, IntFeasTol = 1e-9, MIPGapAbs = 1e-5, MIPGap = 0))
        M = 20
        epsilon = 1
        # M2 = 1000
    
        @variable(m2, xbar[j = 1:NoS, k = 1:n])
        # for j = 1:NoS-1
        #     for k = 1:n
        #         setvalue(xbar[j,k], initx[j,k])
        #     end
        # end
        @variable(m2, lambda[1:NoS, 1:c] >= 0)
        @variable(m2, s[1:NoS, 1:c] >= 0)
        @variable(m2, z[j = 1:NoS, k = 1:c], Bin)
        for j = 1:NoS-1
            for k = 1:c
                setvalue(z[j,k], initz[j,k])
            end
        end
        @variable(m2, C[1:n]>=1e-4)
        @variable(m2, t[1:J, 1:n, 1:NoS] >= 0)
        # @variable(m2, w[1:n], Bin)
        # @variable(m2, y[1:n] >= 0)
    
        @constraint(m2, DF[k = 1:NoS, j = 1:n], C[j] - sum(lambda[k, i]*A[i, j, k] for i = 1:c) == 0)
        @constraint(m2, PF[k = 1:NoS, i = 1:c], sum(A[i, j, k]*xbar[k, j] for j = 1:n) + s[k, i] == b[k, i])
        @constraint(m2, CS1[k = 1:NoS, i = 1:c], s[k, i] <= M*(1-z[k, i]))
        @constraint(m2, CS2[k = 1:NoS, i = 1:c], lambda[k, i] <= M*z[k, i])
        @constraint(m2, VC[k = 1:NoS] , sum(z[k,:]) >= n)
        # @constraint(m2, NC1[j = 1:n], y[j] <= M2*(1-w[j]) + C[j])
        # @constraint(m2, NC2[j = 1:n], y[j] <= M2*w[j] - C[j])
        # @constraint(m2, sum(y) == epsilon)
        @constraint(m2, epsilon == sum(C))
        @constraint(m2, linearization1[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= xbar[k,i] - x[j,i,k])
        @constraint(m2, linearization2[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= -(xbar[k,i] - x[j,i,k]))
        JuMP.build(m2)
        # grb = JuMP.internalmodel(m2).inner
        # Gurobi.write_model(grb, "test.mps")
    
        @objective(m2, Min, sum(t))
        status = solve(m2; suppress_warnings = true)
        loss1 = getobjectivevalue(m2)
        return getvalue(xbar), loss1, getvalue(z), status
    end
    
    #-------------------------------------------------------------------------------
    function accuracy(xtest, xhat)
    NoS = length(xtest[:, 1])
    n = length(xtest[1,:])
    error = 0
        for i = 1:NoS
            # println(norm(xtest[i,:] - xhat[i,:]), "\n")
            if !isapprox(norm(xtest[i,:] - xhat[i,:]), 0, atol = 1e-3)
                error += 1
            end
        end
        return error/NoS
    end
    
    