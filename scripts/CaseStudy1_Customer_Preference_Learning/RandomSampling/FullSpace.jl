function SolveP1b(A, b, x, NoS, J, iteration, istight)
    n = length(A[1, :, 1])

    # m2 = Model(solver = CplexSolver(CPX_PARAM_SCRIND = 1, CPX_PARAM_EPGAP = 0))
    m2 = Model(solver = GurobiSolver(OutputFlag = 0, TimeLimit = 7200, ConcurrentMIP = 3, IntFeasTol = 1e-9, MIPGapAbs = 1e-5, MIPGap = 0))
    M = 50
    epsilon = 1
    M2 = 50

    @variable(m2, xbar[1:NoS, 1:n])
    @variable(m2, lambda[1:NoS, 1:c] >= 0)
    @variable(m2, s[1:NoS, 1:c] >= 0)
    @variable(m2, z[1:NoS, 1:c], Bin)
    @variable(m2, C[1:n] >= 1e-4)
    @variable(m2, d[1:NoS, 1:c, 1:n] >= 0)
    @variable(m2, w, Bin)
    @variable(m2, t[1:J, 1:n, 1:NoS] >= 0)

    @constraint(m2, DF[k = 1:NoS, j = 1:n], C[j] - sum(lambda[k, i]*A[i, j, k] for i = 1:c) == 0)
    @constraint(m2, PF[k = 1:NoS, i = 1:c], sum(A[i, j, k]*xbar[k, j] for j = 1:n) + s[k, i] == b[k, i])
    @constraint(m2, CS1[k = 1:NoS, i = 1:c], s[k, i] <= M*(1-z[k, i]))
    @constraint(m2, CS2[k = 1:NoS, i = 1:c], lambda[k, i] <= M*z[k, i])
    @constraint(m2, VC[k = 1:NoS] , sum(z[k,:]) >= n)
    # @constraint(m2, epsilon <= M2*w + sum(C))
    # @constraint(m2, epsilon <= M2*(1-w) - sum(C))
    @constraint(m2, sum(C) == epsilon)

    @constraint(m2, linearization1[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= xbar[k,i] - x[j,i,k])
    @constraint(m2, linearization2[j = 1:J, i = 1:n, k = 1:NoS], t[j,i,k] >= -(xbar[k,i] - x[j,i,k]))
    if iteration > 1
        for m = 1:(iteration-1)
            @constraint(m2, sum(z[k, istight[k+(m-1)*NoS][i]] for k = 1:NoS for i = 1:length(istight[k][:])) <= n*NoS-1)
        end
    end

    @objective(m2, Min, sum(t))
    status = solve(m2; suppress_warnings = true)
    println(status)
    loss1 = getobjectivevalue(m2)

    return getvalue(xbar), loss1, getvalue(z), getobjbound(m2)
end

###############################################################################
function SolveP2b(A, Cprime, istight, NoS, iteration)
    v = []
    hatc = []
    for m = 1:iteration
        # m3 = Model(solver = CplexSolver(CPX_PARAM_SCRIND = 0))
        m3 = Model(solver = GurobiSolver(OutputFlag = 0))
        @variable(m3, hat_c[1:n])
        @variable(m3, gamma[1:NoS, 1:c] >= 0)

        for k = 1:NoS
            for j = 1:n
                @constraint(m3, hat_c[j] == sum(A[istight[(m-1)*NoS+k][t], j, k]*gamma[k, t] for t = 1:length(istight[(m-1)*NoS+k])))
            end
        end
        @objective(m3, Min, sum((hat_c[j] - Cprime[j])^2 for j = 1:n))
        status = solve(m3; suppress_warnings = true)
        push!(hatc, getvalue(hat_c))
        push!(v, getobjectivevalue(m3))
    end
    return v, hatc
end

function batchlearning(obj, noisyp, n,c, I, J, Aprime, bprime, truex, noisyx, Atest, btest, testx)
    global istight = []
    global rstar = 100
    global r = 100
    global iteration = 1
    global cont = 0

    # Phase 1
    ts = time()
    while cont == 0
        println("Iteration \t", iteration)
        ti_s = time()
        hat_x, r, z,gap = SolveP1b(Aprime, bprime, noisyx, I, J, iteration, istight)
        ti_e = time()
        iteration_time= ti_e-ti_s
        println("Best Bound \t", gap)
        println("Time taken \t", iteration_time)
        println("Error \t", norm(hat_x-truex))
        println("loss \t", r)
        if iteration == 1
            global rstar = r
        end
        for i = 1:I
            temp = []
            for k = 1:c
                if isapprox(z[i, k], 1, atol = 1e-3)
                    push!(temp, k)
                end
            end
            push!(istight, temp)
        end
        global iteration += 1
        if !isapprox(rstar, r, atol = 1e-4)
            global cont = 1
        end
    end

    # Phase 2
    iteration = iteration - 1

    v, hatc = SolveP2b(Aprime, obj, istight, I, iteration)
    println("Error wrt the true objective \t", v)
    te = time()
    DeltaT = (te-ts)
    println("Runtime\t", DeltaT)

    #Random Testing, not related to the algorithm
    v1, hatc1 = SolveP2b(Aprime, noisyp, istight, I, iteration)
    println("Error wrt the random objective \t", v1)
    xhat  = zeros(Float32, TS, n)

    # Prediction error calculation
    for test = 1:TS
        xhat[test, 1:n] = SolveModel(Atest[:, :, test], hatc1[1], btest[test, :], n)
    end
    errorrate = accuracy(testx, xhat)
    println("Test set error \t", errorrate)
    println("--------------------------------------")
end
