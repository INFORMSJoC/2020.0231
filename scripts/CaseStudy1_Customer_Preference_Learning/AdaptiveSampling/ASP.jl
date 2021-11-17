function maxmin(t, cone, a, z)
    # Solves (ASP) as a part of Algorithm 3 (Step 4)
    cd(string(root,"Results//ex",id,"//opts",t))
    n = length(a[1,:])
    c = length(a[:,1])
    iteration = length(cone[:])
    m2 = Model(solver=BaronSolver(CplexLibName="/panfs/roc/groups/10/qizh/rishabh/Mangi/CPLEX_Studio128/cplex/bin/x86-64_linux/libcplex1280.so", PrLevel = 0, MaxTime = 100))
    # m2 = Model(solver=BaronSolver(PrLevel = 0, MaxTime = 100))

    M = 4000
    epsilon = 1

    @variable(m2, 160 >= y[1:n] >= -5)
    @variable(m2, d[1:2*n] >= 0)
    @variable(m2,  1 >= gamma[1:iteration, 1:c] >= 0)
    @variable(m2, 200 >= t[1:n] >= 0)
    @variable(m2, gamma2[1:c] >= 0)

    @objective(m2, Min, -sum(t[i] for i = 1:n))

    @NLconstraint(m2, sum(y[i]*d[i] for i = 1:n) - sum(y[i]*d[i+n] for i = 1:n) == sum(t[i] for i = 1:n))

    for i = 1:n
        @constraint(m2, -t[i] - sum(gamma2[j]*a[j,i]*z[j] for j = 1:c) <= -y[i])
    end

    for i = 1:n
        @constraint(m2, -t[i] + sum(gamma2[j]*a[j,i]*z[j] for j = 1:c) <= y[i])
    end

    for i = 1:n
        @constraint(m2, d[i]+d[n+i] <= 1)
    end

    for j = 1:c
        if z[j] >= 0.1
            @constraint(m2, sum(a[j,i]*d[i] for i = 1:n) - sum(a[j,i]*d[i+n] for i = 1:n) <= 0)
        end
    end

    @constraint(m2, CC[i = 1:iteration, j = 1:n], y[j] == sum(gamma[i,k]*cone[i][k,j] for k = 1:length(cone[i][:,1])))

    status = solve(m2; suppress_warnings = true)
    if status != :Infeasible
        return -1*getobjectivevalue(m2), -1*getobjbound(m2)
    else
        return 0, 0
    end
end
