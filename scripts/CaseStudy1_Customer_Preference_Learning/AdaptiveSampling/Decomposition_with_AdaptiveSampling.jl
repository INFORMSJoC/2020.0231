function adaptive_sampling(A,b,obj,noisyp,Atest,btest,testx,instance,S)
    Experiment = 1

    I = 100
	TS = 100
    re_solve_counter = 0
    cone = fill(zeros(n,n), 1)
    Pred_Error_Evolution = zeros(Float64, I, 10)
    ind = 0

    A_train = zeros(Float64, 2*n+1, n, I)
    b_train = zeros(Float64, I, 2*n+1)
    x_train = zeros(Float64, 30, n, I)

    # Generate random folders to store BARON temp files
    id=randstring(5)
    @eval @everywhere id=$id
    mkdir(string(root,"Results//ex",id))
    for t=1:S
        mkdir(string(root,"Results//ex",id,"//opts",t))
    end

    while(Experiment <= I)
        T = Experiment
        A_train[:,:,Experiment] = deepcopy(A);
        # Scale A and b to use a smaller big-M parameter while solving (P1)
        A_train[n+1, :, iteration] = A_train[n+1, :, Experiment]/10000;
        b_train[iteration, :] = deepcopy(b);
        b_train[iteration, n+1] = b_train[iteration, n+1]/10000;
        x = SolveModel(A, obj, b, n)
        J = 30;
        noisyx = zeros(Float64, J, n)
        for j = 1:J
            for k = 1:n
                noisyx[j, k] = x[k] + randn(rng, Float64)/100
            end
        end
        x_train[:,:,iteration] = deepcopy(noisyx);

        hat_x = zeros(Float64, I, n)
        hat_z = zeros(Float64, I, c)

        hat_x[T, :], r, z = SolveP1(A_train[:,:,T], b_train[T,:], noisyx, 1, J)
        hat_z[T, :] = z
        istight = []
        for i = 1:c
            if isapprox(z[1, i], 1, atol = 1e-3) == 1
                push!(istight, i)
            end
        end

        if Experiment == 1
            for j = 1:length(istight)
                cone[iteration][j,:] = A[istight[j], :]/norm(A[istight[j], :])
            end
        else
            B = zeros(Float64, length(istight), n)
            for j = 1:length(istight)
                B[j,:] = A[istight[j], :]/norm(A[istight[j], :])
            end
            push!(cone, B)
        end
        status = SolveFP(cone, T)

        if status == :Infeasible
            re_solve_counter += 1
            z = []
            x, r, z, status = SolveP1m(A_train[:,:,1:T], b_train[1:T,:], x_train[:,:,1:T], T, J, instance, hat_x, hat_z)

            if status != :Optimal
                break
            end
            hat_x[1:T, :] = x
            hat_z[1:T, :] = z

            cone = fill(zeros(n,n), 1)

            for k = 1:Experiment
                istight = []
                for i = 1:c
                    if isapprox(z[k, i], 1, atol = 1e-3)
                        push!(istight, i)
                    end
                end
                if k == 1
                    for j = 1:length(istight)
                        cone[k][j,:] = A_train[istight[j], :, k]/norm(A_train[istight[j], :, k])
                    end
                else
                    B = zeros(Float64, length(istight), n)
                    for j = 1:length(istight)
                        B[j,:] = A_train[istight[j], :, k]/norm(A_train[istight[j], :, k])
                    end
                    push!(cone, B)
                end
            end
        end

        hatc1, loss1 = SolveP2(noisyp, cone, iteration)
        xhat  = zeros(Float32, TS, n)
        for test = 1:TS
            xhat[test, 1:n] = SolveModel(Atest[:, :, test], hatc1, btest[test, :], n)
        end
        errorrate = accuracy(testx, xhat)
        push!(Pred_Error_Evolution,errorrate)

        # Initialize arrays to store samples for heuristic-based adaptive sampling
        global bprime = zeros(Float64, S, c)
        global Aprime = zeros(Float64, c, n, S)
        xprime = zeros(Float64, S, dim)
        global zprime = zeros(Int16, S, c)
        phi = zeros(Float64, S)  # To store the objective values of (ASP)
        Boun = zeros(Float64, S) # To store the best bound for (ASP)
        G = zeros(Float64, S)

        for T = 1:S
            bprime[T, :] = b
            Aprime[:, :, T] = A
            for i = 1:dim
                Aprime[dim+1, i, T] = rand(rng, 5000:15000)/100
            end
            hatc2 = SampleC(cone) # Sample a random vector from the admissible set
            xprime[T, :] = SolveModel(Aprime[:,:,T], hatc2, bprime[T,:], n)
            temp, zprime[T, :] = tightind(Aprime[:,:,T], bprime[T,:], xprime[T,:])
        end

        # Solve the adaptive sampling problem for the samples in Aprime, bprime
        Results = zeros(Float64, S, 3)
        Results = pmap(T->maxmin(T, cone, Aprime[:,:,T], zprime[T,:]), 1:S, retry_delays=ones(10))
        for i = 1:S
            phi[i] = Results[i][1]
            Boun[i] = Results[i][2]
        end
        ind = findmax(phi)

        # Update A to the sample with the highest value of \eta
        A = Aprime[:,:,ind[2]]
        Experiment += 1
    end

    cd(string(root,"Results"))
    df1 = DataFrame(Predic_Error = Pred_Error_Evolution)
    s1 = @sprintf "Adaptive_Parallel_dual_%03.d_%03.d_%04.d.csv" instance dim S
    CSV.write(s1, df1)
end
