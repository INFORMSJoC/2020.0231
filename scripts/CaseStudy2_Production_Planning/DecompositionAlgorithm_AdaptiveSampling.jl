function adaptive_sampling(A,b,obj,noisyp,Atest,btest,testx,instance,S)
    Experiment = 1

    I = 100
	TS = 100
    re_solve_counter = 0
    cone = fill(zeros(c,n), 1)
    Pred_Error_Evolution = zeros(Float64, I, 10)
    ind = 0

    A_train = zeros(Float64, c, n, I)
    b_train = zeros(Float64, I, c)
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
        b_train[Experiment, :] = deepcopy(b);
        x = SolveModel(A, obj, b, n)
        J = 30;
        noisy_x = zeros(Float64, J, n)
        for j = 1:J
            for k = 1:n
                noisy_x[j, k] = x[k] + 3*randn(rng, Float64)
            end
        end
        x_train[:,:,Experiment] = deepcopy(noisy_x);

        hat_x = zeros(Float64, I, n)
        hat_z = zeros(Float64, I, c)

        hat_x[T, :], r, z = SolveP1(A_train[:,:,T], b_train[T,:], noisy_x, 1, J)
        istight = []
        istight, z = tightind(A_train[:,:,T], b_train[T,:], hat_x[T, :])
        hat_z[T, :] = z

        if Experiment == 1
            for j = 1:length(istight)
                cone[Experiment][j,:] = A[istight[j], :]/norm(A[istight[j], :])
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

            cone = fill(zeros(c,n), 1)

            for k = 1:T
                istight = []
                istight, z = tightind(A_train[:,:,k], b_train[k,:], hat_x[k, :])
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

        for estimate = 1:10
            hatc, loss = SolveP2(noisyp[estimate, :], cone, T)
            x_pred  = zeros(Float16, TS, n)

            for test = 1:TS
                x_pred[test, 1:n] = SolveModel(Atest[:, :, test], hatc, btest[test, :], n)
            end
            Pred_Error_Evolution[T, estimate] = infnorm(testx, x_pred)
        end

        # Initialize arrays to store samples for heuristic-based adaptive sampling
        global bprime = zeros(Float64, S, c)
        global Aprime = zeros(Float64, c, n, S)
        xprime = zeros(Float64, S, dim)
        global zprime = zeros(Int16, S, c)
        phi = zeros(Float64, S) # To store the objective values of (ASP)
        Boun = zeros(Float64, S) # To store the best bound for (ASP)
        G = zeros(Float64, S)

        for sample = 1:S
            mutrain = zeros(Float64, II, JJ)
            Dtrain = zeros(Float64, JJ, TT)
            for i = 1:II
                for j = 1:JJ
                    mutrain[i,j] = (rand(rng, 50:100)/100)*Mumax[i,j]
                    for t = 1:TT
                        Dtrain[j, t] = (rand(rng, 90:110)/100)*Nominal_Demand[j]
                    end
                end
            end
            Aprime[:,:,sample], bprime[sample, :] = ModelParameters(II,JJ,TT,mutrain,Dtrain)
            hat_c = SampleC(cone) # Sample a random vector from the admissible set
            xprime[sample, :] = SolveModel(Aprime[:,:,sample], hat_c, bprime[sample,:], n)
            temp, zprime[sample, :] = tightind(Aprime[:,:,sample], bprime[sample,:], xprime[sample,:])
        end

        # Solve the adaptive sampling problem for the samples in Aprime, bprime
        Results = zeros(Float64, S, 3)
        Results = pmap(Tr->maxmin(Tr, cone, Aprime[:,:,Tr], zprime[Tr,:]), 1:S, retry_delays=ones(10))
        for i = 1:S
            phi[i] = Results[i][1]
            Boun[i] = Results[i][2]
        end
        ind = findmax(phi)
        
        # Update A, b to the sample with the highest value of \eta
        b = bprime[ind[2],:]
        A = Aprime[:,:, ind[2]]
        Experiment += 1
    end
    cd(string(root,"Results"))
    df = DataFrame(Pred_Error_Evolution)
    s = @sprintf "PredictionError_dim_%03.d_apaptivesamp_instance_%03.d.csv" dim instance
    CSV.write(s, df)
end
