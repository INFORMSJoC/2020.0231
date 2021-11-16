function random_sampling(A,b,obj,noisyp,Atest,btest,testx,instance)
    I = 100 # I experiments
    J = 30 # J samples for each experiment
    TS = 100 # Number of test samples

    true_x  = zeros(Float64, I, n)
    noisy_x = zeros(Float64, J, n, I)
    b_train = zeros(Float64, I, c)
    A_train = zeros(Float64, c, n, I)

    # Generate training data
    for T = 1:I
        mu_train = zeros(Float64, II, JJ) # Generate random inputs
        D_train = zeros(Float64, JJ, TT)
        for i = 1:II
            for j = 1:JJ
                mu_train[i,j] = (rand(rng, 50:100)/100)*Mumax[i,j]
                for t = 1:TT
                    D_train[j, t] = (rand(rng, 90:110)/100)*Nominal_Demand[j]
                end
            end
        end
        A_train[:,:,T], b_train[T, :] = ModelParameters(II,JJ,TT,mu_train,D_train)
        if T == 1
            A_train[:,:,1] = deepcopy(A)
            b_train[1, :] = deepcopy(b)
        end
        true_x[T, 1:n] = SolveModel(A_train[:, :, T], obj, b_train[T, :], n)
    end

    t_instance_s = time()
    re_solve_counter = 0 
    global cone = fill(zeros(c,n), 1)
    Pred_Error_Evolution = zeros(Float64, I, 10)
    hat_x = zeros(Float64, I, n)
    hat_z = zeros(Float64, I, c)

    for T = 1:I
        istight = []

        for j = 1:J
            for k = 1:n
                noisy_x[j, k, T] = true_x[T,k] + 3*randn(rng, Float64)
            end
        end

        hat_x[T, :], r, z = SolveP1(A_train[:,:,T], b_train[T,:], noisy_x[:,:,T], 1, J)
        istight, z = tightind(A_train[:,:,T], b_train[T,:], hat_x[T, :])
        hat_z[T, :] = z

        if T == 1
            for j = 1:length(istight)
                cone[T][j,:] = A_train[istight[j], :, T]/norm(A_train[istight[j], :, T])
            end
        else
            B = zeros(Float64, length(istight), n)
            for j = 1:length(istight)
                B[j,:] = A_train[istight[j], :, T]/norm(A_train[istight[j], :, T])
            end
            push!(cone, B)
        end

        status = SolveFP(cone, T)

        if status == :Infeasible
            re_solve_counter += 1
            z = []
            x, r, z, status = SolveP1m(A_train[:,:,1:T], b_train[1:T,:], noisy_x[:,:,1:T], T, J, instance, hat_x, hat_z)

            if status2 != :Optimal
                break
            end
            hat_x[1:T, :] = x
            hat_z[1:T, :] = z

            global cone = []
            global cone = fill(zeros(c,n), 1)

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
            status = SolveFP(cone, T)
        end

        for estimate = 1:10
            hatc, loss = SolveP2(noisyp[estimate, :], cone, T)
            x_pred  = zeros(Float16, TS, n)

            for test = 1:TS
                x_pred[test, 1:n] = SolveModel(Atest[:, :, test], hatc, btest[test, :], n)
            end
            Pred_Error_Evolution[T, estimate] = infnorm(testx, x_pred)
        end
    end
    cd(string(root,"Results"))
    df = DataFrame(Pred_Error_Evolution)
    s = @sprintf "PredictionError_dim_%03.d_randomsamp_instance_%03.d.csv" dim instance
    CSV.write(s, df)

    println("Runtime ", time() - t_instance_s)
    println("Total no of times full problem was solved, ", re_solve_counter)
    println("-------------------------------------------------")
end
