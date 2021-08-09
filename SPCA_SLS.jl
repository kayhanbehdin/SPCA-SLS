using Gurobi, JuMP, LinearAlgebra, TSVD, Dates, MAT



function SPCA_SLS(X, s, z0, lambda, M, env, T_limit, tol)

	T_limit = 10*60
	main_X = copy(X)
	n,p = size(X)
	z = copy(z0)

	GGG = X'*X
	model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0))
	set_optimizer_attribute(model, "TimeLimit", T_limit)
	@variable(model, Z[1:p],Bin)
	@variable(model, eta)
	@constraint(model, sum(Z[i] for i=1:p) <= s)
	sense = MOI.MIN_SENSE
	@objective(model, sense,eta)

	c_min = 1e15
	z_min = zeros(p)
	B_min = zeros(p,p)
	Xalpha = zeros(p,s)
	B = zeros(p,p)
	G = zeros(p)
	maxIter = 20000
	gap = zeros(maxIter)
	counter = 1
	ETA = -1000
	c = 1e20
	diff = zeros(maxIter)
	cost = zeros(maxIter)

	sumM_1 = zeros(p)
	sumM_2 = zeros(p)

	Xmult = copy(GGG)
	normXj = zeros(p)
	for j = 1:p
		Xmult[j,j] = 0
		normXj[j] = norm(X[:,j])^2
	end
	absXmult = abs.(Xmult)
	t0 = now()

	G_tot = zeros(p,maxIter)
	Z_tot = zeros(p,maxIter)
	c_tot = zeros(maxIter)


	while(counter <= maxIter)

		B = zeros(p,p)
		G = zeros(p)
		c = 0
		idx1 = findall(x-> x.>0.1, z)
		idx2 = findall(x-> x.<0.1, z)

		sumM_1 = zeros(p)
		sumM_2 = zeros(p)

		c = 0.5*sum(normXj[idx2])
		sumM_1[idx2] = sumM_1[idx2] + sum(absXmult[idx2,idx2], dims = 2)/2
		sumM_2[idx2] = sumM_2[idx2] + sum(absXmult[idx2,idx2], dims = 1)'/2
		sumM_2[idx2] = sumM_2[idx2] + sum(absXmult[idx1,idx2], dims = 1)'




		XpX = GGG[idx1, idx1]
		for counter_j = 1:length(idx1)
			j = idx1[counter_j]
			X[:,j] = zeros(n)
			OBJ, BETA, ALPHA, SIGMA = subGrad(X[:,idx1], main_X[:,j], XpX, Xmult[idx1,j], lambda, M)
			idxx = findall(x -> abs.(x) <1, BETA)
			c = c + OBJ
			Xalpha[:,counter_j] = (X'*ALPHA)
			Xalpha[idx1,counter_j] = abs.(Xalpha[idx1,counter_j]-2*BETA*lambda)
			Xalpha[:,counter_j] = abs.(Xalpha[:,counter_j])

			B[idx1,j] = BETA[:]
			B[j,j] = max(SIGMA - 1,0.0)
			X[:,j] = main_X[:,j]
			G[j] = G[j] - lambda*BETA'*BETA
		end
		sumM_1[idx1] = sumM_1[idx1] + sum(Xalpha[idx1[:],1:length(idx1)], dims = 2)/2
		sumM_1[idx2] = sumM_1[idx2] + sum(Xalpha[idx2[:],1:length(idx1)], dims = 2)
		sumM_2[idx1] = sumM_2[idx1] + sum(Xalpha[idx1[:],1:length(idx1)], dims = 1)'/2

		G = G -M*sumM_1 - M*sumM_2

		if (c<c_min)
			z_min = copy(z)
			c_min = copy(c)
			B_min = copy(B)
		end


		cost[counter] = c
		diff[counter] = ETA - c
		G_tot[:,counter] = G
		Z_tot[:,counter] = z
		c_tot[counter] = c
		@constraint(model, eta >= c + sum( G[i]*(Z[i]-z[i]) for i=1:p))


		status=optimize!(model)
		z = value.(Z)
		ETA = value.(eta)
		gap[counter] = ((ETA-c_min)/c_min)
		println(-gap[counter])
		t2 = now()
		diff_t = Dates.value(convert(Dates.Millisecond, t2-t0))
		if (diff_t/1000 > T_limit || gap[counter] > -tol)
			break
		end
		counter = counter + 1
	end
	z = z_min
	B = B_min

	a,b,c = tsvd(B)

	return a,gap[1:counter-1],B
end




function subGrad(X, x, XpX, Xx, lambda, M)
    n,s = size(X)
    beta = zeros(s)
    alpha = x
    OBJ = 0.5*alpha'*alpha
    a1,a2,a3 = tsvd(X')
    a2 = a2[1]
    t = 1/(a2.^2+2*lambda)
    t = t/2
    v = beta
    counter = 1
    beta_old = 1000*ones(s)
    while (norm(beta - beta_old)/norm(beta_old) > 1e-3 && counter <= 500)
        theta = 2/(counter + 1)
        beta_old = copy(beta)
        alpha_old = copy(alpha)
        OBJ_old = OBJ
        y = beta + theta*(v-beta)
        temp = y - t*(XpX*y - Xx + 2*lambda*y)
        neg_idx = findall(x -> x.<0, temp)
        temp = abs.(temp)
        bad_idx = findall(x -> x.>M, temp)
        temp[bad_idx] = M*ones(length(bad_idx))
        temp[neg_idx] = -temp[neg_idx]
        beta = copy(temp)
        v = beta + (beta - beta_old)/theta
        counter = counter + 1
        alpha = x-X*beta
        OBJ = 0.5*alpha'*alpha + lambda*beta'*beta
        if (abs(OBJ-OBJ_old)/OBJ_old<1e-3)
            beta = copy(beta_old)
            alpha = copy(alpha_old)
            OBJ = OBJ_old
            break
        end
    end
    sigma = alpha'*alpha/n
    return OBJ, beta, alpha, sigma
end
