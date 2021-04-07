using Gurobi, JuMP, LinearAlgebra, TSVD, Dates, MAT
include("subgradLinReg.jl")


function spcaLinReg(X, s, z0, maxIter, maxT, minGap)


	main_X = copy(X)
	n, p = size(X)
	z = copy(z0)

	GGG = X'*X
	model = Model(with_optimizer(Gurobi.Optimizer,OutputFlag = 0, TimeLimit=300))
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
			OBJ, BETA, ALPHA, SIGMA = subgradLinReg(X[:,idx1], main_X[:,j], XpX, Xmult[idx1,j])
			c = c + OBJ

			Xalpha[:,counter_j] = abs.(X'*ALPHA)


			B[idx1,j] = BETA[:]
			B[j,j] = max(SIGMA - 1,0.0)
			X[:,j] = main_X[:,j]
		end
		sumM_1[idx1] = sumM_1[idx1] + sum(Xalpha[idx1[:],1:length(idx1)], dims = 2)/2
		sumM_1[idx2] = sumM_1[idx2] + sum(Xalpha[idx2[:],1:length(idx1)], dims = 2)
		sumM_2[idx1] = sumM_2[idx1] + sum(Xalpha[idx1[:],1:length(idx1)], dims = 1)'/2

		G = G -sumM_1 - sumM_2

		if (c<c_min)
			z_min = z
			c_min = c
			B_min = B
		end


		cost[counter] = c
		diff[counter] = ETA - c
		if((ETA-c)/c >=  -minGap)
			break
		end
		@constraint(model, eta >= c + sum( G[i]*(Z[i]-z[i]) for i=1:p))


		status=optimize!(model)
		z = value.(Z)
		ETA = value.(eta)
		gap[counter] = (ETA-c_min)/c_min
		t2 = now()
		diff_t = Dates.value(convert(Dates.Millisecond, t2-t0))
		if (diff_t/1000 > maxT*60)
			break
		end
		counter = counter + 1
	end
	z = copy(z_min)
	B = copy(B_min)

	a,b,c = tsvd(B)
	return a,-gap[1:counter-1]
end
