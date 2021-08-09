using Gurobi, JuMP, LinearAlgebra, TSVD, Dates, MAT


function SLSInitialization(X, G, s, N_tune, env, M_max)

	k = 2
	n, p = size(X)
	u0 =  diagTresh(G, k*s, p)
	pp = sortperm(abs.(u0), rev = true)
	z0 = zeros(p)
	z0[pp[1:s]] = ones(s)



	model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0))
    set_optimizer_attribute(model, "TimeLimit", 30)
	@variable(model, Z[1:k*s],Bin)
	@variable(model, beta[1:k*s,1:k*s])
	@constraint(model, sum(Z[ii] for ii=1:k*s) <= s)
	@constraint(model, [ii in 1:k*s, jj in 1:k*s], beta[ii,jj] <= M_max*Z[ii])
	@constraint(model, [ii in 1:k*s, jj in 1:k*s], beta[ii,jj] <= M_max*Z[jj])
	@constraint(model, [ii in 1:k*s, jj in 1:k*s], beta[ii,jj] >= -M_max*Z[ii])
	@constraint(model, [ii in 1:k*s, jj in 1:k*s], beta[ii,jj] >= -M_max*Z[jj])
	@constraint(model, [ii in 1:k*s], beta[ii,ii] == 0)
	sense = MOI.MIN_SENSE
	@objective(model, sense, sum( sum( (X[ii,pp[jj]]-sum( X[ii,pp[jjj]]*beta[jjj,jj]  for jjj = 1:k*s)   )^2 for ii = 1:n) for jj = 1:k*s) )
	status=optimize!(model)
	z00 = value.(Z)
	bet = value.(beta)
	beta_base = copy(bet)


	idx = findall(x-> x>=0.9, z00)
	z0 = zeros(p)
	z0[pp[idx]] = ones(size(idx))
	u_rec =  TrunPow(G, s, z0/norm(z0), p)
	u_rec = u_rec[:,1]
	pp = sortperm(abs.(u_rec), rev = true)
	z0 = zeros(p)
	z0[pp[1:s]] = ones(s)

	if (N_tune == 0)
		return z0
	end
	M = M_max
	decay = 0.75
	for i = 1:N_tune
		M = M*decay
		@constraint(model, [ii in 1:k*s, jj in 1:k*s], beta[ii,jj] <= M*Z[ii])
		@constraint(model, [ii in 1:k*s, jj in 1:k*s], beta[ii,jj] <= M*Z[jj])
		@constraint(model, [ii in 1:k*s, jj in 1:k*s], beta[ii,jj] >= -M*Z[ii])
		@constraint(model, [ii in 1:k*s, jj in 1:k*s], beta[ii,jj] >= -M*Z[jj])
		status=optimize!(model)
		beta_new = value.(beta)
		if( M < 0.01 || norm(beta_new-beta_base)/norm(beta_base) > 0.1 )
			break
		end

	end
	return z0, M/decay

end




function diagTresh(X, s, p)
    v = zeros(p)
    v = diag(X)
    pp = sortperm(v, rev = true)
    Y = zeros(s,s)
    Y[:,:] = X[pp[1:s],pp[1:s]]
    a,b,c = tsvd(Y)
    u = zeros(p)
    u[pp[1:s]] = a[:]
    return u
end



function TrunPow(X, s, u0, p)
    u = copy(u0)
    u_old = zeros(p)
    counter = 1
    diff = 1000
    while (counter <= 500 && diff >= 0.001)
        u_old = copy(u)
        temp = X*u
        x = temp/norm(temp)
        pp = sortperm(abs.(x), rev = true)
        u = zeros(p)
        u[pp[1:s]] = x[pp[1:s]]
        u = u/norm(u)
        diff = norm(u-u_old)
        counter = counter + 1
    end
    return u
end
