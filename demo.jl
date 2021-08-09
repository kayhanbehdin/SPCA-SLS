using LinearAlgebra, JuMP, Distributions, Random, MAT, TSVD, Gurobi
include("SPCA_SLS.jl")
include("SLSInitialization.jl")
env = Gurobi.Env()

p = 10000
n = 10000
s = 5
## Data Generation
z = zeros(p)
z[1:s] = ones(s)
u = rand(p).*z
u = u./norm(u)
theta = 1
Sigma = Matrix(I, p, p)*1.0 + theta*u*u'
dist = MvNormal(Sigma)
X = rand(dist, n)'
G = X'*X/n
## Initialization
z0, M = SLSInitialization(X, G, s, 20, env, 1)
println("Intialization Completed.")
## Solving SPCA
u_hat,gap = SPCA_SLS(X, s, z0, 0, M, env, 30, 0.01)

error = sqrt(1 - (u_hat'*u)^2)

println("Estimation error:")
println(error)
println("Optimality gap:")
println(gap[end])
