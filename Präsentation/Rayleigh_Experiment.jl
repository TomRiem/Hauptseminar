using Manopt, Manifolds, Random
Random.seed!(42)
n = 50
A = randn(n,n)
A = (A+A')/2
F(::Sphere, p::Array{Float64,1}) = p'*A*p
gradF(::Sphere, p::Array{Float64,1}) = 2*(A*p-p*p'*A*p)
M = Sphere(n-1)
x = random_point(M)

trust_regions!(M, F, gradF,
ApproxHessianSymmetricRankOne(M, x, gradF; nu=1e-8),
x; retraction_method=ProjectionRetraction())
