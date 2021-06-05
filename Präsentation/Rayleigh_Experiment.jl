using Manopt, Manifolds, Random
Random.seed!(42)
n = 50
A = randn(n,n)
A = ( A + A' ) / 2
F(::Sphere, p::Array{Float64,1}) = p' * A * p
gradF(::Sphere, p::Array{Float64,1}) = 2 * (A * p - p * p' * A * p)
HessF(::Sphere, p::Array{Float64,1}, X::Array{Float64,1}) = 2 * (A * X - p * p' * A * X - X * p' * A * p - p * p' * X * p' * A * p)
M = Sphere(n - 1)
x = random_point(M)
trust_regions!(
    M,
    F,
    gradF,
    ApproxHessianSymmetricRankOne(M, x, gradF; nu = sqrt(eps(Float64))),
    x;
    stopping_criterion=StopWhenAny(
        StopAfterIteration(500), StopWhenGradientNormLess(10^(-6))
    ),
    retraction_method=ProjectionRetraction(),
    θ=0.1,
    κ=0.9,
    trust_region_radius=1.0
)
