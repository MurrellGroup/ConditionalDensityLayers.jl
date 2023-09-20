module GaussianMixtureLayers

using Flux
using LinearAlgebra
using ConditionalDensityLayers: AbstractConditionalDensityLayer
using Distributions: log2π
using Distributions
import StatsBase

export VMMLayer

struct VMMLayer <: AbstractConditionalDensityLayer
    central_network::Chain
    weight_network::Chain
    μx_network::Chain
    μy_network::Chain
    K::Int 
    N_dims::Int
end

Flux.@functor VMMLayer

function VMMLayer(; K::Integer, N_dims, sizeof_conditionvector::Integer, numembeddings::Integer = 256, numhiddenlayers::Integer = 20, σ = relu, p = 0.05f0)
    lays = []
    for i in 2:3*numhiddenlayers
        if i % 3 == 1
            push!(lays, Dense(numembeddings => numembeddings, σ))
        elseif i % 3 == 2
            iseven(i) ? push!(lays, LayerNorm(numembeddings)) : push!(lays, Dropout(p))
        end
    end
    central_network = Chain(
            Dense(sizeof_conditionvector => numembeddings, σ),
            lays...,
        )
    μx_network = Chain(
            Dense(numembeddings => K*N_dims)
        )

    μy_network = Chain(
            Dense(numembeddings => K*N_dims)
        )
    weight_network = Chain(
            Dense(numembeddings => K)
        )
    VMMLayer(central_network, weight_network, μx_network, μy_network, K, N_dims)
end


# Degree 7 polynomial approximations of I₀(x), where 0 ≤ x < 7.5 and x ≥ 7.5 for small and large respectively.
bsl_small(x) = 1.0f0 .+ 0.25f0*x + 0.027777778f0*x.^2 + 0.0017361111f0*x.^3 + 6.9444446f-5*x.^4 + 1.9290123f-6*x.^5 + 3.93676f-8*x.^6 
bsl_large(x) = 0.3989423f0 .+ 0.049867786f0*x + 0.02805063f0*x.^2 + 0.029219503f0*x.^3 + 0.044718623f0*x.^4 + 0.0940852f0*x.^5 - 0.106990956f0*x.^6 

"""
Logarithm of the modified Bessel function of the first kind: log(I₀(x)). Note that log(besselix(0,x)) = log(I₀(x)) - x.
"""
function logbessi0(x)
    small_mask = x .< 7.5
    large_mask = x .>= 7.5
    xsmall = small_mask .* (x ./ 2) .^2 
    xlarge = large_mask .* (1 ./ x)
    ysmall = log.(xsmall .* bsl_small(xsmall) .+ 1 ).* small_mask
    ylarge = (log.(bsl_large(xlarge)) .+ log.(xlarge) ./ 2 .+ x) .* large_mask
    return ysmall .+ ylarge
end 

"""
Negative log likelihood of a Von-Mises mixture: -log(∑ᵢ (wᵢ × (∏ⱼ p_j(θ | μᵢ, κᵢ)), where pⱼ is the Von-Mises probability density function along dimension j.  
"""
function NLLVonMisesMixture(θ, μ, κ, w)
    batch = size(μ, 3)
    C₀ = logbessi0(κ) .+ log(2π)
    θ_broadcast = Flux.unsqueeze(θ, dims = 2)
    w_broadcast = Flux.unsqueeze(w, dims = 1)
    # note that summing over dim = 1 before the logsumexp gives us the cross-dim product. 
    terms =  log.(w_broadcast) .+ sum(κ .* cos.(θ_broadcast .- μ) .- C₀, dims = 1) 
    max_term = maximum(terms, dims = 2)
    log_sum = log.(sum(exp.(terms .- max_term), dims=2)) .+ max_term
    return -sum(log_sum) / (batch)
end

"""
Gets the Von-Mises parameters for the mixture from an embedding vector S. 
    κ = μx^2 + μy^2, and
    μ = atan(μx, μy)
"""
function get_vmm_params(v::VMMLayer, S)
    S_emb = v.central_network(S)
    weights = Flux.softmax(reshape(v.weight_network(S_emb), v.K, :),dims = 1)
    μx = reshape(v.μx_network(S_emb), 2, v.K, :)
    μy = reshape(v.μy_network(S_emb), 2, v.K, :)
    κ = μx.^2 .+ μy.^2
    μ = atan.(μx, μy)
    return μ, κ, weights
end

"""
Computes the mixture-NLL loss of the Von-Mises Mixture Layer, given an embedding vector S and true values θ. 
"""
function loss(v::VMMLayer, θ::AbstractVecOrMat{Float32}, S::AbstractVecOrMat{Float32})
    μ, κ, w = get_vmm_params(v, S)
    return NLLVonMisesMixture(θ, μ, κ, w)
end

"""
Generates one sample from a Von Mises mixture. 
"""
function VonMisesSample(μ, κ, w)
    c = StatsBase.sample(1:size(μ,2), StatsBase.Weights(w))
    VMx = VonMises(μ[1,c], κ[1,c])
    VMy = VonMises(μ[2,c], κ[2,c])
    x = rand(VMx)
    y = rand(VMy)
    return mod.([x, y] .+ π, 2π) .- π
end

""" 
Given an embedding vector S, generates N_samples samples from the resulting mixtures.
"""
function VMMSample(v::VMMLayer, S::AbstractVecOrMat{T}; N_samples::Integer = 1000) where T <: Real
    Sgpu = S |> gpu
    μ, κ, w = cpu(get_vmm_params(v, Sgpu))
    μ, κ, w = Float64.(μ), Float64.(κ), Float64.(w)
    samps = [stack([VonMisesSample(μ[:,:,i], κ[:,:,i], w[:,i]) for j in 1:N_samples]) for i in axes(S,2)]
    return samps
end

end