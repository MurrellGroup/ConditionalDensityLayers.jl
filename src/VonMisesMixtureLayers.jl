module VonMisesMixtureLayers

using Flux
using LinearAlgebra
using ConditionalDensityLayers: AbstractConditionalDensityLayer
using Distributions: log2π
using Distributions
import StatsBase

export VMMLayer

struct VMMLayer 
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
            Dense(floor(Int, numembeddings) => K*N_dims)
        )

    μy_network = Chain(
            Dense(floor(Int, numembeddings) => K*N_dims)
        )
    weight_network = Chain(
            Dense(floor(Int, numembeddings) => K)
        )
    VMMLayer(central_network, weight_network, μx_network, μy_network, K, N_dims)
end
VMMLayer(K=10, N_dims=3, sizeof_conditionvector=3)


# Degree 6 polynomial approximations of I₀(x), where 0 ≤ x < 7.5 and x ≥ 7.5 for small and large respectively.
bsl_small(x) = 1.0f0 .+ 0.25f0*x + 0.027777778f0*x.^2 + 0.0017361111f0*x.^3 + 6.9444446f-5*x.^4 + 1.9290123f-6*x.^5 + 3.93676f-8*x.^6 
bsl_large(x) = 0.3989423f0 .+ 0.049867786f0*x + 0.02805063f0*x.^2 + 0.029219503f0*x.^3 + 0.044718623f0*x.^4 + 0.0940852f0*x.^5 - 0.106990956f0*x.^6 

#= Higher degrees for testing purposes. 
bsl_small(x) = 1.0f0 .+ 0.25f0*x + 0.027777778f0*x.^2 + 0.0017361111f0*x.^3 + 6.9444446f-5*x.^4 + 1.9290123f-6*x.^5 + 3.93676f-8*x.^6 + 6.1511873f-10*x.^7 + 7.594058f-12*x.^8 + 7.594058f-14*x.^9 + 6.276084f-16*x.^10 + 4.3583592f-18*x.^11
bsl_large(x) = 0.3989423f0.+ 0.049867786f0*x + 0.02805063f0*x.^2 + 0.029219503f0*x.^3 + 0.044718623f0*x.^4 + 0.0940852f0*x.^5 - 0.106990956f0*x.^6 + 22.725199f0*x.^7 - 1002.689f0*x.^8 + 31275.74f0*x.^9 - 593550.25f0*x.^10 + 2.6092888f6*x.^11 
=#

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
function NLLVonMisesMixture(θ, μ, κ, w; mask = 0)
    batch = prod(size(μ)[3:end])
    C₀ = logbessi0(κ) .+ log(2π)
    θ_broadcast = Flux.unsqueeze(θ, dims = 2)
    w_broadcast = Flux.unsqueeze(w, dims = 1)
    # note that summing over dim = 1 before the logsumexp gives us the cross-dim product. 
    terms =  log.(w_broadcast) .+ sum(κ .* cos.(θ_broadcast .- μ) .- C₀, dims = 1) 
    max_term = maximum(terms, dims = 2)
    if mask != 0 
        log_sum = reshape(mask,1,1,size(μ)[3:end]...) .* (log.(sum(exp.(terms .- max_term), dims=2)) .+ max_term)
    else 
        log_sum = log.(sum(exp.(terms .- max_term), dims=2)) .+ max_term
    end
    return -sum(log_sum) / (batch)
end

"""
Gets the Von-Mises parameters for the mixture from an embedding vector S. 
    κ = μx^2 + μy^2, and
    μ = atan(μx, μy)
"""
function get_vmm_params(v::VMMLayer, S)
    batch = size(S)[2:end]
    S_emb = v.central_network(S)
    N_dims = v.N_dims
    weights = Flux.softmax(reshape(v.weight_network(S_emb), v.K, batch...),dims = 1)
    μx = reshape(v.μx_network(S_emb), N_dims, v.K, batch...)
    μy = reshape(v.μy_network(S_emb), N_dims, v.K, batch...)
    κ = min.(max.(μx.^2 .+ μy.^2, 0.001),4000)
    μ = atan.(μx, μy)
    return μ, κ, weights
end

"""
Computes the mixture-NLL loss of the Von-Mises Mixture Layer, given an embedding vector S and true values θ. 
"""
function loss(v::VMMLayer, θ, S)
    μ, κ, w = get_vmm_params(v, S)
    return NLLVonMisesMixture(θ, μ, κ, w), (θ,μ)
end

"""
Generates one sample from a Von Mises mixture. 
"""
function VonMisesSample(μ, κ, w; n = 1)
    c = StatsBase.sample(1:size(μ,2), StatsBase.Weights(w))
    VMx = VonMises(μ[1,c], κ[1,c])
    VMy = VonMises(μ[2,c], κ[2,c])
    VMz = VonMises(μ[3,c], κ[3,c])
    if n == 1
        x = rand(VMx)
        y = rand(VMy)
        z = rand(VMz)

        return mod.([x, y, z] .+ π, 2π) .- π
    else 
        x = rand(VMx, n)
        y = rand(VMy, n)
        z = rand(VMz, n)
        return mod.(hcat(x,y,z) .+ π, 2π) .- π
    end
end


""" 
Given an embedding vector S, generates N_samples samples from the resulting mixtures.
"""
function VMMSample(v::VMMLayer, S::AbstractVecOrMat{T}; N_samples::Integer = 1000) where T <: Real
    μ, κ, w = get_vmm_params(v, S)
    μ, κ, w = Float64.(μ), Float64.(κ), Float64.(w)
    samps = [stack([VonMisesSample(μ[:,:,i], κ[:,:,i], w[:,i]) for j in 1:N_samples]) for i in axes(S,2)]
    return samps
end

"""
Nucleus sample from draws given probs. 
"""
function nucleussample(draws,probs; p = 0.9, num_samps = 1)
    cap = Int(round(length(probs)*p))
    keeps = sortperm(probs, rev = true)[1:cap]
    return draws[:,sample(keeps, num_samps)]
end

"""
Nucleus sample from the Von Mises distribution defined by μ, κ, w.
"""
function VonMisesNucleusSample(μ, κ, w; N_samps = 10000, Pd = 0.8, same_dist_samp = false)
    if same_dist_samp
        sampsd = VonMisesSample(μ, κ, w, n = N_samps)
    else
        # this corresponds to nucleus sampling over the whole mixture distribution, and is the default
        sampsd = stack([VonMisesSample(μ, κ, w) for i in 1:N_samps])
    end
    sampsd = stack([VonMisesSample(μ, κ, w) for i in 1:N_samps])
    samp_lossesd = [NLLVonMisesMixture(reshape(sampsd[:,i],:,1,1), μ, κ, w) for i in 1:N_samps]
    probs = exp.(-1 .* samp_lossesd)
    dihs_sampled = reshape(nucleussample(sampsd, probs, p = Pd, num_samps = 1),3,:)

    return dihs_sampled
end

function (v::VMMLayer)(S)
    return get_vmm_params(v, S)
end

end