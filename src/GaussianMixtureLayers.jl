module GaussianMixtureLayers

using Flux
using LinearAlgebra
using ConditionalDensityLayers: AbstractConditionalDensityLayer
import StatsBase

export GNNLayer
export GNNSettings

struct GNNLayer 
    central_network::Chain
    weight_network::Chain
    centroid_network::Chain
    std_network::Chain
    K::Int 
    N_dims::Int
end
Flux.@functor GNNLayer

function GNNSettings(sizeof_conditionvector; K = 20, N_dims = 3, numembeddings = 256, numhiddenlayers = 6, σ = relu, p = 0.05f0)
    return (
        K = K, 
        N_dims = N_dims,
        sizeof_conditionvector = sizeof_conditionvector,
        numembeddings = numembeddings,
        numhiddenlayers = numhiddenlayers, 
        σ = σ,
        p = p
        )
end

function GNNLayer(settings::NamedTuple)
    s = settings
    return GNNLayer(
        K = s.K,
        N_dims = s.N_dims,
        sizeof_conditionvector = s.sizeof_conditionvector,
        numembeddings = s.numembeddings,
        numhiddenlayers = s.numhiddenlayers,
        σ = s.σ,
        p = s.p
    )
end

function GNNLayer(; K::Integer, N_dims::Integer, sizeof_conditionvector::Integer, numembeddings::Integer = 256, numhiddenlayers::Integer = 20, σ = relu, p = 0.05f0)
    lays = []
    for i in 2:3*numhiddenlayers
        # alternate dense -> layernorm -> dense -> dropout -> ...
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
    weight_network = Chain(
            Dense(numembeddings => K)
        )

    centroid_network = Chain(
            Dense(numembeddings => K*N_dims)
        )
    std_network = Chain(
            Dense(numembeddings => K, softplus)
        )
    GNNLayer(central_network, weight_network, centroid_network, std_network, K, N_dims)
end

function (g::GNNLayer)(conditionvector)
    batch = size(conditionvector)[2:end]
    S_emb = g.central_network(conditionvector)
    weights = Flux.softmax(reshape(g.weight_network(S_emb), g.K, batch...), dims = 1)
    means = reshape(g.centroid_network(S_emb), g.N_dims, g.K, batch...)
    stds = reshape(g.std_network(S_emb), g.K, batch...) .+ 0.01f0 
    return weights, means, stds
end
#m.layers.DensityTransformer.gmm(randn(256,100,70))
function NLLIsotropicGMM(x, w, μ, σ)
    sz = size(μ)
    N = sz[1]
    K = sz[2]
    batch = prod(sz[3:end])
    x_broadcast = Flux.unsqueeze(x,dims=2)
    dots = sum((x_broadcast .- μ).*(x_broadcast .- μ), dims=1) 
    logpdfs = -(N/2)*log.(2π .* σ.^2).-(1 ./ (2 .* σ.^2)) .* reshape(dots, sz[2:end]...) 
    weighted_logpdfs = logpdfs .+ log.(w)   
    M = maximum(weighted_logpdfs, dims=1)  
    log_likelihood = M .+ log.(sum(exp.(weighted_logpdfs .- M), dims=1))  
    return -sum(log_likelihood)./(K .* batch)
end

function loss(g::GNNLayer, Y::AbstractVecOrMat{Float32}, conditionvector::AbstractVecOrMat{Float32})
    w, μ, σ = get_gmm_params(g, conditionvector)
    return NLLIsotropicGMM(Y, w, μ, σ)
end

end