# Code for training and testing ConditionalDensityLayers

module Gym

import Random, StatsBase

#==== DATA GENERATION ====#

function generate_simple_1D_gaussian_dataset(; numobs = 10000, μ_interval = (-5, 5), σ²_interval = (0.5, 1.5))
    μs = [rand(Uniform(first(μ_interval), last(μ_interval))) for _ in 1:numobs]
    σ²s = [rand(Uniform(first(σ²_interval), last(σ²_interval))) for _ in 1:numobs]

    data = [(point = rand(Normal(μ, σ²), 1), conditionvector = [μ, σ²]) for (μ, σ²) in zip(μs, σ²s)]

    filter!(data) do (point, _conditionvector)
        first(only(logdensitylayer.intervals)) < only(point) < last(only(logdensitylayer.intervals))
    end

    data
end


mutable struct ConditionalDensityArena
    manifoldD::Int
    SD::Int #The S dimension
    mixK::Int #Number of components in the distribution
    outputD::Int #Dimension of the domain of the distribution
    eW::Matrix{Float64}
    dW::Matrix{Float64}
    function ConditionalDensityArena(manifoldD, SD, mixK, outputD)
        #These two matrices fully encode the structure of the mapping, which is Fourier-ish
        eW = randn(manifoldD, SD ÷ 2)
        dW = randn(SD,2mixK + mixK*outputD) #weights, iso stds (per component), plus one outputD mu for each component
        return new(manifoldD, SD, mixK, outputD,eW,dW)
    end
end

stan(x) = x ./ sum(x)

sensible_t(x) = log(exp(x) + 1.5) #Transform to sensible positive range

#Fourier-ish from low dim manifold to S
function lowD2S(X, cdg::ConditionalDensityArena)
    eWtX = cdg.eW'X
    return [cos.(eWtX); sin.(eWtX)]
end

function sample_isoGNN(w,mus,std)
    c = StatsBase.sample(1:length(w),StatsBase.Weights(w))
    mus[:,c] .+ (randn(length(mus[:,c])) .* std[c])
end

function sample_from_conditional_density(S,n,cdg::ConditionalDensityArena)
   theta = cdg.dW'S
    samps = []
    for i in 1:size(theta,2)
        v = theta[:,i]
        w = stan(exp.(v[end-2cdg.mixK+1:end-cdg.mixK] ./ 2))
        stds = sensible_t.(v[end-cdg.mixK+1:end]) .* 3
        mus = reshape(v[1:cdg.mixK*cdg.outputD],cdg.outputD,cdg.mixK)
        push!(samps,stack([sample_isoGNN(w,mus,stds) for k in 1:n]))
    end
    return samps
end

function conditional_distribution(S,cdg::ConditionalDensityArena)
    #To do - not used for training, but will be needed to validate
    #Returns the distribution equivalent to what is sampled from in sample_from_conditional_density(...)
    #Should return a Julia distribution
end

function generate_dataset(num_conditions, CDG::ConditionalDensityArena)
    mX = randn(CDG.manifoldD, num_conditions)
    S = lowD2S(mX, CDG)
    samps = reduce(hcat, sample_from_conditional_density(S, 1, CDG))
    return Float32.(samps), Float32.(S)
end

end