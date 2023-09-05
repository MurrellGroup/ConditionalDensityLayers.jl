module LogDensityLayers

using ConditionalDensityLayers: AbstractConditionalDensityLayer
using Flux
import StatsBase

export LogDensityLayer

struct LogDensityLayer <: AbstractConditionalDensityLayer
    intervals
    logprobability_estimation
    constantfactor_estimation
end

Flux.@functor LogDensityLayer

LogDensityLayer(;intervals, logprobability_estimation, constantfactor_estimation) = LogDensityLayer(intervals, logprobability_estimation, constantfactor_estimation)

function LogDensityLayer(;intervals, sizeof_conditionvector::Integer, numembeddings::Integer = 16, numhiddenlayers::Integer = 1, σ = relu)
    (intervals .|> (interval -> first(interval) <= last(interval)) |> all) || throw("last(interval) is less than first(interval)")
    
    sizeof_pointvector = length(intervals)
    logprobability_estimation = Chain(
        Flux.Bilinear((sizeof_pointvector, sizeof_conditionvector) => numembeddings, σ),
        [iseven(i) ? Dense(numembeddings => numembeddings, σ) : LayerNorm(numembeddings) for i in 2:2*numhiddenlayers]...,
        Dense(numembeddings => 1)
    )
    constantfactor_estimation = Chain(
        Dense(sizeof_conditionvector => numembeddings, σ),
        [iseven(i) ? Dense(numembeddings => numembeddings, σ) : LayerNorm(numembeddings) for i in 2:2*numhiddenlayers]...,
        Dense(numembeddings => 1)
    )
    return LogDensityLayer(intervals, logprobability_estimation, constantfactor_estimation)
end

domainvolume(l::LogDensityLayer) = prod(last.(l.intervals) .- first.(l.intervals))

uniformrand(T::Type, l::LogDensityLayer) = first.(l.intervals) .+ rand(T, length(l.intervals)) .* (last.(l.intervals) .- first.(l.intervals))

logprobabilityloss(logprobability) = -sum(logprobability)

constantfactorloss(constantfactor, ∫pdf) = Flux.mse(constantfactor, ∫pdf)

normalizationloss(∫pdf) = Flux.mse(∫pdf, Float32[1])

function loss(l::LogDensityLayer, point, conditionvector; montecarlo_n = 100)
    logp = l.logprobability_estimation(point, conditionvector)
    
    k = l.constantfactor_estimation(conditionvector)

    ∫pdf = estimate_∫pdf(l, conditionvector, montecarlo_n = montecarlo_n)

    return logprobabilityloss(logp) + normalizationloss(∫pdf) + constantfactorloss(k, ∫pdf)
end

function sample(l::LogDensityLayer, conditionvector; montecarlo_n = 100)
    sample_points = [uniformrand(Float32, l) for _ in 1:montecarlo_n]
    sample_weights = [exp(only(model((p, conditionvector)).logp)) for p in sample_points]

    mask = [!isnan(x) && !isinf(x) for x in sample_weights]

    return StatsBase.sample(sample_points[mask], Weights(sample_weights[mask]))
end

end