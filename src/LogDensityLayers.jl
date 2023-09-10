module LogDensityLayers

using ConditionalDensityLayers: AbstractConditionalDensityLayer
using Flux
import StatsBase

export LogDensityLayer

# Add a point a encoder and a condition encoder, or just a condition encoder
struct LogDensityLayer <: AbstractConditionalDensityLayer
    interval_infimums
    interval_supremums
    logprobability_estimation
    constantfactor_estimation
end

Flux.@functor LogDensityLayer

function LogDensityLayer(;intervals, sizeof_conditionvector::Integer, numembeddings::Integer = 16, numhiddenlayers::Integer = 1, σ = relu)
    (intervals .|> (interval -> first(interval) <= last(interval) && length(interval) == 2) |> all) || throw("last(interval) is less than first(interval)")

    interval_infimums = first.(intervals)
    interval_supremums = last.(intervals)

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
    return LogDensityLayer(interval_infimums, interval_supremums, logprobability_estimation, constantfactor_estimation)
end

domainvolume(l::LogDensityLayer) = prod(l.interval_supremums .- l.interval_infimums)

uniformrand(T::Type, l::LogDensityLayer) = l.interval_infimums .+ rand(T, length(l.interval_infimums)) .* (l.interval_supremums .- l.interval_infimums)

uniformrand(T::Type, l::LogDensityLayer, n) = reduce(hcat, [uniformrand(T, l) for _ in 1:n])

function estimate_∫pdf(l, condition; montecarlo_n)
    if condition isa AbstractMatrix
        uniform_random_points = uniformrand(Float32, l, montecarlo_n * size(condition, 2))
        copied_condition = reduce(hcat, [condition for _ in 1:montecarlo_n])
    else
        uniform_random_points = uniformrand(Float32, l, montecarlo_n)
        copied_condition = reduce(hcat, [condition for _ in 1:montecarlo_n])
    end

    mean_probdensity = StatsBase.mean(l.logprobability_estimation((uniform_random_points, copied_condition)) .|> exp, dims = 1)
    return reduce(mean_probdensity .* domainvolume(l), size(mean_probdensity, 2))
end

logprobabilityloss(logprobability) = -sum(logprobability)

constantfactorloss(constantfactor, ∫pdf) = Flux.mse(constantfactor, ∫pdf)

normalizationloss(∫pdf) = Flux.mse(∫pdf, Float32[1])

function loss(l::LogDensityLayer, point, condition; montecarlo_n = 100)
    logp = l.logprobability_estimation((point, condition))
    
    k = l.constantfactor_estimation(condition)
    return mean(point)
    #∫pdf = estimate_∫pdf(l, condition, montecarlo_n = montecarlo_n)

    #return logprobabilityloss(logp)# + normalizationloss(∫pdf) + constantfactorloss(k, ∫pdf)
end

function sample(l::LogDensityLayer, conditionvector; montecarlo_n = 100)
    sample_points = [uniformrand(Float32, l) for _ in 1:montecarlo_n]
    sample_weights = [exp(only(model((p, conditionvector)).logp)) for p in sample_points]

    mask = [!isnan(x) && !isinf(x) for x in sample_weights]

    return StatsBase.sample(sample_points[mask], Weights(sample_weights[mask]))
end

end