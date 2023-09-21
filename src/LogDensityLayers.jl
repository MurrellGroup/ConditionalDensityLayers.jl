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
end

Flux.@functor LogDensityLayer

function LogDensityLayer(;intervals, sizeof_conditionvector::Integer, numembeddings::Integer = 16, numhiddenlayers::Integer = 1, σ = relu, p = 0.0f0)
    (intervals .|> (interval -> first(interval) <= last(interval) && length(interval) == 2) |> all) || throw("last(interval) is less than first(interval)")

    interval_infimums = first.(intervals)
    interval_supremums = last.(intervals)
    layertypes = [Dense(numembeddings => numembeddings, σ), LayerNorm(numembeddings), Dropout(p)] 

    sizeof_pointvector = length(intervals)
    logprobability_estimation = Chain(
        Flux.Bilinear((sizeof_pointvector, sizeof_conditionvector) => numembeddings, σ),
        [layertypes[1 + (i % 3)] for i in 3:3*numhiddenlayers]...,
        Dense(numembeddings => 1)
    )
    return LogDensityLayer(interval_infimums, interval_supremums, logprobability_estimation)
end

domainvolume(; sups, infs) = prod(sups .- infs)

domainvolume(l::LogDensityLayer) = prod(l.interval_supremums .- l.interval_infimums)

uniformrand(T::Type, infimums, supremums) = infimums .+ rand(T, length(infimums)) .* (supremums .- infimums)

uniformrand(T::Type, infimums, supremums, n) = reduce(hcat, [uniformrand(T, infimums, supremums) for _ in 1:n])

function stratified_subgroups(supremums, infimums, strata_count)
    # Number of dimensions
    dim = length(supremums)
       
    # Calculate width of each stratum for each dimension
    widths = [(supremums[i] - infimums[i]) / strata_count for i in 1:dim]
    
    # Compute the sub-intervals for each dimension
    intervals = [[(infimums[i] + (j-1) * widths[i], infimums[i] + j * widths[i]) for j in 1:strata_count] for i in 1:dim]

    # Create combinations of intervals for each subgroup
    stratas = []
    for combo in Iterators.product(intervals...)
        infimums_group = Float32[elem[1] for elem in combo]
        supremums_group = Float32[elem[2] for elem in combo]
        subgroup = [infimums_group, supremums_group]
        push!(stratas, subgroup)
    end
    
    return stratas
end

function estimate_∫pdf(l, condition, infimums, supremums)
    if condition isa AbstractMatrix
        uniform_random_points = uniformrand(Float32, infimums, supremums, size(condition, 2))
    else
        uniform_random_points = uniformrand(Float32, infimums, supremums)
    end

    logprobs = l.logprobability_estimation((uniform_random_points, condition))

    probs = exp.(logprobs)
    
    mean_probdensity = StatsBase.mean(probs, dims = 1)
    
    return reshape(mean_probdensity .* domainvolume(l), size(mean_probdensity, 2))
end

logprobabilityloss(logprobability, ∫pdf) = StatsBase.mean(-(logprobability .- log.(∫pdf)))

normalizationloss(∫pdf) = Flux.mse(log.(∫pdf), Float32(0))

function loss(l::LogDensityLayer, point, condition; subgroups)
    logp = l.logprobability_estimation((point, condition))
    
    ∫pdf = StatsBase.mean(
			[estimate_∫pdf(l, condition, infimums, supremums) for (infimums, supremums) in subgroups], 
			dims = 1
		)

    logp = reduce(vcat, logp)
    ∫pdf = reduce(vcat, ∫pdf)

    return logprobabilityloss(logp, ∫pdf) + normalizationloss(∫pdf)
end

gumbel(n) = -log.(-log.(rand(n)))
gumbel_max_sample(x) = argmax(x + gumbel(length(x)))

function sample(l::LogDensityLayer, conditionvector; montecarlo_n = 100)
    sample_points = [uniformrand(Float32, l.interval_infimums, l.interval_supremums) for _ in 1:montecarlo_n]
    sample_weights = [only(l.logprobability_estimation((p, conditionvector))) for p in sample_points]

    mask = [!isnan(x) && !isinf(x) for x in sample_weights]

    return sample_points[mask][gumbel_max_sample(sample_weights[mask])]
end

end
