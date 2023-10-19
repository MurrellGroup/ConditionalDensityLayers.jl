# MonteCarlo utils
#TODO: Add `sample` for `CompositeMonteCarloMethod`.
#TODO: Add sampling of CompositeMonteCarloMethod

module MonteCarloMethods

using StatsBase: mean

export
	# Structs
	MonteCarloMethod,
	CompositeMonteCarloMethod,
	# Functions
	estimate_integral,
	sample,
	# Functions for creating common MonteCarloMethods
	uniform_montecarlo,
	stratified_montecarlo

abstract type AbstractMonteCarloMethod end

struct MonteCarloMethod <: AbstractMonteCarloMethod
	logpdf
	sample
end

MonteCarloMethod(; logpdf, sample) = MonteCarloMethod(logpdf, sample)

function estimate_integral(logf, condition, montecarlo_method::MonteCarloMethod)
	point = montecarlo_method.sample(condition)	

    integral = exp.(logf(point, condition) .- montecarlo_method.logpdf(point, condition))

    return reshape(integral, size(integral, 2))
end

function estimate_integral(logf, condition, montecarlo_method::MonteCarloMethod, n::Integer)
	n_integral_estimates = ( _ -> estimate_integral(logf, condition, montecarlo_method) ).(1:n)
	
	return mean(n_integral_estimates)
end

gumbel(n) = -log.(-log.(rand(n)))
gumbel_max_sample(x) = argmax(x + gumbel(length(x)))
    
function sample(logf, conditionvector, montecarlo_method::MonteCarloMethod, n)
    sample_points = [montecarlo_method.sample(conditionvector) for _ in 1:n]
    sample_weights = [only( logf(p, conditionvector) ) for p in sample_points]
    
    mask = [!isnan(x) && !isinf(x) for x in sample_weights]
    
    return sample_points[mask][gumbel_max_sample(sample_weights[mask])]
end

struct CompositeMonteCarloMethod <: AbstractMonteCarloMethod
	submethods::Tuple
	weights
	
	CompositeMonteCarloMethod(submethods, weights) = new(submethods, weights ./ sum(weights))
end

function estimate_integral(logf, condition, composite_montecarlo_method::CompositeMonteCarloMethod, n = 1)
	integral_estimates = (submethod -> estimate_integral(logf, condition, submethod, n)).(composite_montecarlo_method.submethods)
	
	mean_integral_estimate = sum(integral_estimates .* composite_montecarlo_method.weights)
	
	return mean_integral_estimate
end

#TODO add more error handling
function uniform_montecarlo(; infimums, supremums, sample_infimums = infimums, sample_supremums = supremums)
	all(infimums .< supremums) || throw("Infimums must be strict less than supremums, for uniform_montecarlo.")

	MonteCarloMethod(
		logpdf = (point, condition) -> fill(log(1 / prod(supremums .- infimums)), 1, size(condition, 2)),
		sample = (condition)  -> sample_infimums .+ rand(length(sample_infimums), size(condition, 2)) .* (sample_supremums .- sample_infimums) .|> Float32	
	)
end
	
function stratified_montecarlo(; infimums, supremums, subgroups_per_dim)
    
    T = eltype(infimums)
    # Number of dimensions
    dim = length(infimums)

    # Calculate width of each stratum for each dimension
    widths = [(supremums[i] - infimums[i]) / subgroups_per_dim for i in 1:dim]
    
    # Compute the sub-intervals for each dimension
    intervals = [[(infimums[i] + (j-1) * widths[i], infimums[i] + j * widths[i]) for j in 1:subgroups_per_dim] for i in 1:dim]

    # Create combinations of intervals for each subgroup
    subgroups = []
    for combo in Iterators.product(intervals...)
        infimums_of_group = T[elem[1] for elem in combo]
        supremums_of_group = T[elem[2] for elem in combo]

        push!(subgroups, uniform_montecarlo(sample_infimums = infimums_of_group, sample_supremums = supremums_of_group, infimums = infimums, supremums = supremums))
    end
    
    return CompositeMonteCarloMethod(Tuple(subgroups), ones(T, length(subgroups)))
end

end
