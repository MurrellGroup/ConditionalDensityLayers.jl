# MonteCarlo utils
#TODO: Add `sample` for `CompositeMonteCarloMethod`.
#TODO: Add unit tests.

module MonteCarloMethods

using StatsBase: mean

struct MonteCarloMethod
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

struct CompositeMonteCarloMethod
	submethods::Tuple
	weights
end

function estimate_integral(logf, condition, composite_montecarlo_method::CompositeMonteCarloMethod, n = 1)
	integral_estimates = (submethod -> estimate_integral(logf, condition, submethod, n)).(composite_montecarlo_method.submethods)
	
	mean_integral_estimate = sum(integral_estimates .* composite_montecarlo.weights) ./ sum(composite_montecarlo.weights)
	
	return mean_integral_estimate
end

#TODO add error handling
uniform_montecarlo(; infimums, supremums) =
	MonteCarlo(
		logpdf = (point, condition) -> fill(1 / prod(supremums .- infimums), size(condition, 2)),
		sample = (condition)  -> infimums .+ rand(eltype(infimums), length(infimums), size(condition, 2)) .* (supremums .- infimums)	
	)
	
function stratified_montecarlo(; infimums, supremums, subgroups_per_dim)
    
    T = eltype(infimums)
    # Number of dimensions
    dim = length(intervals)

    # Calculate width of each stratum for each dimension
    widths = [(supremums[i] - infimums[i]) / strata_count for i in 1:dim]
    
    # Compute the sub-intervals for each dimension
    intervals = [[(infimums[i] + (j-1) * widths[i], infimums[i] + j * widths[i]) for j in 1:strata_count] for i in 1:dim]

    # Create combinations of intervals for each subgroup
    subgroups = []
    for combo in Iterators.product(intervals...)
        infimums_of_group = T[elem[1] for elem in combo]
        supremums_of_group = T[elem[2] for elem in combo]

        push!(subgroups, uniform_montecarlo(infimums = infimums_of_group, supremums = supremums_of_group))
    end
    
    return CompositeMonteCarlo(Tuple(subgroups), ones(T, length(subgroups)))
end
