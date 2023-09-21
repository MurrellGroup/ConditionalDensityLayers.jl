using Pkg; Pkg.activate("trainscripts")

using ConditionalDensityLayers.LogDensityLayers, Flux, Distributions, Plots, ParameterSchedulers

#
intervals = [[-10, 10]]

num_dists = 25000

conditions = [collect(n) for n in zip(rand(Float32, num_dists) * 10 .- 5, rand(Float32, num_dists) .+ 1)]
points = (x -> rand(Normal(x...))).(conditions)

conditions = reduce(hcat, conditions)
points = reduce(hcat, points)

traindata = (conditions, points)

model = LogDensityLayers.LogDensityLayer(
    intervals = intervals,
    numembeddings = 16,
    numhiddenlayers = 2,
    sizeof_conditionvector = 2
)

opt_state = Flux.setup(Adam(0.02), model)
schedule = Cos(λ0 = 1e-4, λ1 = 1e-2, period = 10)
@show typeof(points) typeof(conditions)
subgroups = LogDensityLayers.stratified_subgroups(model.interval_infimums, model.interval_supremums, 10)
for (eta, epoch) in zip(schedule, 1:30)
    Flux.adjust!(opt_state, eta)
 	
	∑ls = 0
	for i in 1:100:num_dists-99
    	ls, grads = Flux.withgradient(model) do m
			LogDensityLayers.loss(m, points[:, i:i+99], conditions[:, i:i+99]; subgroups = subgroups)
    	end


    	if isfinite(ls)
			Flux.update!(opt_state, model, grads[1])
    	end

		∑ls += ls
	end
    @info epoch ∑ls/(num_dists/100) 
end

cond = [-2, 1.3]
d = Normal(cond...)

xs = -10:0.1:10
true_ys = pdf.((d,), xs)
model_ys = [exp(only(model.logprobability_estimation(([x], cond)))) for x in xs]
@show model_ys
pl = plot(xs, true_ys)
plot!(pl, xs, model_ys) 
