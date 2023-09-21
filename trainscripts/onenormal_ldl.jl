using Pkg; Pkg.activate("trainscripts")

using ConditionalDensityLayers.LogDensityLayers, Flux, Distributions, Plots;

#
intervals = [[-10, 10]]

d = Normal(0, 3)
num_points = 20000

condition = [1, 2]
points = reduce(hcat, [Float32.(rand(d)) for _ in 1:num_points])
conditions = reduce(hcat, [Float32.(condition) for _ in 1:num_points]) 

batches = [(points[:, i:i+99], conditions[:, i:i+99]) for i in 1:100:num_points-101]

model = LogDensityLayers.LogDensityLayer(
    intervals = intervals,
    numembeddings = 16,
    numhiddenlayers = 1,
    p = 0.3,
    sizeof_conditionvector = 2,
    σ = Flux.swish
)

opt_state = Flux.setup(Adam(0.05), model)

subgroups = LogDensityLayers.stratified_subgroups(model.interval_infimums, model.interval_supremums, 2)
etaidx = 0
for epoch in 1:10
    global etaidx
    ∑ls = 0

    for x in batches
        ls, grads = Flux.withgradient(model) do m
	    LogDensityLayers.loss(m, x...; subgroups = subgroups)
        end

        if isfinite(ls)
	    Flux.update!(opt_state, model, grads[1])
        end

        ∑ls += ls
    
    end
 
    ∑ls / length(batches) < 4 && etaidx == 0 && (etaidx += 1; Flux.adjust!(opt_state, 0.01))
    ∑ls / length(batches) < 3 && etaidx == 1 && (etaidx += 1; Flux.adjust!(opt_state, 0.001))
    #∑ls / length(batches) < 2.53 && break


    @info epoch ∑ls/length(batches)
end



xs = -10:0.1:10
true_ys = pdf.((d,), xs)
model_ys = [exp(only(model.logprobability_estimation(([x], condition)))) for x in xs]

pl = plot(xs, true_ys)
histogram!(pl, reduce(vcat, points), nbins = 100, normalize = :pdf, alpha = 0.3)
plot!(pl, xs, model_ys) 
scatter!(pl, [xs[argmax(model_ys)]], [maximum(model_ys)], label = "max is $(maximum(model_ys))")
