# Dependencies
using Pkg; Pkg.activate("trainscripts")

using Flux, Distributions, Random, Plots, TOML, Dates, CUDA, ParameterSchedulers
import ConditionalDensityLayers.Gym, ConditionalDensityLayers.LogDensityLayers

# Read metadata
toml_metadata = TOML.parsefile("parameters/logdensitytrain.toml")

hyperparameters = toml_metadata["hyperparameters"]::Dict{String, Any}
trainingparameters = toml_metadata["trainingparameters"]::Dict{String, Any}
datagen = toml_metadata["datageneration"]::Dict{String, Any}
misc =  toml_metadata["misc"]::Dict{String, Any}

runID = join(misc["experimentID"], "_", Dates.now())
runpath = joinpath("runs", runID)

# Generate dataset - NOTE: Data generation is not set to a strict interval, which is expected by the LogDensityLayer
CDG = Gym.ConditionalDensityArena(
    datagen["manifold_dimension"],
    datagen["S_dimension"], 
    datagen["distribution_component_count"],
    datagen["output_dimension"]
)

#This line is broken for flux
#traindata, testdata = Flux.splitobs(Gym.generate_dataset(datagen["num_obs"], CDG), at = datagen["split_at"])
traindata = Gym.generate_dataset(round(Int, datagen["num_obs"] * datagen["split_at"]), CDG)
testdata = Gym.generate_dataset(round(Int, datagen["num_obs"] * (1 - datagen["split_at"])), CDG) 

# Set up tracking plots
num_samps = misc["plot_sample_size"]
num_conditions = 25

plot_S = Gym.generate_dataset(num_conditions, CDG)[2]
true_samples = Gym.sample_from_conditional_density(plot_S, num_samps, CDG)

tracking_plots = []

function make_plot(m)
	model_samples = [reduce(hcat, [LogDensityLayers.sample(m.ldl, m.encoder(condition); montecarlo_n = 1000) for _ in 1:num_samps]) for condition in eachcol(plot_S)]

	xlim = (-25, 25)#(m.ldl.interval_infimums[1], m.ldl.interval_supremums[1])
	ylim = (-25, 25)#(m.ldl.interval_infimums[2], m.ldl.interval_supremums[2])
	alpha = 0.1

	true_plots = [scatter(samps[1, :], samps[2, :], xlim = xlim, ylim = ylim, alpha = alpha, markersize = 2, markerstrokwidth = 0, label = :none) for samps in true_samples]
	tplot = plot(true_plots...)
	model_plots = [scatter(samps[1, :], samps[2, :], xlim = xlim, ylim = ylim, alpha = alpha, markersize = 2, markerstrokewidth = 0, label = :none, color = :red) for (i, samps) in enumerate(model_samples)]
	mplot = plot(model_plots...)

	overlay_plots = [scatter!(true_plots[i], samps[1, :], samps[2, :], xlim = xlim, ylim = ylim, alpha = alpha, markersize = 2, markerstrokewidth = 0, label = :none, color = :red) for (i, samps) in enumerate(model_samples)]
	oplot = plot(overlay_plots...)

	push!(tracking_plots, plot(tplot, oplot, size = (2000, 1000)))
	
	gui(tracking_plots[end])
end
	
# Initialize model
emb = hyperparameters["num_embeddings"]

eps = 1f-2
model = (
	encoder = Chain(
		Dense(datagen["S_dimension"] => emb, relu), LayerNorm(emb),
		Dense(emb => emb, relu), LayerNorm(emb), Dropout(0.1f0),
		Dense(emb => emb, relu), LayerNorm(emb), Dropout(0.1f0),
		Dense(emb => 16)
	),
	ldl = LogDensityLayers.LogDensityLayer(
		intervals = datagen["intervals"],
		sizeof_conditionvector = 16,
		numembeddings = emb,
		numhiddenlayers = 1,
		σ = eval(Symbol(hyperparameters["activation_function"])),
		p = 0.1f0
	)
)

# Send to gpu if stated in the toml file
# Note that gpu support is currently not working
device = Bool(trainingparameters["run_on_gpu"]) ? gpu : cpu

model = device(model)
traindata = device(traindata)
testdata = device(testdata)

# Training loop

stratas = LogDensityLayers.stratified_subgroups(model.ldl.interval_infimums, model.ldl.interval_supremums, trainingparameters["montecarlo_n"])

batches = [(traindata[1][:, i:i+999], traindata[2][:, i:i+999]) for i in 1:1000:size(traindata[1], 2)-999]

opt_state = Flux.setup(Flux.Optimiser(WeightDecay(1f-3), Adam()), model)
schedule = CosAnneal(λ0 = 1e-4, λ1 = 1e-2, period = 10, restart = false)
for (eta, epoch) in zip(schedule, 1:150)

    Flux.adjust!(opt_state, eta)

    trainmode!(model)
	
    ∑train_loss = 0
	n_losses = 0
    for (points, conditions) in batches
        train_loss, grads = Flux.withgradient(model) do m
			embeddings = m.encoder(conditions)
			LogDensityLayers.loss(m.ldl, points, embeddings; subgroups = stratas)
        end
    
        if isfinite(train_loss)
            Flux.update!(opt_state, model, grads[1])

			∑train_loss += train_loss
			n_losses += 1
		end
    end

    avg_train_loss = ∑train_loss / n_losses
    
    testmode!(model)

    test_loss = LogDensityLayers.loss(model.ldl, testdata[1], model.encoder(testdata[2]); subgroups = stratas)

    @info epoch avg_train_loss test_loss n_losses
	
	make_plot(model)
	
	n_losses == 0 && break
end
