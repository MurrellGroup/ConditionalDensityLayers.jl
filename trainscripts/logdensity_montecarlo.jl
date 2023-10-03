# Dependencies
using Pkg; Pkg.activate("trainscripts")

using Flux, Distributions, Random, Plots, TOML, Dates, CUDA, ParameterSchedulers
using ConditionalDensityLayers.Gym, ConditionalDensityLayers.LogDensityLayers, ConditionalDensityLayers.MonteCarloMethods

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

batches = [(traindata[1][:, i:i+999], traindata[2][:, i:i+999]) for i in 1:1000:size(traindata[1], 2)-999]

# --- Set up model --- #
emb = hyperparameters["num_embeddings"]

eps = 1f-2
model = LogDensityLayer(
	point_encoder = identity,
	condition_encoder = LogDensityLayers.deepNN(
		din = datagen["S_dimension"],
		demb = emb,
		dout = 16,
		nhidden = 1,
		σ = eval(Symbol(hyperparameters["activation_function"])),
		dropout = 0.1f0 #hyperparameters["dropout"]
	),
	logprobability_decoder = LogDensityLayers.deepNN(
		din = datagen["output_dimension"] + 16,
		demb = emb,
		nhidden = 1,
		dout = 1,
		σ = eval(Symbol(hyperparameters["activation_function"])),
		dropout = 0.1f0
	)
)   

montecarlo_method = stratified_montecarlo(infimums=first.(datagen["intervals"]), supremums=last.(datagen["intervals"]), subgroups_per_dim=trainingparameters["subgroups_per_dim"])

opt_state = Flux.setup(Flux.Optimiser(WeightDecay(1f-3), Adam()), model)

schedule = CosAnneal(λ0 = 1e-4, λ1 = 1e-2, period = 10, restart = false)

# --- Training loop --- #
for (eta, epoch) in zip(schedule, 1:150)

    Flux.adjust!(opt_state, eta)

    trainmode!(model)
       
    ∑train_loss = 0 
    n_losses = 0 
    for (point, condition) in batches
        train_loss, grads = Flux.withgradient(model) do m
            logp = m(point, condition)
			∫f = estimate_integral(m, condition, montecarlo_method, trainingparameters["montecarlo_n"])
            
			LogDensityLayers.standardloss(logp=logp, ∫f=∫f)
        end 

        if isfinite(train_loss)
            Flux.update!(opt_state, model, grads[1])

            ∑train_loss += train_loss
            n_losses += 1
        end 
    end 

    avg_train_loss = ∑train_loss / n_losses
    
	@info epoch avg_train_loss
end
