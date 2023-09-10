# Dependencies
using Pkg; Pkg.activate(".")

using Flux, Distributions, Random, Plots, TOML, Dates, CUDA
import ConditionalDensityLayers.Gym, ConditionalDensityLayers.LogDensityLayers

# Read metadata
toml_metadata = TOML.parsefile("../parameters/logdensitytrain.toml")

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

# Initialize model
model = LogDensityLayers.LogDensityLayer(
    intervals = datagen["intervals"],
    sizeof_conditionvector = datagen["S_dimension"],
    numembeddings = hyperparameters["num_embeddings"],
    numhiddenlayers = hyperparameters["num_hidden_layers"],
    σ = eval(Symbol(hyperparameters["activation_function"]))
)

opt_state = Flux.setup(Adam(), model)

# Send to gpu if stated in the toml file
device = Bool(trainingparameters["run_on_gpu"]) ? gpu : cpu

model = device(model)
traindata = device(traindata)
testdata = device(testdata)

# Training loop
for epoch in 1:trainingparameters["num_epochs"]

    trainmode!(model)

    train_loss, grads = Flux.withgradient(model) do m
        LogDensityLayers.loss(m, traindata...; montecarlo_n = trainingparameters["montecarlo_n"])
    end

    grads = grads |> cpu

    if isfinite(train_loss)
        Flux.update!(opt_state, model, grads[1])
    end

    testmode!(model)

    test_loss = LogDensityLayers.loss(model, testdata...; montecarlo_n = trainingparameters["montecarlo_n"])

    @info epoch train_loss test_loss
end

# TODO: Setup plotting to check if it looks right