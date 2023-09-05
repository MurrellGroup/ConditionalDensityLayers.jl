using Pkg; Pkg.activate(".")

using ConditionalDensityLayers.LogDensityLayers

model = LogDensityLayer(;
    intervals = [(-3, 3)],
    sizeof_conditionvector = 8,
    numembeddings = 16,
    numhiddenlayers = 2
)

