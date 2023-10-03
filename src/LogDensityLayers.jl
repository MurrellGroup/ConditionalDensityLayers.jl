module LogDensityLayers

using ConditionalDensityLayers: AbstractConditionalDensityLayer
using StatsBase: mean
using Flux


export LogDensityLayer

# --- LogDensityLayer --- #

struct LogDensityLayer <: AbstractConditionalDensityLayer
    point_encoder
	condition_encoder
	logprobability_decoder
end

Flux.@functor LogDensityLayer

LogDensityLayer(; point_encoder, condition_encoder, logprobability_decoder) = LogDensityLayer(point_encoder, condition_encoder, logprobability_decoder)

(l::LogDensityLayer)(point, condition) = vcat(l.point_encoder(point), l.condition_encoder(condition)) |> l.logprobability_decoder

# --- Losses --- #

truesampleloss(logp, ∫f) = mean(-(logp .- log.(∫f)))

normalizationloss(∫f) = Flux.mse(log.(∫f), Float32(0))

standardloss(; logp, ∫f) = truesampleloss(logp, ∫f) + normalizationloss(∫f)

# --- Utilities --- #

function deepNN(;din, dout, demb::Integer = 16, nhidden::Integer = 1, σ = relu, dropout = 0.0f0)

    hiddenlayer_parts = (Dense(demb => demb, σ), LayerNorm(demb), Dropout(dropout))

    return Chain(
        Dense(din => demb, σ),
        [hiddenlayer_parts[1 + (i % 3)] for i in 3:3nhidden]...,
        Dense(demb => dout)
    )
end

end
