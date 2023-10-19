module LogDensityLayers

using ConditionalDensityLayers: AbstractConditionalDensityLayer
using ConditionalDensityLayers.MonteCarloMethods: AbstractMonteCarloMethod, estimate_integral
using StatsBase: mean
using Flux

export LogDensityLayer

# --- LogDensityLayer --- #

@kwdef struct LogDensityLayer{T1, T2, T3} <: AbstractConditionalDensityLayer where {T1, T2, T3}
    point_encoder::T1 = identity
	condition_encoder::T2 = identity
	logprobability_decoder::T3
end

Flux.@functor LogDensityLayer

(l::LogDensityLayer)(point, condition) = vcat(l.point_encoder(point), l.condition_encoder(condition)) |> l.logprobability_decoder

# --- Losses --- #

truesampleloss(logp, ∫f) = mean(-(logp .- log.(∫f)))

normalizationloss(∫f) = Flux.mse(log.(∫f), Float32(0))

standardloss(; logp, ∫f) = truesampleloss(logp, ∫f) + normalizationloss(∫f)

standardloss(l::LogDensityLayer, point, condition, montecarlomethod::AbstractMonteCarloMethod, montecarlo_n=1) = 
	standardloss(logp = l(point, condition), ∫f = estimate_integral(l, condition, montecarlomethod, montecarlo_n))

	
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
