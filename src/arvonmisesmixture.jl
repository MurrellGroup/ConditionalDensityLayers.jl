using Flux
using LinearAlgebra
using Distributions: log2π
using Distributions
import StatsBase

export NARVMM
export NVMM
export NVMM_settings
export loss

struct VMM
    #Mixture components K -by- dimensions -by- batch
    μ::AbstractArray
    κ::AbstractArray
    logw::AbstractArray
end
Flux.@functor VMM

# Degree 6 polynomial approximations of I₀(x), where 0 ≤ x < 7.5 and x ≥ 7.5 for small and large respectively.
bsl_small(x) = 1.0f0 + 0.25f0 * x + 0.027777778f0 * x^2 + 0.0017361111f0 * x^3 + 6.9444446f-5 * x^4 + 1.9290123f-6 * x^5 + 3.93676f-8 * x^6
bsl_large(x) = 0.3989423f0 + 0.049867786f0 * x + 0.02805063f0 * x^2 + 0.029219503f0 * x^3 + 0.044718623f0 * x^4 + 0.0940852f0 * x^5 - 0.106990956f0 * x^6

"""
approxlogbesselix0(x)

An approximation of log(besselix(0,x)), but GPU+gradient friendly
"""
function approxlogbesselix0(x)
    if x < 7.5
        xsmall = (x / 2)^2
        return log(xsmall * bsl_small(xsmall) + 1) - x
    else
        xlarge = 1 / x
        return log(bsl_large(xlarge)) + log(xlarge) / 2
    end
end

#θ dims must be 1-by-dims-by-batch; vmm.? dims must be K-by-dims-by-batch
function logpdf(vmm::VMM, θ::AbstractArray{T}) where T <: Real
    @assert size(vmm.μ)[2:end] == size(θ)[2:end] && size(θ)[1] == 1
    log_ps = vmm.logw .+ vmm.κ .* (cos.(θ .- vmm.μ) .- 1) .- approxlogbesselix0.(vmm.κ) .- log(T(2π))
    max_term = maximum(log_ps, dims = 1)
    return log.(sum(exp.(log_ps .- max_term), dims=1)) .+ max_term
end

#θ dims must be 1-by-dims-by-batch; vmm.? dims must be K-by-dims-by-batch
function loss(vmm::VMM, θ; mask = ones(Bool, size(θ)))
    lls = logpdf(vmm, θ)
    return -sum(lls .* mask) / sum(mask)
end

function NVMM_settings(in_dim; K = 20, core_dim = 256, core_layers = 3, layer_norm = true, σ = Flux.leakyrelu)
    return (
        K = K, 
        in_dim = in_dim,
        core_dim = core_dim,
        core_layers = core_layers,
        layer_norm = true,
        σ = σ
        )
end
export NVMM_settings

#This is a single 1D neural Von-Mises mixture layer
struct NVMM{A, B, C} 
    K::A
    core::B
    out::C
end
Flux.@functor NVMM
NVMM(in_dim, K, core_dim, core_layers, layer_norm, σ,
    ) = NVMM(K,
            ifelse(layer_norm,
            Chain(Dense(in_dim, core_dim, σ),[Chain(LayerNorm(core_dim),Dense(core_dim, core_dim, σ)) for _ in 1:core_layers]...), #Adding LayerNorm here.
            Chain(Dense(in_dim, core_dim, σ),[Dense(core_dim, core_dim, σ) for _ in 1:core_layers]...)), #No LayerNorm here.
            Dense(core_dim, K*3) #Because we have 3 parameters for each mixture component (μ_x, μ_y, and weight)
            )
NVMM(settings::NamedTuple) = NVMM(settings.in_dim, settings.K, settings.core_dim, settings.core_layers, settings.layer_norm, settings.σ)
NVMM(in_dim::Int, settings::NamedTuple) = NVMM(in_dim, settings.K, settings.core_dim, settings.core_layers, settings.layer_norm, settings.σ) #For case where we inherit settings but change in_dim
function (l::NVMM)(x::AbstractArray{T}; maxκ = T(1000)) where T
    latent = l.core(x)
    vmm = reshape(l.out(latent), l.K, 3, size(x)[2:end]...)
    μx,μy,logw = selectdim(vmm, 2, 1), selectdim(vmm, 2, 2), Flux.logsoftmax(selectdim(vmm, 2, 3))
    κ = min.(μx.^2 .+ μy.^2,maxκ) #To prevent spikes to Inf
    μ = atan.(μx, μy)
    return VMM(μ, κ, logw) #, latent #Unsure yet if we want to return the mid state for stacking these somehow.
end


"""
anglefeats(θ, feature_width = 4)

θ is length(-by-batch). We want to return f(feature_width)-by-length(-by-batch)
"""
function anglefeats(θ::AbstractArray{T}; feature_width = 4) where T
    rsθ = reshape(θ, 1, size(θ)...)
    scals = rsθ.*gpu([1,2,3,4,7,11,17,23,31]) #Bad to have gpu in here. Needs to be figured out.
    return vcat(cos.(scals),sin.(scals))
end
#=
#To viz, and check continuity:
rad2ang(x) = 180*(x/pi)
x = vcat(collect(-pi:pi/180:pi),collect((-pi + pi/180):pi/180:pi))
plot(anglefeats(x)', label = :none, size = (1000,300), xticks = (1:45:length(x),Int.(round.(rad2ang.(x))[1:45:length(x)])), xrotation = 90, margin = 10Plots.mm)
=#

struct NARVMM
    NVMMs::Vector{NVMM}
end
Flux.@functor NARVMM
function NARVMM(L,settings; feature_width = 4)
    nvmms = [NVMM(settings.in_dim + (i-1)*(feature_width*4+2), settings)
        for i in 1:L]
    return NARVMM(nvmms)
end

#This function takes an S, and a sequence of angles, and returns an autoregressive sequence of VMMs, for each angle.
#Importantly, the N+1th VMM is conditioned on the Nth angle, and the Nth VMM.
#The reason for this is so we can recycle it for (inefficient) sampling, which we will rewrite later.
#Dims: the first dim of θ is each observed angle in sequence.
#So for standard AAs this would be 6-by-AAs(-by-batch)
function (l::NARVMM)(S::AbstractArray{T}, θ::AbstractArray{T}) where T
    angfeats = anglefeats(selectdim(θ, 1, 1))
    vmms = [l.NVMMs[1](S), l.NVMMs[2](vcat(S, angfeats))]
    for i in 3:length(l.NVMMs)
        angfeats = vcat(angfeats, anglefeats(selectdim(θ, 1, i-1)))
        push!(vmms, l.NVMMs[i](vcat(S, angfeats)))
    end
    return vmms
end

#Write a version of the above that directly accumulates the loss.
function loss(l::NARVMM, S::AbstractArray{T}, θ::AbstractArray{T}, mask) where T
    vmm = l.NVMMs[1](S)
    lls = logpdf(vmm, Flux.unsqueeze(selectdim(θ, 1, 1), dims = 1))
    angfeats = anglefeats(selectdim(θ, 1, 1))
    for i in 2:length(l.NVMMs)
        vmm = l.NVMMs[i](vcat(S, angfeats))
        lls = lls .+ logpdf(vmm, Flux.unsqueeze(selectdim(θ, 1, i), dims = 1))
        if i < length(l.NVMMs)
            angfeats = vcat(angfeats, anglefeats(selectdim(θ, 1, i))) #THIS WAS A BUG - THE FIRST DIHEDRAL IS INCLUDED TWICE IN THE ANGFEATS
        end
    end
    return -sum(lls .* mask)/sum(mask)
end

#with dihedral weights. Avoiding trying to package this into one function because of the gpu array issue
function weighted_triplet_loss(l::NARVMM, S::AbstractArray{T}, θ::AbstractArray{T}, mask, d2weight::T) where T
    vmm = l.NVMMs[1](S)
    lls = logpdf(vmm, Flux.unsqueeze(selectdim(θ, 1, 1), dims = 1))
    angfeats = anglefeats(selectdim(θ, 1, 1))
    vmm = l.NVMMs[2](vcat(S, angfeats))
    lls = lls .+ logpdf(vmm, Flux.unsqueeze(selectdim(θ, 1, 2), dims = 1)) .* T(d2weight) #downweighting the second dihedral
    angfeats = vcat(angfeats, anglefeats(selectdim(θ, 1, 2)))
    vmm = l.NVMMs[3](vcat(S, angfeats))
    lls = lls .+ logpdf(vmm, Flux.unsqueeze(selectdim(θ, 1, 3), dims = 1))
    return -sum(lls .* mask)/sum(mask)
end
export weighted_triplet_loss, loss

function discrete_sample(logits, cats)
    sm = softmax(logits)
    return [sample(cats, Weights(c)) for c in eachcol(sm)]
end

function vmm_sample(vmm::VMM)
    #First sample a component
    T = eltype(vmm.μ)
    c = Dihedragen.discrete_sample(vmm.logw, 1:size(vmm.logw,1))
    samps = T.([rand(VonMises(Float64(vmm.μ[c[i],i]), Float64(vmm.κ[c[i],i]))) for i in 1:size(vmm.logw,2)])
    #Rewrap the angles to be between -pi and pi
    return T.([samps[i] - 2π*floor((samps[i] + π)/(2π)) for i in 1:length(samps)])
end

function vmm_prob_curve(vmm::VMM, angle_range)
    T = eltype(vmm.μ)
    width = size(vmm.μ,2)
    angles = T.(collect(angle_range))
    return vcat([logpdf(vmm, fill(a, 1, width)) for a in angles]...)
end

export vmm_prob_curve

function nucleus_vmm_sample(vmm::VMM; p = 0.95, n = 200)
    samps = [vmm_sample(vmm) for i in 1:n]
    probs = [exp.(logpdf(vmm, reshape(s,1,:))) for s in samps]
    return [nucleussample([s[i] for s in samps],[pr[i] for pr in probs], p = p) for i in 1:length(samps[1])]
end

function model_sample(l::NARVMM, S::AbstractArray{T}; device = cpu, vmm_sampler = v -> nucleus_vmm_sample(v, p = T(0.95))) where T
    #First sample the first angle
    angles = zeros(T, length(l.NVMMs), size(S,2))
    vmm = l.NVMMs[1](S) |> cpu
    θ = vmm_sampler(vmm)
    angles[1,:] .= θ
    S = vcat(S, anglefeats(device(θ)))
    for i in 2:length(l.NVMMs)
        vmm = l.NVMMs[i](S) |> cpu
        θ = vmm_sampler(vmm)
        angles[i,:] .= θ
        if i < length(l.NVMMs)
            S = vcat(S, anglefeats(device(θ[:])))
        end
    end
    return angles
end
