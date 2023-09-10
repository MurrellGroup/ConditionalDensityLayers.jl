"""
    vonmiseslogpdf(μ::T,κ::T,θ::Real) = κ * (cos(θ - μ) - 1) - log(besselix(zero(T), κ)) - Distributions.log2π

Log PDF of the Von-Mises distribution, with implicit wrapping (ie. no bounds on the PDF).

μ: anglular mean, in radians 

κ: concentration parameter. κ=0 is uniform
"""
vonmiseslogpdf(μ::T,κ::T,θ::Real) where T <: Real = κ * (cos(θ - μ) - 1) - log(besselix(zero(T), κ)) - Distributions.log2π


"""
    vonmisesplanelogpdf(μx::T,μy::T,θ::Real) = vonmiseslogpdf(atan(μx,μy),μy^2 + μx^2,θ)

Planar parameterization of the Von-Mises distribution.

The angle of the <μx,μy> vector (from <0,1>) controls the angular mean of the Von-Mises and the concentration is μx^2+μy^2.
"""
planarvonmiseslogpdf(μx::T,μy::T,θ::Real) where T <: Real = vonmiseslogpdf(atan(μx,μy),μy^2 + μx^2,θ)

#=
using Plots

function plot_wrapped!(f; kwargs...)
    ang = -pi:0.01:pi
    verts = -pi:pi/15:pi
    plot!(sin.(ang), cos.(ang),f.(ang), color = "blue", label = :none; kwargs...)
    plot!(sin.(ang), cos.(ang),f.(ang) .* 0.0, color = "blue", label = :none; kwargs...)
    for v in verts
        plot!([sin(v),sin(v)],[cos(v),cos(v)],[0.0,f(v)], linestyle = :dash, color = "blue", label = :none; kwargs...)
    end
end

#Plotting the <μx,μy>, and the resulting distributions
pl = plot()
μx,μy = 2.0,2.0
plot!([0.0,μx],[0.0,μy],[0.0,0.0], linestyle = :dashdot, color = "blue", label = "μx,μy = $μx,$μy")
plot_wrapped!(θ -> exp(planarvonmiseslogpdf(μx,μy,θ)), color = "blue")

μx,μy = -0.4,-0.1
plot!([0.0,μx],[0.0,μy],[0.0,0.0], linestyle = :dashdot, color = "red", label = "μx,μy = $μx,$μy")
plot_wrapped!(θ -> exp(planarvonmiseslogpdf(μx,μy,θ)), color = "red")
pl
=#
