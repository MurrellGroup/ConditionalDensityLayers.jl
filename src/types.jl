abstract type AbstractConditionalDensityLayer end

"""
MyConditionalDensityLayer <: AbstractConditionalDensityLayer should implement the functions:
- loss(l::MyConditionalDensityLayer, point, conditionvector)
- sample(l::MyConditionalDensityLayer, conditionvector)
"""