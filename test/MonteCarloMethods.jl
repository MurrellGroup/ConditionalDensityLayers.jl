using ConditionalDensityLayers.MonteCarloMethods

@testset "MonteCarloMethods" verbose=true begin
	infimums = zeros(Float32, 3)
	supremums = Float32[10.0, 3.0, 2π]
	
	@testset "Uniform" begin
		logf(points, condition) = fill(1 / prod(supremums .- infimums), size(condition, 2))

		montecarlo_method = uniform_montecarlo(infimums=infimums, supremums=supremums)
		
		condition = rand(2)
		
		integral_estimates = (n -> only(estimate_integral(logf, condition, montecarlo_method, n))).([1, 100, 1000])

		eps = abs.(integral_estimates .- 1)
		
		@test all(==(0), eps)

		num_samples = 1000

		samps = [sample(logf, condition, montecarlo_method, 2) for _ in 1:num_samples]

		percentage_of_samps_larger_than_mean = count(x -> x[1] > (supremums[1] - infimums[1]) / 2, samps) / num_samples

		@test isapprox(percentage_of_samps_larger_than_mean, 0.5) atol=0.05
	end
	
	@testset "Stratified (CompositeMonteCarloMethod)" begin
		num_subgroups = 10

		logf(points, condition) = fill(1 / prod(supremums/num_subgroups .- infimums/num_subgroups), size(condition, 2))

        montecarlo_method = stratified_montecarlo(infimums=infimums, supremums=supremums, subgroups_per_dim=num_subgroups)
  
        condition = rand(2)
  
        integral_estimates = (n -> only(estimate_integral(logf, condition, montecarlo_method, n))).([1, 100, 1000])
  
        eps = abs.(integral_estimates .- 1)
          
        @test all(==(0), eps)
	end
end	
		

		

	

