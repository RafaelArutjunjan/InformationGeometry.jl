
using SafeTestsets


@safetestset "Probability Objects" begin 
    include("testsets/probability_objects.jl")
end

@safetestset "Confidence Regions" begin 
    include("testsets/confidence_regions.jl")
end

@safetestset "More Boundary tests" begin 
    include("testsets/more_boundary_tests.jl")
end

@safetestset "ODE-based models" begin 
    include("testsets/diffeq_based_models.jl")
end

@safetestset "Model and Data Transformations" begin 
    include("testsets/model_data_transformations.jl")
end

@safetestset "In-place ModelMaps" begin 
    include("testsets/inplace_modelmaps.jl")
end

@safetestset "Inputting Datasets of various shapes" begin 
    include("testsets/inputting_datasets_various_shapes.jl")
end

@safetestset "DataSetUncertain" begin 
    include("testsets/datasetuncertain.jl")
end

@safetestset "Priors" begin 
    include("testsets/priors.jl")
end

# @safetestset "PEtabExtension Tests" begin 
#     include("testsets/petab_tests.jl")
# end

@safetestset "Kullback-Leibler Divergences" begin 
    include("testsets/kullback_leibler_divergences.jl")
end

@safetestset "Optimization Functions" begin 
    include("testsets/optimization_functions.jl")
end

@safetestset "MultistartFits" begin 
    include("testsets/multistartfits.jl")
end

@safetestset "ParameterProfiles" begin 
    include("testsets/parameterprofiles.jl")
end

@safetestset "Numerical Helper Functions" begin 
    include("testsets/numerical_helper_functions.jl")
end

@safetestset "Differential Geometry - Geodesics" begin 
    include("testsets/differential_geometry_geodesics.jl")
end

@safetestset "Differential Geometry - Curvature" begin 
    include("testsets/differential_geometry_curvature.jl")
end

