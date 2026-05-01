
using InformationGeometry, PEtab, FiniteDifferences, Plots, Test

import InformationGeometry: Trafos, dims
function TestConversion(petab_prob::PEtabODEProblem, DM::AbstractDataModel=ConditionGrid(petab_prob), X::AbstractVector=MLE(DM); atol=1e-10, atol2=0.1)
    ##### Ensure correct copy over
    @test sum(abs, -InformationGeometry.loglikelihood(DM,X) - petab_prob.nllh(X)) < atol
    @test sum(abs, -Score(DM, X) - petab_prob.grad(X)) < atol
    @test sum(abs, FisherMetric(DM, X) - petab_prob.FIM(X)) < atol
    @test sum(abs, InformationGeometry.CostHessian(DM)(X) - petab_prob.hess(X)) < atol
    
    GradRes1 = rand(length(X));    GradRes2 = rand(length(X));    HessRes1 = rand(length(X), length(X));    HessRes2 = rand(length(X), length(X))
    Score(DM)(GradRes1, X);   petab_prob.grad!(GradRes2, X);    @test sum(abs, -GradRes1 -GradRes2) < atol
    FisherMetric(DM)(HessRes1, X);   petab_prob.FIM!(HessRes2, X);    @test sum(abs, HessRes1 -HessRes2) < atol
    InformationGeometry.CostHessian(DM)(HessRes1, X);   petab_prob.hess!(HessRes2, X);    @test sum(abs, HessRes1 -HessRes2) < atol

    ## Score seems to be slightly dissimilar
    if length(Conditions(DM)) > 1
        # Consistency of likelihoods of individual conditions with total
        @test sum(loglikelihood(Conditions(DM)[i], Trafos(DM)[i](X)) for i in eachindex(Conditions(DM))) == loglikelihood(DM, X)
        @test sum(abs, sum(Score(Conditions(DM)[i], Trafos(DM)[i](X)) for i in eachindex(Conditions(DM))) .- Score(DM, X)) < atol2
        @test sum(abs, sum(FisherMetric(Conditions(DM)[i], Trafos(DM)[i](X)) for i in eachindex(Conditions(DM))) .- FisherMetric(DM, X)) < atol
    end

    ### Check consistency of log-likelihood?
    cids = InformationGeometry.ConditionNames(DM);  j = 1
    # Test prediction
    @test !all(iszero, EmbeddingMap(DM, X, cids[j]))
    # Test derivative of prediction (currently FiniteDifferences)
    @test !all(iszero, EmbeddingMatrix(DM, X, cids[j]))

    # Xj = Trafos(DM)[j](X)
    # ## Check that reconstructed objective function and Score corresponds to PEtab.jl for simple model
    # ## Assuming normal data!
    # # dmj = DataModel(CompositeDataSet(xdata(Data(Conditions(DM)[j])), InformationGeometry.ReconstructDataMatrices(Data(Conditions(DM)[j]))[2], InformationGeometry.ysigmaProper(Conditions(DM)[j], Xj)), Predictor(Conditions(DM)[j]), Xj, true)
    # dmj = DataModel(Data(Conditions(DM)[j]), Predictor(Conditions(DM)[j]), Xj, true)
    # @test abs(loglikelihood(dmj, Xj) - loglikelihood(Conditions(DM)[j], Xj)) < atol
    # @test sum(abs, Score(dmj, Xj) - Score(Conditions(DM)[j], Xj)) < atol
    # # @test sum(abs, Score(dmj, Xj) - Score(Conditions(DM)[j], Xj)) < atol

    # # Compute reduced chi^2
    # @test sum(abs2, (EmbeddingMap(DMj, Xj) - ydata(DMj)) ./ ysigma(DMj, Xj)) / InformationGeometry.DataspaceDim(Conditions(DM)[j]) < 5
    
end


begin
    ## For local execution only
    const BenchmarkModels_PEtab_Path = expanduser("~/Software/Benchmark-Models-PEtab/Benchmark-Models")
    GetModelYaml(path::AbstractString=BenchmarkModels_PEtab_Path) = map(x->GetModelYaml(path, x), readdir(path))
    function GetModelYaml(path::AbstractString, ModelFolderName::AbstractString)
        Names = readdir(joinpath(path, ModelFolderName))
        Yamls = map(endswith(".yaml"), Names)
        !any(Yamls) && throw("Model Folder $ModelFolderName contains no yaml files!")
        if sum(Yamls) == 1
            Names[Yamls][1]
        else
            if isfile(joinpath(path, ModelFolderName,ModelFolderName *".yaml"))
                ModelFolderName *".yaml"
            else
                @warn("More than one yaml in $(ModelFolderName)! Taking first and hoping for the best!")
                Names[Yamls][1]
            end
        end
    end
    ModelYamlsContaining(S::AbstractString) = ModelYamlsContaining([S])
    ModelYamlsContaining(X::AbstractVector{<:AbstractString}) = filter(x->any(k->occursin(k, x), X), GetModelYaml())
    ModelYamlsContaining(N::Nothing=nothing) = GetModelYaml()
    extractBefore(S::AbstractString, token) = contains(S, token) ? (@view S[1:findfirst(token, S)[1]-1]) : S
    extractAfter(S::AbstractString, token) = contains(S, token) ? (@view S[findfirst(token, S)[1]+1:end]) : S
    extractBefore(tok::AbstractString) = S->extractBefore(S, tok)
    extractAfter(tok::AbstractString) = S->extractAfter(S, tok)
    # Sort A similarly to B but B only contains substrings of A
    FuzzySort(A::AbstractArray, B::AbstractArray) = A[reduce(vcat,[findall(a->occursin(b, a), A) for b in B])]

    # Name without .yaml suffix, unsafe loading
    LoadSinglePEtabModel(ModelFolderName::AbstractString) = (FolderName=extractBefore(extractAfter(ModelFolderName, "/"),".yaml");   PEtabModel(joinpath(BenchmarkModels_PEtab_Path, FolderName, FolderName *".yaml")))

    #### Fuzzy:
    ## Safely loads only PEtab.PEtabModel from string WITH yaml
    LoadPEtabModel(Name::AbstractString; kwargs...) = LoadPEtabModel([Name]; kwargs...)
    function LoadPEtabModel(Names::Union{Nothing,AbstractVector{<:AbstractString}}; kwargs...)
        ModelFolderNames = map(extractBefore(".yaml"), ModelYamlsContaining(Names))
        DMs = progress_pmap(ModelFolderName->try LoadSinglePEtabModel(ModelFolderName; kwargs...) catch E; println(E);  ModelFolderName end, ModelFolderNames)
        length(DMs) == 1 ? DMs[1] : DMs
    end
end


Bruno = PEtabODEProblem(LoadPEtabModel("Bruno"))
Boehm = PEtabODEProblem(LoadPEtabModel("Boehm"))
## BrunoDM = ConditionGrid(Bruno)
## BoehmDM = ConditionGrid(Boehm)

TestConversion(Bruno)
TestConversion(Boehm)
