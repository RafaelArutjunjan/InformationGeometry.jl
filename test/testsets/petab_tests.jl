
using InformationGeometry, PEtab, FiniteDifferences
using Test, Distributions, LinearAlgebra, Optim, ProgressMeter

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

BrunoDM = ConditionGrid(Bruno)

using Plots
plot(BrunoDM) |> display

sleep(30)