module InformationGeometryNonlinearSolveExt


using InformationGeometry, DerivableFunctionsBase, SciMLBase
using NonlinearSolveFirstOrder

import InformationGeometry: SolveConstrainedOptimisationProblem


function SolveConstrainedOptimisationProblem(objective_fixedt0::Function, constraint::Function, θguess::AbstractVector{<:Number}, N::Nothing; meth::SciMLBase.AbstractNonlinearAlgorithm=NonlinearSolveFirstOrder.TrustRegion(), kwargs...)
    SolveConstrainedOptimisationProblem(objective_fixedt0, constraint, θguess, meth; kwargs...)
end

end # module