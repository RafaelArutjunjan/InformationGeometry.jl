module InformationGeometryNonlinearSolveExt


using InformationGeometry, DerivableFunctionsBase, SciMLBase
using NonlinearSolveFirstOrder

import InformationGeometry: SolveConstrainedOptimisationProblem


function SolveConstrainedOptimisationProblem(objective_fixedt0::Function, constraint::Function, θguess::AbstractVector{<:Real}, C::Real, N::Nothing; meth::SciMLBase.AbstractNonlinearAlgorithm=NonlinearSolveFirstOrder.TrustRegion(), kwargs...)
    SolveConstrainedOptimisationProblem(objective_fixedt0, constraint, θguess, C, meth; kwargs...)
end

end # module