module InformationGeometryDelayDiffEqExt

using InformationGeometry, SciMLBase, DelayDiffEq


InformationGeometry.LazyMethodOfSteps(M::SciMLBase.AbstractODEAlgorithm) = DelayDiffEq.MethodOfSteps(M)


end # module