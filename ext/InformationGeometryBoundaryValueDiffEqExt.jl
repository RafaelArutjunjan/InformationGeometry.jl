module InformationGeometryBoundaryValueDiffEqExt

using InformationGeometry, SciMLBase, BoundaryValueDiffEqShooting


InformationGeometry.LazyShooting(M::SciMLBase.AbstractODEAlgorithm) = BoundaryValueDiffEqShooting.Shooting(M)


end # module