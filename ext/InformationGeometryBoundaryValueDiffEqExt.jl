module InformationGeometryBoundaryValueDiffEqExt

using InformationGeometry, SciMLBase, BoundaryValueDiffEqShooting


InformationGeometry.LazyShooting(M::SciMLBase.AbstractSciMLAlgorithm) = BoundaryValueDiffEqShooting.Shooting(M)


end # module