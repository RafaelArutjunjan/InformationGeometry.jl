module InformationGeometryLibGEOSExt


using InformationGeometry, LibGEOS

import InformationGeometry: ToGeos, UnionPolygons, ConsistentElDims


function ToGeos(pointlist::AbstractVector{<:AbstractVector{<:Number}})
    @assert 2 == ConsistentElDims(pointlist)
    text = "POLYGON(("
    for point in pointlist
        text *= string(point[1]) *" "* string(point[2]) *","
    end
    text *= string(pointlist[1][1]) *" "* string(pointlist[1][2]) * "))"
    LibGEOS.readgeom(text)
end
UnionPolygons(p1::AbstractVector{<:AbstractVector{<:Number}}, p2::AbstractVector{<:AbstractVector{<:Number}}) = LibGEOS.GeoInterface.coordinates(UnionPolygons(ToGeos(p1), ToGeos(p2)))[1]
UnionPolygons(p1::LibGEOS.Polygon, p2::LibGEOS.Polygon) = LibGEOS.union(p1,p2)


end # module