using Documenter
using InformationGeometry

makedocs(
    sitename = "InformationGeometry",
    authors = "Rafael Arutjunjan",
    format = Documenter.HTML(),
    modules = [InformationGeometry],
    pages = Any[
                "Getting Started" => "index.md",
                "Basics of Information Geometry" => "basics.md",
                "Tutorial" => Any[ "Providing Data and Models" => "datamodels.md",
                                    "Confidence Regions" => "confidence-regions.md",
                                    "Kullback-Leibler Divergences" => "kullback-leibler.md",
                                    "Exporting" => "exporting.md"],
                # Advanced Tutorial: Confidence Bands, Geodesics, Profile Likelihood, DataSetExact, Plotting, PDE / Stochastic Examples, Exporting
                "Advanced Examples" => Any["ODE based models" => "ODEmodels.md"],
                "List of useful methods" => "methodlist.md",
                "Contributing" => "todo.md",
            ],
)


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/RafaelArutjunjan/InformationGeometry.jl.git",
    # julia = "1.5",
    # osname = "linux",
    # target = "build",
)
