using Documenter
using InformationGeometry

makedocs(
    sitename = "InformationGeometry",
    authors = "Rafael Arutjunjan",
    format = Documenter.HTML(),
    modules = [InformationGeometry],
    pages = Any["Getting Started" => "index.md",
                "Basics of Information Geometry" => "basics.md",
                "Tutorial" => [     "Providing Data and Models" => "datamodels.md",
                                    "Maximum Likelihood Estimation" => "optimization.md",
                                    "Confidence Regions" => "confidence-regions.md",
                                    "Profile Likelihoods" => "parameter-profiles.md",
                                    "Model Transformations" => "transformations.md",
                                    "Useful Diagnostic Plots" => "plotting.md",
                                    "Parallelization" => "parallelization.md",
                                    "Exporting" => "exporting.md",
                                    # "Kullback-Leibler Divergences" => "kullback-leibler.md",
                                    ],
                # Advanced Tutorial: Confidence Bands, Geodesics, DataSetExact, Plotting, PDE / Stochastic Examples
                "Advanced Examples" => ["ODE-based Models" => "ODEmodels.md", "Advanced Dataset Types" => "AdvancedData.md", 
                                        "Linking Multiple Models and Datasets" => "ConditionGrids.md"],
                "Package Extensions" => ["PEtab.jl" => "PEtabExt.md"], #, "ProfileLikelihood.md"],
                "List of useful methods" => "methodlist.md",
                "Contributing" => "todo.md",
            ],
    warnonly=true # Do not throw error for docstrings which are not included in the documentation
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
