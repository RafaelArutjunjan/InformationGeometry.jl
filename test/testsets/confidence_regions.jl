using InformationGeometry, Test, Plots

DS = DataSet([0,0.5,1,1.5],[1.,3.,7.,8.1],[1.2,2.,0.6,1.]);    DM = DataModel(DS,LinearModel)
DME = DataModel(DataSetExact([0,0.5,1,1.5],0.1ones(4),[1.,3.,7.,8.1],[1.2,2.,0.6,1.]), LinearModel)

sols = ConfidenceRegions(DM,1:2; tol=1e-6)
@test InformationGeometry.IsStructurallyIdentifiableAlong(DM, sols[1]) == true
@test size(SaveConfidence(sols,50)) == (50,4)
@test size(SaveGeodesics(sols,50)) == (50,2)
@test size(SaveDataSet(DM)) == (4,3)
@test ConfidenceRegionVolume(DM,sols[1];N=5000) < ConfidenceRegionVolume(DM,sols[2];N=5000,WE=true)

@test size(ConfidenceBands(DM,sols[1]; N=50, plot=false)) == (50,3)
@test size(PlotMatrix(inv(FisherMetric(DM,MLE(DM))),MLE(DM); N=50,plot=false)) == (50,2)

@test Plots.plot(DM) isa Plots.Plot
@test Plots.plot(DME) isa Plots.Plot
@test ResidualPlot(DM) isa Plots.Plot
@test VisualizeGeos([MBAM(DM)]) isa Plots.Plot

simplermodel(x,p) = p[1]*x;    DMSimp = DataModel(DS,simplermodel)
@test length(ConfidenceRegion(DMSimp,1.)) == 2
@test ModelComparison(DM,DMSimp)[2] > 0.

@test FindFBoundary(DM,1)[1] - FindConfBoundary(DM,1)[1] > 0
z = 8rand()
@test FindConfBoundary(DM, z; BoolTest=true)[1] - FindConfBoundary(DM, z; BoolTest=false)[1] < 1e-9