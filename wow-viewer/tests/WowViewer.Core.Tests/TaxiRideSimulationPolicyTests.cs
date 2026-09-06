using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests;

public sealed class TaxiRideSimulationPolicyTests
{
    [Fact]
    public void ActiveRideRouteSurvivesVisibilityAndSelectionFilters()
    {
        Assert.True(TaxiRideSimulationPolicy.ShouldSimulateRoute(
            routeId: 42,
            activeRideRouteId: 42,
            showTaxi: false,
            showTaxiActors: false,
            hasTaxiSelection: false,
            routeVisible: false));
        Assert.True(TaxiRideSimulationPolicy.ShouldSimulateRoute(
            routeId: 7,
            activeRideRouteId: 42,
            showTaxi: true,
            showTaxiActors: true,
            hasTaxiSelection: true,
            routeVisible: true));
    }

    [Fact]
    public void InactiveRoutesKeepTheExistingPresentationGates()
    {
        Assert.True(TaxiRideSimulationPolicy.ShouldSimulateRoute(
            routeId: 7,
            activeRideRouteId: -1,
            showTaxi: true,
            showTaxiActors: true,
            hasTaxiSelection: true,
            routeVisible: true));
        Assert.False(TaxiRideSimulationPolicy.ShouldSimulateRoute(
            routeId: 7,
            activeRideRouteId: -1,
            showTaxi: true,
            showTaxiActors: true,
            hasTaxiSelection: true,
            routeVisible: false));
    }
}
