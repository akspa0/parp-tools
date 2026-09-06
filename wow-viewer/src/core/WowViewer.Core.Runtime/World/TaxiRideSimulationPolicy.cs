namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Determines which taxi route poses must be simulated for a frame. An active
/// ride is an explicit runtime request and therefore bypasses presentation
/// filters such as route selection and visibility checkboxes.
/// </summary>
public static class TaxiRideSimulationPolicy
{
    public static bool ShouldSimulateRoute(
        int routeId,
        int activeRideRouteId,
        bool showTaxi,
        bool showTaxiActors,
        bool hasTaxiSelection,
        bool routeVisible)
    {
        bool visiblePresentationRoute = showTaxi && showTaxiActors && hasTaxiSelection && routeVisible;
        bool activeRideRoute = activeRideRouteId >= 0 && routeId == activeRideRouteId;
        return visiblePresentationRoute || activeRideRoute;
    }
}
