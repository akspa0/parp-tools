namespace WowViewer.Core.Runtime.PromoVideo;

/// <summary>Factory for the first built-in renderer-tour recipe.</summary>
public static class BuiltinFeatureTourRecipes
{
    /// <summary>
    /// Creates the initial clean-scene overview from the loaded camera-path label. Only the file
    /// name is retained, so callers may pass an imported M2/MDX path without leaking its directory.
    /// </summary>
    public static FeatureTourRecipe CreateCameraPathOverview(string? cameraPathIdentity, int fps)
    {
        string pathName = GetSafePathName(cameraPathIdentity);
        return new FeatureTourRecipe(
            FeatureTourRecipe.SchemaV1,
            "camera-path-overview",
            $"Camera Path Overview — {pathName}",
            "1",
            RequiresWarmPath: true,
            new FeatureTourCaptureSettings(fps, FullFrameWithTourOverlay: true),
            [
                new FeatureTourBeat(0, FeatureTourPresentationKind.Callout, "camera-path", "Cinematic Camera Path", "Playback follows the loaded client camera path after bounded path warmup.", 1.5),
                new FeatureTourBeat(2.0, FeatureTourPresentationKind.Callout, "renderer-capture", "Direct Renderer Capture", "The video receives frames from the viewer renderer, not desktop capture.", 1.5),
                new FeatureTourBeat(4.0, FeatureTourPresentationKind.Callout, "renderer-benchmark", "Renderer Benchmark", "The adjacent receipt preserves frame-time samples and hitches from this run.", 1.5),
            ]);
    }

    /// <summary>
    /// Creates a feature-tour recipe with callout beats for an authentic flight along a taxi path.
    /// </summary>
    public static FeatureTourRecipe CreateTaxiRouteOverview(
        int routeId,
        string? routeLabel,
        string? fromStation,
        string? toStation,
        int fps)
    {
        string label = string.IsNullOrWhiteSpace(routeLabel) ? $"Route #{routeId}" : routeLabel;
        string origin = string.IsNullOrWhiteSpace(fromStation) ? "Origin" : fromStation;
        string destination = string.IsNullOrWhiteSpace(toStation) ? "Destination" : toStation;

        return new FeatureTourRecipe(
            FeatureTourRecipe.SchemaV1,
            "taxi-route-overview",
            $"Taxi Flight — {label}",
            "1",
            RequiresWarmPath: false,
            new FeatureTourCaptureSettings(fps, FullFrameWithTourOverlay: true),
            [
                new FeatureTourBeat(0.0, FeatureTourPresentationKind.Callout, "departure", $"Departing: {origin}", $"In-flight on taxi route #{routeId}.", 2.0),
                new FeatureTourBeat(3.0, FeatureTourPresentationKind.Callout, "cruise", "Authentic Flight Path", "Following the DBC taxi flight path with smoothed directional camera.", 2.5),
                new FeatureTourBeat(6.5, FeatureTourPresentationKind.Callout, "destination", $"Approaching: {destination}", "Inbound approach to destination flight node.", 2.5),
            ]);
    }

    private static string GetSafePathName(string? cameraPathIdentity)
    {
        string fileName = Path.GetFileNameWithoutExtension(cameraPathIdentity?.Trim() ?? string.Empty);
        return string.IsNullOrWhiteSpace(fileName) ? "Loaded Camera Path" : fileName;
    }
}
