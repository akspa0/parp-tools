namespace WowViewer.Core.Runtime.Marketing;

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

    private static string GetSafePathName(string? cameraPathIdentity)
    {
        string fileName = Path.GetFileNameWithoutExtension(cameraPathIdentity?.Trim() ?? string.Empty);
        return string.IsNullOrWhiteSpace(fileName) ? "Loaded Camera Path" : fileName;
    }
}
