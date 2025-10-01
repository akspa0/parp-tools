namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Configuration container for viewer generation. Values mirror the defaults in
/// <c>memory-bank/plans/viewer-diff-plan.md</c> and can be expanded alongside implementation.
/// </summary>
public sealed record ViewerOptions(
    string? DefaultVersion,
    (string From, string To)? DiffPair,
    int MinimapWidth,
    int MinimapHeight,
    double DiffDistanceThreshold,
    double MoveEpsilonRatio,
    string MinimapFormat,     // png | jpg | webp
    int MinimapQuality        // 1..100, used for lossy formats
)
{
    public static ViewerOptions CreateDefault()
        => new(
            DefaultVersion: "0.5.3",
            DiffPair: null,
            MinimapWidth: 512,
            MinimapHeight: 512,
            DiffDistanceThreshold: 10.0,
            MoveEpsilonRatio: 0.005,
            MinimapFormat: "jpg",
            MinimapQuality: 85);
}
