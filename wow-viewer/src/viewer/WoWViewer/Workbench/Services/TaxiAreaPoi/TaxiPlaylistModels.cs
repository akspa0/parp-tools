namespace WoWViewer;

/// <summary>
/// A single route segment within a taxi flight playlist.
/// </summary>
public sealed record TaxiPlaylistItem(
    int PathId,
    int FromNodeId,
    int ToNodeId,
    string FromName,
    string ToName,
    float RouteLength)
{
    public string DisplayLabel => $"{FromName} \u2192 {ToName} (#{PathId})";
}

/// <summary>
/// Status summary of current taxi playlist playback.
/// </summary>
public readonly record struct TaxiPlaylistStatus(
    bool IsPlaying,
    bool IsRecording,
    int CurrentIndex,
    int TotalCount,
    TaxiPlaylistItem? CurrentItem,
    float SegmentProgressFraction,
    float OverallProgressFraction,
    string StatusText);
