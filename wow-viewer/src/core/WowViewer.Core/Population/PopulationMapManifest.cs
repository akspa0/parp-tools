namespace WowViewer.Core.Population;

public sealed class PopulationMapSummary
{
    public int MapId { get; init; }
    public string MapName { get; init; } = string.Empty;
    public int CreatureCount { get; init; }
    public int GameObjectCount { get; init; }
    public string RelativeFilePath { get; init; } = string.Empty;
}

/// <summary>
/// Manifest describing a full multi-map population dataset extracted for a specific client version.
/// </summary>
public sealed class PopulationMapManifest
{
    public string Version { get; init; } = string.Empty;
    public DateTime GeneratedUtc { get; init; } = DateTime.UtcNow;
    public int TotalCreatures { get; init; }
    public int TotalGameObjects { get; init; }
    public IReadOnlyList<PopulationMapSummary> Maps { get; init; } = Array.Empty<PopulationMapSummary>();
}
