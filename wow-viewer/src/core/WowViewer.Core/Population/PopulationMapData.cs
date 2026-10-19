namespace WowViewer.Core.Population;

/// <summary>
/// Container for a partitioned map's population dataset.
/// </summary>
public sealed class PopulationMapData
{
    public int MapId { get; init; }
    public string MapName { get; init; } = string.Empty;
    public string ClientVersion { get; init; } = string.Empty;
    public DateTime ExportedUtc { get; init; } = DateTime.UtcNow;
    public IReadOnlyList<PopulationRecord> Spawns { get; init; } = Array.Empty<PopulationRecord>();
}
