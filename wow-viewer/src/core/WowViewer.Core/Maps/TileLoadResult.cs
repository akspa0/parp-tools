namespace WowViewer.Core.Maps;

public sealed class TileLoadResult
{
    public required IReadOnlyList<TerrainChunkData> Chunks { get; init; }
    public required IReadOnlyList<MddfPlacement> MddfPlacements { get; init; }
    public required IReadOnlyList<ModfPlacement> ModfPlacements { get; init; }

    /// <summary>
    /// Spec 232 FR-9: true when the placements in this result already carry the full layer
    /// transform (pose + offset + cell delta) — the merge must not translate them again.
    /// </summary>
    public bool PlacementsPreTransformed { get; init; }
}