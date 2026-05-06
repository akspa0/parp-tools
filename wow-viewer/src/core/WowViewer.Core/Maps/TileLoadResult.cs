namespace WowViewer.Core.Maps;

public sealed class TileLoadResult
{
    public required IReadOnlyList<TerrainChunkData> Chunks { get; init; }
    public required IReadOnlyList<MddfPlacement> MddfPlacements { get; init; }
    public required IReadOnlyList<ModfPlacement> ModfPlacements { get; init; }
}