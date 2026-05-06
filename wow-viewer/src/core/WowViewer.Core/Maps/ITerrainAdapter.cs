using System.Collections.Concurrent;
using System.Numerics;

namespace WowViewer.Core.Maps;

public interface ITerrainAdapter
{
    IReadOnlyList<int> ExistingTiles { get; }
    bool TileExists(int tileX, int tileY);
    TileLoadResult LoadTileWithPlacements(int tileX, int tileY);
    bool TryGetPlacementSourceData(int tileX, int tileY, out string sourcePath, out byte[] sourceBytes);
    bool TryGetPlacementWritablePath(int tileX, int tileY, out string? fullPath);
    ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; }
    IReadOnlyList<string> MdxModelNames { get; }
    IReadOnlyList<string> WmoModelNames { get; }
    List<MddfPlacement> MddfPlacements { get; }
    List<ModfPlacement> ModfPlacements { get; }
    bool IsWmoBased { get; }
    List<Vector3> LastLoadedChunkPositions { get; }
}