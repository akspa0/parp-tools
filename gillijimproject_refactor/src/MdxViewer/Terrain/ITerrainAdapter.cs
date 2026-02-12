using System.Collections.Concurrent;
using System.Numerics;

namespace MdxViewer.Terrain;

/// <summary>
/// Common interface for terrain adapters (Alpha WDT, Standard WDT+ADT, etc.).
/// Provides tile enumeration, loading, and placement data.
/// </summary>
public interface ITerrainAdapter
{
    /// <summary>Flat list of existing tile indices (tileX*64+tileY).</summary>
    IReadOnlyList<int> ExistingTiles { get; }

    /// <summary>Check if a tile exists at grid coordinates.</summary>
    bool TileExists(int tileX, int tileY);

    /// <summary>Load a tile and return terrain chunks + per-tile placements.</summary>
    TileLoadResult LoadTileWithPlacements(int tileX, int tileY);

    /// <summary>Texture names per tile (MTEX).</summary>
    ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; }

    /// <summary>MDX model name table.</summary>
    IReadOnlyList<string> MdxModelNames { get; }

    /// <summary>WMO model name table.</summary>
    IReadOnlyList<string> WmoModelNames { get; }

    /// <summary>Collected MDDF placements (deduplicated).</summary>
    List<MddfPlacement> MddfPlacements { get; }

    /// <summary>Collected MODF placements (deduplicated).</summary>
    List<ModfPlacement> ModfPlacements { get; }

    /// <summary>True if WMO-only map (no terrain tiles).</summary>
    bool IsWmoBased { get; }

    /// <summary>Chunk world positions for diagnostics.</summary>
    List<Vector3> LastLoadedChunkPositions { get; }
}
