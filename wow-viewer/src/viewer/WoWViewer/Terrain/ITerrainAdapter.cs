using System.Collections.Concurrent;
using System.Numerics;
using WowViewer.Core.Maps;

namespace WoWViewer.Terrain;

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

    /// <summary>Read the placement-bearing ADT payload for a tile when the format supports writing it.</summary>
    bool TryGetPlacementSourceData(int tileX, int tileY, out string sourcePath, out byte[] sourceBytes);

    /// <summary>Resolve the placement-bearing ADT payload to a writable loose-file path when available.</summary>
    bool TryGetPlacementWritablePath(int tileX, int tileY, out string? fullPath);

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

    /// <summary>
    /// Optional secondary overlay map name for phased terrain.
    /// </summary>
    /// <remarks>
    /// Convenience shim over <see cref="PhaseLayers"/>: reading returns the first enabled layer,
    /// and assigning replaces the whole stack with that single layer (or clears it). Adapters that
    /// support more than one simultaneous phase should be driven through <see cref="PhaseLayers"/>.
    /// </remarks>
    string? OverlayMapName { get; set; }

    /// <summary>
    /// The ordered phase overlay stack. Layers are applied in list order, so a later layer wins on
    /// any channel it shares with an earlier one. An adapter that does not support phasing exposes
    /// an empty list.
    /// </summary>
    IList<PhaseLayerSettings> PhaseLayers { get; }

    /// <summary>
    /// Cartography (Spec 222): the tile coordinates a named map occupies in its own grid, as
    /// (tileX, tileY) pairs. This is the donor footprint drawn on the minimap and the source of
    /// the donor tile-picker grid. Returns an empty set when the map cannot be resolved — callers
    /// should consult <see cref="TryResolveMap"/> to distinguish "resolved but empty" from
    /// "unresolved".
    /// </summary>
    IReadOnlyList<(int TileX, int TileY)> GetOccupiedTiles(string mapName);

    /// <summary>
    /// Cartography (Spec 222): whether a named map can be resolved to readable terrain data
    /// through this adapter's data source. Never throws; a false result is a displayable state,
    /// not an error.
    /// </summary>
    bool TryResolveMap(string mapName);

    /// <summary>
    /// Cartography (Spec 222): true when the named map is WMO-based (a dungeon/global-WMO map).
    /// Such maps carry no terrain tiles - their WDT MAIN entries are leftovers - so layers sourced
    /// from them must not claim terrain tiles or draw footprints. Unresolvable maps return false;
    /// combine with <see cref="TryResolveMap"/> for the displayable state.
    /// </summary>
    bool IsMapWmoBased(string mapName);

    /// <summary>
    /// Spec 232 FR-11: the channels the BASE map contributes to its own tiles. Gates off the base
    /// map's liquids, shadows, objects, etc. per channel so the operator controls exactly what
    /// the base keeps. Defaults to everything.
    /// </summary>
    PhaseDataChannel BaseChannelKeep { get => PhaseDataChannel.All; set { } }
}

/// <summary>
/// Spec 232 FR-11: strips base-map channels the operator gated off, operating on the viewer's
/// local tile-result shape (both WDT adapters share it).
/// </summary>
internal static class BaseChannelStrip
{
    public static void Apply(TileLoadResult result, PhaseDataChannel keep)
    {
        var chunks = new List<TerrainChunkData>(result.Chunks.Count);
        foreach (TerrainChunkData chunk in result.Chunks)
        {
            chunks.Add(new TerrainChunkData
            {
                McinIndex = chunk.McinIndex,
                TileX = chunk.TileX,
                TileY = chunk.TileY,
                ChunkX = chunk.ChunkX,
                ChunkY = chunk.ChunkY,
                Heights = chunk.Heights,
                Normals = chunk.Normals,
                HoleMask = keep.HasFlag(PhaseDataChannel.Holes) ? chunk.HoleMask : 0,
                Layers = keep.HasFlag(PhaseDataChannel.TextureLayers) ? chunk.Layers : Array.Empty<TerrainLayer>(),
                AlphaMaps = keep.HasFlag(PhaseDataChannel.TextureLayers) ? chunk.AlphaMaps : new Dictionary<int, byte[]>(),
                ShadowMap = keep.HasFlag(PhaseDataChannel.Shadows) ? chunk.ShadowMap : null,
                MccvColors = keep.HasFlag(PhaseDataChannel.VertexColors) ? chunk.MccvColors : null,
                Liquid = keep.HasFlag(PhaseDataChannel.Liquid) ? chunk.Liquid : null,
                WorldPosition = chunk.WorldPosition,
                AreaId = keep.HasFlag(PhaseDataChannel.AreaId) ? chunk.AreaId : 0,
                McnkFlags = keep.HasFlag(PhaseDataChannel.Liquid) ? chunk.McnkFlags : (chunk.McnkFlags & ~0x3C),
                AlphaSourceFlags = chunk.AlphaSourceFlags,
                McrdReferences = keep.HasFlag(PhaseDataChannel.Doodads) ? chunk.McrdReferences : Array.Empty<int>(),
                McrwReferences = keep.HasFlag(PhaseDataChannel.WorldObjects) ? chunk.McrwReferences : Array.Empty<int>(),
            });
        }

        result.Chunks.Clear();
        result.Chunks.AddRange(chunks);
        if (!keep.HasFlag(PhaseDataChannel.Doodads))
            result.MddfPlacements.Clear();
        if (!keep.HasFlag(PhaseDataChannel.WorldObjects))
            result.ModfPlacements.Clear();
    }
}
