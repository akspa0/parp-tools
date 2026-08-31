namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// Baseline anchor datum used to reference compressed height deltas during stratigraphy restoration.
/// </summary>
public enum StratigraphyAnchorMode
{
    /// <summary>
    /// Anchors scaling relative to the lowest elevation floor (Min Z) of the tile/chunk.
    /// Recommended for upward-compressed terrain where relief rises from ocean/valley floor.
    /// </summary>
    LowestZ_Floor = 0,

    /// <summary>
    /// Anchors scaling relative to the highest elevation ceiling (Max Z) of the tile/chunk.
    /// Recommended for inverted / excavated compressed developmental terrain (e.g. Dragon Isles).
    /// </summary>
    HighestZ_Ceiling = 1,

    /// <summary>
    /// Anchors scaling relative to the arithmetic mean elevation of the active tile/chunk.
    /// </summary>
    MeanZ = 2,

    /// <summary>
    /// Anchors scaling relative to adjoining active neighbor mesh boundary vertices.
    /// </summary>
    NeighborMeshBorder = 3,

    /// <summary>
    /// Anchors scaling relative to low-frequency WDL macro lattice elevation surface.
    /// </summary>
    WdlLattice = 4,

    /// <summary>
    /// Custom user-specified absolute world elevation datum.
    /// </summary>
    CustomDatum = 5
}
