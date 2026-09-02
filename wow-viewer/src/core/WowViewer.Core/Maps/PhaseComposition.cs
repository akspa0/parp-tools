namespace WowViewer.Core.Maps;

/// <summary>
/// The independently-selectable kinds of data a phase map can contribute to the map beneath it.
/// </summary>
/// <remarks>
/// One flag per thing a user would reasonably want to take or leave from a phase, which is what
/// makes a per-layer checkbox panel expressible. The same type doubles as a <em>presence</em>
/// bitmask describing what a given chunk actually carries -- see
/// <see cref="PhaseCompositionPolicy.ResolveChannelsToTake"/>.
/// </remarks>
[Flags]
public enum PhaseDataChannel
{
    None = 0,

    /// <summary>MCVT heights.</summary>
    Heightmap = 1 << 0,

    /// <summary>MCNR normals.</summary>
    Normals = 1 << 1,

    /// <summary>MCLY layers plus their MCAL alpha maps. The "texturing" channel.</summary>
    TextureLayers = 1 << 2,

    /// <summary>MCCV vertex colours.</summary>
    VertexColors = 1 << 3,

    /// <summary>MCNK hole mask.</summary>
    Holes = 1 << 4,

    /// <summary>MCSH baked shadow map.</summary>
    Shadows = 1 << 5,

    /// <summary>MH2O / MCLQ liquid.</summary>
    Liquid = 1 << 6,

    /// <summary>MDDF doodad placements.</summary>
    Doodads = 1 << 7,

    /// <summary>MODF world-object (WMO) placements.</summary>
    WorldObjects = 1 << 8,

    /// <summary>MCNK AreaID.</summary>
    AreaId = 1 << 9,

    Terrain = Heightmap | Normals | Holes,
    Texturing = TextureLayers | VertexColors | Shadows,
    Objects = Doodads | WorldObjects,

    All = Terrain | Texturing | Liquid | Objects | AreaId,
}

/// <summary>
/// One entry in the phase overlay stack: which map, whether it is active, and what it may
/// contribute. Layers are applied in list order, so a later layer overrides an earlier one on any
/// channel they both supply.
/// </summary>
public sealed class PhaseLayerSettings
{
    public required string MapName { get; init; }

    /// <summary>Unchecked layers are skipped entirely without being removed from the stack.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>The channels this layer is permitted to contribute.</summary>
    public PhaseDataChannel Channels { get; set; } = PhaseDataChannel.All;

    /// <summary>
    /// When true (the default), a channel is taken only if the phase actually carries it.
    /// </summary>
    /// <remarks>
    /// This is the guard against a phase's blank placeholder chunks erasing good base terrain and
    /// texturing. A phase tile that ships no MCLY contributes no texturing; one that ships no MCVT
    /// contributes no heights. Turning it off makes the phase authoritative for every enabled
    /// channel, including the channels it is empty for -- which is occasionally what you want when
    /// a phase is deliberately clearing something, and is destructive otherwise.
    /// </remarks>
    public bool OnlyTakeWhatThePhaseCarries { get; set; } = true;

    /// <summary>
    /// Tile-space shift applied to this layer's content, in tiles.
    /// </summary>
    /// <remarks>
    /// Positive X moves the layer's content toward higher tile X. Base tile <c>(tx, ty)</c> therefore
    /// reads this layer's tile <c>(tx - TileOffsetX, ty - TileOffsetY)</c>, so content authored at
    /// tile <c>(a, b)</c> lands on base tile <c>(a + TileOffsetX, b + TileOffsetY)</c>.
    /// <para>
    /// Exists because instance and dungeon maps are frequently copies of an earlier revision of a
    /// zone's terrain, and are not always stored at the tile coordinates the original occupies.
    /// Without a shift those maps can only be compared against the original by eye.
    /// </para>
    /// </remarks>
    public int TileOffsetX { get; set; }

    /// <inheritdoc cref="TileOffsetX"/>
    public int TileOffsetY { get; set; }

    /// <summary>True when this layer is read from tile coordinates other than the base map's.</summary>
    public bool HasTileOffset => TileOffsetX != 0 || TileOffsetY != 0;

    public PhaseLayerSettings Clone() => new()
    {
        MapName = MapName,
        Enabled = Enabled,
        Channels = Channels,
        OnlyTakeWhatThePhaseCarries = OnlyTakeWhatThePhaseCarries,
        TileOffsetX = TileOffsetX,
        TileOffsetY = TileOffsetY,
    };
}

/// <summary>
/// The rules deciding what a phase layer contributes to the composed tile.
/// </summary>
/// <remarks>
/// <para>
/// Before this existed, <c>MergePhaseTile</c> replaced the whole base chunk with the phase chunk
/// and preserved exactly one field (liquid). Anything the base carried and the phase did not was
/// lost, which is why phased maps dropped trees and showed blank plates: once split
/// <c>_obj0</c>/<c>_tex0</c> companions landed, a phase tile became a <em>partial</em> chunk, and a
/// partial chunk replacing a complete one loses the difference.
/// </para>
/// <para>
/// Placements had the opposite bug: they were appended unconditionally, so a phase that restages
/// objects produced both the old and the new set.
/// </para>
/// </remarks>
public static class PhaseCompositionPolicy
{
    /// <summary>
    /// Which channels this layer actually contributes for one chunk.
    /// </summary>
    /// <param name="requested">The channels the user left enabled on the layer.</param>
    /// <param name="phaseCarries">The channels the phase chunk actually carries.</param>
    /// <param name="onlyTakeWhatThePhaseCarries">The blank-plate guard.</param>
    public static PhaseDataChannel ResolveChannelsToTake(
        PhaseDataChannel requested,
        PhaseDataChannel phaseCarries,
        bool onlyTakeWhatThePhaseCarries)
        => onlyTakeWhatThePhaseCarries ? requested & phaseCarries : requested;

    /// <summary>
    /// Describe what a chunk carries, from the primitive facts about it.
    /// </summary>
    /// <remarks>
    /// Takes primitives rather than a chunk type so the rule is testable and shared, and so the
    /// same definition of "carries texturing" cannot drift between the renderer and the harvester.
    /// </remarks>
    public static PhaseDataChannel DescribePresence(
        bool hasHeights,
        bool hasNormals,
        int textureLayerCount,
        bool hasVertexColors,
        bool hasHoleMask,
        bool hasShadowMap,
        bool hasLiquid,
        bool hasAreaId)
    {
        PhaseDataChannel present = PhaseDataChannel.None;
        if (hasHeights)
            present |= PhaseDataChannel.Heightmap;
        if (hasNormals)
            present |= PhaseDataChannel.Normals;
        if (textureLayerCount > 0)
            present |= PhaseDataChannel.TextureLayers;
        if (hasVertexColors)
            present |= PhaseDataChannel.VertexColors;
        if (hasHoleMask)
            present |= PhaseDataChannel.Holes;
        if (hasShadowMap)
            present |= PhaseDataChannel.Shadows;
        if (hasLiquid)
            present |= PhaseDataChannel.Liquid;
        if (hasAreaId)
            present |= PhaseDataChannel.AreaId;
        return present;
    }

    /// <summary>
    /// Whether a phase's placements replace the base map's for this tile, rather than adding to them.
    /// </summary>
    /// <remarks>
    /// <b>Presence-gated replace.</b> A phase that ships an object set is restaging the tile, so its
    /// set is authoritative and the base's is dropped; a phase that ships none is not saying
    /// anything about objects, so the base's survive. Appending both -- the previous behaviour --
    /// is the one option that is never right: it duplicates every object the phase restaged.
    /// </remarks>
    public static bool PhaseOwnsPlacements(PhaseDataChannel channelsTaken, PhaseDataChannel channel, int phasePlacementCount)
        => (channelsTaken & channel) != 0 && phasePlacementCount > 0;

    /// <summary>
    /// The world-space translation matching a tile-space shift, in the renderer's axes.
    /// </summary>
    /// <remarks>
    /// Chunk identity is taken from the base chunk, so terrain relocates for free. Placements do not:
    /// MDDF/MODF carry world coordinates, so a shifted layer's objects must be translated by the same
    /// amount or they stay at the donor map's coordinates while its terrain moves.
    /// <para>
    /// Renderer X decreases as tile X increases (<c>worldX = MapOrigin - tileX * TileSize - ...</c>),
    /// and likewise for Y, so the translation is negative in both axes.
    /// </para>
    /// </remarks>
    public static (float X, float Y) TileOffsetToWorldTranslation(int tileOffsetX, int tileOffsetY, float tileSize)
        => (-tileOffsetX * tileSize, -tileOffsetY * tileSize);

    /// <summary>Human-readable channel list, for logs and the layers panel.</summary>
    public static string Describe(PhaseDataChannel channels)
        => channels == PhaseDataChannel.None ? "none" : channels.ToString();
}
