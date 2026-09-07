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
/// Whether the host has checked a layer's donor map and what it found. Cartography (Spec 222)
/// surfaces this as an inline row badge so an unresolvable map is never a silent no-op.
/// </summary>
public enum PhaseLayerResolution
{
    /// <summary>No resolution attempt has been made yet.</summary>
    NotYetChecked = 0,

    /// <summary>The donor map resolved to readable terrain data.</summary>
    Resolved = 1,

    /// <summary>The donor map could not be resolved; the layer contributes nothing.</summary>
    Unresolved = 2,
}

/// <summary>
/// One entry in the phase overlay stack: which map, whether it is active, and what it may
/// contribute. Layers are applied in list order, so a later layer overrides an earlier one on any
/// channel they both supply.
/// </summary>
public sealed class PhaseLayerSettings
{
    public required string MapName { get; init; }

    /// <summary>Cartography (Spec 222): the host's last resolution attempt outcome for this layer's donor map.</summary>
    public PhaseLayerResolution Resolution { get; set; } = PhaseLayerResolution.NotYetChecked;

    /// <summary>
    /// Cartography (Spec 222): index into the host's footprint color palette, assigned by stack
    /// position so the minimap overlay and the row swatch agree. Negative means unassigned.
    /// </summary>
    public int FootprintColorIndex { get; set; } = -1;

    /// <summary>
    /// Rotation of this layer's content, in degrees, about <see cref="RotationOriginTileX"/> /
    /// <see cref="RotationOriginTileY"/>. Zero (the default) short-circuits all rotation logic.
    /// Exact-grid when a multiple of 90; free-rotate (grid-snapped tile lookup, exact placement
    /// points) otherwise. Spec 219.
    /// </summary>
    public float RotationDegrees { get; set; }

    /// <summary>Rotation origin, in donor tile coordinates. Defaults to the centre of the layer's
    /// occupied tile bounds, computed by the adapter when unset (negative).</summary>
    public float RotationOriginTileX { get; set; } = -1f;

    /// <inheritdoc cref="RotationOriginTileX"/>
    public float RotationOriginTileY { get; set; } = -1f;

    /// <summary>Mirror this layer's content along the horizontal axis. Exact-grid involution.</summary>
    public bool MirrorHorizontal { get; set; }

    /// <summary>Mirror this layer's content along the vertical axis. Exact-grid involution.</summary>
    public bool MirrorVertical { get; set; }

    /// <summary>Per-tile donor-to-target mappings; a mapping claims its target over the whole-layer offset.</summary>
    public IList<PhaseTilePlacement> TilePlacements { get; } = new List<PhaseTilePlacement>();

    /// <summary>True when this layer carries any transform (rotation, mirror, or per-tile mapping).</summary>
    public bool HasTransform =>
        RotationDegrees != 0f || MirrorHorizontal || MirrorVertical || TilePlacements.Count > 0;

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

    /// <summary>
    /// Spec 232 FR-1: sub-tile alignment fine-tune in terrain cells (MCNKs, 1/16 tile), applied in
    /// the composed (target) frame after rotation/mirror. Range −15..+15 cells; ±16 cells is one
    /// whole tile — use <see cref="TileOffsetX"/> for that.
    /// </summary>
    public int CellOffsetX { get; set; }

    /// <inheritdoc cref="CellOffsetX"/>
    public int CellOffsetY { get; set; }

    /// <summary>True when a cell-level fine-tune is active.</summary>
    public bool HasCellOffset => CellOffsetX != 0 || CellOffsetY != 0;

    public PhaseLayerSettings Clone()
    {
        var clone = new PhaseLayerSettings
        {
            MapName = MapName,
            RotationDegrees = RotationDegrees,
            RotationOriginTileX = RotationOriginTileX,
            RotationOriginTileY = RotationOriginTileY,
            MirrorHorizontal = MirrorHorizontal,
            MirrorVertical = MirrorVertical,
            Enabled = Enabled,
            Channels = Channels,
            OnlyTakeWhatThePhaseCarries = OnlyTakeWhatThePhaseCarries,
            TileOffsetX = TileOffsetX,
            TileOffsetY = TileOffsetY,
            Resolution = Resolution,
            FootprintColorIndex = FootprintColorIndex,
        };

        foreach (PhaseTilePlacement placement in TilePlacements)
            clone.TilePlacements.Add(placement);

        clone.CellOffsetX = CellOffsetX;
        clone.CellOffsetY = CellOffsetY;

        return clone;
    }
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
    /// Renderer X decreases as tile X increases (<c>worldX = MapOrigin - tileX * ChunkSize - ...</c>),
    /// and likewise for Y, so the translation is negative in both axes.
    /// </para>
    /// <para>
    /// <paramref name="tileSize"/> MUST be the world span of ONE tile in the 64x64 ADT grid.
    /// In this codebase that span is <c>WoWConstants.ChunkSize</c> (533.33) — the constant is
    /// misnamed but is the ADT-tile size (corner = MapOrigin - tileX * ChunkSize). Passing
    /// <c>WoWConstants.TileSize</c> (8533.33 = 16 ADTs) overshoots placements 16x.
    /// </para>
    /// </remarks>
    public static (float X, float Y) TileOffsetToWorldTranslation(int tileOffsetX, int tileOffsetY, float tileSize)
        => (-tileOffsetX * tileSize, -tileOffsetY * tileSize);

    /// <summary>
    /// Spec 231 T072: transforms placement poses in place through a layer's rotation/mirror —
    /// positions via <see cref="ForwardTransformWorldPoint"/>, headings via
    /// <see cref="ForwardTransformYawDegrees"/>. Excludes the whole-layer offset translation;
    /// callers apply <see cref="TileOffsetToWorldTranslation"/> alongside it.
    /// </summary>
    public static (IReadOnlyList<MddfPlacement> Mddf, IReadOnlyList<ModfPlacement> Modf) ForwardTransformPlacementPoses(
        IReadOnlyList<MddfPlacement> mddfPlacements,
        IReadOnlyList<ModfPlacement> modfPlacements,
        PhaseLayerSettings layer,
        float tileSize,
        float mapOrigin)
    {
        var mddf = new MddfPlacement[mddfPlacements.Count];
        for (int i = 0; i < mddfPlacements.Count; i++)
        {
            MddfPlacement placement = mddfPlacements[i];
            (float px, float py) = ForwardTransformWorldPoint(layer, placement.Position.X, placement.Position.Y, tileSize, mapOrigin);
            float yaw = ForwardTransformYawDegrees(layer, placement.Rotation.Z);
            mddf[i] = placement with
            {
                Position = new System.Numerics.Vector3(px, py, placement.Position.Z),
                Rotation = new System.Numerics.Vector3(placement.Rotation.X, placement.Rotation.Y, yaw),
            };
        }

        var modf = new ModfPlacement[modfPlacements.Count];
        for (int i = 0; i < modfPlacements.Count; i++)
        {
            ModfPlacement placement = modfPlacements[i];
            (float px, float py) = ForwardTransformWorldPoint(layer, placement.Position.X, placement.Position.Y, tileSize, mapOrigin);
            float yaw = ForwardTransformYawDegrees(layer, placement.Rotation.Z);
            modf[i] = placement with
            {
                Position = new System.Numerics.Vector3(px, py, placement.Position.Z),
                Rotation = new System.Numerics.Vector3(placement.Rotation.X, placement.Rotation.Y, yaw),
            };
        }

        return (mddf, modf);
    }

    /// <summary>Human-readable channel list, for logs and the layers panel.</summary>
    public static string Describe(PhaseDataChannel channels)
        => channels == PhaseDataChannel.None ? "none" : channels.ToString();

    /// <summary>
    /// Resolve which donor tile fills base-map target tile (targetX, targetY) for one layer, and
    /// what transform to apply to its content. Resolution order (fixed, per Spec 219 R3):
    /// per-tile mappings last-wins first, then the whole-layer offset with rotation/mirror.
    /// </summary>
    /// <param name="layer">The phase layer being resolved.</param>
    /// <param name="targetX">Base-map target tile X.</param>
    /// <param name="targetY">Base-map target tile Y.</param>
    /// <param name="hasDonorTile">
    /// Predicate answering whether the donor map has content at a donor tile coordinate (the
    /// caller supplies this so the policy stays I/O-free).</param>
    public static PhaseTileSource ResolveTileSource(
        PhaseLayerSettings layer,
        int targetX,
        int targetY,
        Func<int, int, bool> hasDonorTile)
    {
        ArgumentNullException.ThrowIfNull(layer);
        ArgumentNullException.ThrowIfNull(hasDonorTile);

        // Spec 231 Phase 7 (operator rule): a layer's composed content is confined to the base
        // map's 64x64 tile grid. Targets outside it contribute nothing, no matter what the
        // offset/rotation math computes.
        if (targetX < 0 || targetX > 63 || targetY < 0 || targetY > 63)
            return PhaseTileSource.Empty;

        // 1. Per-tile mappings: last one claiming this target wins. Count claims so the caller can
        // report a deterministic conflict rather than silently hiding an earlier mapping.
        int claimCount = 0;
        PhaseTilePlacement? selectedPlacement = null;
        for (int i = layer.TilePlacements.Count - 1; i >= 0; i--)
        {
            PhaseTilePlacement placement = layer.TilePlacements[i];
            if (!placement.IsValid)
                continue;

            if (placement.TargetTileX == targetX && placement.TargetTileY == targetY)
            {
                claimCount++;
                selectedPlacement ??= placement;
            }
        }

        if (selectedPlacement is PhaseTilePlacement selected)
        {
            if (!hasDonorTile(selected.DonorTileX, selected.DonorTileY))
                return PhaseTileSource.Empty with { TargetClaimCount = claimCount };

            return new PhaseTileSource(
                true,
                selected.DonorTileX,
                selected.DonorTileY,
                PhaseTileSourceKind.PerTile,
                ComposeTileTransforms(layer),
                ResolveRotationApproximation(layer),
                claimCount);
        }

        // 2. Whole-layer offset, with rotation/mirror composed into the lookup.
        int sourceX = targetX - layer.TileOffsetX;
        int sourceY = targetY - layer.TileOffsetY;
        if (layer.RotationDegrees != 0f || layer.MirrorHorizontal || layer.MirrorVertical)
        {
            (sourceX, sourceY) = InverseTransformTile(sourceX, sourceY, layer);
        }

        if (!hasDonorTile(sourceX, sourceY))
            return PhaseTileSource.Empty;

        bool transformed = layer.RotationDegrees != 0f || layer.MirrorHorizontal || layer.MirrorVertical;
        return new PhaseTileSource(
            true,
            sourceX,
            sourceY,
            transformed ? PhaseTileSourceKind.Rotation : PhaseTileSourceKind.Offset,
            ComposeTileTransforms(layer),
            ResolveRotationApproximation(layer),
            0);
    }

    /// <summary>
    /// The tile-content transforms a layer's sourced content must receive, in the fixed
    /// application order: rotation first (exact-grid when a multiple of 90), then mirrors.
    /// Free-angle rotations contribute no chunk-content transform (placements only; research R2).
    /// Empty when the layer carries no exact-grid chunk transform.
    /// </summary>
    public static IReadOnlyList<TileTransformKind> ComposeTileTransforms(PhaseLayerSettings layer)
    {
        var kinds = new List<TileTransformKind>();

        if (TryGetQuarterTurn(layer.RotationDegrees, out int quarter))
        {
            switch (quarter)
            {
                case 1: kinds.Add(TileTransformKind.Rotate90CW); break;
                case 2: kinds.Add(TileTransformKind.Rotate180); break;
                case 3: kinds.Add(TileTransformKind.Rotate90CCW); break;
            }
        }

        if (layer.MirrorHorizontal)
            kinds.Add(TileTransformKind.MirrorH);
        if (layer.MirrorVertical)
            kinds.Add(TileTransformKind.MirrorV);

        return kinds;
    }

    /// <summary>The declared approximation used for this layer's rotation.</summary>
    public static PhaseRotationApproximation ResolveRotationApproximation(PhaseLayerSettings layer)
    {
        if (layer.RotationDegrees == 0f)
        {
            return layer.MirrorHorizontal || layer.MirrorVertical
                ? PhaseRotationApproximation.ExactGrid
                : PhaseRotationApproximation.None;
        }

        return TryGetQuarterTurn(layer.RotationDegrees, out _)
            ? PhaseRotationApproximation.ExactGrid
            : PhaseRotationApproximation.FreeRotate;
    }

    /// <summary>
    /// Inverse tile lookup: which donor tile fills the target, given the layer's rotation/mirror.
    /// Exact-grid for quadrant turns and mirrors; for free angles this is the grid-snapped
    /// approximation (research R2) and the caller reports the mode.
    /// </summary>
    public static (int X, int Y) InverseTransformTile(int x, int y, PhaseLayerSettings layer)
    {
        double originX = layer.RotationOriginTileX >= 0f ? layer.RotationOriginTileX : 0d;
        double originY = layer.RotationOriginTileY >= 0f ? layer.RotationOriginTileY : 0d;

        // Mirror is its own inverse: apply to the lookup first (order: rotate -> mirror, so the
        // inverse is mirror -> rotate). Mirrors use the same declared origin as rotation.
        double ix = x;
        double iy = y;
        // Tile X is the North-South row; tile Y is the West-East column. A horizontal (left-right)
        // mirror therefore reflects tile Y, while a vertical (top-bottom) mirror reflects tile X.
        if (layer.MirrorHorizontal)
            iy = (2d * originY) - iy;
        if (layer.MirrorVertical)
            ix = (2d * originX) - ix;

        if (layer.RotationDegrees == 0f)
            return ((int)Math.Round(ix), (int)Math.Round(iy));

        double rad = layer.RotationDegrees * Math.PI / 180.0;
        double cos = Math.Cos(rad);
        double sin = Math.Sin(rad);
        double dx = ix - originX;
        double dy = iy - originY;
        // Forward positive rotation is clockwise in tile coordinates:
        // (row, column) -> (column, -row). Its inverse is counter-clockwise.
        return (
            (int)Math.Round(originX + (dx * cos) - (dy * sin)),
            (int)Math.Round(originY + (dx * sin) + (dy * cos)));
    }

    /// <summary>
    /// Spec 232 FR-1: resolves which supplying target tile feeds one cell-shifted composed chunk.
    /// The layer moves as a RIGID map object: the composed chunk (chunkX, chunkY) of target tile
    /// (targetTileX, targetTileY) shows the layer's already-composed content from global cell
    /// (gX, gY), shifted by the layer's cell fine-tune. The caller composes each supplying
    /// target tile once (through <see cref="ResolveTileSource"/> + the content transforms) and
    /// indexes it by the returned chunk slot.
    /// Confined to the base map's 64x64 tile grid: global cells outside it contribute nothing.
    /// </summary>
    public static (bool HasSource, int TileX, int TileY, int ChunkX, int ChunkY) ResolveCellShiftedChunk(
        PhaseLayerSettings layer,
        int targetTileX,
        int targetTileY,
        int chunkX,
        int chunkY)
    {
        int globalX = (targetTileX * 16) + chunkX - layer.CellOffsetX;
        int globalY = (targetTileY * 16) + chunkY - layer.CellOffsetY;
        if (globalX < 0 || globalX > 1023 || globalY < 0 || globalY > 1023)
            return (false, 0, 0, 0, 0);

        int supplyingTileX = (int)Math.Floor(globalX / 16.0);
        int supplyingTileY = (int)Math.Floor(globalY / 16.0);
        return (true, supplyingTileX, supplyingTileY, globalX - (supplyingTileX * 16), globalY - (supplyingTileY * 16));
    }

    /// <summary>
    /// Forward tile map (Spec 231 Phase 7): which base-map target tile the donor tile
    /// <paramref name="donorX"/>/<paramref name="donorY"/> fills under this layer's rotation/mirror,
    /// excluding the whole-layer offset (callers add
    /// <see cref="PhaseLayerSettings.TileOffsetX"/>/<c>TileOffsetY</c> themselves). The exact
    /// inverse of <see cref="InverseTransformTile"/>'s transform; free angles are the grid-snapped
    /// approximation, matching <see cref="ResolveRotationApproximation"/>.
    /// </summary>
    public static (int X, int Y) ForwardTransformTile(int donorX, int donorY, PhaseLayerSettings layer)
    {
        (double x, double y) = ForwardTransformContinuous(donorX, donorY, layer);
        return ((int)Math.Round(x), (int)Math.Round(y));
    }

    /// <summary>Continuous form of <see cref="ForwardTransformTile"/> for world-space consumers.</summary>
    private static (double X, double Y) ForwardTransformContinuous(double donorX, double donorY, PhaseLayerSettings layer)
    {
        double originX = layer.RotationOriginTileX >= 0f ? layer.RotationOriginTileX : 0d;
        double originY = layer.RotationOriginTileY >= 0f ? layer.RotationOriginTileY : 0d;
        double dx = donorX - originX;
        double dy = donorY - originY;

        // Forward = MH ∘ MV ∘ Rcw — the exact inverse of InverseTransformTile = Rccw ∘ MV ∘ MH:
        // clockwise rotation first, then the vertical-axis mirror, then the horizontal-axis mirror.
        if (layer.RotationDegrees != 0f)
        {
            double rad = layer.RotationDegrees * Math.PI / 180.0;
            double cos = Math.Cos(rad);
            double sin = Math.Sin(rad);
            (dx, dy) = (dx * cos + dy * sin, -dx * sin + dy * cos);
        }
        if (layer.MirrorVertical)
            dx = -dx;
        if (layer.MirrorHorizontal)
            dy = -dy;

        return (originX + dx, originY + dy);
    }

    /// <summary>
    /// Forward world-space point map for a layer's rotation/mirror (Spec 231 T072): transforms a
    /// donor-map world position into the composed frame, excluding the whole-layer offset
    /// translation (callers add <see cref="TileOffsetToWorldTranslation"/> themselves). Uses the
    /// adapters' chunk-corner convention: world X = mapOrigin − tileX·tileSize, world Y =
    /// mapOrigin − tileY·tileSize (both axes negate, which conjugates as a 180° turn and
    /// preserves the rotation/mirror directions).
    /// </summary>
    public static (float X, float Y) ForwardTransformWorldPoint(
        PhaseLayerSettings layer, float worldX, float worldY, float tileSize, float mapOrigin)
    {
        double tx = (mapOrigin - worldX) / tileSize;
        double ty = (mapOrigin - worldY) / tileSize;
        (double rx, double ry) = ForwardTransformContinuous(tx, ty, layer);
        return ((float)(mapOrigin - rx * tileSize), (float)(mapOrigin - ry * tileSize));
    }

    /// <summary>
    /// Forward yaw map (Spec 231 T072): applies the layer's composed transforms to a placement
    /// heading in degrees, in the same application order as <see cref="ComposeTileTransforms"/>
    /// (rotation first, then mirrors) — matching <c>TileContentTransform.TransformYawDegrees</c>.
    /// </summary>
    public static float ForwardTransformYawDegrees(PhaseLayerSettings layer, float yawDegrees)
    {
        float yaw = yawDegrees;
        if (TryGetQuarterTurn(layer.RotationDegrees, out int quarter))
        {
            yaw = quarter switch
            {
                1 => yaw - 90f,   // Rotate90CW
                2 => yaw + 180f,  // Rotate180
                3 => yaw + 90f,   // Rotate90CCW
                _ => yaw,
            };
        }
        if (layer.MirrorHorizontal)
            yaw = -yaw;
        if (layer.MirrorVertical)
            yaw = 180f - yaw;
        return yaw;
    }

    private static bool TryGetQuarterTurn(float degrees, out int normalizedQuarterTurn)
    {
        double rawQuarterTurn = degrees / 90d;
        int roundedQuarterTurn = (int)Math.Round(rawQuarterTurn);
        if (Math.Abs(rawQuarterTurn - roundedQuarterTurn) > 1e-5d)
        {
            normalizedQuarterTurn = 0;
            return false;
        }

        normalizedQuarterTurn = NormalizeQuarterTurn(roundedQuarterTurn);
        return true;
    }

    private static int NormalizeQuarterTurn(int quarterTurn)
        => ((quarterTurn % 4) + 4) % 4;
}

/// <summary>
/// One per-tile donor-to-target mapping on a phase layer (Spec 219 US5): the donor tile's full
/// channel content is dropped onto the target tile of the base map, subject to the layer's
/// per-channel gates. A mapping claims its target over the whole-layer offset; two mappings
/// claiming the same target resolve last-wins, with the conflict reported by the caller.
/// </summary>
public readonly record struct PhaseTilePlacement(
    int DonorTileX,
    int DonorTileY,
    int TargetTileX,
    int TargetTileY)
{
    public bool IsValid =>
        DonorTileX is >= 0 and < 64 && DonorTileY is >= 0 and < 64 &&
        TargetTileX is >= 0 and < 64 && TargetTileY is >= 0 and < 64;
}

/// <summary>The result of resolving which donor tile fills a base-map target tile.</summary>
/// <param name="HasSource">False when nothing fills this target.</param>
/// <param name="SourceTileX">Donor tile to read, already in the donor map's grid.</param>
/// <param name="SourceTileY">Donor tile to read, already in the donor map's grid.</param>
/// <param name="Via">Which mechanism supplied the mapping, for logs and conflict reports.</param>
/// <param name="Transforms">The tile-content transforms to apply, in application order.</param>
/// <param name="Approximation">Whether rotation is exact-grid or uses free-rotate lookup.</param>
/// <param name="TargetClaimCount">Valid per-tile mappings claiming this target (greater than one is a reportable conflict).</param>
public readonly record struct PhaseTileSource(
    bool HasSource,
    int SourceTileX,
    int SourceTileY,
    PhaseTileSourceKind Via,
    IReadOnlyList<TileTransformKind> Transforms,
    PhaseRotationApproximation Approximation,
    int TargetClaimCount)
{
    public bool HasConflict => TargetClaimCount > 1;

    public static readonly PhaseTileSource Empty =
        new(false, 0, 0, PhaseTileSourceKind.None, Array.Empty<TileTransformKind>(), PhaseRotationApproximation.None, 0);
}

/// <summary>Declared rotation approximation for diagnostics and the layer UI (Spec 219 FR-009).</summary>
public enum PhaseRotationApproximation
{
    None = 0,
    ExactGrid,
    FreeRotate,
}

/// <summary>Provenance of a target tile's content, for logs and conflict reports.</summary>
public enum PhaseTileSourceKind
{
    None = 0,
    Base,
    Offset,
    Rotation,
    PerTile,
}
