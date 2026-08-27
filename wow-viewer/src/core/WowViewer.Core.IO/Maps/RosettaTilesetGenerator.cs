using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public enum RosettaAssetKind
{
    Model,
    WorldModel,
}

/// <summary>One placeable asset with its source bounds, in model-local units.</summary>
public sealed record RosettaAssetEntry(
    string AssetPath,
    RosettaAssetKind Kind,
    Vector3 BoundsMin,
    Vector3 BoundsMax);

/// <summary>
/// Layout settings for the Rosetta canvas. Every cell on a given tile is the SAME size and every
/// cell edge lands on a chunk boundary, so the grid reads as a grid instead of a ragged shelf pack.
/// </summary>
/// <param name="CellChunks">
/// Cell edge in chunks. Must divide 16 so cells tile a 16x16-chunk ADT exactly: 1, 2, 4, 8 or 16.
/// The default 8 gives 2x2 cells of 266.67 m per tile — the smallest cell that still fits a 30
/// character label at a legible font size.
/// </param>
/// <param name="LabelBandChunks">
/// Chunks reserved at the far edge of every cell for the painted name. One chunk is exactly one
/// text line, so this is also the maximum number of label lines.
/// </param>
public sealed record RosettaGeneratorOptions(
    string MapName,
    int StartTileX = 24,
    int StartTileY = 24,
    int CellChunks = 8,
    int LabelBandChunks = 3,
    float FootprintMarginMeters = 16f,
    bool PaintCellBorders = true,
    bool GroupByDesignkit = true,
    int KitDepth = 0,
    int MaxTilesPerMap = 4096,
    string GroundTexture = @"tileset\ocean\westfallseafloor.blp",
    string InkTexture = @"tileset\generic\black.blp",
    float PedestalHeightMeters = 4f,
    float PedestalBevelMeters = 12.5f,
    int LabelFontTexels = 6)
{
    public const float TileSize = 533.33333f;
    public const int ChunksPerTileAxis = 16;
    public const float ChunkSize = TileSize / ChunksPerTileAxis;
    public const int TilesPerAxis = 64;
    /// <summary>Default cell edge in chunks; mirrors the <c>CellChunks</c> parameter default.</summary>
    public const int DefaultCellChunks = 8;

    /// <summary>Default label band in chunks; mirrors the <c>LabelBandChunks</c> parameter default.</summary>
    public const int DefaultLabelBandChunks = 3;

    /// <summary>Cell edge used for assets too large for the standard cell: one whole tile.</summary>
    public const int OversizeCellChunks = ChunksPerTileAxis;

    /// <summary>
    /// Layer-1 texture the painted label is blended in with. Like the ground texture it must be a
    /// path the target client ships, and it wants to contrast with the ground.
    /// </summary>
    public const string DefaultInkTexture = @"tileset\generic\black.blp";

    /// <summary>Label font pixel measured in MCAL texels; 6 gives ~3.1 m pixels, 14 characters a line.</summary>
    public const int DefaultLabelFontTexels = 6;

    /// <summary>
    /// Layer-0 terrain texture every generated tile references. It has to be a path the TARGET
    /// client actually ships, which is why it is a setting rather than a constant: the whole point
    /// of era-gating this generator is that a tile must not name assets that do not exist.
    /// </summary>
    public const string DefaultGroundTexture = @"tileset\ocean\westfallseafloor.blp";

    /// <summary>Throws if the settings cannot produce a uniform chunk-aligned grid.</summary>
    public void Validate()
    {
        if (CellChunks is not (1 or 2 or 4 or 8 or 16))
            throw new ArgumentOutOfRangeException(nameof(CellChunks),
                $"CellChunks must divide {ChunksPerTileAxis} (1, 2, 4, 8 or 16), got {CellChunks}.");
        if (LabelBandChunks < 1 || LabelBandChunks >= CellChunks)
            throw new ArgumentOutOfRangeException(nameof(LabelBandChunks),
                $"LabelBandChunks must be in [1, {CellChunks - 1}] for a {CellChunks}-chunk cell, got {LabelBandChunks}.");
        ArgumentOutOfRangeException.ThrowIfNegative(FootprintMarginMeters);
        ArgumentOutOfRangeException.ThrowIfNegative(StartTileX);
        ArgumentOutOfRangeException.ThrowIfNegative(StartTileY);
        ArgumentException.ThrowIfNullOrWhiteSpace(GroundTexture);
        ArgumentException.ThrowIfNullOrWhiteSpace(InkTexture);
        ArgumentOutOfRangeException.ThrowIfNegative(PedestalHeightMeters);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(PedestalBevelMeters);
        ArgumentOutOfRangeException.ThrowIfLessThan(LabelFontTexels, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(KitDepth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxTilesPerMap);
        if (MaxTilesPerMap > TilesPerAxis * TilesPerAxis)
            throw new ArgumentOutOfRangeException(nameof(MaxTilesPerMap),
                $"A map holds at most {TilesPerAxis * TilesPerAxis} tiles, got {MaxTilesPerMap}.");
    }
}

/// <summary>
/// Placement record carrying both coordinate spaces. <see cref="RawPosition"/> is the canvas/file
/// coordinate (<c>rawX = tileY*TileSize + u</c>, <c>rawY = tileX*TileSize + v</c>) that ends up in
/// the MDDF/MODF bytes on disk; <see cref="RendererPosition"/> is where the viewer draws it, and is
/// what <see cref="LkAdtWriter"/> takes as input — that writer applies the MapOrigin flip itself.
/// </summary>
public sealed record RosettaPlacementRecord(
    RosettaAssetEntry Asset,
    int TileX,
    int TileY,
    Vector3 RawPosition,
    Vector3 RendererPosition,
    float CellU,
    float CellV,
    float CellSize,
    string LabelText,
    IReadOnlyList<string> LabelLines,
    float LabelPixelMeters,
    int UniqueId);

/// <summary>
/// A planned tile: everything needed to build its ADT, but NOT the ADT itself. A full-client corpus
/// runs to five figures of tiles and a built <see cref="LkAdtData"/> is ~400 KB of arrays, so
/// materialising them all before writing any exhausts memory. Call
/// <see cref="RosettaTilesetGenerator.BuildTileAdt"/> per tile, write it, and let it go.
/// </summary>
/// <param name="AlphaCanvas">
/// Tile-wide MCAL canvas carrying the painted labels at 1024x1024 (see
/// <see cref="RosettaAlphaPainter"/>), or null when nothing was painted.
/// </param>
/// <param name="Pedestals">Object-band rectangles raised into a plinth by the height field.</param>
public sealed record RosettaTilePlan(
    int TileX,
    int TileY,
    IReadOnlyList<RosettaPlacementRecord> Placements,
    IReadOnlyList<RosettaMccvRect> Rects,
    IReadOnlyList<RosettaLabel> Labels,
    byte[]? AlphaCanvas,
    IReadOnlyList<RosettaPedestal> Pedestals);

/// <summary>A raised platform under one cell's object, in tile canvas space.</summary>
public sealed record RosettaPedestal(float U0, float V0, float U1, float V1, float Height);

public sealed record RosettaExcludedAsset(RosettaAssetEntry Asset, string Reason);

/// <summary>
/// Where one designkit — the source folder an asset set came from — ended up. This is the index
/// entry that answers "which map is this kit stashed on".
/// </summary>
/// <param name="Kit">Source folder, backslash-separated, no trailing separator. Empty when grouping is off.</param>
public sealed record RosettaDesignkitPlan(
    string Kit,
    string MapName,
    IReadOnlyList<(int TileX, int TileY)> Tiles,
    int AssetCount,
    int ModelCount,
    int WorldModelCount);

/// <summary>
/// One generated map. The corpus is split across as many of these as it takes, because a single
/// 64x64 map cannot hold a full client's assets at a legible cell size.
/// </summary>
/// <param name="BlockOriginX">Tile X of the square block's top-left corner, after any clamping to fit the map.</param>
/// <param name="BlockOriginY">Tile Y of the square block's top-left corner, after any clamping to fit the map.</param>
/// <param name="BlockSide">Edge length of the square tile block, in tiles.</param>
public sealed record RosettaMapPlan(
    string MapName,
    IReadOnlyList<RosettaTilePlan> Tiles,
    IReadOnlyList<RosettaPlacementRecord> Placements,
    IReadOnlyList<RosettaDesignkitPlan> Designkits,
    int BlockOriginX,
    int BlockOriginY,
    int BlockSide,
    string GroundTexture,
    string InkTexture);

public sealed record RosettaGenerationResult(
    string MapName,
    IReadOnlyList<RosettaMapPlan> Maps,
    IReadOnlyList<RosettaPlacementRecord> Placements,
    IReadOnlyList<RosettaDesignkitPlan> Designkits,
    IReadOnlyList<RosettaExcludedAsset> Exclusions,
    int NextUniqueId);

/// <summary>
/// Deterministic Spec 190 layout engine: packs every asset into its own cell on a uniform
/// designkit-style grid, paints the asset's name into the cell's label band via
/// <see cref="RosettaTextPainter"/>, populates per-chunk MCRF references so placements are admitted
/// by chunk-based pipelines, and emits one flat LK-form ADT per touched tile built on
/// <see cref="BlankAdtFactory"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Uniform grid.</b> A tile carries <c>(16 / CellChunks)²</c> identical cells whose edges land on
/// chunk boundaries. Cell size does not track asset size — an asset that does not fit the standard
/// cell goes to a separate run of oversize (whole-tile) cells, so no tile ever mixes cell sizes.
/// </para>
/// <para>
/// <b>Coordinate spaces.</b> Canvas offset <c>(u, v)</c> spans [0, TileSize]² inside a tile, with
/// <c>u</c> along the chunk-X axis and <c>v</c> along the chunk-Y axis. The viewer places tile
/// terrain at <c>rendererX = MapOrigin − tileX*TileSize − v</c> and
/// <c>rendererY = MapOrigin − tileY*TileSize − u</c>, so that is exactly what
/// <see cref="RosettaPlacementRecord.RendererPosition"/> holds, and it is what is handed to
/// <see cref="LkAdtWriter"/>. Handing the writer <see cref="RosettaPlacementRecord.RawPosition"/>
/// instead double-applies the MapOrigin flip and throws every object onto the perpendicular axis,
/// thousands of units off its own terrain — the "objects run horizontally, tiles run vertically"
/// defect. The tile-alignment tests exist to keep that from coming back.
/// </para>
/// </remarks>
public static class RosettaTilesetGenerator
{
    private const float MapOrigin = 17066.666f;

    /// <summary>Label plate: dark enough that bright glyphs read at a distance (127 is neutral).</summary>
    private const byte PlateChannel = 28;

    /// <summary>Cell boundary rule brightness — visibly above neutral, below the glyph white.</summary>
    private const byte BorderChannel = 196;

    /// <summary>Glyph ink: the brightest MCCV value, doubling terrain colour under the shader.</summary>
    private const byte GlyphChannel = 255;

    /// <summary>MCNK header flag 0x40: this chunk carries an MCCV sub-chunk.</summary>
    private const int McnkHasMccvFlag = 0x40;

    private sealed record LayoutCell(
        RosettaAssetEntry Asset,
        int SeqTile,
        float CellU,
        float CellV,
        float CellSize,
        float ObjectBandSize,
        string LabelText,
        IReadOnlyList<string> LabelLines,
        float LabelPixelMeters,
        int UniqueId);

    /// <summary>One uniform cell geometry: how big a cell is and how its label band is measured.</summary>
    private sealed class GridClass(int cellChunks, int labelBandChunks)
    {
        public int CellChunks { get; } = cellChunks;
        public int LabelBandChunks { get; } = labelBandChunks;
        public float CellSize { get; } = cellChunks * RosettaGeneratorOptions.ChunkSize;
        public int CellsPerAxis { get; } = RosettaGeneratorOptions.ChunksPerTileAxis / cellChunks;
        public List<RosettaAssetEntry> Assets { get; } = [];

        public int CellsPerTile => CellsPerAxis * CellsPerAxis;

        /// <summary>
        /// Square of cell the object is centred in. Normally the cell minus its label band; the
        /// oversize class gives the object the WHOLE cell (its label is painted under the geometry)
        /// because that band was excluding ~90 real assets that fit a tile perfectly well.
        /// </summary>
        public float ObjectBandSize => IsOversize
            ? CellSize
            : (CellChunks - LabelBandChunks) * RosettaGeneratorOptions.ChunkSize;

        /// <summary>Set for the whole-tile class that catches assets too big for a standard cell.</summary>
        public bool IsOversize { get; init; }

        public int TileCount => Assets.Count == 0 ? 0 : ((Assets.Count - 1) / CellsPerTile) + 1;
    }

    public static RosettaGenerationResult Generate(
        IReadOnlyList<RosettaAssetEntry> assets,
        RosettaGeneratorOptions options,
        IReadOnlySet<(int X, int Y)>? occupiedTiles = null)
    {
        ArgumentNullException.ThrowIfNull(assets);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();

        var exclusions = new List<RosettaExcludedAsset>();

        // Pass 1: split into designkits (the source folder IS the kit - Blizzard shipped them that
        // way) and lay each kit out on whole tiles of its own. A tile never mixes kits, so a kit is
        // always addressable as a contiguous run of tiles on exactly one map.
        List<KitLayout> kits = LayoutDesignkits(assets, options, exclusions);

        // Pass 2: pack kits into maps. A single 64x64 map cannot hold a full client's corpus at a
        // legible cell size, so kits spill onto RosettaDev00, RosettaDev01, ... and the index says
        // which map each one landed on.
        List<List<KitLayout>> mapGroups = PackKitsIntoMaps(kits, options);

        var maps = new List<RosettaMapPlan>(mapGroups.Count);
        var allPlacements = new List<RosettaPlacementRecord>();
        var allKits = new List<RosettaDesignkitPlan>(kits.Count);
        int uniqueId = 0;

        for (int mapIndex = 0; mapIndex < mapGroups.Count; mapIndex++)
        {
            string mapName = mapGroups.Count == 1
                ? options.MapName
                : $"{options.MapName}{mapIndex:00}";

            RosettaMapPlan plan = BuildMap(mapName, mapGroups[mapIndex], options, occupiedTiles, ref uniqueId);
            maps.Add(plan);
            allPlacements.AddRange(plan.Placements);
            allKits.AddRange(plan.Designkits);
        }

        return new RosettaGenerationResult(options.MapName, maps, allPlacements, allKits, exclusions, uniqueId);
    }

    /// <summary>A designkit's assets already bucketed into uniform cell classes and sized in tiles.</summary>
    private sealed record KitLayout(
        string Kit,
        IReadOnlyList<RosettaAssetEntry> StandardAssets,
        IReadOnlyList<RosettaAssetEntry> OversizeAssets,
        GridClass Standard,
        GridClass Oversize,
        int TileCount);

    /// <summary>
    /// Groups assets by source folder and measures each kit's tile footprint. Grouping can be turned
    /// off, in which case the whole corpus is one unnamed kit and the layout is exactly what it was
    /// before kits existed.
    /// </summary>
    private static List<KitLayout> LayoutDesignkits(
        IReadOnlyList<RosettaAssetEntry> assets,
        RosettaGeneratorOptions options,
        List<RosettaExcludedAsset> exclusions)
    {
        var byKit = new SortedDictionary<string, List<RosettaAssetEntry>>(StringComparer.Ordinal);
        foreach (RosettaAssetEntry asset in assets)
        {
            string kit = options.GroupByDesignkit ? DesignkitOf(asset.AssetPath, options.KitDepth) : string.Empty;
            if (!byKit.TryGetValue(kit, out List<RosettaAssetEntry>? list))
            {
                list = [];
                byKit[kit] = list;
            }

            list.Add(asset);
        }

        var kits = new List<KitLayout>(byKit.Count);
        foreach ((string kit, List<RosettaAssetEntry> kitAssets) in byKit)
        {
            kitAssets.Sort(static (a, b) => string.CompareOrdinal(a.AssetPath, b.AssetPath));

            var standard = new GridClass(options.CellChunks, options.LabelBandChunks);
            GridClass oversize = options.CellChunks >= RosettaGeneratorOptions.OversizeCellChunks
                ? standard
                : new GridClass(
                    RosettaGeneratorOptions.OversizeCellChunks,
                    Math.Min(options.LabelBandChunks, RosettaGeneratorOptions.OversizeCellChunks - 1))
                {
                    IsOversize = true,
                };

            foreach (RosettaAssetEntry asset in kitAssets)
            {
                float extentU = MathF.Abs(asset.BoundsMax.X - asset.BoundsMin.X);
                float extentV = MathF.Abs(asset.BoundsMax.Y - asset.BoundsMin.Y);

                // Geometry-less M2s (camera paths, some effects) carry degenerate bounds. Say so
                // plainly instead of reporting an infinite footprint against the cell size, and keep
                // NaN/infinity out of the layout arithmetic entirely.
                if (!float.IsFinite(extentU) || !float.IsFinite(extentV))
                {
                    exclusions.Add(new RosettaExcludedAsset(asset,
                        $"Bounds are not finite ({asset.BoundsMin} .. {asset.BoundsMax}); the asset carries no placeable geometry."));
                    continue;
                }

                float required = MathF.Max(extentU, extentV) + options.FootprintMarginMeters;

                if (required <= standard.ObjectBandSize)
                    standard.Assets.Add(asset);
                else if (required <= oversize.ObjectBandSize)
                    oversize.Assets.Add(asset);
                else
                    exclusions.Add(new RosettaExcludedAsset(asset,
                        $"Footprint {required - options.FootprintMarginMeters:F1}m (+{options.FootprintMarginMeters:F0}m margin) " +
                        $"exceeds the largest cell's {oversize.ObjectBandSize:F1}m object band."));
            }

            bool sharedClass = ReferenceEquals(oversize, standard);
            List<RosettaAssetEntry> standardAssets = [.. standard.Assets];
            List<RosettaAssetEntry> oversizeAssets = sharedClass ? [] : [.. oversize.Assets];
            if (standardAssets.Count == 0 && oversizeAssets.Count == 0)
                continue;

            kits.AddRange(MakeKitLayouts(kit, standardAssets, oversizeAssets, options));
        }

        return kits;
    }

    /// <summary>
    /// Turns one kit's assets into map-sized layout parts. Normally that is a single part; a kit
    /// bigger than a whole map (the client has a few, e.g. <c>item\objectcomponents</c>) is split
    /// into as many parts as it takes so the run never dead-ends. Split parts keep the kit's name,
    /// so the index lists the kit once per map it spans instead of pretending it fits one.
    /// </summary>
    private static IEnumerable<KitLayout> MakeKitLayouts(
        string kit,
        List<RosettaAssetEntry> standardAssets,
        List<RosettaAssetEntry> oversizeAssets,
        RosettaGeneratorOptions options)
    {
        int budget = Math.Min(options.MaxTilesPerMap, RosettaGeneratorOptions.TilesPerAxis * RosettaGeneratorOptions.TilesPerAxis);
        int cellsPerTile = ClassCellsPerTile(options.CellChunks);

        var parts = new List<KitLayout>();
        for (int i = 0; i < standardAssets.Count; i += budget * cellsPerTile)
        {
            List<RosettaAssetEntry> slice = standardAssets.GetRange(i, Math.Min(budget * cellsPerTile, standardAssets.Count - i));
            parts.Add(MakePart(kit, slice, [], options));
        }

        for (int i = 0; i < oversizeAssets.Count; i += budget)
        {
            List<RosettaAssetEntry> slice = oversizeAssets.GetRange(i, Math.Min(budget, oversizeAssets.Count - i));
            parts.Add(MakePart(kit, [], slice, options));
        }

        return parts;
    }

    private static KitLayout MakePart(
        string kit,
        List<RosettaAssetEntry> standardAssets,
        List<RosettaAssetEntry> oversizeAssets,
        RosettaGeneratorOptions options)
    {
        var standard = new GridClass(options.CellChunks, options.LabelBandChunks);
        standard.Assets.AddRange(standardAssets);

        GridClass oversize = options.CellChunks >= RosettaGeneratorOptions.OversizeCellChunks
            ? standard
            : new GridClass(
                RosettaGeneratorOptions.OversizeCellChunks,
                Math.Min(options.LabelBandChunks, RosettaGeneratorOptions.OversizeCellChunks - 1))
            {
                IsOversize = true,
            };

        if (!ReferenceEquals(oversize, standard))
            oversize.Assets.AddRange(oversizeAssets);

        bool sharedClass = ReferenceEquals(oversize, standard);
        int tileCount = standard.TileCount + (sharedClass ? 0 : oversize.TileCount);

        return new KitLayout(kit, standardAssets, sharedClass ? [] : oversizeAssets, standard, oversize, tileCount);
    }

    private static int ClassCellsPerTile(int cellChunks)
    {
        int perAxis = RosettaGeneratorOptions.ChunksPerTileAxis / cellChunks;
        return perAxis * perAxis;
    }

    /// <summary>
    /// Greedily fills maps with whole kits. A kit is never split across maps, so the index entry for
    /// a kit names exactly one map.
    /// </summary>
    private static List<List<KitLayout>> PackKitsIntoMaps(List<KitLayout> kits, RosettaGeneratorOptions options)
    {
        var groups = new List<List<KitLayout>>();
        var current = new List<KitLayout>();
        int currentTiles = 0;

        foreach (KitLayout kit in kits)
        {
            if (!FitsOnOneMap(kit.TileCount, options))
                throw new InvalidOperationException(
                    $"Designkit part '{kit.Kit}' needs {kit.TileCount} tiles, which does not fit a map " +
                    $"(budget {options.MaxTilesPerMap}). MakeKitLayouts should have split it - this is a bug.");

            if (current.Count > 0 && !FitsOnOneMap(currentTiles + kit.TileCount, options))
            {
                groups.Add(current);
                current = [];
                currentTiles = 0;
            }

            current.Add(kit);
            currentTiles += kit.TileCount;
        }

        if (current.Count > 0)
            groups.Add(current);

        return groups;
    }

    private static bool FitsOnOneMap(int tileCount, RosettaGeneratorOptions options)
    {
        if (tileCount > options.MaxTilesPerMap)
            return false;

        int side = (int)MathF.Ceiling(MathF.Sqrt(tileCount));
        return side <= RosettaGeneratorOptions.TilesPerAxis;
    }

    /// <summary>Resolves one map's kits onto a square tile block and builds every tile's ADT.</summary>
    private static RosettaMapPlan BuildMap(
        string mapName,
        List<KitLayout> kits,
        RosettaGeneratorOptions options,
        IReadOnlySet<(int X, int Y)>? occupiedTiles,
        ref int uniqueId)
    {
        var cells = new List<LayoutCell>();
        var kitRanges = new List<(string Kit, int FirstSeqTile, int TileCount, int Models, int WorldModels)>(kits.Count);
        int tileBase = 0;

        foreach (KitLayout kit in kits)
        {
            int firstSeqTile = tileBase;
            AppendClass(kit.Standard, ref tileBase, ref uniqueId, cells);
            if (!ReferenceEquals(kit.Oversize, kit.Standard))
                AppendClass(kit.Oversize, ref tileBase, ref uniqueId, cells);

            int models = kit.StandardAssets.Count(static a => a.Kind == RosettaAssetKind.Model)
                + kit.OversizeAssets.Count(static a => a.Kind == RosettaAssetKind.Model);
            int worldModels = kit.StandardAssets.Count(static a => a.Kind == RosettaAssetKind.WorldModel)
                + kit.OversizeAssets.Count(static a => a.Kind == RosettaAssetKind.WorldModel);
            kitRanges.Add((kit.Kit, firstSeqTile, tileBase - firstSeqTile, models, worldModels));
        }

        RosettaTileBlock block = AssignSquareBlockTileCoords(
            tileBase, options.StartTileX, options.StartTileY, occupiedTiles);
        Dictionary<int, (int TileX, int TileY)> tileCoords = block.Coords;

        var placementsByTile = new SortedDictionary<(int X, int Y), List<RosettaPlacementRecord>>();
        var labelsByTile = new Dictionary<(int X, int Y), List<RosettaLabel>>();
        var rectsByTile = new Dictionary<(int X, int Y), List<RosettaMccvRect>>();
        var placements = new List<RosettaPlacementRecord>(cells.Count);

        foreach (LayoutCell cell in cells)
        {
            (int tileX, int tileY) = tileCoords[cell.SeqTile];
            var key = (tileX, tileY);
            if (!placementsByTile.TryGetValue(key, out List<RosettaPlacementRecord>? list))
            {
                list = [];
                placementsByTile[key] = list;
                labelsByTile[key] = [];
                rectsByTile[key] = [];
            }

            float centerU = cell.CellU + (cell.CellSize / 2f);
            float centerV = cell.CellV + (cell.ObjectBandSize / 2f);

            // Canvas/file coords, and the renderer coords the viewer derives from them.
            var raw = new Vector3(
                (tileY * RosettaGeneratorOptions.TileSize) + centerU,
                (tileX * RosettaGeneratorOptions.TileSize) + centerV,
                0f);
            var renderer = new Vector3(MapOrigin - raw.Y, MapOrigin - raw.X, 0f);

            var record = new RosettaPlacementRecord(
                cell.Asset, tileX, tileY, raw, renderer,
                cell.CellU, cell.CellV, cell.CellSize,
                cell.LabelText, cell.LabelLines, cell.LabelPixelMeters, cell.UniqueId);
            list.Add(record);
            placements.Add(record);

            AppendCellPaint(cell, labelsByTile[key], rectsByTile[key], options.PaintCellBorders);
        }

        var tiles = new List<RosettaTilePlan>(placementsByTile.Count);
        foreach ((var key, List<RosettaPlacementRecord> tilePlacements) in placementsByTile)
            tiles.Add(new RosettaTilePlan(key.X, key.Y, tilePlacements, rectsByTile[key], labelsByTile[key]));

        // One index entry per (kit, map). A kit contributes several layout parts here — its standard
        // cells and its oversize cells are laid out separately — but those are an implementation
        // detail of the packing, not two kits, so they merge back into one entry.
        var designkits = new List<RosettaDesignkitPlan>(kitRanges.Count);
        var entryByKit = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach ((string kit, int firstSeqTile, int tileCount, int models, int worldModels) in kitRanges)
        {
            var kitTiles = new List<(int TileX, int TileY)>(tileCount);
            for (int i = 0; i < tileCount; i++)
                kitTiles.Add(tileCoords[firstSeqTile + i]);

            if (entryByKit.TryGetValue(kit, out int existing))
            {
                RosettaDesignkitPlan merged = designkits[existing];
                designkits[existing] = merged with
                {
                    Tiles = [.. merged.Tiles, .. kitTiles],
                    AssetCount = merged.AssetCount + models + worldModels,
                    ModelCount = merged.ModelCount + models,
                    WorldModelCount = merged.WorldModelCount + worldModels,
                };
                continue;
            }

            entryByKit[kit] = designkits.Count;
            designkits.Add(new RosettaDesignkitPlan(
                kit, mapName, kitTiles, models + worldModels, models, worldModels));
        }

        return new RosettaMapPlan(
            mapName, tiles, placements, designkits, block.OriginX, block.OriginY, block.Side,
            options.GroundTexture, options.InkTexture);
    }

    /// <summary>
    /// Builds one planned tile's painted, placement-carrying LK ADT. Deliberately on demand: the
    /// caller writes each tile and drops it, which is what keeps a full-client corpus inside memory.
    /// </summary>
    public static LkAdtData BuildTileAdt(string mapName, RosettaTilePlan tile, string? groundTexture = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);
        ArgumentNullException.ThrowIfNull(tile);

        int tileX = tile.TileX;
        int tileY = tile.TileY;
        IReadOnlyList<RosettaPlacementRecord> tilePlacements = tile.Placements;
        IReadOnlyList<RosettaMccvRect> rects = tile.Rects;
        IReadOnlyList<RosettaLabel> labels = tile.Labels;

        LkAdtData blank = string.IsNullOrWhiteSpace(groundTexture)
            ? BlankAdtFactory.CreateBlank(mapName, tileX, tileY)
            : BlankAdtFactory.CreateBlank(mapName, tileX, tileY, groundTexture);
        IReadOnlyList<LkMcnkData> paintedChunks = RosettaTextPainter.PaintTile(
            blank.Chunks, rects, labels, RosettaGeneratorOptions.ChunkSize);

        var mddf = new List<LkMddfEntry>();
        var modf = new List<LkModfEntry>();
        var modelNames = new List<string>();
        var wmoNames = new List<string>();

        foreach (RosettaPlacementRecord placement in tilePlacements)
        {
            // LkAdtWriter takes RENDERER coordinates and applies the MapOrigin flip itself.
            if (placement.Asset.Kind == RosettaAssetKind.Model)
            {
                int nameId = IndexOfOrAdd(modelNames, placement.Asset.AssetPath);
                mddf.Add(new LkMddfEntry(nameId, placement.UniqueId, placement.RendererPosition, Vector3.Zero, 1f));
            }
            else
            {
                int nameId = IndexOfOrAdd(wmoNames, placement.Asset.AssetPath);
                // Rotation is zero, so a square half-extent from the larger horizontal axis is a
                // conservative world-space AABB no matter which model axis maps to which.
                float extentU = MathF.Abs(placement.Asset.BoundsMax.X - placement.Asset.BoundsMin.X);
                float extentV = MathF.Abs(placement.Asset.BoundsMax.Y - placement.Asset.BoundsMin.Y);
                float half = MathF.Max(extentU, extentV) / 2f;
                float minZ = MathF.Min(placement.Asset.BoundsMin.Z, 0f);
                float maxZ = MathF.Max(placement.Asset.BoundsMax.Z, 0f);
                Vector3 center = placement.RendererPosition;
                var boundsMin = new Vector3(center.X - half, center.Y - half, center.Z + minZ);
                var boundsMax = new Vector3(center.X + half, center.Y + half, center.Z + maxZ);
                modf.Add(new LkModfEntry(
                    nameId, placement.UniqueId, center, Vector3.Zero,
                    boundsMin, boundsMax, Flags: 0, DoodadSet: 0, NameSet: 0, Scale: 1f));
            }
        }

        // Per-chunk MCRF references (placement indices, matching BlankAdtFactory's convention)
        // so chunk-admitted render pipelines see every object.
        var mddfRefsByChunk = AssignRefsToChunks(paintedChunks.Count, tilePlacements
            .Select(p => (Index: mddf.IndexOfByUniqueId(p.UniqueId), U: p.RawPosition.X - (p.TileY * RosettaGeneratorOptions.TileSize), V: p.RawPosition.Y - (p.TileX * RosettaGeneratorOptions.TileSize)))
            .Where(static t => t.Index >= 0));
        var modfRefsByChunk = AssignRefsToChunks(paintedChunks.Count, tilePlacements
            .Select(p => (Index: modf.IndexOfByUniqueId(p.UniqueId), U: p.RawPosition.X - (p.TileY * RosettaGeneratorOptions.TileSize), V: p.RawPosition.Y - (p.TileX * RosettaGeneratorOptions.TileSize)))
            .Where(static t => t.Index >= 0));

        var finalChunks = new LkMcnkData[paintedChunks.Count];
        for (int i = 0; i < paintedChunks.Count; i++)
        {
            LkMcnkData c = paintedChunks[i];
            finalChunks[i] = new LkMcnkData
            {
                IndexX = c.IndexX,
                IndexY = c.IndexY,
                // 0x40 = has_mccv. The readers here locate MCCV by scanning sub-chunk FourCCs,
                // but a tile that carries vertex paint without advertising it is a trap for any
                // consumer that trusts the flag - and this corpus exists to be consumed.
                Flags = c.MccvColors is not null ? c.Flags | McnkHasMccvFlag : c.Flags,
                AreaId = c.AreaId,
                NLayers = c.NLayers,
                HoleMask = c.HoleMask,
                BaseHeight = c.BaseHeight,
                Heights = c.Heights,
                Normals = c.Normals,
                ShadowMap = c.ShadowMap,
                AlphaMapData = c.AlphaMapData,
                AlphaMapSize = c.AlphaMapSize,
                Layers = c.Layers,
                DoodadRefs = mddfRefsByChunk.TryGetValue(i, out List<int>? dr) ? dr : [],
                WorldModelRefs = modfRefsByChunk.TryGetValue(i, out List<int>? wr) ? wr : [],
                LiquidData = c.LiquidData,
                MccvColors = c.MccvColors,
                MclvLighting = c.MclvLighting,
                PosX = c.PosX,
                PosY = c.PosY,
                PosZ = c.PosZ,
            };
        }

        return new LkAdtData
        {
            MapName = blank.MapName,
            TileX = blank.TileX,
            TileY = blank.TileY,
            TextureNames = blank.TextureNames,
            ModelNames = modelNames,
            WorldModelNames = wmoNames,
            ModelPlacements = mddf,
            WorldModelPlacements = modf,
            Chunks = finalChunks,
            MhdrFlags = blank.MhdrFlags,
            MfboFlightBounds = blank.MfboFlightBounds,
        };
    }

    /// <summary>
    /// The designkit an asset belongs to: its source folder, backslash-separated and lowercased so
    /// the client's inconsistent casing (CREATURE\ vs Creature\) does not split one kit in two.
    /// </summary>
    /// <param name="depth">
    /// How many leading path segments make up the kit. 0 (the default) uses the whole folder — the
    /// finest grain, which gives a whole tile to folders holding a single asset. A small depth
    /// coarsens the grouping (depth 1 puts all of <c>creature\*</c> in one kit), trading kit
    /// precision for a far smaller tile count.
    /// </param>
    public static string DesignkitOf(string assetPath, int depth = 0)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(assetPath);
        ArgumentOutOfRangeException.ThrowIfNegative(depth);

        string normalized = assetPath.Replace('/', '\\');
        int lastSeparator = normalized.LastIndexOf('\\');
        if (lastSeparator <= 0)
            return string.Empty;

        string folder = normalized[..lastSeparator].ToLowerInvariant();
        if (depth == 0)
            return folder;

        string[] segments = folder.Split('\\', StringSplitOptions.RemoveEmptyEntries);
        return string.Join('\\', segments.Take(depth));
    }

    /// <summary>
    /// Lays a class's assets out row-major across whole tiles of identical cells, advancing the
    /// shared tile counter so the next class starts on a fresh tile.
    /// </summary>
    private static void AppendClass(GridClass cls, ref int tileBase, ref int uniqueId, List<LayoutCell> cells)
    {
        float pixelMeters = RosettaTextPainter.SubCellFor(RosettaGeneratorOptions.ChunkSize);
        int charsPerLine = RosettaTextPainter.CharsPerLine(cls.CellSize, pixelMeters);

        for (int index = 0; index < cls.Assets.Count; index++)
        {
            RosettaAssetEntry asset = cls.Assets[index];
            int seqTile = tileBase + index / cls.CellsPerTile;
            int cellIndex = index % cls.CellsPerTile;
            float cellU = (cellIndex % cls.CellsPerAxis) * cls.CellSize;
            float cellV = (cellIndex / cls.CellsPerAxis) * cls.CellSize;

            // Keep the extension. It is the asset TYPE, which is the one thing a painted label has
            // to carry that the name alone does not - .MDX vs .WMO changes what the cell even is.
            // Middle-elision puts it at the surviving tail, so it shows even on a clipped name; the
            // plate tint is a redundant cue, not a replacement.
            string labelText = SanitizeLabel(asset.AssetPath);
            IReadOnlyList<string> lines = WrapLabel(labelText, charsPerLine, cls.LabelBandChunks);

            uniqueId++;
            cells.Add(new LayoutCell(
                asset, seqTile, cellU, cellV, cls.CellSize, cls.ObjectBandSize,
                labelText, lines, pixelMeters, uniqueId));
        }

        tileBase += cls.TileCount;
    }

    /// <summary>
    /// Emits the cell's MCCV paint: a dark plate behind the label band, the wrapped text lines
    /// centered on the lattice, and (optionally) a one-sub-cell rule along the cell boundary so the
    /// grid is visibly a grid.
    /// </summary>
    private static void AppendCellPaint(
        LayoutCell cell, List<RosettaLabel> labels, List<RosettaMccvRect> rects, bool paintBorders)
    {
        float pixel = cell.LabelPixelMeters;
        float bandV0 = cell.CellV + cell.ObjectBandSize;
        float lineAdvance = RosettaTextPainter.LineAdvanceRows * pixel;
        (byte plateR, byte plateG, byte plateB) = PlateColorFor(cell.Asset.Kind);

        rects.Add(new RosettaMccvRect(
            cell.CellU, bandV0, cell.CellU + cell.CellSize, cell.CellV + cell.CellSize,
            plateR, plateG, plateB));

        if (paintBorders)
        {
            float half = pixel / 2f;
            float u0 = cell.CellU, v0 = cell.CellV;
            float u1 = cell.CellU + cell.CellSize, v1 = cell.CellV + cell.CellSize;
            rects.Add(new RosettaMccvRect(u0 - half, v0 - half, u1 + half, v0 + half, BorderChannel, BorderChannel, BorderChannel));
            rects.Add(new RosettaMccvRect(u0 - half, v1 - half, u1 + half, v1 + half, BorderChannel, BorderChannel, BorderChannel));
            rects.Add(new RosettaMccvRect(u0 - half, v0 - half, u0 + half, v1 + half, BorderChannel, BorderChannel, BorderChannel));
            rects.Add(new RosettaMccvRect(u1 - half, v0 - half, u1 + half, v1 + half, BorderChannel, BorderChannel, BorderChannel));
        }

        for (int line = 0; line < cell.LabelLines.Count; line++)
        {
            string text = cell.LabelLines[line];
            if (text.Length == 0)
                continue;

            float width = RosettaTextPainter.MeasureWidthMeters(text, pixel);
            // Snap the left edge to the vertex lattice: with an on-lattice origin the inner vertices
            // land on font-pixel centres and the outer ones on the corners, which is what makes the
            // antialiased resolve clean instead of smeared.
            float inset = MathF.Floor(MathF.Max(0f, (cell.CellSize - width) / 2f) / pixel) * pixel;
            labels.Add(new RosettaLabel(
                text,
                cell.CellU + inset,
                bandV0 + (line * lineAdvance),
                pixel,
                GlyphChannel, GlyphChannel, GlyphChannel,
                plateR, plateG, plateB));
        }
    }

    /// <summary>
    /// Plate tint per asset kind. MCCV multiplies the terrain texture, so these read as a cool
    /// (world model) or warm (model) cast on the dark label plate — enough to tell the two apart
    /// at a glance now that the extension is no longer painted.
    /// </summary>
    private static (byte R, byte G, byte B) PlateColorFor(RosettaAssetKind kind) => kind switch
    {
        RosettaAssetKind.WorldModel => (PlateChannel, (byte)(PlateChannel + 6), (byte)(PlateChannel + 18)),
        _ => ((byte)(PlateChannel + 18), (byte)(PlateChannel + 8), PlateChannel),
    };

    /// <summary>
    /// Hard-wraps a label (asset names carry no spaces) to at most <paramref name="maxLines"/> lines.
    /// A name too long for the band is elided in the MIDDLE, not cut at the end: WoW asset names put
    /// the family up front and the discriminator at the back
    /// (<c>ICECROWN_WALL_SEMICIRCLE_PIECE_02_LONG_HOLLOW</c>), so a tail cut throws away exactly the
    /// part that tells two cells apart.
    /// </summary>
    public static IReadOnlyList<string> WrapLabel(string text, int charsPerLine, int maxLines)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (charsPerLine <= 0 || maxLines <= 0)
            return [];

        int capacity = charsPerLine * maxLines;
        if (text.Length > capacity)
        {
            // Keep the leading identity and the trailing discriminator, marked with '-'.
            int tail = (capacity - 1) / 2;
            int head = capacity - 1 - tail;
            text = string.Concat(text.AsSpan(0, head), "-", text.AsSpan(text.Length - tail));
        }

        var lines = new List<string>(maxLines);
        for (int i = 0; i < text.Length && lines.Count < maxLines; i += charsPerLine)
            lines.Add(text.Substring(i, Math.Min(charsPerLine, text.Length - i)));

        return lines;
    }

    /// <summary>
    /// Assigns tiles contiguously in row-major order from the start tile, wrapping across the
    /// 64x64 grid and skipping occupied coordinates. The full ~14k-asset corpus needs hundreds of
    /// tiles, so contiguous fill naturally grows to cover the map — no scattering.
    /// </summary>
    /// <summary>
    /// Lays the tiles out as a SQUARE block anchored at the start tile: side = ceil(sqrt(count)),
    /// filled row-major, so the finished map is a predictable rectangle from (StartTileX, StartTileY)
    /// instead of a strip that wraps at column 63 and lands wherever the asset count happens to put
    /// it. Occupied coordinates are skipped, which spills into extra rows below the block rather than
    /// perturbing its width. If the block would run off the 64x64 grid the origin is pulled back to
    /// the largest coordinate that fits, and the caller is told where it actually landed.
    /// </summary>
    private static RosettaTileBlock AssignSquareBlockTileCoords(
        int tileCount,
        int startX,
        int startY,
        IReadOnlySet<(int X, int Y)>? occupiedTiles)
    {
        int axis = RosettaGeneratorOptions.TilesPerAxis;
        if (tileCount <= 0)
            return new RosettaTileBlock([], startX, startY, 0);

        int side = (int)MathF.Ceiling(MathF.Sqrt(tileCount));
        if (side > axis)
            throw new InvalidOperationException(
                $"Cannot place {tileCount} Rosetta tiles: a square block needs a {side}x{side} area but the map is only {axis}x{axis}. " +
                "LOWER --cell-chunks so more cells fit per tile (each halving quadruples the cells per tile, " +
                "at the cost of a narrower label band), split the corpus across maps with --map-name, or cap it with --max-assets.");

        int originX = Math.Clamp(startX, 0, axis - side);
        int originY = Math.Clamp(startY, 0, axis - side);

        var result = new Dictionary<int, (int TileX, int TileY)>(tileCount);
        int assigned = 0;
        int slot = 0;
        while (assigned < tileCount)
        {
            int tileX = originX + (slot % side);
            int tileY = originY + (slot / side);
            slot++;

            if (tileY >= axis)
                throw new InvalidOperationException(
                    $"Cannot place {tileCount} Rosetta tiles in a {side}-wide block from ({originX},{originY}): " +
                    "too many occupied tiles pushed the block off the bottom of the map.");

            if (occupiedTiles is not null && occupiedTiles.Contains((tileX, tileY)))
                continue;

            result[assigned++] = (tileX, tileY);
        }

        return new RosettaTileBlock(result, originX, originY, side);
    }

    private sealed record RosettaTileBlock(
        Dictionary<int, (int TileX, int TileY)> Coords,
        int OriginX,
        int OriginY,
        int Side);

    /// <summary>Reduces an asset path to the label-friendly characters the bitmap font supports.</summary>
    public static string SanitizeLabel(string assetPath, bool stripExtension = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(assetPath);

        string name = stripExtension
            ? Path.GetFileNameWithoutExtension(assetPath)
            : Path.GetFileName(assetPath);
        if (name.Length == 0)
            name = Path.GetFileName(assetPath);
        char[] chars = new char[name.Length];
        int count = 0;
        foreach (char c in name)
        {
            char upper = char.ToUpperInvariant(c);
            bool supported = char.IsAsciiLetterOrDigit(upper) || upper is '_' or '-' or '.';
            chars[count++] = supported ? upper : '_';
        }

        return new string(chars, 0, count);
    }

    private static Dictionary<int, List<int>> AssignRefsToChunks(
        int chunkCount,
        IEnumerable<(int Index, float U, float V)> items)
    {
        var result = new Dictionary<int, List<int>>();
        foreach (var item in items)
        {
            int cx = Math.Clamp((int)(item.U / RosettaGeneratorOptions.ChunkSize), 0, 15);
            int cy = Math.Clamp((int)(item.V / RosettaGeneratorOptions.ChunkSize), 0, 15);
            int chunkIndex = cy * 16 + cx;
            if (chunkIndex < 0 || chunkIndex >= chunkCount)
                continue;

            if (!result.TryGetValue(chunkIndex, out var list))
            {
                list = [];
                result[chunkIndex] = list;
            }

            list.Add(item.Index);
        }

        return result;
    }

    private static int IndexOfByUniqueId(this IReadOnlyList<LkMddfEntry> entries, int uniqueId)
    {
        for (int i = 0; i < entries.Count; i++)
        {
            if (entries[i].UniqueId == uniqueId)
                return i;
        }

        return -1;
    }

    private static int IndexOfByUniqueId(this IReadOnlyList<LkModfEntry> entries, int uniqueId)
    {
        for (int i = 0; i < entries.Count; i++)
        {
            if (entries[i].UniqueId == uniqueId)
                return i;
        }

        return -1;
    }

    private static int IndexOfOrAdd(List<string> names, string name)
    {
        // MMDX/MWMO entries use backslash-separated paths (real ADT convention); listfiles and
        // archive enumeration use forward slashes.
        string normalized = name.Replace('/', '\\');
        int index = names.IndexOf(normalized);
        if (index >= 0)
            return index;
        names.Add(normalized);
        return names.Count - 1;
    }
}
