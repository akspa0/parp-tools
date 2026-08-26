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

public sealed record RosettaGeneratorOptions(
    string MapName,
    int StartTileX = 24,
    int StartTileY = 24,
    float CellMarginMeters = 32f,
    float MinCellSize = 48f,
    float MaxCellSize = 512f,
    float LabelCapHeightMeters = 42f)
{
    public const float TileSize = 533.33333f;
    public const float ChunkSize = TileSize / 16f;
}

public sealed record RosettaPlacementRecord(
    RosettaAssetEntry Asset,
    int TileX,
    int TileY,
    Vector3 WorldPosition,
    float CellSize,
    string LabelText,
    int UniqueId);

public sealed record RosettaTilePlan(
    int TileX,
    int TileY,
    LkAdtData AdtData,
    IReadOnlyList<RosettaPlacementRecord> Placements);

public sealed record RosettaExcludedAsset(RosettaAssetEntry Asset, string Reason);

public sealed record RosettaGenerationResult(
    string MapName,
    IReadOnlyList<RosettaTilePlan> Tiles,
    IReadOnlyList<RosettaPlacementRecord> Placements,
    IReadOnlyList<RosettaExcludedAsset> Exclusions,
    int NextUniqueId);

/// <summary>
/// Deterministic Spec 190 layout engine: packs every asset into its own non-overlapping cell on a
/// continuous designkit-style canvas (uniform spacing, tile boundaries crossed freely), paints the
/// asset's name below it via <see cref="RosettaTextPainter"/>, and emits one flat LK-form ADT per
/// touched tile built on <see cref="BlankAdtFactory"/>.
/// </summary>
public static class RosettaTilesetGenerator
{
    private const float MapOrigin = 17066.666f;

    public static RosettaGenerationResult Generate(
        IReadOnlyList<RosettaAssetEntry> assets,
        RosettaGeneratorOptions options,
        IReadOnlySet<(int X, int Y)>? occupiedTiles = null)
    {
        ArgumentNullException.ThrowIfNull(assets);
        ArgumentNullException.ThrowIfNull(options);

        List<RosettaAssetEntry> ordered = [.. assets];
        ordered.Sort(static (a, b) => string.CompareOrdinal(a.AssetPath, b.AssetPath));

        var exclusions = new List<RosettaExcludedAsset>();
        var placementsByTile = new SortedDictionary<(int X, int Y), List<RosettaPlacementRecord>>();
        var labelsByTile = new Dictionary<(int X, int Y), List<RosettaLabel>>();
        var placements = new List<RosettaPlacementRecord>();

        int tileX = options.StartTileX;
        int tileY = options.StartTileY;
        float cursorU = 0f;
        float cursorV = 0f;
        float rowHeight = 0f;
        int uniqueId = 0;

        void AdvanceTile()
        {
            do
            {
                tileY++;
                if (tileY >= 64)
                {
                    tileY = 0;
                    tileX++;
                }
            }
            while (occupiedTiles is not null && occupiedTiles.Contains((tileX, tileY)) && tileX < 64);

            cursorU = 0f;
            cursorV = 0f;
            rowHeight = 0f;
        }

        if (occupiedTiles is not null && occupiedTiles.Contains((tileX, tileY)))
            AdvanceTile();

        foreach (RosettaAssetEntry asset in ordered)
        {
            float extentX = MathF.Abs(asset.BoundsMax.X - asset.BoundsMin.X);
            float extentY = MathF.Abs(asset.BoundsMax.Y - asset.BoundsMin.Y);
            float footprint = MathF.Max(extentX, extentY);
            float cell = Math.Clamp(
                (footprint * 1.25f) + options.CellMarginMeters,
                options.MinCellSize,
                options.MaxCellSize);

            if (footprint > RosettaGeneratorOptions.TileSize - 2f)
            {
                exclusions.Add(new RosettaExcludedAsset(
                    asset, $"Footprint {footprint:F1}m exceeds one tile."));
                continue;
            }

            // Label must fit inside the cell too; shrink cap height or truncate rather than overlap.
            string labelText = SanitizeLabel(asset.AssetPath);
            float labelWidth = RosettaTextPainter.MeasureWidthMeters(
                labelText, options.LabelCapHeightMeters, RosettaGeneratorOptions.ChunkSize);
            while (labelWidth > cell && labelText.Length > 4)
            {
                labelText = labelText[..^1];
                labelWidth = RosettaTextPainter.MeasureWidthMeters(
                    labelText, options.LabelCapHeightMeters, RosettaGeneratorOptions.ChunkSize);
            }

            if (cell > RosettaGeneratorOptions.TileSize - cursorU)
            {
                // wrap to next row within this tile
                cursorV += rowHeight;
                cursorU = 0f;
                rowHeight = 0f;
            }

            if (cell > RosettaGeneratorOptions.TileSize - cursorV)
                AdvanceTile();

            uniqueId++;
            var key = (tileX, tileY);
            if (!placementsByTile.TryGetValue(key, out List<RosettaPlacementRecord>? list))
            {
                list = [];
                placementsByTile[key] = list;
                labelsByTile[key] = [];
            }

            float centerU = cursorU + cell / 2f;
            float centerV = cursorV + cell / 2f;
            var world = new Vector3(
                tileX * RosettaGeneratorOptions.TileSize + centerU,
                MapOrigin - tileY * RosettaGeneratorOptions.TileSize - centerV,
                0f);

            var record = new RosettaPlacementRecord(asset, tileX, tileY, world, cell, labelText, uniqueId);
            list.Add(record);
            placements.Add(record);

            float labelCapHeight = MathF.Min(options.LabelCapHeightMeters, cell / 8f);
            float labelV = cursorV + cell - labelCapHeight * RosettaTextPainter.GlyphRows - 4f;
            float labelU = cursorU + MathF.Max(0f, (cell - labelWidth) / 2f);
            labelsByTile[key].Add(new RosettaLabel(labelText, labelU, labelV, labelCapHeight));

            cursorU += cell;
            rowHeight = MathF.Max(rowHeight, cell);
        }

        var tiles = new List<RosettaTilePlan>();
        foreach (var kvp in placementsByTile)
        {
            LkAdtData blank = BlankAdtFactory.CreateBlank(options.MapName, kvp.Key.X, kvp.Key.Y);
            IReadOnlyList<LkMcnkData> paintedChunks = RosettaTextPainter.PaintTileLabels(
                blank.Chunks, labelsByTile[kvp.Key], RosettaGeneratorOptions.ChunkSize);

            var mddf = new List<LkMddfEntry>();
            var modf = new List<LkModfEntry>();
            var modelNames = new List<string>();
            var wmoNames = new List<string>();

            foreach (RosettaPlacementRecord placement in kvp.Value)
            {
                if (placement.Asset.Kind == RosettaAssetKind.Model)
                {
                    int nameId = IndexOfOrAdd(modelNames, placement.Asset.AssetPath);
                    mddf.Add(new LkMddfEntry(nameId, placement.UniqueId, placement.WorldPosition, Vector3.Zero, 1f));
                }
                else
                {
                    int nameId = IndexOfOrAdd(wmoNames, placement.Asset.AssetPath);
                    var boundsMin = placement.WorldPosition + placement.Asset.BoundsMin;
                    var boundsMax = placement.WorldPosition + placement.Asset.BoundsMax;
                    modf.Add(new LkModfEntry(
                        nameId, placement.UniqueId, placement.WorldPosition, Vector3.Zero,
                        boundsMin, boundsMax, Flags: 0, DoodadSet: 0, NameSet: 0, Scale: 1f));
                }
            }

            var adt = new LkAdtData
            {
                MapName = blank.MapName,
                TileX = blank.TileX,
                TileY = blank.TileY,
                TextureNames = blank.TextureNames,
                ModelNames = modelNames,
                WorldModelNames = wmoNames,
                ModelPlacements = mddf,
                WorldModelPlacements = modf,
                Chunks = paintedChunks,
                MhdrFlags = blank.MhdrFlags,
                MfboFlightBounds = blank.MfboFlightBounds,
            };

            tiles.Add(new RosettaTilePlan(kvp.Key.X, kvp.Key.Y, adt, kvp.Value));
        }

        return new RosettaGenerationResult(options.MapName, tiles, placements, exclusions, uniqueId);
    }

    /// <summary>Reduces an asset path to the label-friendly characters the bitmap font supports.</summary>
    public static string SanitizeLabel(string assetPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(assetPath);

        string name = Path.GetFileName(assetPath);
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

    private static int IndexOfOrAdd(List<string> names, string name)
    {
        int index = names.IndexOf(name);
        if (index >= 0)
            return index;
        names.Add(name);
        return names.Count - 1;
    }
}
