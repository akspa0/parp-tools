using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Procedural;

/// <summary>
/// Density preset controlling how closely exhibits are arranged within procedural courtyards.
/// </summary>
public enum DensityPreset
{
    /// <summary>Maximum exhibit packing density (Micro: 16.66m, Small: 33.33m).</summary>
    Compact,
    /// <summary>Balanced density with clear landscaped paths (Micro: 33.33m, Small: 66.66m).</summary>
    Balanced,
    /// <summary>Grand exhibition spacing with wide ceremonial walkways.</summary>
    Spacious
}

/// <summary>
/// Configuration options for the adaptive layout packing engine.
/// </summary>
public sealed record AdaptiveLayoutOptions(
    DensityPreset Density = DensityPreset.Balanced,
    float GlobalM2ScaleMultiplier = 1.0f,
    bool ForceUniformScale = false,
    float UniformScaleValue = 2.5f,
    float PedestalMarginMeters = 3.0f,
    float BasePedestalHeightMeters = 4.0f);

/// <summary>
/// A concrete placement plan for an individual exhibit computed by the adaptive layout engine.
/// </summary>
public sealed record AdaptiveExhibitPlacement(
    RosettaAssetEntry Asset,
    SemanticAssetClassification Classification,
    float ComputedScale,
    Vector3 ScaledBoundsMin,
    Vector3 ScaledBoundsMax,
    float CellSize,
    float CellU,
    float CellV,
    int TileX,
    int TileY,
    Vector3 WorldPosition,
    Vector3 PedestalMin,
    Vector3 PedestalMax,
    float PedestalHeight);

/// <summary>
/// Multi-tier spatial partitioning engine that packs assets into appropriately sized
/// courtyards based on semantic archetype, scaled bounding volume, and density preset.
/// </summary>
public static class AdaptiveLayoutPacker
{
    public const float TileSizeMeters = 533.33333f;
    public const float ChunkSizeMeters = 33.33333f;
    private const float MapOrigin = 17066.666f;

    /// <summary>
    /// Computes the effective scale multiplier for an asset based on options and semantic classification.
    /// </summary>
    public static float ComputeScale(RosettaAssetEntry asset, AdaptiveLayoutOptions options)
    {
        if (asset.Kind == RosettaAssetKind.WorldModel)
            return 1.0f;

        if (options.ForceUniformScale)
            return Math.Clamp(options.UniformScaleValue, 0.5f, 10.0f);

        SemanticAssetClassification classification = SemanticAssetClassifier.Classify(
            asset.AssetPath, asset.BoundsMin, asset.BoundsMax, asset.Kind);

        return Math.Clamp(classification.RecommendedScale * options.GlobalM2ScaleMultiplier, 0.5f, 10.0f);
    }

    /// <summary>
    /// Resolves the cell size in meters for an asset based on density preset and semantic classification.
    /// </summary>
    public static float ResolveCellSize(
        SemanticAssetClassification classification,
        Vector3 scaledBoundsMin,
        Vector3 scaledBoundsMax,
        DensityPreset preset)
    {
        float maxSpan = MathF.Max(
            MathF.Abs(scaledBoundsMax.X - scaledBoundsMin.X),
            MathF.Abs(scaledBoundsMax.Y - scaledBoundsMin.Y));

        // Ensure cell is at least 1.4x larger than the scaled object's horizontal span
        float requiredSpan = maxSpan * 1.4f;

        return (preset, classification.RecommendedDensityTier) switch
        {
            (DensityPreset.Compact, CellDensityTier.Micro) when requiredSpan <= 14f => 16.66666f,
            (DensityPreset.Compact, CellDensityTier.Small) when requiredSpan <= 28f => 33.33333f,
            (DensityPreset.Compact, CellDensityTier.Medium) when requiredSpan <= 58f => 66.66666f,
            (DensityPreset.Compact, CellDensityTier.Large) when requiredSpan <= 118f => 133.33333f,

            (DensityPreset.Balanced, CellDensityTier.Micro) when requiredSpan <= 14f => 16.66666f,
            (DensityPreset.Balanced, CellDensityTier.Micro) when requiredSpan <= 28f => 33.33333f,
            (DensityPreset.Balanced, CellDensityTier.Small) when requiredSpan <= 28f => 33.33333f,
            (DensityPreset.Balanced, CellDensityTier.Small) when requiredSpan <= 58f => 66.66666f,
            (DensityPreset.Balanced, CellDensityTier.Medium) when requiredSpan <= 58f => 66.66666f,
            (DensityPreset.Balanced, CellDensityTier.Medium) when requiredSpan <= 118f => 133.33333f,
            (DensityPreset.Balanced, CellDensityTier.Large) when requiredSpan <= 118f => 133.33333f,

            (DensityPreset.Spacious, CellDensityTier.Micro) => 33.33333f,
            (DensityPreset.Spacious, CellDensityTier.Small) => 66.66666f,
            (DensityPreset.Spacious, CellDensityTier.Medium) => 133.33333f,
            (DensityPreset.Spacious, CellDensityTier.Large) => 266.66666f,

            _ when requiredSpan <= 14f => 16.66666f,
            _ when requiredSpan <= 28f => 33.33333f,
            _ when requiredSpan <= 58f => 66.66666f,
            _ when requiredSpan <= 118f => 133.33333f,
            _ when requiredSpan <= 238f => 266.66666f,
            _ => 533.33333f
        };
    }

    /// <summary>
    /// Packs an ordered list of assets into a structured grid of multi-tier exhibition tiles.
    /// </summary>
    public static List<AdaptiveExhibitPlacement> PackExhibits(
        IReadOnlyList<RosettaAssetEntry> assets,
        AdaptiveLayoutOptions options,
        int originTileX = 24,
        int originTileY = 24)
    {
        var placements = new List<AdaptiveExhibitPlacement>(assets.Count);
        if (assets.Count == 0)
            return placements;

        // Group assets by resolved cell size to pack uniformly sized courtyards together
        var prepared = assets.Select(asset =>
        {
            var classification = SemanticAssetClassifier.Classify(asset.AssetPath, asset.BoundsMin, asset.BoundsMax, asset.Kind);
            float scale = ComputeScale(asset, options);
            Vector3 scaledMin = asset.BoundsMin * scale;
            Vector3 scaledMax = asset.BoundsMax * scale;
            float cellSize = ResolveCellSize(classification, scaledMin, scaledMax, options.Density);
            return (Asset: asset, Classification: classification, Scale: scale, ScaledMin: scaledMin, ScaledMax: scaledMax, CellSize: cellSize);
        }).GroupBy(p => p.CellSize).OrderBy(g => g.Key);

        int currentTileX = originTileX;
        int currentTileY = originTileY;
        int maxTilesPerRow = 16;
        int tilesInCurrentRow = 0;

        foreach (var group in prepared)
        {
            float cellSize = group.Key;
            int cellsPerTileEdge = Math.Max(1, (int)MathF.Round(TileSizeMeters / cellSize));
            int cellsPerTile = cellsPerTileEdge * cellsPerTileEdge;

            int cellIndex = 0;

            foreach (var item in group)
            {
                int localCellIndex = cellIndex % cellsPerTile;
                if (cellIndex > 0 && localCellIndex == 0)
                {
                    // Advance to next tile
                    tilesInCurrentRow++;
                    if (tilesInCurrentRow >= maxTilesPerRow)
                    {
                        tilesInCurrentRow = 0;
                        currentTileX = originTileX;
                        currentTileY++;
                    }
                    else
                    {
                        currentTileX++;
                    }
                }

                int gridX = localCellIndex % cellsPerTileEdge;
                int gridY = localCellIndex / cellsPerTileEdge;

                float cellU = gridX * cellSize;
                float cellV = gridY * cellSize;

                float centerU = cellU + (cellSize * 0.5f);
                float centerV = cellV + (cellSize * 0.5f);

                // Compute world coordinates
                float tileWorldOriginX = MapOrigin - (currentTileX * TileSizeMeters);
                float tileWorldOriginY = MapOrigin - (currentTileY * TileSizeMeters);

                float worldX = tileWorldOriginX - centerU;
                float worldY = tileWorldOriginY - centerV;

                // Center object bounds offset
                Vector3 boundsCenter = (item.ScaledMin + item.ScaledMax) * 0.5f;
                float exhibitWorldX = worldX - boundsCenter.X;
                float exhibitWorldY = worldY - boundsCenter.Y;

                // Z elevation: place comfortably above pedestal
                float pedestalHeight = options.BasePedestalHeightMeters;
                float worldZ = pedestalHeight + MathF.Max(0f, -item.ScaledMin.Z) + 0.5f;

                // Pedestal footprint
                float halfCell = cellSize * 0.5f;
                Vector3 pedMin = new(worldX - halfCell + options.PedestalMarginMeters, worldY - halfCell + options.PedestalMarginMeters, 0f);
                Vector3 pedMax = new(worldX + halfCell - options.PedestalMarginMeters, worldY + halfCell - options.PedestalMarginMeters, pedestalHeight);

                placements.Add(new AdaptiveExhibitPlacement(
                    item.Asset,
                    item.Classification,
                    item.Scale,
                    item.ScaledMin,
                    item.ScaledMax,
                    cellSize,
                    cellU,
                    cellV,
                    currentTileX,
                    currentTileY,
                    new Vector3(exhibitWorldX, exhibitWorldY, worldZ),
                    pedMin,
                    pedMax,
                    pedestalHeight));

                cellIndex++;
            }

            // Move to next tile for the next cell size group so courtyards have uniform geometry
            tilesInCurrentRow++;
            if (tilesInCurrentRow >= maxTilesPerRow)
            {
                tilesInCurrentRow = 0;
                currentTileX = originTileX;
                currentTileY++;
            }
            else
            {
                currentTileX++;
            }
        }

        return placements;
    }
}
