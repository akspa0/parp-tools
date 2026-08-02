using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Produces the paired liquid-bearing synthesized-minimap target from the terrain-only baseline.
/// The overlay reuses the viewer's flat liquid type palette and opacity; it deliberately does not
/// claim client-exact water textures, waves, specular effects, or animation.
/// </summary>
public static class TerrainMinimapLiquidCompositor
{
    /// <summary>Profile identifier for the default palette, recorded in the synthesis manifest.</summary>
    public static string RenderProfile => MinimapLiquidPalette.Default.RenderProfile;

    public static Image<Rgba32> Compose(
        Image<Rgba32> terrainBaseline,
        TerrainTileTensorPack pack,
        out int liquidPixelCount,
        MinimapLiquidPalette? palette = null)
    {
        ArgumentNullException.ThrowIfNull(terrainBaseline);
        ArgumentNullException.ThrowIfNull(pack);
        palette ??= MinimapLiquidPalette.Default;

        Image<Rgba32> result = terrainBaseline.Clone();
        liquidPixelCount = 0;

        float[,]? liquidMask = pack.UnifiedLiquidMask;
        if (liquidMask is null || liquidMask.GetLength(0) == 0 || liquidMask.GetLength(1) == 0)
            return result;

        float[,]? liquidHeight = pack.UnifiedLiquidHeight;
        float[,]? terrainHeight = pack.Height257;
        byte[,]? liquidTypes = pack.LiquidBasicType257;
        int maskHeight = liquidMask.GetLength(0);
        int maskWidth = liquidMask.GetLength(1);

        for (int y = 0; y < result.Height; y++)
        {
            for (int x = 0; x < result.Width; x++)
            {
                if (!TryResolveCellCoverage(
                        liquidMask,
                        x,
                        y,
                        result.Width,
                        result.Height,
                        out int cellX,
                        out int cellY,
                        out float coverage))
                continue;

                // Gate by terrain height: skip liquid pixels where the liquid surface
                // is below the terrain surface. Liquid data is stored as planes that
                // extend across the full cell, but terrain may rise above the water
                // level — painting liquid over those pixels produces floating water.
                if (liquidHeight is not null && terrainHeight is not null)
                {
                    int terrainY = Math.Clamp(cellY, 0, terrainHeight.GetLength(0) - 1);
                    int terrainX = Math.Clamp(cellX, 0, terrainHeight.GetLength(1) - 1);
                    float terrainZ = terrainHeight[terrainY, terrainX];
                    float liquidZ = liquidHeight[cellY, cellX];
                    if (float.IsFinite(terrainZ) && float.IsFinite(liquidZ) && liquidZ < terrainZ)
                        continue;
                }

                liquidPixelCount++;
                MinimapLiquidStyle style = palette.Resolve(ResolveType(liquidTypes, cellX, cellY, maskWidth, maskHeight));
                float alpha = Math.Clamp(coverage, 0f, 1f) * style.Opacity;
                result[x, y] = Blend(result[x, y], style, alpha);
            }
        }

        return result;
    }

    private static bool TryResolveCellCoverage(
        float[,] mask,
        int outputX,
        int outputY,
        int outputWidth,
        int outputHeight,
        out int cellX,
        out int cellY,
        out float coverage)
    {
        int maskHeight = mask.GetLength(0);
        int maskWidth = mask.GetLength(1);
        cellX = 0;
        cellY = 0;
        coverage = 0f;

        if (maskWidth < 2 || maskHeight < 2)
            return false;

        // UnifiedLiquidMask is PER-VERTEX 257x257 decoded from MH2O / MCLQ / WL* -- real
        // sub-chunk-precision coverage, not MCNK chunk flags (which are metadata only and never
        // invent coverage).
        //
        // The previous sampling threw that precision away: it took the MINIMUM of a cell's four
        // corner vertices, so any cell touching dry land was dropped entirely. That eroded every
        // water body by a full cell and quantised the shoreline to the cell grid, which is what
        // makes synthesized coastlines read blocky and inset against authored minimaps.
        //
        // Bilinear sampling of the same vertex data instead puts the waterline at the half-way
        // point between a wet and a dry vertex, which is the correct reading of vertex-sampled
        // coverage: neither eroded by a cell nor smeared a cell out over dry ground (the failure
        // the min-of-four was guarding against).
        float sampleX = (outputX + 0.5f) * (maskWidth - 1f) / outputWidth;
        float sampleY = (outputY + 0.5f) * (maskHeight - 1f) / outputHeight;
        int x0 = Math.Clamp((int)MathF.Floor(sampleX), 0, maskWidth - 2);
        int y0 = Math.Clamp((int)MathF.Floor(sampleY), 0, maskHeight - 2);
        float fx = Math.Clamp(sampleX - x0, 0f, 1f);
        float fy = Math.Clamp(sampleY - y0, 0f, 1f);

        // Type lookup uses the nearest vertex, so a shoreline pixel takes the class of the water it
        // actually belongs to rather than a neighbouring body's.
        cellX = fx < 0.5f ? x0 : x0 + 1;
        cellY = fy < 0.5f ? y0 : y0 + 1;

        float topLeft = mask[y0, x0];
        float topRight = mask[y0, x0 + 1];
        float bottomLeft = mask[y0 + 1, x0];
        float bottomRight = mask[y0 + 1, x0 + 1];
        if (!float.IsFinite(topLeft)
            || !float.IsFinite(topRight)
            || !float.IsFinite(bottomLeft)
            || !float.IsFinite(bottomRight))
        {
            return false;
        }

        float top = topLeft + ((topRight - topLeft) * fx);
        float bottom = bottomLeft + ((bottomRight - bottomLeft) * fx);
        coverage = top + ((bottom - top) * fy);
        return coverage > 0f;
    }

    private static AdtLiquidBasicType ResolveType(
        byte[,]? liquidTypes,
        int maskX,
        int maskY,
        int maskWidth,
        int maskHeight)
    {
        if (liquidTypes is null || liquidTypes.GetLength(0) == 0 || liquidTypes.GetLength(1) == 0)
            return AdtLiquidBasicType.Water;

        int typeY = ScaleCoordinate(maskY, maskHeight, liquidTypes.GetLength(0));
        int typeX = ScaleCoordinate(maskX, maskWidth, liquidTypes.GetLength(1));
        byte value = liquidTypes[typeY, typeX];
        return value <= LiquidBasicTypeConstants.MaxBasicType
            ? (AdtLiquidBasicType)value
            : AdtLiquidBasicType.Water;
    }

    private static Rgba32 Blend(Rgba32 terrain, MinimapLiquidStyle liquid, float alpha)
    {
        float terrainWeight = 1f - alpha;
        return new Rgba32(
            ToByte((terrain.R / 255f * terrainWeight) + (liquid.Red * alpha)),
            ToByte((terrain.G / 255f * terrainWeight) + (liquid.Green * alpha)),
            ToByte((terrain.B / 255f * terrainWeight) + (liquid.Blue * alpha)),
            terrain.A);
    }

    private static int ScaleCoordinate(int coordinate, int sourceSize, int targetSize)
    {
        if (targetSize <= 1 || sourceSize <= 1)
            return 0;

        return Math.Clamp((int)MathF.Round(coordinate * (targetSize - 1f) / (sourceSize - 1f)), 0, targetSize - 1);
    }

    private static byte ToByte(float value) => (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);
}
