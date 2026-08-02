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

        // Liquid flags describe cells, while UnifiedLiquidMask retains vertex coverage. Render
        // only cells whose four vertices are known liquid; directly sampling one vertex paints
        // thin liquid strips over adjacent dry terrain along cell boundaries.
        cellX = Math.Clamp((int)MathF.Floor((outputX + 0.5f) * (maskWidth - 1f) / outputWidth), 0, maskWidth - 2);
        cellY = Math.Clamp((int)MathF.Floor((outputY + 0.5f) * (maskHeight - 1f) / outputHeight), 0, maskHeight - 2);
        float topLeft = mask[cellY, cellX];
        float topRight = mask[cellY, cellX + 1];
        float bottomLeft = mask[cellY + 1, cellX];
        float bottomRight = mask[cellY + 1, cellX + 1];
        if (!float.IsFinite(topLeft)
            || !float.IsFinite(topRight)
            || !float.IsFinite(bottomLeft)
            || !float.IsFinite(bottomRight))
        {
            return false;
        }

        coverage = MathF.Min(MathF.Min(topLeft, topRight), MathF.Min(bottomLeft, bottomRight));
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
