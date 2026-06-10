using WowViewer.Core.PM4;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.PM4.Models;
using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.PM4;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.PM4.Models;

namespace WoWViewer.Terrain.Vlm;

/// <summary>
/// Builds a precise PM4/MPRL reference mask that can be consumed as model context.
/// </summary>
public static class VlmPm4MaskService
{
    private const float TileMarginTiles = 0.05f;
    private const float TileBoundsToleranceWorld = 2f;
    private const float MapOrigin = 32f * Pm4CoordinateService.TileSize;

    public static byte[] BuildPm4Mask(string adtTile, IReadOnlyList<Pm4MprlEntry> positionRefs, int width, int height)
    {
        if (width <= 0 || height <= 0 || positionRefs.Count == 0)
            return Array.Empty<byte>();

        if (!TryParseTileCoordinates(adtTile, out int tileX, out int tileY))
            return Array.Empty<byte>();

        using Image<L8> mask = new(width, height);
        bool hasCoverage = false;
        int pointRadius = EstimateReferenceRadiusPixels(width, height);

        foreach (Pm4MprlEntry positionRef in positionRefs)
        {
            Vector3 placement = Pm4CoordinateService.MprlToAdtPlacement(positionRef.Position);
            if (!Pm4CoordinateService.IsWithinPlacementTileBounds(placement, tileX, tileY, TileBoundsToleranceWorld))
                continue;

            if (!TryProjectPlacementToTilePixel(placement, tileX, tileY, width, height, out int centerX, out int centerY))
                continue;

            DrawFilledCircle(mask, centerX, centerY, pointRadius);
            hasCoverage = true;
        }

        if (!hasCoverage)
            return Array.Empty<byte>();

        using MemoryStream ms = new();
        mask.SaveAsPng(ms);
        return ms.ToArray();
    }

    private static int EstimateReferenceRadiusPixels(int width, int height)
    {
        int minDimension = Math.Min(width, height);
        return Math.Clamp((int)MathF.Round(minDimension / 128f), 2, 8);
    }

    private static bool TryProjectPlacementToTilePixel(
        Vector3 placement,
        int tileX,
        int tileY,
        int width,
        int height,
        out int centerX,
        out int centerY)
    {
        centerX = 0;
        centerY = 0;

        List<(float U, float V)> candidates =
        [
            .. BuildTileUvCandidates(placement.X, placement.Z, tileX, tileY)
        ];

        if (candidates.Count == 0)
            return false;

        (float U, float V) best = candidates[0];
        float bestOverflow = float.PositiveInfinity;

        foreach ((float u, float v) in candidates)
        {
            float overflow =
                MathF.Max(0f, -u) + MathF.Max(0f, u - 1f) +
                MathF.Max(0f, -v) + MathF.Max(0f, v - 1f);
            if (overflow < bestOverflow)
            {
                best = (u, v);
                bestOverflow = overflow;
                if (overflow <= 0.000001f)
                    break;
            }
        }

        if (best.U < -TileMarginTiles || best.U > 1f + TileMarginTiles ||
            best.V < -TileMarginTiles || best.V > 1f + TileMarginTiles)
        {
            return false;
        }

        centerX = Math.Clamp((int)MathF.Round(Math.Clamp(best.U, 0f, 1f) * (width - 1)), 0, width - 1);
        centerY = Math.Clamp((int)MathF.Round(Math.Clamp(best.V, 0f, 1f) * (height - 1)), 0, height - 1);
        return true;
    }

    private static IEnumerable<(float U, float V)> BuildTileUvCandidates(float worldA, float worldB, int tileX, int tileY)
    {
        float tileSize = Pm4CoordinateService.TileSize;
        yield return ((worldA / tileSize) - tileX, (worldB / tileSize) - tileY);
        yield return (((MapOrigin - worldB) / tileSize) - tileX, ((MapOrigin - worldA) / tileSize) - tileY);
    }

    private static bool TryParseTileCoordinates(string adtTile, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;

        string[] parts = adtTile.Split('_');
        if (parts.Length < 3)
            return false;

        return int.TryParse(parts[^2], out tileX) && int.TryParse(parts[^1], out tileY);
    }

    private static void DrawFilledCircle(Image<L8> image, int centerX, int centerY, int radius)
    {
        int radiusSquared = radius * radius;
        int minX = Math.Max(0, centerX - radius);
        int maxX = Math.Min(image.Width - 1, centerX + radius);
        int minY = Math.Max(0, centerY - radius);
        int maxY = Math.Min(image.Height - 1, centerY + radius);

        for (int y = minY; y <= maxY; y++)
        {
            int deltaY = y - centerY;
            for (int x = minX; x <= maxX; x++)
            {
                int deltaX = x - centerX;
                if ((deltaX * deltaX) + (deltaY * deltaY) <= radiusSquared)
                    image[x, y] = new L8(255);
            }
        }
    }
}
