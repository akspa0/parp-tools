using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Rasterizes WL* 4x4 vertex blocks as their nine contiguous surface quads.
/// WL* blocks are geometry, not sparse point samples: marking a block origin or its vertices
/// creates the visible 16-by-16 checkerboard seen in liquid outputs.
/// </summary>
public static class WlLiquidRasterizer
{
    public const int DefaultResolution = 257;
    public const float TileWorldSize = 533.33333f;
    public const float MapOrigin = 17066.666f;
    /// <summary>Dataset-provenance marker for masks produced by this contiguous-surface rasterizer.</summary>
    public const string SurfaceRasterizationSignal = "wl_liquid_surface_quads_v1";
    /// <summary>Dataset-provenance marker for WL* surfaces clipped against terrain elevation.</summary>
    public const string AboveTerrainSignal = "wl_liquid_above_terrain_v1";
    /// <summary>Dataset-provenance marker for the per-pixel WL* basic-liquid-type resolver.</summary>
    public const string BasicTypeSignal = "wl_liquid_basic_type_header_v1";

    public static bool TryRasterize(
        IEnumerable<WlFile> files,
        int targetTileX,
        int targetTileY,
        out float[,]? mask,
        out float[,]? heights,
        out byte[,]? basicTypes,
        int resolution = DefaultResolution)
    {
        ArgumentNullException.ThrowIfNull(files);
        if (targetTileX is < 0 or > 63)
            throw new ArgumentOutOfRangeException(nameof(targetTileX));
        if (targetTileY is < 0 or > 63)
            throw new ArgumentOutOfRangeException(nameof(targetTileY));
        if (resolution < 2)
            throw new ArgumentOutOfRangeException(nameof(resolution));

        float[,] rasterMask = new float[resolution, resolution];
        float[,] rasterHeights = new float[resolution, resolution];
        byte[,] rasterTypes = new byte[resolution, resolution];
        for (int y = 0; y < resolution; y++)
        {
            for (int x = 0; x < resolution; x++)
                rasterTypes[y, x] = LiquidBasicTypeConstants.NoLiquid;
        }
        bool any = false;

        foreach (WlFile file in files)
        {
            byte basicType = ResolveBasicType(file.Header);
            foreach (WlBlock block in file.Blocks)
            {
                if (!BelongsToTile(block, targetTileX, targetTileY) || block.Vertices.Length < 16)
                    continue;

                for (int row = 0; row < 3; row++)
                {
                    for (int column = 0; column < 3; column++)
                    {
                        // WL vertices are serialized from lower-right to upper-left. Reversing the
                        // linear index restores the row-major 4x4 surface used by adjacent quads.
                        Vector3 topLeft = ToRaster(block.Vertices[15 - (row * 4 + column)], targetTileX, targetTileY, resolution);
                        Vector3 topRight = ToRaster(block.Vertices[15 - (row * 4 + column + 1)], targetTileX, targetTileY, resolution);
                        Vector3 bottomLeft = ToRaster(block.Vertices[15 - ((row + 1) * 4 + column)], targetTileX, targetTileY, resolution);
                        Vector3 bottomRight = ToRaster(block.Vertices[15 - ((row + 1) * 4 + column + 1)], targetTileX, targetTileY, resolution);

                        any |= RasterizeTriangle(topLeft, bottomLeft, bottomRight, rasterMask, rasterHeights, rasterTypes, basicType);
                        any |= RasterizeTriangle(topLeft, bottomRight, topRight, rasterMask, rasterHeights, rasterTypes, basicType);
                    }
                }
            }
        }

        if (!any)
        {
            mask = null;
            heights = null;
            basicTypes = null;
            return false;
        }

        mask = rasterMask;
        heights = rasterHeights;
        basicTypes = rasterTypes;
        return true;
    }

    /// <summary>
    /// Removes WL* candidates that are below the aligned terrain vertex at the same raster sample.
    /// WL* is auxiliary recovered geometry rather than authoritative visible-water coverage; without
    /// a finite matching terrain height, it cannot be allowed to paint water through terrain.
    /// </summary>
    public static int KeepOnlyAboveTerrain(
        float[,] mask,
        float[,] heights,
        float[,]? terrainHeights,
        byte[,]? basicTypes = null)
    {
        ArgumentNullException.ThrowIfNull(mask);
        ArgumentNullException.ThrowIfNull(heights);
        if (mask.GetLength(0) != heights.GetLength(0) || mask.GetLength(1) != heights.GetLength(1))
            throw new ArgumentException("WL mask and height grids must have identical dimensions.");
        if (basicTypes is not null
            && (basicTypes.GetLength(0) != mask.GetLength(0) || basicTypes.GetLength(1) != mask.GetLength(1)))
        {
            throw new ArgumentException("WL basic-type grid must match the mask dimensions.", nameof(basicTypes));
        }

        bool alignedTerrain = terrainHeights is not null
            && terrainHeights.GetLength(0) == mask.GetLength(0)
            && terrainHeights.GetLength(1) == mask.GetLength(1);
        int retained = 0;

        for (int y = 0; y < mask.GetLength(0); y++)
        {
            for (int x = 0; x < mask.GetLength(1); x++)
            {
                if (!(mask[y, x] > 0f))
                    continue;

                float surface = heights[y, x];
                float terrain = alignedTerrain ? terrainHeights![y, x] : float.NaN;
                if (!float.IsFinite(surface) || !float.IsFinite(terrain) || surface < terrain)
                {
                    mask[y, x] = 0f;
                    heights[y, x] = 0f;
                    if (basicTypes is not null)
                        basicTypes[y, x] = LiquidBasicTypeConstants.NoLiquid;
                    continue;
                }

                retained++;
            }
        }

        return retained;
    }

    private static bool BelongsToTile(WlBlock block, int tileX, int tileY)
    {
        Vector3 position = block.WorldPosition;
        if (!IsFinite(position))
            return false;

        int blockTileX = Math.Clamp((int)MathF.Floor((MapOrigin - position.Y) / TileWorldSize), 0, 63);
        int blockTileY = Math.Clamp((int)MathF.Floor((MapOrigin - position.X) / TileWorldSize), 0, 63);
        return blockTileX == tileX && blockTileY == tileY;
    }

    private static Vector3 ToRaster(Vector3 world, int tileX, int tileY, int resolution)
    {
        float localX = (MapOrigin - world.Y) - (tileX * TileWorldSize);
        float localY = (MapOrigin - world.X) - (tileY * TileWorldSize);
        float scale = (resolution - 1f) / TileWorldSize;
        return new Vector3(localX * scale, localY * scale, world.Z);
    }

    private static bool RasterizeTriangle(
        Vector3 a,
        Vector3 b,
        Vector3 c,
        float[,] mask,
        float[,] heights,
        byte[,] basicTypes,
        byte basicType)
    {
        if (!IsFinite(a) || !IsFinite(b) || !IsFinite(c))
            return false;

        float signedTwiceArea = Cross(b - a, c - a);
        if (MathF.Abs(signedTwiceArea) <= 1e-6f)
            return false;

        int width = mask.GetLength(1);
        int height = mask.GetLength(0);
        int minX = Math.Clamp((int)MathF.Floor(MathF.Min(a.X, MathF.Min(b.X, c.X))), 0, width - 1);
        int maxX = Math.Clamp((int)MathF.Ceiling(MathF.Max(a.X, MathF.Max(b.X, c.X))), 0, width - 1);
        int minY = Math.Clamp((int)MathF.Floor(MathF.Min(a.Y, MathF.Min(b.Y, c.Y))), 0, height - 1);
        int maxY = Math.Clamp((int)MathF.Ceiling(MathF.Max(a.Y, MathF.Max(b.Y, c.Y))), 0, height - 1);
        bool wrote = false;

        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                var point = new Vector2(x + 0.5f, y + 0.5f);
                float weightB = Cross(point - new Vector2(a.X, a.Y), new Vector2(c.X - a.X, c.Y - a.Y)) / signedTwiceArea;
                float weightC = Cross(new Vector2(b.X - a.X, b.Y - a.Y), point - new Vector2(a.X, a.Y)) / signedTwiceArea;
                float weightA = 1f - weightB - weightC;
                const float epsilon = -1e-5f;
                if (weightA < epsilon || weightB < epsilon || weightC < epsilon)
                    continue;

                // WL* families can overlap. Keep the canonical type with the
                // stronger liquid identity instead of letting filesystem order
                // arbitrarily recolor the same recovered surface.
                if (mask[y, x] > 0f && GetTypePrecedence(basicTypes[y, x]) <= GetTypePrecedence(basicType))
                {
                    wrote = true;
                    continue;
                }

                mask[y, x] = 1f;
                heights[y, x] = (weightA * a.Z) + (weightB * b.Z) + (weightC * c.Z);
                basicTypes[y, x] = basicType;
                wrote = true;
            }
        }

        return wrote;
    }

    private static float Cross(Vector3 left, Vector3 right) => (left.X * right.Y) - (left.Y * right.X);

    private static float Cross(Vector2 left, Vector2 right) => (left.X * right.Y) - (left.Y * right.X);

    private static byte ResolveBasicType(WlHeader header)
    {
        // WLM is magma and WLL is lava. The renderer's canonical four-type
        // palette deliberately represents lava as Magma; WLW/WLQ use their
        // parsed header liquid type for water/ocean/slime classification.
        if (header.FileType is WlFileType.WLM or WlFileType.WLL)
            return (byte)AdtLiquidBasicType.Magma;

        return header.LiquidType switch
        {
            WlLiquidType.Ocean => (byte)AdtLiquidBasicType.Ocean,
            WlLiquidType.Magma => (byte)AdtLiquidBasicType.Magma,
            WlLiquidType.Slime => (byte)AdtLiquidBasicType.Slime,
            _ => (byte)AdtLiquidBasicType.Water
        };
    }

    private static int GetTypePrecedence(byte basicType) => basicType switch
    {
        (byte)AdtLiquidBasicType.Magma => 0,
        (byte)AdtLiquidBasicType.Slime => 1,
        (byte)AdtLiquidBasicType.Ocean => 2,
        (byte)AdtLiquidBasicType.Water => 3,
        _ => int.MaxValue
    };

    private static bool IsFinite(Vector3 value) =>
        float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}
