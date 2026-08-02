using System.Numerics;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Converts between WoW world coordinates and a minimap tile's pixel raster.
///
/// The mapping is taken from <c>AdtTensorPackBuilder</c>'s vertex world-position derivation, which is
/// the authoritative source in this codebase:
/// <code>
///   worldX = MapOrigin - (tileY * TileWorldSize) - (rowWithinTile * step)
///   worldY = MapOrigin - (tileX * TileWorldSize) - (colWithinTile * step)
/// </code>
/// Note the crossover that makes this easy to get backwards: world <b>X</b> (North) varies with the
/// tile ROW, and world <b>Y</b> (West) varies with the tile COLUMN. Both decrease as the index grows.
/// </summary>
public static class MinimapTileProjection
{
    public const float TileWorldSize = 533.33333f;

    /// <summary>World coordinate of tile (0,0)'s origin corner. The 64x64 grid centres on the origin.</summary>
    public const float MapOrigin = 32f * TileWorldSize;

    /// <summary>
    /// Projects a world position onto a tile's normalized raster coordinates, where (0,0) is the
    /// tile's first pixel and (1,1) its last. Values outside 0..1 mean the position lies on another
    /// tile -- returned rather than rejected, because a light centred on a neighbouring tile can
    /// still cast influence across the seam into this one.
    /// </summary>
    public static void Project(Vector3 worldPosition, int tileX, int tileY, out float u, out float v)
    {
        v = ((MapOrigin - worldPosition.X) / TileWorldSize) - tileY;
        u = ((MapOrigin - worldPosition.Y) / TileWorldSize) - tileX;
    }

    /// <summary>True when the projected coordinates fall inside the tile.</summary>
    public static bool IsWithinTile(float u, float v) => u is >= 0f and <= 1f && v is >= 0f and <= 1f;

    /// <summary>
    /// World position at the centre of a tile pixel. The inverse of <see cref="Project"/>, used by
    /// per-pixel overlays that need each pixel's world location.
    /// </summary>
    public static Vector3 Unproject(int pixelX, int pixelY, int resolution, int tileX, int tileY, float worldZ = 0f)
    {
        float u = resolution <= 0 ? 0f : (pixelX + 0.5f) / resolution;
        float v = resolution <= 0 ? 0f : (pixelY + 0.5f) / resolution;
        return new Vector3(
            MapOrigin - ((tileY + v) * TileWorldSize),
            MapOrigin - ((tileX + u) * TileWorldSize),
            worldZ);
    }
}
