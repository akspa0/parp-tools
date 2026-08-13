using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Distance helpers shared by streaming and render admission. A tile or chunk
/// is spatially present when any part of its bounds is within the configured
/// range; testing its center can reject nearby geometry at a tile edge.
/// </summary>
public static class WorldDistanceMath
{
    public static float DistanceSquaredPointToAabb(Vector3 point, Vector3 min, Vector3 max)
    {
        float minX = MathF.Min(min.X, max.X);
        float maxX = MathF.Max(min.X, max.X);
        float minY = MathF.Min(min.Y, max.Y);
        float maxY = MathF.Max(min.Y, max.Y);
        float minZ = MathF.Min(min.Z, max.Z);
        float maxZ = MathF.Max(min.Z, max.Z);

        float dx = point.X < minX ? minX - point.X : point.X > maxX ? point.X - maxX : 0f;
        float dy = point.Y < minY ? minY - point.Y : point.Y > maxY ? point.Y - maxY : 0f;
        float dz = point.Z < minZ ? minZ - point.Z : point.Z > maxZ ? point.Z - maxZ : 0f;
        return dx * dx + dy * dy + dz * dz;
    }
}
