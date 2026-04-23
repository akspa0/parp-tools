using System.Numerics;

namespace WowViewer.Core.IO.Maps;

public static class AdtTerrainMath
{
    public const int TileHeightmapSize = 257;
    private const float TileWorldSize = 533.33333f;
    private const float HalfStepWorldSize = TileWorldSize / (TileHeightmapSize - 1);

    public static Vector3 ComputeNormal(IReadOnlyList<float> tileHeightmap257, int sampleX, int sampleY)
    {
        ArgumentNullException.ThrowIfNull(tileHeightmap257);

        int leftX = Math.Max(sampleX - 1, 0);
        int rightX = Math.Min(sampleX + 1, TileHeightmapSize - 1);
        int upY = Math.Max(sampleY - 1, 0);
        int downY = Math.Min(sampleY + 1, TileHeightmapSize - 1);

        float leftHeight = tileHeightmap257[(sampleY * TileHeightmapSize) + leftX];
        float rightHeight = tileHeightmap257[(sampleY * TileHeightmapSize) + rightX];
        float upHeight = tileHeightmap257[(upY * TileHeightmapSize) + sampleX];
        float downHeight = tileHeightmap257[(downY * TileHeightmapSize) + sampleX];

        float deltaX = (rightX - leftX) * HalfStepWorldSize;
        float deltaY = (downY - upY) * HalfStepWorldSize;
        if (deltaX <= 0f || deltaY <= 0f)
            return Vector3.UnitZ;

        Vector3 tangentX = new(deltaX, 0f, rightHeight - leftHeight);
        Vector3 tangentY = new(0f, deltaY, downHeight - upHeight);
        Vector3 normal = Vector3.Cross(tangentX, tangentY);
        if (normal.LengthSquared() <= 1e-12f)
            return Vector3.UnitZ;

        normal = Vector3.Normalize(normal);
        return normal.Z < 0f ? -normal : normal;
    }
}