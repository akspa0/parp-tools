using System.Numerics;

namespace WowViewer.Core.Terrain;

/// <summary>
/// Shared authored solar direction in terrain world space.
/// </summary>
public static class TerrainSolarDirection
{
    public static Vector3 Evaluate(float gameTime)
    {
        float wrappedTime = gameTime - MathF.Floor(gameTime);
        float sunAngle = wrappedTime * MathF.Tau;
        float sunHeight = MathF.Sin(sunAngle - (MathF.PI * 0.5f));
        float sunHorizontal = MathF.Cos(sunAngle - (MathF.PI * 0.5f));

        // World +X/+Y project toward the top-left of the synthesized terrain raster.
        // At 12:00 horizontal components are zero; after noon the source projects top-left.
        const float diagonalHorizontalScale = 0.3535534f; // 0.5 / sqrt(2)
        float horizontal = -sunHorizontal * diagonalHorizontalScale;
        return Vector3.Normalize(new Vector3(
            horizontal,
            horizontal,
            MathF.Max(sunHeight, 0.05f)));
    }
}
