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

        // MCNR hillshade proof establishes that a positive terrain-X source reads as south in the
        // synthesized minimap, which inverts basin/hill perception. Keep the source on negative
        // terrain X (raster north) throughout the day: it traverses north-west before noon and
        // north-east after. Letting the source enter positive X makes shading project upward.
        const float diagonalHorizontalScale = 0.3535534f; // 0.5 / sqrt(2)
        float horizontalX = -MathF.Abs(sunHorizontal) * diagonalHorizontalScale;
        float horizontalY = sunHorizontal * diagonalHorizontalScale;
        return Vector3.Normalize(new Vector3(
            horizontalX,
            horizontalY,
            MathF.Max(sunHeight, 0.05f)));
    }
}
