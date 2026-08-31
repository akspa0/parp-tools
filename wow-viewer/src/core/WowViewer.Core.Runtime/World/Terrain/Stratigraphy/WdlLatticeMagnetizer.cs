using System.Numerics;

namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// Service that magnetizes high-frequency ADT micro-relief terrain onto the low-frequency WDL elevation lattice.
/// </summary>
public static class WdlLatticeMagnetizer
{
    /// <summary>
    /// Evaluates the interpolated WDL macro height at a given normalized position (0.0 to 1.0) within a 64x64 world map tile.
    /// </summary>
    public static float SampleWdlHeight(short[,] height17, float normalizedX, float normalizedY)
    {
        ArgumentNullException.ThrowIfNull(height17);

        float gx = Math.Clamp(normalizedX * 16f, 0f, 16f);
        float gy = Math.Clamp(normalizedY * 16f, 0f, 16f);

        int x0 = (int)MathF.Floor(gx);
        int y0 = (int)MathF.Floor(gy);
        int x1 = Math.Min(x0 + 1, 16);
        int y1 = Math.Min(y0 + 1, 16);

        float fx = gx - x0;
        float fy = gy - y0;

        float h00 = height17[y0, x0];
        float h10 = height17[y0, x1];
        float h01 = height17[y1, x0];
        float h11 = height17[y1, x1];

        float top = (1f - fx) * h00 + fx * h10;
        float bottom = (1f - fx) * h01 + fx * h11;

        return (1f - fy) * top + fy * bottom;
    }

    /// <summary>
    /// Magnetizes a 145-vertex MCNK chunk onto the WDL macro elevation surface for the chunk at (chunkX, chunkY) in [0..15].
    /// </summary>
    public static float[] MagnetizeChunkHeights(
        ReadOnlySpan<float> sourceHeights145,
        short[,] height17,
        int chunkX,
        int chunkY,
        float reliefFactor = 1.0f,
        bool polarityInverted = false,
        float magnetizationStrength = 1.0f,
        float anchorHeight = 0f)
    {
        if (sourceHeights145.Length < 145)
            throw new ArgumentException("Source heights must have at least 145 elements.", nameof(sourceHeights145));
        ArgumentNullException.ThrowIfNull(height17);

        var result = new float[145];
        float sign = polarityInverted ? -1f : 1f;
        float strength = Math.Clamp(magnetizationStrength, 0f, 1f);

        // Outer 9x9 lattice
        for (int row = 0; row < 9; row++)
        {
            float normY = (chunkY + (row / 8f)) / 16f;
            for (int col = 0; col < 9; col++)
            {
                float normX = (chunkX + (col / 8f)) / 16f;
                int idx = row * 17 + col;

                float src = sourceHeights145[idx];
                float wdlMacro = SampleWdlHeight(height17, normX, normY);
                float microRelief = sign * reliefFactor * (src - anchorHeight);

                float magnetized = wdlMacro + microRelief;
                float directRestored = anchorHeight + microRelief;

                result[idx] = (1f - strength) * directRestored + (strength * magnetized);
            }
        }

        // Inner 8x8 lattice
        for (int row = 0; row < 8; row++)
        {
            float normY = (chunkY + ((row + 0.5f) / 8f)) / 16f;
            for (int col = 0; col < 8; col++)
            {
                float normX = (chunkX + ((col + 0.5f) / 8f)) / 16f;
                int idx = 9 + row * 17 + col;

                float src = sourceHeights145[idx];
                float wdlMacro = SampleWdlHeight(height17, normX, normY);
                float microRelief = sign * reliefFactor * (src - anchorHeight);

                float magnetized = wdlMacro + microRelief;
                float directRestored = anchorHeight + microRelief;

                result[idx] = (1f - strength) * directRestored + (strength * magnetized);
            }
        }

        return result;
    }
}
