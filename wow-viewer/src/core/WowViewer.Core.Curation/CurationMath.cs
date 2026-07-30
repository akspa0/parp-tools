namespace WowViewer.Core.Curation;

/// <summary>
/// Shared per-tile array math ported from <c>data-harvester/src/harvester/v16_curation.py</c> and
/// <c>mismatch_detector.py</c> (numpy) to C# (2D arrays over <c>TerrainTileTensorPack</c> fields).
/// Every classifier/detector in this library builds on these same primitives rather than each
/// reimplementing edge/relief/coverage math, mirroring why this whole library exists: one
/// definition per metric, not one per consumer.
/// </summary>
internal static class CurationMath
{
    public static float Clamp01(float value) => Math.Clamp(value, 0f, 1f);

    /// <summary>Ports <c>v16_curation.edge_strength</c>: max of horizontal/vertical forward
    /// differences.</summary>
    public static float[,] EdgeStrength(float[,] x)
    {
        int h = x.GetLength(0), w = x.GetLength(1);
        var result = new float[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int col = 0; col < w; col++)
            {
                float gx = col + 1 < w ? MathF.Abs(x[y, col + 1] - x[y, col]) : 0f;
                float gy = y + 1 < h ? MathF.Abs(x[y + 1, col] - x[y, col]) : 0f;
                result[y, col] = MathF.Max(gx, gy);
            }
        }
        return result;
    }

    /// <summary>Ports <c>v16_curation.normal_relief</c>: horizontal-normal magnitude, masked.</summary>
    public static float[,] NormalRelief(float[,,] normals, float[,] mask)
    {
        int h = normals.GetLength(0), w = normals.GetLength(1);
        var result = new float[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int col = 0; col < w; col++)
            {
                float nx = normals[y, col, 0];
                float ny = normals[y, col, 1];
                float relief = MathF.Sqrt(MathF.Max(0f, (nx * nx) + (ny * ny)));
                result[y, col] = relief * mask[y, col];
            }
        }
        return result;
    }

    /// <summary>Ports <c>mismatch_detector.normal_edge_strength</c>: max edge strength across the
    /// three masked normal channels.</summary>
    public static float[,] NormalEdgeStrength(float[,,] normals, float[,] mask)
    {
        int h = normals.GetLength(0), w = normals.GetLength(1);
        var nx = new float[h, w];
        var ny = new float[h, w];
        var nz = new float[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int col = 0; col < w; col++)
            {
                float m = mask[y, col];
                nx[y, col] = normals[y, col, 0] * m;
                ny[y, col] = normals[y, col, 1] * m;
                nz[y, col] = normals[y, col, 2] * m;
            }
        }
        float[,] ex = EdgeStrength(nx), ey = EdgeStrength(ny), ez = EdgeStrength(nz);
        var result = new float[h, w];
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
                result[y, col] = MathF.Max(ex[y, col], MathF.Max(ey[y, col], ez[y, col]));
        return result;
    }

    /// <summary>Ports <c>mismatch_detector.minimap_grayscale</c> (Rec.601 luma).</summary>
    public static float[,] MinimapGrayscale(byte[,,] minimapRgb)
    {
        int h = minimapRgb.GetLength(0), w = minimapRgb.GetLength(1);
        var result = new float[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int col = 0; col < w; col++)
            {
                float r = minimapRgb[y, col, 0] / 255f;
                float g = minimapRgb[y, col, 1] / 255f;
                float b = minimapRgb[y, col, 2] / 255f;
                result[y, col] = (0.299f * r) + (0.587f * g) + (0.114f * b);
            }
        }
        return result;
    }

    public static float Mean(float[,] x)
    {
        int h = x.GetLength(0), w = x.GetLength(1);
        if (h == 0 || w == 0) return 0f;
        double sum = 0;
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
                sum += x[y, col];
        return (float)(sum / (h * (double)w));
    }

    public static float FractionAtLeast(float[,] x, float threshold)
    {
        int h = x.GetLength(0), w = x.GetLength(1);
        if (h == 0 || w == 0) return 0f;
        int count = 0;
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
                if (x[y, col] >= threshold) count++;
        return count / (float)(h * w);
    }

    public static float StdDev(float[,] x)
    {
        int h = x.GetLength(0), w = x.GetLength(1);
        if (h == 0 || w == 0) return 0f;
        float mean = Mean(x);
        double sumSq = 0;
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
            {
                double d = x[y, col] - mean;
                sumSq += d * d;
            }
        return (float)Math.Sqrt(sumSq / (h * (double)w));
    }

    public static (float Min, float Max) MinMax(float[,] x)
    {
        int h = x.GetLength(0), w = x.GetLength(1);
        float min = float.MaxValue, max = float.MinValue;
        for (int y = 0; y < h; y++)
        {
            for (int col = 0; col < w; col++)
            {
                float v = x[y, col];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        if (min > max) return (0f, 0f);
        return (min, max);
    }

    /// <summary>Crops a 257x257 array to its top-left 256x256 (ports <c>crop_257_to_256</c>).</summary>
    public static float[,] Crop257To256(float[,] x)
    {
        var result = new float[256, 256];
        for (int y = 0; y < 256; y++)
            for (int col = 0; col < 256; col++)
                result[y, col] = x[y, col];
        return result;
    }

    /// <summary>Coverage fraction of a 4-layer alpha pack painted above a small threshold, ports
    /// <c>alpha_painted</c> + a &gt;=0.05 threshold used inline by <c>v16_curation</c> callers
    /// (channels 1..3 = additional-layer blend weight; falls back to channel 0 if no additional
    /// layer ever paints, matching the Python function's exact fallback).</summary>
    public static float AlphaPaintedCoverage(float[,,] alpha)
    {
        int h = alpha.GetLength(0), w = alpha.GetLength(1), channels = alpha.GetLength(2);
        if (channels <= 0) return 0f;

        var painted = new float[h, w];
        bool anyAdditionalLayerPainted = false;
        if (channels > 1)
        {
            for (int y = 0; y < h && !anyAdditionalLayerPainted; y++)
                for (int col = 0; col < w && !anyAdditionalLayerPainted; col++)
                    for (int c = 1; c < channels; c++)
                        if (alpha[y, col, c] > 0f) { anyAdditionalLayerPainted = true; break; }
        }

        for (int y = 0; y < h; y++)
        {
            for (int col = 0; col < w; col++)
            {
                float max = anyAdditionalLayerPainted ? 0f : alpha[y, col, 0];
                if (anyAdditionalLayerPainted)
                {
                    for (int c = 1; c < channels; c++)
                        max = MathF.Max(max, alpha[y, col, c]);
                }
                painted[y, col] = max;
            }
        }
        return FractionAtLeast(painted, 0.05f);
    }

    /// <summary>Ports <c>mcly_painted_coverage</c>: fraction of chunks with any active layer above
    /// a small threshold, from the boolean layer-presence mask (not the alpha pack).</summary>
    public static float MclyPaintedCoverage(bool[,,] layerMask)
    {
        int h = layerMask.GetLength(0), w = layerMask.GetLength(1), layers = layerMask.GetLength(2);
        if (h == 0 || w == 0 || layers <= 0) return 0f;
        int count = 0;
        for (int y = 0; y < h; y++)
        {
            for (int col = 0; col < w; col++)
            {
                bool any = false;
                for (int l = 0; l < layers; l++)
                    if (layerMask[y, col, l]) { any = true; break; }
                if (any) count++;
            }
        }
        return count / (float)(h * w);
    }

    /// <summary>Ports <c>v16_curation.is_blank_what_plate</c>: a tile is blank when height has
    /// essentially no variance AND every painted-signal coverage is essentially zero.</summary>
    public static bool IsBlankWhatPlate(
        float[,] height,
        float alphaCov,
        float mclyCov,
        float liquidCov,
        float objectCov,
        float heightAbsMaxEps = 1e-6f,
        float heightStdEps = 1e-6f)
    {
        (float min, float max) = MinMax(height);
        float heightAbsMax = MathF.Max(MathF.Abs(min), MathF.Abs(max));
        float heightStd = StdDev(height);
        return heightAbsMax <= heightAbsMaxEps
            && heightStd <= heightStdEps
            && alphaCov <= 1e-4f
            && mclyCov <= 1e-4f
            && liquidCov <= 1e-4f
            && objectCov <= 1e-4f;
    }
}
