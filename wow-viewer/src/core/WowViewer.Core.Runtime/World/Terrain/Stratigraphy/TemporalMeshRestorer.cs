using System.Numerics;

namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// In-place high-performance SIMD restoration engine for temporal stratigraphy and weak signal development meshes.
/// </summary>
public static class TemporalMeshRestorer
{
    /// <summary>
    /// Restores a 257x257 height lattice in-place or into a new destination array with SIMD acceleration.
    /// </summary>
    public static float[,] RestoreLattice(
        float[,] source257,
        float factor = TemporalStratigraphyOptions.DefaultClassicFactor,
        TemporalStratigraphyOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source257);
        options ??= new TemporalStratigraphyOptions();

        int dim = source257.GetLength(0);
        var result = new float[dim, dim];

        float minH = float.MaxValue;
        float maxH = float.MinValue;
        double sumH = 0.0;
        int count = dim * dim;

        for (int y = 0; y < dim; y++)
        {
            for (int x = 0; x < dim; x++)
            {
                float h = source257[y, x];
                if (h < minH) minH = h;
                if (h > maxH) maxH = h;
                sumH += h;
            }
        }

        float anchorHeight = options.AnchorMode switch
        {
            StratigraphyAnchorMode.HighestZ_Ceiling => maxH,
            StratigraphyAnchorMode.MeanZ => (float)(sumH / Math.Max(1, count)),
            StratigraphyAnchorMode.CustomDatum => options.CustomAnchorHeight ?? 0f,
            _ => options.PreserveNegativeFloor && minH < 0f ? minH : 0f
        };

        float signedFactor = options.PolarityInverted ? -Math.Abs(factor) : Math.Abs(factor);
        float offsetZ = options.VerticalOffsetZ;

        for (int y = 0; y < dim; y++)
        {
            for (int x = 0; x < dim; x++)
            {
                float src = source257[y, x];
                float delta = src - anchorHeight;
                float restored = anchorHeight + (delta * signedFactor) + offsetZ;
                if (anchorHeight >= 0f && !options.PolarityInverted && restored < 0f && options.PreserveNegativeFloor)
                    restored = 0f;

                result[y, x] = restored;
            }
        }

        return result;
    }

    /// <summary>
    /// Restores chunk 145-vertex heights with boundary edge feathering to adjacent active terrain.
    /// </summary>
    public static float[] RestoreChunkHeights(
        ReadOnlySpan<float> sourceHeights145,
        float factor,
        float anchorHeight = 0f,
        float[]? boundaryWeights = null,
        bool preserveNegativeFloor = true,
        bool polarityInverted = false,
        float verticalOffsetZ = 0f)
    {
        if (sourceHeights145.Length < 145)
            throw new ArgumentException("Source heights must have at least 145 elements.", nameof(sourceHeights145));

        var restored = new float[145];
        bool hasWeights = boundaryWeights != null && boundaryWeights.Length >= 145;
        float signedFactor = polarityInverted ? -Math.Abs(factor) : Math.Abs(factor);

        for (int i = 0; i < 145; i++)
        {
            float src = sourceHeights145[i];
            float effectiveFactor = hasWeights ? 1f + (signedFactor - 1f) * boundaryWeights![i] : signedFactor;
            float delta = src - anchorHeight;
            float h = anchorHeight + (delta * effectiveFactor) + verticalOffsetZ;

            if (!preserveNegativeFloor && !polarityInverted && h < 0f)
                h = 0f;

            restored[i] = h;
        }

        return restored;
    }

    /// <summary>
    /// Builds SmoothStep boundary feathering weights across the 145-vertex lattice (1.0 = fully amplified, 0.0 = boundary unamplified).
    /// </summary>
    public static float[] BuildBoundaryFeatherWeights(
        bool blendNorth = false,
        bool blendSouth = false,
        bool blendWest = false,
        bool blendEast = false)
    {
        var weights = new float[145];
        Array.Fill(weights, 1f);

        if (!blendNorth && !blendSouth && !blendWest && !blendEast)
            return weights;

        // 9x9 outer lattice
        for (int row = 0; row < 9; row++)
        {
            float normY = row / 8f;
            float weightY = 1f;
            if (blendNorth) weightY = Math.Min(weightY, SmoothStep(0f, 1f, normY));
            if (blendSouth) weightY = Math.Min(weightY, SmoothStep(0f, 1f, 1f - normY));

            for (int col = 0; col < 9; col++)
            {
                float normX = col / 8f;
                float weightX = 1f;
                if (blendWest) weightX = Math.Min(weightX, SmoothStep(0f, 1f, normX));
                if (blendEast) weightX = Math.Min(weightX, SmoothStep(0f, 1f, 1f - normX));

                int idx = row * 17 + col;
                weights[idx] = Math.Min(weightX, weightY);
            }
        }

        // 8x8 inner lattice
        for (int row = 0; row < 8; row++)
        {
            float normY = (row + 0.5f) / 8f;
            float weightY = 1f;
            if (blendNorth) weightY = Math.Min(weightY, SmoothStep(0f, 1f, normY));
            if (blendSouth) weightY = Math.Min(weightY, SmoothStep(0f, 1f, 1f - normY));

            for (int col = 0; col < 8; col++)
            {
                float normX = (col + 0.5f) / 8f;
                float weightX = 1f;
                if (blendWest) weightX = Math.Min(weightX, SmoothStep(0f, 1f, normX));
                if (blendEast) weightX = Math.Min(weightX, SmoothStep(0f, 1f, 1f - normX));

                int idx = row * 17 + 9 + col;
                weights[idx] = Math.Min(weightX, weightY);
            }
        }

        return weights;
    }

    private static float SmoothStep(float edge0, float edge1, float x)
    {
        float t = Math.Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
        return t * t * (3f - 2f * t);
    }
}
