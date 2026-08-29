using System.Numerics;

namespace WowViewer.Core.Editor.Procedural;

/// <summary>
/// Configuration parameters for the procedural garden terrain sculptor.
/// </summary>
public sealed record TerrainSculptorOptions(
    float Roughness = 0.35f,
    float MaxSlopeDegrees = 25.0f,
    float BaseElevation = 0.0f,
    float HillAmplitudeMeters = 8.0f,
    float PodiumHeightMeters = 2.0f,
    int Seed = 1337);

/// <summary>
/// Sculpts continuous, playable, navmesh-grade garden terrain heightmaps
/// with harmonic noise, gradient slope constraints, and smooth pedestal podiums.
/// </summary>
public static class ProceduralTerrainSculptor
{
    public const float ChunkSizeMeters = 33.333333f;
    public const float OuterVertexSpacing = ChunkSizeMeters / 8.0f; // ~4.166666m
    public const float InnerVertexSpacing = OuterVertexSpacing;
    public const float InnerOffset = OuterVertexSpacing * 0.5f;     // ~2.083333m

    /// <summary>
    /// Evaluates smooth harmonic Simplex/Perlin-like 2D noise with octave persistence.
    /// </summary>
    public static float SampleHarmonicNoise(float x, float y, float frequency, int octaves, float persistence, int seed)
    {
        float total = 0f;
        float amplitude = 1f;
        float maxVal = 0f;
        float currentFreq = frequency;

        for (int i = 0; i < octaves; i++)
        {
            total += SmoothNoise(x * currentFreq, y * currentFreq, seed + (i * 313)) * amplitude;
            maxVal += amplitude;
            amplitude *= persistence;
            currentFreq *= 2.0f;
        }

        return total / MathF.Max(0.001f, maxVal);
    }

    /// <summary>
    /// Generates the standard WoW 145-vertex MCVT height array for a specific chunk.
    /// Outer 9x9 (indices 0..80) followed by Inner 8x8 (indices 81..144).
    /// </summary>
    public static float[] GenerateChunkHeights(
        int chunkX,
        int chunkY,
        IReadOnlyList<AdaptiveExhibitPlacement> tilePlacements,
        TerrainSculptorOptions options)
    {
        float[] heights = new float[145];
        float chunkOriginU = chunkX * ChunkSizeMeters;
        float chunkOriginV = chunkY * ChunkSizeMeters;

        // 1. Evaluate Base Landscape Heights for Outer 9x9
        for (int row = 0; row < 9; row++)
        {
            for (int col = 0; col < 9; col++)
            {
                int index = (row * 9) + col;
                float localU = chunkOriginU + (col * OuterVertexSpacing);
                float localV = chunkOriginV + (row * OuterVertexSpacing);

                heights[index] = SampleBlendedHeight(localU, localV, tilePlacements, options);
            }
        }

        // 2. Evaluate Base Landscape Heights for Inner 8x8
        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                int index = 81 + (row * 8) + col;
                float localU = chunkOriginU + (col * OuterVertexSpacing) + InnerOffset;
                float localV = chunkOriginV + (row * OuterVertexSpacing) + InnerOffset;

                heights[index] = SampleBlendedHeight(localU, localV, tilePlacements, options);
            }
        }

        // 3. Apply Gradient Slope Constraint (clamp steep steps to guarantee character walkability)
        ApplyWalkableSlopeConstraints(heights, options.MaxSlopeDegrees);

        return heights;
    }

    /// <summary>
    /// Samples the smooth blended height at tile coordinate (u, v), seamlessly incorporating
    /// garden rolling hills and raised exhibit podiums with natural SmoothStep falloff ramps.
    /// </summary>
    public static float SampleBlendedHeight(
        float u,
        float v,
        IReadOnlyList<AdaptiveExhibitPlacement> tilePlacements,
        TerrainSculptorOptions options)
    {
        // 1. Base Garden Landscape Height
        float noise = SampleHarmonicNoise(u * 0.005f, v * 0.005f, 1.0f, 3, 0.45f, options.Seed);
        float landscapeZ = options.BaseElevation + ((noise - 0.5f) * options.HillAmplitudeMeters * options.Roughness);

        if (tilePlacements == null || tilePlacements.Count == 0)
            return landscapeZ;

        float finalZ = landscapeZ;
        float highestPodiumWeight = 0f;

        // 2. Blend with Exhibit Podiums
        for (int i = 0; i < tilePlacements.Count; i++)
        {
            AdaptiveExhibitPlacement p = tilePlacements[i];
            float centerU = p.CellU + (p.CellSize * 0.5f);
            float centerV = p.CellV + (p.CellSize * 0.5f);

            float dx = u - centerU;
            float dy = v - centerV;
            float dist = MathF.Sqrt((dx * dx) + (dy * dy));

            float rInner = p.CellSize * 0.34f;
            float rOuter = p.CellSize * 0.48f;

            if (dist < rOuter)
            {
                float t = SmoothStep(rOuter, rInner, dist);
                float targetPodiumZ = landscapeZ + options.PodiumHeightMeters;

                if (t > highestPodiumWeight)
                {
                    highestPodiumWeight = t;
                    finalZ = MathF.Max(finalZ, Lerp(landscapeZ, targetPodiumZ, t));
                }
            }
        }

        return finalZ;
    }

    private static void ApplyWalkableSlopeConstraints(float[] heights, float maxSlopeDegrees)
    {
        float maxSlopeRad = maxSlopeDegrees * (MathF.PI / 180f);
        float maxStepOuter = OuterVertexSpacing * MathF.Tan(maxSlopeRad); // ~1.94m at 25 deg

        // Smooth pass across outer 9x9 grid
        for (int pass = 0; pass < 2; pass++)
        {
            for (int row = 0; row < 9; row++)
            {
                for (int col = 0; col < 8; col++)
                {
                    int i0 = (row * 9) + col;
                    int i1 = i0 + 1;
                    float diff = heights[i1] - heights[i0];
                    if (MathF.Abs(diff) > maxStepOuter)
                    {
                        float sign = MathF.Sign(diff);
                        heights[i1] = heights[i0] + (sign * maxStepOuter);
                    }
                }
            }

            for (int col = 0; col < 9; col++)
            {
                for (int row = 0; row < 8; row++)
                {
                    int i0 = (row * 9) + col;
                    int i1 = ((row + 1) * 9) + col;
                    float diff = heights[i1] - heights[i0];
                    if (MathF.Abs(diff) > maxStepOuter)
                    {
                        float sign = MathF.Sign(diff);
                        heights[i1] = heights[i0] + (sign * maxStepOuter);
                    }
                }
            }
        }

        // Keep inner vertices smoothly bounded by their 4 surrounding outer corners
        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                int innerIdx = 81 + (row * 8) + col;
                int oTL = (row * 9) + col;
                int oTR = oTL + 1;
                int oBL = ((row + 1) * 9) + col;
                int oBR = oBL + 1;

                float cornerAvg = (heights[oTL] + heights[oTR] + heights[oBL] + heights[oBR]) * 0.25f;
                float clampedInner = Math.Clamp(heights[innerIdx], cornerAvg - maxStepOuter, cornerAvg + maxStepOuter);
                heights[innerIdx] = (heights[innerIdx] * 0.4f) + (clampedInner * 0.6f);
            }
        }
    }

    private static float SmoothNoise(float x, float y, int seed)
    {
        int x0 = (int)MathF.Floor(x);
        int y0 = (int)MathF.Floor(y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = x - x0;
        float fy = y - y0;

        float sx = SmoothStep(0f, 1f, fx);
        float sy = SmoothStep(0f, 1f, fy);

        float n00 = Hash2D(x0, y0, seed);
        float n10 = Hash2D(x1, y0, seed);
        float n01 = Hash2D(x0, y1, seed);
        float n11 = Hash2D(x1, y1, seed);

        float ix0 = Lerp(n00, n10, sx);
        float ix1 = Lerp(n01, n11, sx);

        return Lerp(ix0, ix1, sy);
    }

    private static float Hash2D(int x, int y, int seed)
    {
        int h = seed + (x * 374761393) + (y * 668265263);
        h = (h ^ (h >> 13)) * 1274126177;
        return ((h & 0x7fffffff) / (float)0x7fffffff);
    }

    private static float SmoothStep(float edge0, float edge1, float x)
    {
        float diff = edge1 - edge0;
        if (MathF.Abs(diff) < 0.0001f)
            return x >= edge1 ? 1f : 0f;

        float t = Math.Clamp((x - edge0) / diff, 0f, 1f);
        return t * t * (3f - (2f * t));
    }

    private static float Lerp(float a, float b, float t) => a + ((b - a) * t);
}
