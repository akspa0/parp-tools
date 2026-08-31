namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// Profiler for evaluating C0 (height step) and C1 (slope gradient) discontinuities across
/// internal MCNK boundaries (1..15) to detect historical sub-tile merge seams (2x2 vs 4x4 vs MCNK cell worlds).
/// </summary>
public static class SeamDiscontinuityProfiler
{
    public const int BoundaryCount = 15;
    public const int ChunkSizeVertices = 16;
    public const int LatticeDim = 257;

    /// <summary>
    /// Analyzes internal boundary seams across a 257x257 height lattice.
    /// </summary>
    public static SeamDiscontinuityResult AnalyzeSeams(float[,] heights257)
    {
        ArgumentNullException.ThrowIfNull(heights257);

        var hStep = new float[BoundaryCount];
        var hSlope = new float[BoundaryCount];
        var vStep = new float[BoundaryCount];
        var vSlope = new float[BoundaryCount];

        // Measure vertical boundaries (spanning across X, index 1..15)
        for (int b = 0; b < BoundaryCount; b++)
        {
            int seamX = (b + 1) * ChunkSizeVertices;
            float stepAccum = 0f;
            float slopeAccum = 0f;
            int count = 0;

            for (int y = 0; y < LatticeDim; y++)
            {
                if (seamX > 0 && seamX < LatticeDim - 1)
                {
                    float left = heights257[y, seamX - 1];
                    float center = heights257[y, seamX];
                    float right = heights257[y, seamX + 1];

                    stepAccum += Math.Abs(right - left);
                    slopeAccum += Math.Abs((right - center) - (center - left));
                    count++;
                }
            }

            vStep[b] = count > 0 ? stepAccum / count : 0f;
            vSlope[b] = count > 0 ? slopeAccum / count : 0f;
        }

        // Measure horizontal boundaries (spanning across Y, index 1..15)
        for (int b = 0; b < BoundaryCount; b++)
        {
            int seamY = (b + 1) * ChunkSizeVertices;
            float stepAccum = 0f;
            float slopeAccum = 0f;
            int count = 0;

            for (int x = 0; x < LatticeDim; x++)
            {
                if (seamY > 0 && seamY < LatticeDim - 1)
                {
                    float top = heights257[seamY - 1, x];
                    float center = heights257[seamY, x];
                    float bottom = heights257[seamY + 1, x];

                    stepAccum += Math.Abs(bottom - top);
                    slopeAccum += Math.Abs((bottom - center) - (center - top));
                    count++;
                }
            }

            hStep[b] = count > 0 ? stepAccum / count : 0f;
            hSlope[b] = count > 0 ? slopeAccum / count : 0f;
        }

        // Evaluate Spike Patterns
        bool spikeAt8 = IsSpikeAt(vSlope, hSlope, index: 7, neighborIndices: [5, 6, 8, 9]); // Index 7 = Boundary 8
        bool spikeAt4_8_12 = spikeAt8
            && IsSpikeAt(vSlope, hSlope, index: 3, neighborIndices: [1, 2, 4, 5]) // Boundary 4
            && IsSpikeAt(vSlope, hSlope, index: 11, neighborIndices: [9, 10, 12, 13]); // Boundary 12

        float meanSlope = (vSlope.Average() + hSlope.Average()) * 0.5f;
        bool allElevated = meanSlope > 0.05f && !spikeAt8 && !spikeAt4_8_12;

        string origin = spikeAt4_8_12
            ? "4x4 Sub-Tile Merge (Quarter-Tile Seams)"
            : spikeAt8
                ? "2x2 Sub-Tile Merge (Half-Tile Seam at Chunk 8)"
                : allElevated
                    ? "128x128 MCNK-Cell World Fragment"
                    : "Unified Single-Pass Authored";

        return new SeamDiscontinuityResult
        {
            HorizontalBoundaryStepC0 = hStep,
            HorizontalBoundarySlopeC1 = hSlope,
            VerticalBoundaryStepC0 = vStep,
            VerticalBoundarySlopeC1 = vSlope,
            Has2x2MergeSpikeAt8 = spikeAt8,
            Has4x4MergeSpikesAt4_8_12 = spikeAt4_8_12,
            IsElevatedAcrossAllBoundaries = allElevated,
            InferredMergeOrigin = origin,
        };
    }

    private static bool IsSpikeAt(float[] vSlope, float[] hSlope, int index, int[] neighborIndices)
    {
        if ((uint)index >= vSlope.Length)
            return false;

        float target = (vSlope[index] + hSlope[index]) * 0.5f;
        if (target < 0.005f)
            return false;

        float neighborSum = 0f;
        int count = 0;
        foreach (int n in neighborIndices)
        {
            if (n >= 0 && n < vSlope.Length)
            {
                neighborSum += (vSlope[n] + hSlope[n]) * 0.5f;
                count++;
            }
        }

        float baseline = count > 0 ? neighborSum / count : 0.001f;
        return target >= baseline * 1.5f && (target - baseline) > 0.01f;
    }
}
