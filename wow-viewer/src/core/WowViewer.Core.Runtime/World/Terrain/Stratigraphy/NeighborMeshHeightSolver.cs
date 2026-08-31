using System.Numerics;

namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// Result of an automated boundary height and scale alignment fit to surrounding active terrain.
/// </summary>
public sealed record NeighborAutoFitResult(
    bool FoundNeighbor,
    float BestFactor,
    bool BestPolarityInverted,
    StratigraphyAnchorMode BestAnchorMode,
    float BestVerticalOffsetZ,
    float ResidualRmseMeters,
    int BoundarySampleCount,
    string NeighborLocationDescription)
{
    public static NeighborAutoFitResult None { get; } = new(
        FoundNeighbor: false,
        BestFactor: TemporalStratigraphyOptions.DefaultClassicFactor,
        BestPolarityInverted: false,
        BestAnchorMode: StratigraphyAnchorMode.LowestZ_Floor,
        BestVerticalOffsetZ: 0f,
        ResidualRmseMeters: float.MaxValue,
        BoundarySampleCount: 0,
        NeighborLocationDescription: "No active neighboring terrain found within search radius");
}

/// <summary>
/// Boundary vertex pair for error calculation along shared chunk edges.
/// </summary>
public readonly record struct BoundaryVertexPair(
    float ActiveNeighborHeight,
    float CompressedTargetHeight);

/// <summary>
/// Solves for optimal scale factor, polarity inversion, anchor datum, and vertical offset delta
/// by minimizing boundary seam root-mean-square error (RMSE) against adjacent active terrain.
/// </summary>
public static class NeighborMeshHeightSolver
{
    private static readonly float[] CandidateScaleBands =
    [
        1.0f,
        3.333f,
        10.0f,
        16.0f,
        33.334f,
        64.0f,
        80.0f,
        128.0f,
        256.0f,
        512.0f
    ];

    /// <summary>
    /// Solves for the optimal restoration parameters given a collection of shared boundary vertex pairs.
    /// </summary>
    public static NeighborAutoFitResult SolveFromBoundaryPairs(
        IReadOnlyList<BoundaryVertexPair> boundaryPairs,
        float[]? customCandidateScales = null,
        string locationDescription = "Adjacent terrain boundary")
    {
        ArgumentNullException.ThrowIfNull(boundaryPairs);

        if (boundaryPairs.Count == 0)
            return NeighborAutoFitResult.None;

        float[] scales = customCandidateScales ?? CandidateScaleBands;
        float bestRmse = float.MaxValue;
        float bestFactor = 33.334f;
        bool bestPolarity = false;
        var bestAnchor = StratigraphyAnchorMode.LowestZ_Floor;
        float bestOffset = 0f;

        // Sample target stats for anchor candidates
        float targetMin = float.MaxValue;
        float targetMax = float.MinValue;
        double targetSum = 0.0;

        for (int i = 0; i < boundaryPairs.Count; i++)
        {
            float h = boundaryPairs[i].CompressedTargetHeight;
            if (h < targetMin) targetMin = h;
            if (h > targetMax) targetMax = h;
            targetSum += h;
        }

        float targetMean = (float)(targetSum / boundaryPairs.Count);

        StratigraphyAnchorMode[] anchorModes =
        [
            StratigraphyAnchorMode.LowestZ_Floor,
            StratigraphyAnchorMode.HighestZ_Ceiling,
            StratigraphyAnchorMode.MeanZ
        ];

        bool[] polarities = [false, true];

        foreach (var anchorMode in anchorModes)
        {
            float anchorHeight = anchorMode switch
            {
                StratigraphyAnchorMode.HighestZ_Ceiling => targetMax,
                StratigraphyAnchorMode.MeanZ => targetMean,
                _ => targetMin
            };

            foreach (bool inverted in polarities)
            {
                float sign = inverted ? -1f : 1f;

                foreach (float scale in scales)
                {
                    // Compute optimal vertical offset Delta Z: mean(Active - ScaledTarget)
                    double deltaSum = 0.0;
                    for (int i = 0; i < boundaryPairs.Count; i++)
                    {
                        var pair = boundaryPairs[i];
                        float scaledTarget = anchorHeight + (sign * scale * (pair.CompressedTargetHeight - anchorHeight));
                        deltaSum += (pair.ActiveNeighborHeight - scaledTarget);
                    }

                    float offsetZ = (float)(deltaSum / boundaryPairs.Count);

                    // Compute RMSE
                    double sqErrSum = 0.0;
                    for (int i = 0; i < boundaryPairs.Count; i++)
                    {
                        var pair = boundaryPairs[i];
                        float scaledTarget = anchorHeight + (sign * scale * (pair.CompressedTargetHeight - anchorHeight)) + offsetZ;
                        float diff = pair.ActiveNeighborHeight - scaledTarget;
                        sqErrSum += diff * diff;
                    }

                    float rmse = MathF.Sqrt((float)(sqErrSum / boundaryPairs.Count));

                    if (rmse < bestRmse)
                    {
                        bestRmse = rmse;
                        bestFactor = scale;
                        bestPolarity = inverted;
                        bestAnchor = anchorMode;
                        bestOffset = offsetZ;
                    }
                }
            }
        }

        return new NeighborAutoFitResult(
            FoundNeighbor: true,
            BestFactor: bestFactor,
            BestPolarityInverted: bestPolarity,
            BestAnchorMode: bestAnchor,
            BestVerticalOffsetZ: bestOffset,
            ResidualRmseMeters: bestRmse,
            BoundarySampleCount: boundaryPairs.Count,
            NeighborLocationDescription: locationDescription);
    }

    /// <summary>
    /// Extracts the 9 boundary vertices along a specified edge of a 145-vertex MCNK chunk.
    /// </summary>
    public static void ExtractEdge145(ReadOnlySpan<float> chunk145, Direction edge, Span<float> destination9)
    {
        if (chunk145.Length < 145)
            throw new ArgumentException("Chunk heights must have at least 145 elements.", nameof(chunk145));
        if (destination9.Length < 9)
            throw new ArgumentException("Destination span must have at least 9 elements.", nameof(destination9));

        switch (edge)
        {
            case Direction.North:
                // Row 0 of outer 9x9 lattice
                for (int col = 0; col < 9; col++)
                    destination9[col] = chunk145[col];
                break;

            case Direction.South:
                // Row 8 of outer 9x9 lattice (idx = 8 * 17 + col = 136 + col)
                for (int col = 0; col < 9; col++)
                    destination9[col] = chunk145[136 + col];
                break;

            case Direction.West:
                // Col 0 of outer 9x9 lattice (idx = row * 17)
                for (int row = 0; row < 9; row++)
                    destination9[row] = chunk145[row * 17];
                break;

            case Direction.East:
                // Col 8 of outer 9x9 lattice (idx = row * 17 + 8)
                for (int row = 0; row < 9; row++)
                    destination9[row] = chunk145[row * 17 + 8];
                break;
        }
    }

    public enum Direction
    {
        North,
        South,
        West,
        East
    }
}
