namespace WowViewer.Core.PM4.Models;

public enum Pm4AxisConvention
{
    XZPlaneYUp,
    XYPlaneZUp,
    YZPlaneXUp
}

public enum Pm4CoordinateMode
{
    TileLocal,
    WorldSpace
}

public readonly record struct Pm4PlanarTransform(bool SwapPlanarAxes, bool InvertU, bool InvertV)
{
    public bool InvertsWinding
    {
        get
        {
            int parity = 0;
            if (SwapPlanarAxes)
                parity++;
            if (InvertU)
                parity++;
            if (InvertV)
                parity++;

            return (parity & 1) != 0;
        }
    }
}

public readonly record struct Pm4PlacementSolution(
    int TileX,
    int TileY,
    Pm4CoordinateMode CoordinateMode,
    Pm4AxisConvention AxisConvention,
    Pm4PlanarTransform PlanarTransform,
    System.Numerics.Vector3 WorldPivot,
    float WorldYawCorrectionRadians)
{
    public bool HasWorldYawCorrection => MathF.Abs(WorldYawCorrectionRadians) > 1e-6f;
}

public readonly record struct Pm4LinkedPositionRefSummary(
    int TotalCount,
    int NormalCount,
    int TerminatorCount,
    int FloorMin,
    int FloorMax,
    float HeadingMinDegrees,
    float HeadingMaxDegrees,
    float HeadingMeanDegrees)
{
    public bool HasNormalHeadings => NormalCount > 0 && !float.IsNaN(HeadingMeanDegrees);
}

public readonly record struct Pm4ConnectorKey(int X, int Y, int Z);

public readonly record struct Pm4ObjectGroupKey(int TileX, int TileY, uint Ck24);

public sealed record Pm4ConnectorMergeCandidate(
    Pm4ObjectGroupKey Key,
    System.Numerics.Vector3 BoundsMin,
    System.Numerics.Vector3 BoundsMax,
    System.Numerics.Vector3 Center,
    IReadOnlySet<Pm4ConnectorKey> ConnectorKeys);

public readonly record struct Pm4CoordinateModeResolution(
    Pm4CoordinateMode CoordinateMode,
    Pm4PlanarTransform PlanarTransform,
    float TileLocalScore,
    float WorldSpaceScore,
    bool UsedFallback)
{
    public bool HasTileLocalScore => float.IsFinite(TileLocalScore);

    public bool HasWorldSpaceScore => float.IsFinite(WorldSpaceScore);

    public float SelectedScore => CoordinateMode == Pm4CoordinateMode.TileLocal
        ? TileLocalScore
        : WorldSpaceScore;
}