using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Matching;

[Flags]
public enum Pm4SegmentConfidenceFlags
{
    None = 0,
    ZeroCk24Seed = 1 << 0,
    UsedConnectivityFallback = 1 << 1,
    MultipleLinkGroupIds = 1 << 2,
    MissingPositionRefs = 1 << 3,
    ReusedLow16ObjectId = 1 << 4,
    SpansMultipleField04Values = 1 << 5,
    HasUnlinkedSurfaces = 1 << 6,
}

public enum Pm4AssetMatchStatus
{
    Matched,
    Ambiguous,
    Unresolved,
    Ineligible,
}

public sealed record Pm4ObjectSegment(
    string SegmentId,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    IReadOnlyList<string> TileCoordinates,
    IReadOnlyList<uint> Field04Values,
    int SurfaceCount,
    int TotalIndexCount,
    IReadOnlyList<uint> LinkGroupIds,
    uint DominantLinkGroupId,
    Pm4SegmentConfidenceFlags ConfidenceFlags);

public sealed record Pm4SegmentHeightStats(
    float MinimumPlaneDistance,
    float MaximumPlaneDistance,
    float AveragePlaneDistance);

public sealed record Pm4SegmentTopologyStats(
    int SurfaceCount,
    int TotalIndexCount,
    int AnchorPointCount,
    int AnchorNormalCount);

public sealed record Pm4SegmentAnchorSignals(
    int LinkedPositionRefCount,
    int NormalHeadingCount,
    int TerminatorCount,
    int FloorMinimum,
    int FloorMaximum,
    float? HeadingMinimumDegrees,
    float? HeadingMaximumDegrees,
    float? HeadingMeanDegrees);

public sealed record Pm4SegmentSignalRecord(
    string SegmentId,
    Pm4Bounds3? Bounds,
    IReadOnlyList<Vector2> FootprintHull,
    Pm4SegmentHeightStats HeightStats,
    IReadOnlyDictionary<string, int> SurfaceFamilyHistogram,
    Pm4SegmentTopologyStats TopologyStats,
    Pm4SegmentAnchorSignals AnchorSignals,
    string SignalVersion,
    string? SignalStoreRow);

public sealed record Pm4ObjectSegmentSurface(
    int SurfaceIndex,
    byte GroupKey,
    byte AttributeMask,
    byte IndexCount,
    float PlaneDistance,
    uint MsviFirstIndex,
    uint MscnRefIndex,
    uint PackedParams,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    Vector3 Normal);

public sealed record Pm4BuiltObjectSegment(
    Pm4ObjectSegment Segment,
    Pm4SegmentSignalRecord Signal,
    Pm4CorrelationObjectState CorrelationState,
    Pm4LinkedPositionRefSummary AnchorSummary,
    IReadOnlyList<Vector2> AnchorPlanarPoints,
    IReadOnlyList<Pm4ObjectSegmentSurface> Surfaces,
    Pm4CoordinateMode CoordinateMode,
    Pm4AxisConvention AxisConvention,
    Pm4PlanarTransform PlanarTransform,
    float FrameYawDegrees);

public sealed record Pm4SegmentExportFile(
    string SourcePath,
    int TileX,
    int TileY,
    IReadOnlyList<Pm4BuiltObjectSegment> Segments)
{
    public int SegmentCount => Segments.Count;
}

public sealed record Pm4SegmentExportRun(
    string RunId,
    string InputPath,
    int FileCount,
    int SegmentCount,
    IReadOnlyList<Pm4SegmentExportFile> Files,
    IReadOnlyList<string> Warnings);

public sealed record Pm4AssetMatchCandidate(
    string AssetId,
    string AssetPath,
    string AssetKind,
    int Rank,
    double OverallScore,
    Pm4AssetMatchStatus Status,
    IReadOnlyDictionary<string, double> ScoreBreakdown,
    IReadOnlyList<string> Rationale);

public sealed record Pm4AssetReferenceSignalRecord(
    string AssetId,
    string AssetPath,
    string AssetKind,
    string? ClientBuild,
    IReadOnlyList<string> TileCoordinates,
    Pm4Bounds3? Bounds,
    Vector3 Center,
    IReadOnlyList<Vector2> FootprintHull,
    float FootprintArea,
    Vector3? ReferencePosition,
    Vector3? ReferenceRotation,
    float? ReferenceScale,
    IReadOnlyDictionary<string, int> SurfaceFamilyHistogram,
    IReadOnlyDictionary<string, double> RenderOrCollisionSignals,
    string SignalVersion,
    string? SignalStoreRow,
    IReadOnlyList<string>? ValidationTags);

public sealed record Pm4SegmentMatchResult(
    Pm4BuiltObjectSegment Segment,
    string? ExpectedAssetKind,
    Pm4AssetMatchStatus Status,
    bool ReviewRequired,
    IReadOnlyList<string> Rationale,
    IReadOnlyList<Pm4AssetMatchCandidate> Candidates);

public sealed record Pm4ReplacementPlacementProposal(
    string ProposalId,
    string SegmentId,
    string AssetId,
    IReadOnlyList<string> TargetTileCoordinates,
    Vector3? WorldPosition,
    Vector3? WorldRotation,
    float? WorldScale,
    double Confidence,
    bool ReviewRequired,
    IReadOnlyList<string> Provenance);
