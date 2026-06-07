namespace WowViewer.Tools.Shared.Pm4Matching;

public sealed record Pm4MatchReportCandidate(
    string AssetId,
    string AssetPath,
    string AssetKind,
    int Rank,
    double OverallScore,
    string Status,
    IReadOnlyDictionary<string, double>? ScoreBreakdown,
    IReadOnlyList<string>? Rationale);

public sealed record Pm4MatchReportBounds(
    Pm4MatchVector3 Min,
    Pm4MatchVector3 Max);

public sealed record Pm4MatchReportHeightStats(
    double MinimumPlaneDistance,
    double MaximumPlaneDistance,
    double AveragePlaneDistance);

public sealed record Pm4MatchReportTopologyStats(
    int SurfaceCount,
    int TotalIndexCount,
    int AnchorPointCount,
    int AnchorNormalCount);

public sealed record Pm4MatchReportAnchorSignals(
    int LinkedPositionRefCount,
    int NormalHeadingCount,
    int TerminatorCount,
    int FloorMinimum,
    int FloorMaximum,
    double? HeadingMinimumDegrees,
    double? HeadingMaximumDegrees,
    double? HeadingMeanDegrees);

public sealed record Pm4MatchReportPlacementProposal(
    string ProposalId,
    string AssetId,
    IReadOnlyList<string> TargetTileCoordinates,
    Pm4MatchVector3? WorldPosition,
    Pm4MatchRotation? WorldRotation,
    double? WorldScale,
    double Confidence,
    bool ReviewRequired,
    IReadOnlyList<string>? Provenance);

public sealed record Pm4MatchReportSegment(
    string SegmentId,
    string Ck24,
    int Ck24Type,
    int Ck24ObjectId,
    IReadOnlyList<string> TileCoordinates,
    IReadOnlyList<uint>? Field04Values,
    string? ExpectedAssetKind,
    string? Status,
    bool ReviewRequired,
    IReadOnlyList<string>? Rationale,
    IReadOnlyList<string>? ConfidenceFlags,
    int SurfaceCount,
    int TotalIndexCount,
    IReadOnlyList<string>? LinkGroupIds,
    string? DominantLinkGroupId,
    string? CoordinateMode,
    string? AxisConvention,
    double? FrameYawDegrees,
    Pm4MatchReportBounds? Bounds,
    Pm4MatchVector3? Center,
    IReadOnlyList<Pm4MatchVector2>? FootprintHull,
    double? FootprintArea,
    Pm4MatchReportHeightStats? HeightStats,
    Pm4MatchReportTopologyStats? TopologyStats,
    Pm4MatchReportAnchorSignals? AnchorSignals,
    IReadOnlyDictionary<string, int>? SurfaceFamilyHistogram,
    IReadOnlyList<Pm4MatchReportCandidate> Candidates,
    Pm4MatchReportPlacementProposal? PlacementProposal);

public sealed record Pm4MatchRunManifest(
    string RunId,
    string InputPm4Root,
    int SegmentCount,
    IReadOnlyList<Pm4MatchReportSegment> Segments,
    string? AssetReferenceCorpus = null,
    string? SegmentSignalCorpus = null,
    int? MatchedCount = null,
    int? AmbiguousCount = null,
    int? UnresolvedCount = null,
    int? IneligibleCount = null,
    IReadOnlyList<string>? Warnings = null);

public sealed record Pm4MatchVector3(double X, double Y, double Z);

public sealed record Pm4MatchVector2(double X, double Y);

public sealed record Pm4MatchRotation(double? Yaw = null, double? Pitch = null, double? Roll = null);
