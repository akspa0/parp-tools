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
    string? Status,
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
    int? UnresolvedCount = null);

public sealed record Pm4MatchVector3(double X, double Y, double Z);

public sealed record Pm4MatchRotation(double? Yaw = null, double? Pitch = null, double? Roll = null);
