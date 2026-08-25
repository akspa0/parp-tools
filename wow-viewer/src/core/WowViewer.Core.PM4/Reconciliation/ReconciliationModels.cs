using System.Numerics;

namespace WowViewer.Core.PM4.Reconciliation;

public enum ExpectedAssetKind
{
    Model,
    WorldModel,
    Unknown,
}

public enum ReconciliationAction
{
    Align,
    Substitute,
    Clone,
}

public enum ProposalStatus
{
    ReviewRequired,
    Unsupported,
    Conflict,
    Accepted,
    Rejected,
}

public enum CandidateStatus
{
    Matched,
    Ambiguous,
    Unresolved,
    Ineligible,
}

public enum ReviewDisposition
{
    Accept,
    Reject,
}

/// <summary>A single named signal contributing to a proposal, never an opaque score only.</summary>
public readonly record struct ReconciliationEvidence(string Signal, double Value, string Source);

/// <summary>Stable identity of one PM4 object used as a guide (immutable for a preview lifetime).</summary>
public sealed record Pm4GuideIdentity(
    string GuideId,
    string SourcePath,
    string BuildFingerprint,
    string MapName,
    int TileX,
    int TileY,
    uint Ck24 = 0,
    int ObjectPart = 0,
    string? GeometryFingerprint = null);

/// <summary>Measured geometry and placement-space constraints for a guide object.</summary>
public sealed record Pm4GuideObservation(
    Pm4GuideIdentity Guide,
    Vector3 Position,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    IReadOnlyList<Vector3> Footprint,
    double? HeightSignal,
    ExpectedAssetKind ExpectedAssetKind,
    string SignalVersion,
    IReadOnlyList<ReconciliationEvidence> Evidence);

/// <summary>An existing game-object corpus candidate, reused from the existing PM4 matching concepts.</summary>
public sealed record ReconciliationCandidate(
    string AssetId,
    string AssetPath,
    ExpectedAssetKind Kind,
    int Rank,
    double Score,
    IReadOnlyDictionary<string, double> ScoreBreakdown,
    IReadOnlyList<string> Rationale,
    CandidateStatus Status);

/// <summary>Stable identity of one source placement row in a loaded Museum ADT/WDT.</summary>
public sealed record PlacementIdentity(
    string SourcePath,
    string MapName,
    int TileX,
    int TileY,
    ExpectedAssetKind Kind,
    int EntryIndex,
    int UniqueId,
    string AssetPath,
    string BuildFingerprint);

/// <summary>Source placement values captured for comparison and undo.</summary>
public sealed record PlacementSnapshot(
    PlacementIdentity Identity,
    Vector3 Position,
    Vector3 Rotation,
    float Scale);

/// <summary>An immutable, side-effect-free suggestion shown in the viewport.</summary>
public sealed record ReconciliationProposal(
    string ProposalId,
    Pm4GuideIdentity Guide,
    PlacementIdentity? ExistingPlacement,
    PlacementSnapshot? Current,
    ReconciliationAction Action,
    ReconciliationCandidate? Candidate,
    Vector3 ProposedPosition,
    Vector3 ProposedRotation,
    float ProposedScale,
    IReadOnlyDictionary<string, double> Residual,
    double Confidence,
    IReadOnlyList<ReconciliationEvidence> Evidence,
    ProposalStatus Status);

/// <summary>The user's explicit choice for one proposal. There is no AutoAccept disposition.</summary>
public sealed record ReviewDecision(
    string DecisionId,
    string ProposalId,
    ReviewDisposition Disposition,
    string? Reviewer,
    string? Reason,
    DateTimeOffset CreatedUtc);

/// <summary>The atomic editor operation submitted to the session/undo service.</summary>
public sealed record ReconciliationBatch(
    string BatchId,
    IReadOnlyList<string> GuideSources,
    IReadOnlyList<ReviewDecision> Decisions,
    IReadOnlyList<string> Targets,
    IReadOnlyDictionary<string, string> SourceHashes);