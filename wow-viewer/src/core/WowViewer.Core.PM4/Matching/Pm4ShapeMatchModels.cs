using System;
using System.Collections.Generic;
using System.Numerics;

namespace WowViewer.Core.PM4.Matching;

/// <summary>One ranked asset candidate for a PM4 hull.</summary>
public sealed record Pm4ShapeCandidate(
    string ModelPath,
    string ModelName,
    string AssetKind,
    double Score,
    float SolvedScale,
    Vector3 SolvedRotation,
    Vector3 SolvedPosition,
    string Explanation)
{
    public bool IsCredible => Score >= Pm4ShapeMatchAnalysisService.MinimumCredibleScore;
}

/// <summary>The ranked candidate set for one PM4 hull, with the evidence needed to review it.</summary>
public sealed record Pm4ShapeMatchProposal(
    uint HullKey,
    Vector3 HullCentroid,
    Vector3 HullBoundsMin,
    Vector3 HullBoundsMax,
    IReadOnlyList<Pm4ShapeCandidate> Candidates,
    double Confidence,
    bool IsLowConfidence,
    bool IsUnmatched,
    int ConflictCount)
{
    public Pm4ShapeCandidate? TopCandidate => Candidates.Count > 0 ? Candidates[0] : null;

    public bool HasConflict => ConflictCount > 1;
}

/// <summary>Aggregate result of measuring solved-vs-authored placement offsets.</summary>
public sealed record Pm4CentroidMeasurement(
    int SampleCount,
    Vector3 MeanOffset,
    Vector3 MinOffset,
    Vector3 MaxOffset,
    string Source);

/// <summary>The operator's decision for one proposal.</summary>
public enum Pm4ShapeMatchDecisionKind
{
    Pending = 0,
    Accepted,
    Rejected,
    Replaced,
}

/// <summary>One persisted review decision with provenance.</summary>
public sealed record Pm4ShapeMatchDecision(
    uint HullKey,
    string? ModelPath,
    Pm4ShapeMatchDecisionKind Kind,
    double Score,
    string Source,
    DateTimeOffset Timestamp,
    string? ReplacedModelPath = null);

/// <summary>One hull-to-object conflict: two hulls whose top candidates solve to the same place.</summary>
public sealed record Pm4ShapeMatchConflict(
    uint HullKeyA,
    uint HullKeyB,
    string ModelPath,
    float Distance);

/// <summary>Health state for one shape-library asset.</summary>
public sealed record Pm4ShapeLibraryHealthEntry(
    string ModelPath,
    bool IsHealthy,
    string? Reason);
