using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Footprint;

namespace WowViewer.Core.PM4.Matching;

/// <summary>Composite structural identity uniquely identifying an authored object component in PM4.</summary>
public readonly record struct Pm4ObjectIdentity(
    int TileX,
    int TileY,
    int ComponentId,
    uint LinkGroupObjectId,
    float PlacementZ,
    Vector3 Centroid,
    Vector3 BoundingSize,
    byte SurfaceClass = 0)
{
    public ulong UniqueHullId => Pm4HullShapeInput.MakeUniqueHullId(TileX, TileY, ComponentId);

    public string FormatTag() =>
        $"[{TileX}_{TileY}] Obj #{ComponentId} (Z={PlacementZ:F1}yd, Lnk={LinkGroupObjectId})";

    public override string ToString() => FormatTag();
}

/// <summary>Tile-scoped PM4 hull descriptor input for shape-match analysis.</summary>
public sealed record Pm4HullShapeInput(
    int TileX,
    int TileY,
    uint Ck24,
    int ObjectPartId,
    uint LinkGroupObjectId,
    Pm4GeometricShapeDescriptor Descriptor)
{
    public ulong UniqueHullId => MakeUniqueHullId(TileX, TileY, ObjectPartId);

    public float PlacementZ => Descriptor != null ? Descriptor.PrismCentroid.Z : 0f;

    public Pm4ObjectIdentity Identity => new(
        TileX,
        TileY,
        ObjectPartId,
        LinkGroupObjectId,
        PlacementZ,
        Descriptor != null ? Descriptor.PrismCentroid : Vector3.Zero,
        Descriptor != null ? Descriptor.BoundingBoxDimensions : Vector3.Zero);

    public static ulong MakeUniqueHullId(int tileX, int tileY, int componentId)
    {
        return ((ulong)(uint)tileX << 48) | ((ulong)(uint)tileY << 32) | (uint)componentId;
    }
}

/// <summary>Model shape candidate carrying a geometric shape descriptor.</summary>
public sealed record Pm4ModelShapeCandidate(
    string ModelPath,
    string AssetKind,
    Pm4GeometricShapeDescriptor Descriptor);

/// <summary>Configurable thresholds for descriptor-based shape matching.</summary>
public sealed record Pm4ShapeMatchOptions(
    float LowConfidenceThreshold = 0.70f,
    float MaxCredibleShapeDistance = 0.85f,
    float ConflictDistance = 2.0f,
    int MaxCandidates = 5);

/// <summary>One ranked asset candidate for a PM4 hull.</summary>
public sealed record Pm4ShapeCandidate(
    string ModelPath,
    string ModelName,
    string AssetKind,
    double Score,
    float SolvedScale,
    Vector3 SolvedRotation,
    Vector3 SolvedPosition,
    string Explanation,
    float ShapeDistance = 0f)
{
    public bool IsCredible => Score >= Pm4ShapeMatchAnalysisService.MinimumCredibleScore;
    public float Confidence => (float)Score;
}

/// <summary>The ranked candidate set for one PM4 hull, with the evidence needed to review it.</summary>
public sealed record Pm4ShapeMatchProposal(
    uint HullKey,
    Vector3 HullCentroid,
    Vector3 HullBoundsMin,
    Vector3 HullBoundsMax,
    IReadOnlyList<Pm4ShapeCandidate> Candidates,
    double ConfidenceScore,
    bool IsLowConfidence,
    bool IsUnmatched,
    int ConflictCount,
    Pm4HullShapeInput? Hull = null,
    Vector3? ExplicitCalibratedCentroid = null,
    ulong UniqueHullId = 0)
{
    public ulong EffectiveUniqueHullId => UniqueHullId != 0
        ? UniqueHullId
        : (Hull != null ? Hull.UniqueHullId : (ulong)HullKey);

    public Pm4ShapeCandidate? TopCandidate => Candidates.Count > 0 ? Candidates[0] : null;

    public bool HasConflict => ConflictCount > 1;

    public float Confidence => (float)ConfidenceScore;

    public Vector3 UncalibratedCentroid => HullCentroid;

    public Vector3 CalibratedCentroid =>
        ExplicitCalibratedCentroid ?? TopCandidate?.SolvedPosition ?? HullCentroid;
}

/// <summary>One tile-scoped hull-to-object duplicate-claim conflict.</summary>
public sealed record Pm4ProposalConflict(
    Pm4HullShapeInput FirstHull,
    Pm4HullShapeInput SecondHull,
    string ModelPath,
    float DistanceBetweenHulls);

/// <summary>Result of descriptor-based batch shape-match analysis.</summary>
public sealed record Pm4ShapeBatchResult(
    IReadOnlyList<Pm4ShapeMatchProposal> Proposals,
    IReadOnlyList<Pm4ProposalConflict> Conflicts);

/// <summary>Health check result for a single candidate entry.</summary>
public sealed record Pm4ShapeLibraryEntryHealth(
    string ModelPath,
    bool IsHealthy,
    string? Reason);

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
