using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>A project-owned snapshot of an object placement curated independently of its source map.</summary>
public sealed record ManualObjectPlacement(
    Guid PlacementId,
    string ModelPath,
    string AssetKind,
    string ModelName,
    int TileX,
    int TileY,
    int PlacementEntryIndex,
    int UniqueId,
    Vector3 Position,
    Vector3 RotationDegrees,
    float Scale,
    bool CreatedFromAssetCatalog,
    DateTimeOffset AuthoredUtc)
{
    public bool IsValid => PlacementId != Guid.Empty &&
        !string.IsNullOrWhiteSpace(ModelPath) &&
        TileX is >= 0 and < 64 && TileY is >= 0 and < 64 &&
        float.IsFinite(Position.X) && float.IsFinite(Position.Y) && float.IsFinite(Position.Z) &&
        float.IsFinite(RotationDegrees.X) && float.IsFinite(RotationDegrees.Y) && float.IsFinite(RotationDegrees.Z) &&
        float.IsFinite(Scale) && Scale > 0f;
}

/// <summary>Stable tile-scoped identity for one inspected PM4 hull component.</summary>
public readonly record struct Pm4HullIdentity(
    int TileX,
    int TileY,
    uint Ck24,
    int ObjectPartId,
    uint LinkGroupObjectId = 0)
{
    public bool IsValid => TileX is >= 0 and < 64 && TileY is >= 0 and < 64 && ObjectPartId >= 0;
}

/// <summary>Explicit user-authored relation; automatic matching never creates this record.</summary>
public sealed record ManualPm4Association(
    Guid PlacementId,
    Pm4HullIdentity Hull,
    DateTimeOffset AuthoredUtc);

public enum ManualShapeMatchDecisionKind
{
    Accepted,
    Rejected,
    Replaced,
}

/// <summary>
/// Spec 295 (FR-009, FR-010): Persisted human-in-the-loop (or bounded auto-accept) pairing decision for a PM4 hull.
/// </summary>
public sealed record ManualShapeMatchDecisionRecord(
    Pm4HullIdentity Hull,
    string ProposedModelPath,
    string? ChosenModelPath,
    string AssetKind,
    ManualShapeMatchDecisionKind Decision,
    float Confidence,
    float ShapeDistance,
    Vector3 CalibratedCentroid,
    string Provenance,
    DateTimeOffset DecidedUtc);

/// <summary>
/// Spec 295 (FR-011): Append-only audit log entry for every accept, reject, or replace pairing decision.
/// </summary>
public sealed record ManualShapeMatchAuditEntry(
    Guid AuditId,
    Pm4HullIdentity Hull,
    string ProposedModelPath,
    string? ChosenModelPath,
    ManualShapeMatchDecisionKind Decision,
    float Confidence,
    string Provenance,
    string? Notes,
    DateTimeOffset TimestampUtc);

/// <summary>One map's curation-only state. It is intentionally separate from source ADT data.</summary>
public sealed class ManualCurationProject
{
    public int Version { get; set; } = 1;
    public string MapName { get; set; } = string.Empty;
    public List<ManualObjectPlacement> Placements { get; set; } = [];
    public List<ManualPm4Association> Associations { get; set; } = [];
    public List<ManualShapeMatchDecisionRecord> ShapeMatchDecisions { get; set; } = [];
    public List<ManualShapeMatchAuditEntry> ShapeMatchAuditHistory { get; set; } = [];
}

public enum ManualAssociationResult
{
    Added,
    Removed,
    Replaced,
    AlreadyExists,
    PlacementNotFound,
    InvalidHull,
    AssociationNotFound,
}

/// <summary>
/// Pure curation policy for explicit object↔hull links. No matching or file I/O occurs here.
/// </summary>
public static class ManualCurationProjectPolicy
{
    public static bool TryAddPlacement(ManualCurationProject project, ManualObjectPlacement placement)
    {
        ArgumentNullException.ThrowIfNull(project);
        ArgumentNullException.ThrowIfNull(placement);

        if (!placement.IsValid)
            return false;

        if (project.Placements.Any(existing => existing.PlacementId == placement.PlacementId))
            return false;

        project.Placements.Add(placement);
        return true;
    }

    public static bool TryUpdatePlacement(ManualCurationProject project, ManualObjectPlacement placement)
    {
        ArgumentNullException.ThrowIfNull(project);
        ArgumentNullException.ThrowIfNull(placement);
        if (!placement.IsValid)
            return false;

        int index = project.Placements.FindIndex(existing => existing.PlacementId == placement.PlacementId);
        if (index < 0)
            return false;

        project.Placements[index] = placement;
        return true;
    }

    public static ManualAssociationResult TryAddAssociation(
        ManualCurationProject project,
        Guid placementId,
        Pm4HullIdentity hull,
        DateTimeOffset? authoredUtc = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        if (!project.Placements.Any(placement => placement.PlacementId == placementId))
            return ManualAssociationResult.PlacementNotFound;
        if (!hull.IsValid)
            return ManualAssociationResult.InvalidHull;
        if (project.Associations.Any(association =>
                association.PlacementId == placementId && association.Hull == hull))
        {
            return ManualAssociationResult.AlreadyExists;
        }

        project.Associations.Add(new ManualPm4Association(
            placementId,
            hull,
            authoredUtc ?? DateTimeOffset.UtcNow));
        return ManualAssociationResult.Added;
    }

    public static ManualAssociationResult TryRemoveAssociation(
        ManualCurationProject project,
        Guid placementId,
        Pm4HullIdentity hull)
    {
        ArgumentNullException.ThrowIfNull(project);
        int index = project.Associations.FindIndex(association =>
            association.PlacementId == placementId && association.Hull == hull);
        if (index < 0)
            return ManualAssociationResult.AssociationNotFound;

        project.Associations.RemoveAt(index);
        return ManualAssociationResult.Removed;
    }

    /// <summary>Explicitly replaces one existing association. The original is retained on invalid input.</summary>
    public static ManualAssociationResult TryReplaceAssociation(
        ManualCurationProject project,
        Guid placementId,
        Pm4HullIdentity oldHull,
        Pm4HullIdentity newHull,
        DateTimeOffset? authoredUtc = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        int oldIndex = project.Associations.FindIndex(association =>
            association.PlacementId == placementId && association.Hull == oldHull);
        if (oldIndex < 0)
            return ManualAssociationResult.AssociationNotFound;
        if (!project.Placements.Any(placement => placement.PlacementId == placementId))
            return ManualAssociationResult.PlacementNotFound;
        if (!newHull.IsValid)
            return ManualAssociationResult.InvalidHull;
        if (oldHull == newHull)
            return ManualAssociationResult.AlreadyExists;
        if (project.Associations.Any(association =>
                association.PlacementId == placementId && association.Hull == newHull))
        {
            return ManualAssociationResult.AlreadyExists;
        }

        project.Associations[oldIndex] = new ManualPm4Association(
            placementId,
            newHull,
            authoredUtc ?? DateTimeOffset.UtcNow);
        return ManualAssociationResult.Replaced;
    }

    public static bool RecordShapeMatchDecision(
        ManualCurationProject project,
        ManualShapeMatchDecisionRecord decision,
        string? notes = null,
        Guid? auditId = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        ArgumentNullException.ThrowIfNull(decision);

        if (!decision.Hull.IsValid || string.IsNullOrWhiteSpace(decision.ProposedModelPath))
            return false;

        if (decision.Decision == ManualShapeMatchDecisionKind.Replaced &&
            string.IsNullOrWhiteSpace(decision.ChosenModelPath))
        {
            return false;
        }

        project.ShapeMatchDecisions ??= [];
        project.ShapeMatchAuditHistory ??= [];

        if (decision.Decision == ManualShapeMatchDecisionKind.Rejected)
        {
            int existingIdx = project.ShapeMatchDecisions.FindIndex(d =>
                d.Decision == ManualShapeMatchDecisionKind.Rejected &&
                d.Hull == decision.Hull &&
                string.Equals(d.ProposedModelPath, decision.ProposedModelPath, StringComparison.OrdinalIgnoreCase));

            if (existingIdx >= 0)
                project.ShapeMatchDecisions[existingIdx] = decision;
            else
                project.ShapeMatchDecisions.Add(decision);
        }
        else
        {
            project.ShapeMatchDecisions.RemoveAll(d =>
                d.Decision != ManualShapeMatchDecisionKind.Rejected &&
                d.Hull == decision.Hull);
            project.ShapeMatchDecisions.Add(decision);
        }

        project.ShapeMatchAuditHistory.Add(new ManualShapeMatchAuditEntry(
            auditId ?? Guid.NewGuid(),
            decision.Hull,
            decision.ProposedModelPath,
            decision.ChosenModelPath,
            decision.Decision,
            decision.Confidence,
            decision.Provenance,
            notes,
            decision.DecidedUtc));

        return true;
    }

    public static bool IsCandidateRejectedForHull(
        ManualCurationProject project,
        Pm4HullIdentity hull,
        string candidateModelPath)
    {
        ArgumentNullException.ThrowIfNull(project);
        if (project.ShapeMatchDecisions == null || string.IsNullOrWhiteSpace(candidateModelPath))
            return false;

        return project.ShapeMatchDecisions.Any(d =>
            d.Decision == ManualShapeMatchDecisionKind.Rejected &&
            d.Hull == hull &&
            string.Equals(d.ProposedModelPath, candidateModelPath, StringComparison.OrdinalIgnoreCase));
    }

    public static bool TryGetActiveShapeMatchDecision(
        ManualCurationProject project,
        Pm4HullIdentity hull,
        out ManualShapeMatchDecisionRecord? decision)
    {
        ArgumentNullException.ThrowIfNull(project);
        decision = project.ShapeMatchDecisions?.LastOrDefault(d =>
            d.Decision != ManualShapeMatchDecisionKind.Rejected &&
            d.Hull == hull);
        return decision != null;
    }

    public static int ResetRejectedShapeMatchDecisions(
        ManualCurationProject project,
        Pm4HullIdentity? hull = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        if (project.ShapeMatchDecisions == null)
            return 0;

        return project.ShapeMatchDecisions.RemoveAll(d =>
            d.Decision == ManualShapeMatchDecisionKind.Rejected &&
            (hull == null || d.Hull == hull.Value));
    }
}
