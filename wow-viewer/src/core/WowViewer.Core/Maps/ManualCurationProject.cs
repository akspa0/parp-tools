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

/// <summary>One map's curation-only state. It is intentionally separate from source ADT data.</summary>
public sealed class ManualCurationProject
{
    public int Version { get; set; } = 1;
    public string MapName { get; set; } = string.Empty;
    public List<ManualObjectPlacement> Placements { get; set; } = [];
    public List<ManualPm4Association> Associations { get; set; } = [];
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
}
