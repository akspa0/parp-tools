using System;
using System.Collections.Generic;
using System.Numerics;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// Candidate PM4 doodad collision hull targeted for object duplication/matching (Spec 287 Phase 1).
/// </summary>
public sealed record Pm4DoodadTarget(
    int TileX,
    int TileY,
    uint Ck24,
    int ObjectPartId,
    Vector3 WorldCentroid,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    float BaseRotationRadians,
    byte DominantGroupKey,
    IReadOnlyList<Vector3>? WorldVertices = null);

/// <summary>
/// Source model template descriptors used to populate unmatched PM4 doodad targets.
/// </summary>
public sealed record ModelTemplateInfo(
    string ModelPath,
    string ModelKey,
    string ModelName,
    Vector3 Position,
    Vector3 RotationDegrees,
    float Scale,
    Vector3 LocalBoundsMin,
    Vector3 LocalBoundsMax,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax);

/// <summary>
/// Synthesized doodad instance fitted to a PM4 collision hull.
/// </summary>
public sealed record DuplicatedDoodadResult(
    int TileX,
    int TileY,
    uint Ck24,
    int ObjectPartId,
    string ModelPath,
    string ModelKey,
    string ModelName,
    Vector3 Position,
    Vector3 RotationDegrees,
    float Scale,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax,
    int UniqueId,
    float FitScore);

/// <summary>
/// World-space bounds of an already-placed model instance, used for hull occupancy tests.
/// <see cref="BoundsResolved"/> is false when the instance's world bounds have not been computed
/// yet; in that case only <see cref="Origin"/> is trustworthy.
/// </summary>
public readonly record struct PlacementBounds(
    Vector3 Min,
    Vector3 Max,
    Vector3 Origin,
    bool BoundsResolved);

/// <summary>
/// Mathematical and spatial engine for duplicating existing model instances to fill unmatched
/// PM4 doodad collision hulls (e.g. repeated cliff rocks in development.wdt 35_38).
/// </summary>
public static class Pm4DoodadDuplicationEngine
{
    public const float DefaultOccupancyRadius = 8.0f;
    public const int SyntheticUniqueIdBase = 0x50000000;

    /// <summary>
    /// Filters candidate PM4 targets, rejecting any targets whose centroid is within
    /// <paramref name="occupancyRadius"/> of an existing model placement.
    /// </summary>
    public static List<Pm4DoodadTarget> FindUnmatchedTargets(
        IReadOnlyList<Pm4DoodadTarget> candidates,
        IReadOnlyList<Vector3> existingPlacements,
        float occupancyRadius = DefaultOccupancyRadius)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(existingPlacements);

        var unmatched = new List<Pm4DoodadTarget>(candidates.Count);
        float radiusSq = occupancyRadius * occupancyRadius;

        for (int i = 0; i < candidates.Count; i++)
        {
            Pm4DoodadTarget target = candidates[i];
            bool occupied = false;

            for (int j = 0; j < existingPlacements.Count; j++)
            {
                if (Vector3.DistanceSquared(target.WorldCentroid, existingPlacements[j]) < radiusSq)
                {
                    occupied = true;
                    break;
                }
            }

            if (!occupied)
            {
                unmatched.Add(target);
            }
        }

        return unmatched;
    }

    /// <summary>
    /// Returns true when <paramref name="target"/> already has a model assigned/paired.
    /// </summary>
    /// <remarks>
    /// Occupancy is tested against world-space AABBs, not hull-centroid-to-model-origin distance.
    /// WMO and most M2 placement origins sit at the model base rather than the hull centroid, so an
    /// origin-distance test routinely reports an already-paired hull as free and spawns a duplicate
    /// copy. AABB intersection in all three axes is the correct "this hull already contains an
    /// object" semantic. Placements whose bounds are not resolved fall back to origin distance.
    /// </remarks>
    public static bool IsTargetOccupied(
        Pm4DoodadTarget target,
        IReadOnlyList<PlacementBounds> placements,
        float originFallbackRadius = DefaultOccupancyRadius)
    {
        ArgumentNullException.ThrowIfNull(target);
        if (placements == null || placements.Count == 0)
            return false;

        float radiusSq = originFallbackRadius * originFallbackRadius;

        for (int i = 0; i < placements.Count; i++)
        {
            PlacementBounds p = placements[i];
            if (p.BoundsResolved)
            {
                if (OverlapsAabb(target.BoundsMin, target.BoundsMax, p.Min, p.Max))
                    return true;
            }
            else if (Vector3.DistanceSquared(target.WorldCentroid, p.Origin) < radiusSq)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// True when two world-space AABBs intersect on all three axes (with a small epsilon so
    /// touching faces count as overlap).
    /// </summary>
    private static bool OverlapsAabb(Vector3 aMin, Vector3 aMax, Vector3 bMin, Vector3 bMax)
    {
        const float epsilon = 0.05f;
        return aMin.X - epsilon <= bMax.X && aMax.X + epsilon >= bMin.X
            && aMin.Y - epsilon <= bMax.Y && aMax.Y + epsilon >= bMin.Y
            && aMin.Z - epsilon <= bMax.Z && aMax.Z + epsilon >= bMin.Z;
    }

    /// <summary>
    /// Filters candidate PM4 targets using world-space bounds occupancy (see
    /// <see cref="IsTargetOccupied"/>). Prefer this overload over the origin-distance one.
    /// </summary>
    public static List<Pm4DoodadTarget> FindUnmatchedTargets(
        IReadOnlyList<Pm4DoodadTarget> candidates,
        IReadOnlyList<PlacementBounds> placements,
        float originFallbackRadius = DefaultOccupancyRadius)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        if (placements == null || placements.Count == 0)
            return [.. candidates];

        var unmatched = new List<Pm4DoodadTarget>(candidates.Count);
        for (int i = 0; i < candidates.Count; i++)
        {
            if (!IsTargetOccupied(candidates[i], placements, originFallbackRadius))
                unmatched.Add(candidates[i]);
        }

        return unmatched;
    }

    /// <summary>
    /// Fits a model template into a PM4 collision target, computing scale ratio, magnetic position,
    /// planar yaw, and bounding box.
    /// </summary>
    public static DuplicatedDoodadResult FitModelToTarget(
        ModelTemplateInfo template,
        Pm4DoodadTarget target,
        int uniqueId)
    {
        ArgumentNullException.ThrowIfNull(template);
        ArgumentNullException.ThrowIfNull(target);

        // 1. Calculate PM4 hull planar span
        Vector3 targetSpan = target.BoundsMax - target.BoundsMin;
        float targetPlanarSize = MathF.Sqrt(targetSpan.X * targetSpan.X + targetSpan.Y * targetSpan.Y);

        // 2. Calculate model template unscaled planar span
        Vector3 modelBaseSpan = template.LocalBoundsMax - template.LocalBoundsMin;
        float modelPlanarSize = MathF.Sqrt(modelBaseSpan.X * modelBaseSpan.X + modelBaseSpan.Y * modelBaseSpan.Y);
        if (modelPlanarSize <= 0.01f)
        {
            Vector3 worldSpan = template.WorldBoundsMax - template.WorldBoundsMin;
            float worldPlanar = MathF.Sqrt(worldSpan.X * worldSpan.X + worldSpan.Y * worldSpan.Y);
            modelPlanarSize = worldPlanar / MathF.Max(0.01f, template.Scale);
        }

        // 3. Compute scale factor. WMOs are authored 1:1 architecture and must never be
        // rescaled (same rule as Pm4DirectModelMatcher); the caller places them at scale 1.0,
        // so the position solve below must use 1.0 too.
        bool isWmoTemplate = template.ModelPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);
        float fittedScale = template.Scale;
        if (!isWmoTemplate && modelPlanarSize > 0.01f && targetPlanarSize > 0.01f)
        {
            float rawRatio = targetPlanarSize / modelPlanarSize;
            fittedScale = Math.Clamp(rawRatio, 0.1f, 10.0f);
        }

        // 4. Compute planar rotation (Yaw)
        float yawDegrees = target.BaseRotationRadians * (180f / MathF.PI);
        if (MathF.Abs(target.BaseRotationRadians) < 1e-4f && target.WorldVertices is { Count: >= 3 } verts)
        {
            yawDegrees = Pm4UnauthoredPlacementResolver.ComputeFootprintYawDegrees(verts);
        }

        // Align the model's long axis with the hull principal axis (same 1.15 rule as
        // AlignPlacementToPm4Hull and Pm4DirectModelMatcher so all three paths agree);
        // without it, Y-long models duplicate crosswise into the hull.
        Vector3 modelSpan = template.LocalBoundsMax - template.LocalBoundsMin;
        if (MathF.Abs(modelSpan.Y) > MathF.Abs(modelSpan.X) * 1.15f)
        {
            yawDegrees += 90f;
            while (yawDegrees > 180f) yawDegrees -= 360f;
            while (yawDegrees < -180f) yawDegrees += 360f;
        }

        Vector3 fittedRotation = new(template.RotationDegrees.X, template.RotationDegrees.Y, yawDegrees);

        // 5. Magnetically align position to collision mesh centroid
        Vector3 scaledLocalMin = template.LocalBoundsMin * fittedScale;
        Vector3 scaledLocalMax = template.LocalBoundsMax * fittedScale;

        // Doodad PM4 surfaces represent the walkable top surface (MSUR.0x00 == 0x03).
        // T001 contract: P_Z = Z_pm4_top - (H_local_top * s). Target bounds are world-space
        // (the caller transforms them by BaseTransform), so BoundsMax.Z is the hull top and
        // BoundsMin.Z the hull floor. WMOs are the exception: their local origin is authored
        // at the placement Z with foundation geometry extending below it, so they base-align.
        float targetZ;
        if (isWmoTemplate)
        {
            targetZ = target.BoundsMin.Z;
        }
        else
        {
            float modelTopZ = scaledLocalMax.Z > 0.01f ? scaledLocalMax.Z : (scaledLocalMax.Z - scaledLocalMin.Z);
            targetZ = target.BoundsMax.Z - modelTopZ;
        }

        Vector3 attractedPosition = Pm4UnauthoredPlacementResolver.AttractPositionToPm4Mesh(
            target.WorldCentroid,
            targetZ,
            fittedRotation.Z,
            scaledLocalMin,
            scaledLocalMax);

        // 6. Compute fitted world bounds
        Vector3 halfSpan = (scaledLocalMax - scaledLocalMin) * 0.5f;
        Vector3 worldBoundsMin = new(target.WorldCentroid.X - halfSpan.X, target.WorldCentroid.Y - halfSpan.Y, attractedPosition.Z + scaledLocalMin.Z);
        Vector3 worldBoundsMax = new(target.WorldCentroid.X + halfSpan.X, target.WorldCentroid.Y + halfSpan.Y, attractedPosition.Z + scaledLocalMax.Z);

        // 7. Calculate fit score
        float fitScore = 1.0f;
        if (targetPlanarSize > 0.01f && modelPlanarSize > 0.01f)
        {
            float ratio = (modelPlanarSize * fittedScale) / targetPlanarSize;
            fitScore = MathF.Max(0.1f, 1.0f - MathF.Abs(1.0f - ratio));
        }

        return new DuplicatedDoodadResult(
            target.TileX,
            target.TileY,
            target.Ck24,
            target.ObjectPartId,
            template.ModelPath,
            template.ModelKey,
            template.ModelName,
            attractedPosition,
            fittedRotation,
            fittedScale,
            worldBoundsMin,
            worldBoundsMax,
            uniqueId,
            fitScore);
    }

    /// <summary>
    /// Duplicates a model template across all unmatched candidate PM4 targets, automatically allocating
    /// sequential unique IDs starting after <paramref name="startUniqueId"/>.
    /// </summary>
    public static List<DuplicatedDoodadResult> DuplicateAcrossTargets(
        ModelTemplateInfo template,
        IReadOnlyList<Pm4DoodadTarget> candidates,
        IReadOnlyList<Vector3> existingPlacements,
        int startUniqueId,
        float occupancyRadius = DefaultOccupancyRadius)
    {
        List<Pm4DoodadTarget> unmatched = FindUnmatchedTargets(candidates, existingPlacements, occupancyRadius);
        var results = new List<DuplicatedDoodadResult>(unmatched.Count);

        int currentUniqueId = startUniqueId > 0 ? startUniqueId : SyntheticUniqueIdBase;
        for (int i = 0; i < unmatched.Count; i++)
        {
            results.Add(FitModelToTarget(template, unmatched[i], currentUniqueId++));
        }

        return results;
    }

    /// <summary>
    /// Duplicates a model template across all unmatched candidate PM4 targets using world-space
    /// bounds occupancy. This is the overload that prevents duplicate copies on hulls that already
    /// have an assigned/paired model.
    /// </summary>
    public static List<DuplicatedDoodadResult> DuplicateAcrossTargets(
        ModelTemplateInfo template,
        IReadOnlyList<Pm4DoodadTarget> candidates,
        IReadOnlyList<PlacementBounds> placements,
        int startUniqueId,
        float originFallbackRadius = DefaultOccupancyRadius)
    {
        List<Pm4DoodadTarget> unmatched = FindUnmatchedTargets(candidates, placements, originFallbackRadius);
        var results = new List<DuplicatedDoodadResult>(unmatched.Count);

        int currentUniqueId = startUniqueId > 0 ? startUniqueId : SyntheticUniqueIdBase;
        for (int i = 0; i < unmatched.Count; i++)
        {
            results.Add(FitModelToTarget(template, unmatched[i], currentUniqueId++));
        }

        return results;
    }
}
