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

        // 3. Compute scale factor
        float fittedScale = template.Scale;
        if (modelPlanarSize > 0.01f && targetPlanarSize > 0.01f)
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

        Vector3 fittedRotation = new(template.RotationDegrees.X, template.RotationDegrees.Y, yawDegrees);

        // 5. Magnetically align position to collision mesh centroid
        Vector3 scaledLocalMin = template.LocalBoundsMin * fittedScale;
        Vector3 scaledLocalMax = template.LocalBoundsMax * fittedScale;

        // Doodad PM4 surfaces represent the walkable top surface (MSUR.0x00 == 0x03).
        // Align the model's top surface to the PM4 collision mesh: P_Z = pm4_top_Z - (model_top * scale).
        float modelTopZ = scaledLocalMax.Z > 0.01f ? scaledLocalMax.Z : (scaledLocalMax.Z - scaledLocalMin.Z);
        float topAlignedZ = target.WorldCentroid.Z - modelTopZ;

        Vector3 attractedPosition = Pm4UnauthoredPlacementResolver.AttractPositionToPm4Mesh(
            target.WorldCentroid,
            topAlignedZ,
            fittedRotation.Z,
            scaledLocalMin,
            scaledLocalMax);

        // 6. Compute fitted world bounds
        Vector3 halfSpan = (scaledLocalMax - scaledLocalMin) * 0.5f;
        Vector3 worldBoundsMin = new(target.WorldCentroid.X - halfSpan.X, target.WorldCentroid.Y - halfSpan.Y, attractedPosition.Z + scaledLocalMin.Z);
        Vector3 worldBoundsMax = new(target.WorldCentroid.X + halfSpan.X, target.WorldCentroid.Y + halfSpan.Y, target.WorldCentroid.Z);

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
}
