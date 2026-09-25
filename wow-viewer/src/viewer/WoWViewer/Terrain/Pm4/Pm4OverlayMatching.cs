using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Population;
using WoWViewer.Rendering;
using WoWViewer.Audio;
using WowViewer.Core.Audio;
using WowViewer.Core.Maps;
using Silk.NET.OpenGL;
using CorePm4AxisConvention = WowViewer.Core.PM4.Models.Pm4AxisConvention;
using CorePm4CorrelationCandidateScore = WowViewer.Core.PM4.Models.Pm4CorrelationCandidateScore;
using CorePm4CorrelationMetrics = WowViewer.Core.PM4.Models.Pm4CorrelationMetrics;
using CorePm4CorrelationObjectDescriptor = WowViewer.Core.PM4.Models.Pm4CorrelationObjectDescriptor;
using CorePm4CorrelationGeometryInput = WowViewer.Core.PM4.Models.Pm4CorrelationGeometryInput;
using CorePm4CorrelationObjectInput = WowViewer.Core.PM4.Models.Pm4CorrelationObjectInput;
using CorePm4CorrelationObjectState = WowViewer.Core.PM4.Models.Pm4CorrelationObjectState;
using CorePm4CorrelationMath = WowViewer.Core.PM4.Services.Pm4CorrelationMath;
using CorePm4ConnectorKey = WowViewer.Core.PM4.Models.Pm4ConnectorKey;
using CorePm4ConnectorMergeCandidate = WowViewer.Core.PM4.Models.Pm4ConnectorMergeCandidate;
using CorePm4CoordinateMode = WowViewer.Core.PM4.Models.Pm4CoordinateMode;
using CorePm4GeometryLineSegment = WowViewer.Core.PM4.Models.Pm4GeometryLineSegment;
using CorePm4GeometryTriangle = WowViewer.Core.PM4.Models.Pm4GeometryTriangle;
using CorePm4LinkedPositionRefSummary = WowViewer.Core.PM4.Models.Pm4LinkedPositionRefSummary;
using CorePm4MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using CorePm4MshdGroupingService = WowViewer.Core.PM4.Services.Pm4MshdGroupingService;
using CorePm4MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using CorePm4MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using CorePm4CoordinateModeResolution = WowViewer.Core.PM4.Models.Pm4CoordinateModeResolution;
using CorePm4ObjectGroupKey = WowViewer.Core.PM4.Models.Pm4ObjectGroupKey;
using CorePm4CachedTile = WowViewer.Core.PM4.Caching.Pm4CachedTile;
using CorePm4CachedObject = WowViewer.Core.PM4.Caching.Pm4CachedObject;
using CorePm4CachedConnectorKey = WowViewer.Core.PM4.Caching.Pm4CachedConnectorKey;
using CorePm4CachedLineSegment = WowViewer.Core.PM4.Caching.Pm4CachedLineSegment;
using CorePm4CachedTriangle = WowViewer.Core.PM4.Caching.Pm4CachedTriangle;
using CorePm4PerFileCacheEntry = WowViewer.Core.PM4.Caching.Pm4PerFileCacheEntry;
using CorePm4PerFileCache = WowViewer.Core.PM4.Caching.Pm4PerFileCache;
using CorePm4PerFileCacheService = WowViewer.Core.PM4.Caching.Pm4PerFileCacheService;
using CorePm4PlacementContract = WowViewer.Core.PM4.Services.Pm4PlacementContract;
using CorePm4PlacementMath = WowViewer.Core.PM4.Services.Pm4PlacementMath;
using CorePm4PlacementSolution = WowViewer.Core.PM4.Models.Pm4PlacementSolution;
using Pm4PlanarTransform = WowViewer.Core.PM4.Models.Pm4PlanarTransform;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using CorePm4DecodeAuditReport = WowViewer.Core.PM4.Models.Pm4DecodeAuditReport;
using CorePm4ExplorationSnapshot = WowViewer.Core.PM4.Models.Pm4ExplorationSnapshot;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using CorePm4ObjectHypothesis = WowViewer.Core.PM4.Models.Pm4ObjectHypothesis;
using MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using Pm4VersionFormatter = WowViewer.Core.PM4.Services.Pm4VersionFormatter;
using MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using Pm4File = WowViewer.Core.PM4.Research.Pm4ResearchDocument;
using CorePm4ReferenceAudit = WowViewer.Core.PM4.Models.Pm4ReferenceAudit;
using CorePm4ResearchAuditAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchAuditAnalyzer;
using CorePm4ResearchHierarchyAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchHierarchyAnalyzer;
using CorePm4ResearchSnapshotBuilder = WowViewer.Core.PM4.Research.Pm4ResearchSnapshotBuilder;
using CorePm4TileObjectHypothesisReport = WowViewer.Core.PM4.Models.Pm4TileObjectHypothesisReport;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WorldFramePassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldFramePassCoordinator;
using WorldFramePassOptions = WowViewer.Core.Runtime.World.Passes.WorldFramePassOptions;
using WorldFramePasses = WowViewer.Core.Runtime.World.Passes.WorldFramePasses;
using WorldObjectPassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldObjectPassCoordinator;
using WorldObjectPassFrame = WowViewer.Core.Runtime.World.Passes.WorldObjectPassFrame;
using WorldModelBatchGate = WowViewer.Core.Runtime.World.Passes.WorldModelBatchGate;
using WorldModelRenderPath = WowViewer.Core.Runtime.World.Passes.WorldModelRenderPath;
using WorldModelSubmissionOutcome = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionOutcome;
using WorldModelSubmissionTally = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionTally;
using VisibleMdxInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleMdxEntry;
using VisibleWmoInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleWmoEntry;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.SceneGraph;
using WowViewer.Core.Runtime.World.Visibility;
using WowViewer.Core.World;
using static WoWViewer.Terrain.Pm4OverlayScene;
using static WoWViewer.Terrain.Pm4OverlayCacheCodec;
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;

namespace WoWViewer.Terrain;

/// <summary>Pure static PM4 overlay helpers moved verbatim from <c>WorldScene</c> (Epic 251 U-01 E1).</summary>
internal static class Pm4OverlayMatching
{

    internal static List<Pm4AssetProfileState> BuildPm4AssetProfileStates(IReadOnlyList<Pm4PlacementMatchState> placements)
    {
        Dictionary<string, Pm4AssetProfileState> profiles = new(StringComparer.OrdinalIgnoreCase);

        foreach (Pm4PlacementMatchState placement in placements)
        {
            for (int index = 0; index < placement.GeometryVariants.Count; index++)
            {
                Pm4PlacementGeometryVariant variant = placement.GeometryVariants[index];
                if (string.IsNullOrWhiteSpace(variant.AssetProfileKey))
                    continue;

                var profile = new Pm4AssetProfileState(
                    variant.AssetProfileKey,
                    placement.Kind,
                    placement.ModelName,
                    placement.ModelPath,
                    placement.ModelKey,
                    variant.EvidenceSource,
                    variant.CorrelatedGroupKey,
                    variant.MeshGroupCount,
                    variant.MeshVertexCount,
                    variant.MeshTriangleCount,
                    variant.FootprintSampleCount,
                    variant.ShapeSignature);

                if (!profiles.TryGetValue(variant.AssetProfileKey, out Pm4AssetProfileState existingProfile)
                    || ComparePm4AssetProfileRichness(profile, existingProfile) < 0)
                {
                    profiles[variant.AssetProfileKey] = profile;
                }
            }
        }

        return profiles.Values.ToList();
    }

    internal static Pm4ObjectMatchObject BuildPm4ObjectMatchObject(
        Pm4ObjectMatchState pm4Object,
        IReadOnlyList<Pm4PlacementMatchState> placements,
        IReadOnlyList<Pm4AssetProfileState> assetProfiles,
        int maxMatchesPerObject)
    {
        HashSet<string>? preferredAssetProfileKeys = ResolvePreferredPm4AssetProfileKeys(pm4Object, assetProfiles, maxMatchesPerObject);

        List<Pm4PlacementMatchEvaluation> evaluatedCandidates = placements
            .Where(placement => Math.Abs(placement.TileX - pm4Object.TileX) <= 1
                && Math.Abs(placement.TileY - pm4Object.TileY) <= 1)
            .Select(placement => EvaluatePm4PlacementMatch(pm4Object, placement, preferredAssetProfileKeys))
            .Where(static candidate => candidate.HasValue)
            .Select(static candidate => candidate!.Value)
            .ToList();

        if (evaluatedCandidates.Count == 0)
        {
            evaluatedCandidates = placements
                .Select(placement => EvaluatePm4PlacementMatch(pm4Object, placement, preferredAssetProfileKeys))
                .Where(static candidate => candidate.HasValue)
                .Select(static candidate => candidate!.Value)
                .ToList();
        }

        if (evaluatedCandidates.Count == 0)
        {
            evaluatedCandidates = placements
                .Where(placement => Math.Abs(placement.TileX - pm4Object.TileX) <= 1
                    && Math.Abs(placement.TileY - pm4Object.TileY) <= 1)
                .Select(placement => EvaluatePm4PlacementMatch(pm4Object, placement, null))
                .Where(static candidate => candidate.HasValue)
                .Select(static candidate => candidate!.Value)
                .ToList();
        }

        if (evaluatedCandidates.Count == 0)
        {
            evaluatedCandidates = placements
                .Select(placement => EvaluatePm4PlacementMatch(pm4Object, placement, null))
                .Where(static candidate => candidate.HasValue)
                .Select(static candidate => candidate!.Value)
                .ToList();
        }

        List<Pm4PlacementMatchEvaluation> rankedCandidates = evaluatedCandidates
            .OrderBy(candidate => new CorePm4CorrelationCandidateScore(
                    candidate.Placement.SameTile(pm4Object.TileX, pm4Object.TileY),
                    candidate.Metrics,
                    candidate.Placement.WorldBoundsMin,
                    candidate.Placement.WorldBoundsMax,
                    candidate.Placement.Center),
                Comparer<CorePm4CorrelationCandidateScore>.Create(CorePm4CorrelationMath.CompareCandidateScores))
            .ThenBy(candidate => pm4Object.Object.LinkedPositionRefCount > 0 ? candidate.AnchorPlanarGap : float.MaxValue)
            .ThenBy(candidate => GetPm4ObjectMatchEvidenceRank(pm4Object, candidate.Placement))
            .ToList();

        int nearCandidateCount = rankedCandidates.Count(static candidate =>
            candidate.Metrics.PlanarOverlapRatio > 0f
            || candidate.Metrics.VolumeOverlapRatio > 0f
            || candidate.AnchorPlanarGap <= 64f
            || (candidate.Metrics.PlanarGap <= 32f && candidate.Metrics.VerticalGap <= 96f));

        List<Pm4ObjectMatchCandidate> candidates = rankedCandidates
            .Take(maxMatchesPerObject)
            .Select(candidate => new Pm4ObjectMatchCandidate(
                candidate.Placement.TileX,
                candidate.Placement.TileY,
                candidate.Placement.Kind,
                candidate.Placement.UniqueId,
                candidate.Placement.ModelName,
                candidate.Placement.ModelPath,
                candidate.Placement.ModelKey,
                candidate.Placement.SameTile(pm4Object.TileX, pm4Object.TileY),
                candidate.Placement.AssetResolved,
                candidate.Placement.EvidenceSource,
                candidate.Placement.PlacementFlags,
                candidate.Placement.PlacementPosition,
                candidate.Placement.PlacementRotation,
                candidate.Placement.PlacementScale,
                candidate.AnchorPlanarGap,
                candidate.Metrics.PlanarGap,
                candidate.Metrics.VerticalGap,
                candidate.Metrics.CenterDistance,
                candidate.Metrics.PlanarOverlapRatio,
                candidate.Metrics.VolumeOverlapRatio,
                candidate.Metrics.FootprintOverlapRatio,
                candidate.Metrics.FootprintAreaRatio,
                candidate.Metrics.FootprintDistance,
                candidate.Placement.WorldBoundsMin,
                candidate.Placement.WorldBoundsMax,
                candidate.Placement.Center,
                candidate.Placement.MeshGroupCount,
                candidate.Placement.MeshVertexCount,
                candidate.Placement.MeshTriangleCount,
                candidate.Placement.FootprintSampleCount,
                candidate.Placement.WorldFootprintArea))
            .ToList();

        return new Pm4ObjectMatchObject(
            pm4Object.TileX,
            pm4Object.TileY,
            pm4Object.Object.Ck24,
            pm4Object.Object.Ck24Type,
            pm4Object.Object.Ck24ObjectId,
            pm4Object.Object.ObjectPartId,
            pm4Object.Object.LinkGroupObjectId,
            pm4Object.Object.SurfaceCount,
            pm4Object.Object.LinkedPositionRefCount,
            pm4Object.Object.DominantGroupKey,
            pm4Object.Object.DominantAttributeMask,
            pm4Object.Object.DominantMscnRefIndex,
            pm4Object.Object.AverageSurfaceHeight,
            pm4Object.Object.LinkedPositionRefSummary,
            pm4Object.PlacementAnchor,
            pm4Object.BoundsMin,
            pm4Object.BoundsMax,
            pm4Object.Center,
            rankedCandidates.Count,
            nearCandidateCount,
            rankedCandidates.Count(candidate => candidate.Placement.Kind == "wmo"),
            rankedCandidates.Count(candidate => candidate.Placement.Kind == "m2"),
            candidates);
    }

    internal static HashSet<string>? ResolvePreferredPm4AssetProfileKeys(
        Pm4ObjectMatchState pm4Object,
        IReadOnlyList<Pm4AssetProfileState> assetProfiles,
        int maxMatchesPerObject)
    {
        if (assetProfiles.Count == 0)
            return null;

        int shortlistSize = Math.Clamp(maxMatchesPerObject * 6, 12, 48);
        List<Pm4AssetProfileMatchEvaluation> rankedProfiles = assetProfiles
            .Select(profile => new Pm4AssetProfileMatchEvaluation(profile, EvaluatePm4AssetProfileMetrics(pm4Object, profile)))
            .OrderBy(evaluation => evaluation, Comparer<Pm4AssetProfileMatchEvaluation>.Create((left, right) => ComparePm4AssetProfiles(pm4Object, left, right)))
            .Take(shortlistSize)
            .ToList();

        if (rankedProfiles.Count == 0)
            return null;

        HashSet<string> preferredKeys = new(StringComparer.OrdinalIgnoreCase);
        for (int index = 0; index < rankedProfiles.Count; index++)
            preferredKeys.Add(rankedProfiles[index].Profile.AssetProfileKey);

        return preferredKeys;
    }

    internal static CorePm4CorrelationMetrics EvaluatePm4AssetProfileMetrics(Pm4ObjectMatchState pm4Object, Pm4AssetProfileState profile)
    {
        return CorePm4CorrelationMath.EvaluateMetrics(
            pm4Object.ShapeSignature.BoundsMin,
            pm4Object.ShapeSignature.BoundsMax,
            Vector3.Zero,
            pm4Object.ShapeSignature.FootprintHull,
            pm4Object.ShapeSignature.FootprintArea,
            profile.ShapeSignature.BoundsMin,
            profile.ShapeSignature.BoundsMax,
            Vector3.Zero,
            profile.ShapeSignature.FootprintHull,
            profile.ShapeSignature.FootprintArea);
    }

    internal static int ComparePm4AssetProfiles(
        Pm4ObjectMatchState pm4Object,
        Pm4AssetProfileMatchEvaluation left,
        Pm4AssetProfileMatchEvaluation right)
    {
        bool leftGroupMatch = left.Profile.CorrelatedGroupKey.HasValue && left.Profile.CorrelatedGroupKey.Value == pm4Object.Object.DominantGroupKey;
        bool rightGroupMatch = right.Profile.CorrelatedGroupKey.HasValue && right.Profile.CorrelatedGroupKey.Value == pm4Object.Object.DominantGroupKey;
        int compareGroupMatch = rightGroupMatch.CompareTo(leftGroupMatch);
        if (compareGroupMatch != 0)
            return compareGroupMatch;

        int compareScore = CorePm4CorrelationMath.CompareCandidateScores(
            new CorePm4CorrelationCandidateScore(false, left.Metrics, left.Profile.ShapeSignature.BoundsMin, left.Profile.ShapeSignature.BoundsMax, Vector3.Zero),
            new CorePm4CorrelationCandidateScore(false, right.Metrics, right.Profile.ShapeSignature.BoundsMin, right.Profile.ShapeSignature.BoundsMax, Vector3.Zero));
        if (compareScore != 0)
            return compareScore;

        int compareEvidence = GetPlacementGeometryEvidenceRank(left.Profile.EvidenceSource).CompareTo(GetPlacementGeometryEvidenceRank(right.Profile.EvidenceSource));
        if (compareEvidence != 0)
            return compareEvidence;

        return right.Profile.MeshTriangleCount.CompareTo(left.Profile.MeshTriangleCount);
    }

    internal static int ComparePm4AssetProfileRichness(Pm4AssetProfileState left, Pm4AssetProfileState right)
    {
        int compareEvidence = GetPlacementGeometryEvidenceRank(left.EvidenceSource).CompareTo(GetPlacementGeometryEvidenceRank(right.EvidenceSource));
        if (compareEvidence != 0)
            return compareEvidence;

        int compareTriangles = right.MeshTriangleCount.CompareTo(left.MeshTriangleCount);
        if (compareTriangles != 0)
            return compareTriangles;

        return right.FootprintSampleCount.CompareTo(left.FootprintSampleCount);
    }

    internal static Pm4PlacementMatchEvaluation? EvaluatePm4PlacementMatch(
        Pm4ObjectMatchState pm4Object,
        Pm4PlacementMatchState placement,
        ISet<string>? preferredAssetProfileKeys)
    {
        if (!TryResolveBestPlacementGeometryVariant(pm4Object, placement, preferredAssetProfileKeys, out Pm4PlacementMatchState effectivePlacement, out CorePm4CorrelationMetrics metrics))
            return null;

        float anchorPlanarGap = ComputePm4ObjectAnchorPlanarGap(pm4Object.PlacementAnchor, effectivePlacement.PlacementPosition);
        return new Pm4PlacementMatchEvaluation(effectivePlacement, anchorPlanarGap, metrics);
    }

    internal static bool TryResolveBestPlacementGeometryVariant(
        Pm4ObjectMatchState pm4Object,
        Pm4PlacementMatchState placement,
        ISet<string>? preferredAssetProfileKeys,
        out Pm4PlacementMatchState resolvedPlacement,
        out CorePm4CorrelationMetrics metrics)
    {
        IReadOnlyList<Pm4PlacementGeometryVariant> variants = placement.GeometryVariants;
        List<Pm4PlacementGeometryVariant>? filteredVariants = null;
        if (preferredAssetProfileKeys != null)
        {
            filteredVariants = variants
                .Where(variant => preferredAssetProfileKeys.Contains(variant.AssetProfileKey))
                .ToList();
            if (filteredVariants.Count > 0)
                variants = filteredVariants;
        }

        if (variants.Count == 0)
        {
            metrics = CorePm4CorrelationMath.EvaluateMetrics(
                pm4Object.BoundsMin,
                pm4Object.BoundsMax,
                pm4Object.Center,
                pm4Object.FootprintHull,
                pm4Object.FootprintArea,
                placement.WorldBoundsMin,
                placement.WorldBoundsMax,
                placement.Center,
                placement.FootprintHull,
                placement.FootprintArea);
            resolvedPlacement = placement;
            return preferredAssetProfileKeys == null;
        }

        bool sameTile = placement.SameTile(pm4Object.TileX, pm4Object.TileY);
        Pm4PlacementGeometryVariant bestVariant = variants[0];
        CorePm4CorrelationMetrics bestMetrics = EvaluatePlacementVariantMetrics(pm4Object, bestVariant);

        for (int index = 1; index < variants.Count; index++)
        {
            Pm4PlacementGeometryVariant candidateVariant = variants[index];
            CorePm4CorrelationMetrics candidateMetrics = EvaluatePlacementVariantMetrics(pm4Object, candidateVariant);
            if (ComparePlacementGeometryVariants(pm4Object, sameTile, candidateVariant, candidateMetrics, bestVariant, bestMetrics) < 0)
            {
                bestVariant = candidateVariant;
                bestMetrics = candidateMetrics;
            }
        }

        metrics = bestMetrics;
        resolvedPlacement = placement with
        {
            AssetProfileKey = bestVariant.AssetProfileKey,
            EvidenceSource = bestVariant.EvidenceSource,
            WorldBoundsMin = bestVariant.WorldBoundsMin,
            WorldBoundsMax = bestVariant.WorldBoundsMax,
            FootprintHull = bestVariant.FootprintHull,
            FootprintArea = bestVariant.FootprintArea,
            MeshGroupCount = bestVariant.MeshGroupCount,
            MeshVertexCount = bestVariant.MeshVertexCount,
            MeshTriangleCount = bestVariant.MeshTriangleCount,
            FootprintSampleCount = bestVariant.FootprintSampleCount,
            WorldFootprintArea = bestVariant.WorldFootprintArea
        };
        return true;
    }

    internal static Pm4ShapeSignature BuildPm4ShapeSignature(Vector3 boundsMin, Vector3 boundsMax, IReadOnlyList<Vector2> footprintHull)
    {
        Vector2[] resolvedFootprintHull = footprintHull.Count > 0
            ? footprintHull.ToArray()
            : BuildPm4BoundsFootprintHull(boundsMin, boundsMax);
        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        float scale = MathF.Max(MathF.Max(boundsMax.X - boundsMin.X, boundsMax.Y - boundsMin.Y), boundsMax.Z - boundsMin.Z);
        if (!float.IsFinite(scale) || scale <= 0.001f)
            scale = 1f;

        Vector3 normalizedBoundsMin = (boundsMin - center) / scale;
        Vector3 normalizedBoundsMax = (boundsMax - center) / scale;
        Vector2 planarCenter = new(center.X, center.Y);
        Vector2[] normalizedFootprintHull = new Vector2[resolvedFootprintHull.Length];
        for (int index = 0; index < resolvedFootprintHull.Length; index++)
            normalizedFootprintHull[index] = (resolvedFootprintHull[index] - planarCenter) / scale;

        float normalizedFootprintArea = CorePm4CorrelationMath.ComputeFootprintArea(normalizedFootprintHull);
        return new Pm4ShapeSignature(normalizedBoundsMin, normalizedBoundsMax, normalizedFootprintHull, normalizedFootprintArea);
    }

    internal static string BuildPm4AssetProfileKey(string kind, string modelKey, string evidenceSource, byte? correlatedGroupKey)
    {
        string groupKey = correlatedGroupKey.HasValue
            ? correlatedGroupKey.Value.ToString(CultureInfo.InvariantCulture)
            : "-";
        return $"{kind}|{modelKey}|{evidenceSource}|{groupKey}";
    }

    internal static CorePm4CorrelationMetrics EvaluatePlacementVariantMetrics(Pm4ObjectMatchState pm4Object, Pm4PlacementGeometryVariant variant)
    {
        return CorePm4CorrelationMath.EvaluateMetrics(
            pm4Object.BoundsMin,
            pm4Object.BoundsMax,
            pm4Object.Center,
            pm4Object.FootprintHull,
            pm4Object.FootprintArea,
            variant.WorldBoundsMin,
            variant.WorldBoundsMax,
            variant.Center,
            variant.FootprintHull,
            variant.FootprintArea);
    }

    internal static int ComparePlacementGeometryVariants(
        Pm4ObjectMatchState pm4Object,
        bool sameTile,
        Pm4PlacementGeometryVariant leftVariant,
        CorePm4CorrelationMetrics leftMetrics,
        Pm4PlacementGeometryVariant rightVariant,
        CorePm4CorrelationMetrics rightMetrics)
    {
        bool leftGroupMatch = leftVariant.CorrelatedGroupKey.HasValue && leftVariant.CorrelatedGroupKey.Value == pm4Object.Object.DominantGroupKey;
        bool rightGroupMatch = rightVariant.CorrelatedGroupKey.HasValue && rightVariant.CorrelatedGroupKey.Value == pm4Object.Object.DominantGroupKey;
        int compareGroupMatch = rightGroupMatch.CompareTo(leftGroupMatch);
        if (compareGroupMatch != 0)
            return compareGroupMatch;

        int compareScore = CorePm4CorrelationMath.CompareCandidateScores(
            new CorePm4CorrelationCandidateScore(sameTile, leftMetrics, leftVariant.WorldBoundsMin, leftVariant.WorldBoundsMax, leftVariant.Center),
            new CorePm4CorrelationCandidateScore(sameTile, rightMetrics, rightVariant.WorldBoundsMin, rightVariant.WorldBoundsMax, rightVariant.Center));
        if (compareScore != 0)
            return compareScore;

        return GetPlacementGeometryEvidenceRank(leftVariant.EvidenceSource).CompareTo(GetPlacementGeometryEvidenceRank(rightVariant.EvidenceSource));
    }

    internal static int GetPlacementGeometryEvidenceRank(string evidenceSource)
    {
        return evidenceSource.ToLowerInvariant() switch
        {
            "wmo-group-mesh" => 0,
            "mdx-collision" => 1,
            "wmo-mesh" => 2,
            "modf-bounds" => 3,
            _ => 4,
        };
    }

    internal static int GetPm4ObjectMatchEvidenceRank(Pm4ObjectMatchState pm4Object, Pm4PlacementMatchState placement)
    {
        bool zeroOrRootObject = pm4Object.Object.Ck24 == 0 || pm4Object.Object.LinkGroupObjectId == 0;
        if (zeroOrRootObject)
        {
            if (pm4Object.Object.LinkedPositionRefCount > 0)
                return string.Equals(placement.Kind, "m2", StringComparison.OrdinalIgnoreCase) ? 0 : 1;

            return 0;
        }

        if (placement.Kind == "wmo" && string.Equals(placement.EvidenceSource, "wmo-group-mesh", StringComparison.OrdinalIgnoreCase))
            return 0;

        if (string.Equals(placement.EvidenceSource, "mdx-collision", StringComparison.OrdinalIgnoreCase))
            return 1;

        if (placement.Kind == "wmo" && string.Equals(placement.EvidenceSource, "wmo-mesh", StringComparison.OrdinalIgnoreCase))
            return 2;

        if (placement.Kind == "wmo")
            return 3;

        return 4;
    }

    internal static float ComputePm4ObjectAnchorPlanarGap(Vector3 anchor, Vector3 placementPosition)
    {
        if (!float.IsFinite(anchor.X) || !float.IsFinite(anchor.Y) || !float.IsFinite(placementPosition.X) || !float.IsFinite(placementPosition.Y))
            return float.MaxValue;

        return Vector2.Distance(new Vector2(anchor.X, anchor.Y), new Vector2(placementPosition.X, placementPosition.Y));
    }

    internal static Vector2[] BuildPm4BoundsFootprintHull(Vector3 boundsMin, Vector3 boundsMax)
    {
        return
        [
            new Vector2(boundsMin.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMax.Y),
            new Vector2(boundsMin.X, boundsMax.Y),
        ];
    }
}
