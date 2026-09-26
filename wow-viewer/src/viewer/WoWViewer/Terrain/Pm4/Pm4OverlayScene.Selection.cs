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
using static WoWViewer.Terrain.Pm4OverlayMatching;
using static WoWViewer.Terrain.Pm4OverlayCacheCodec;
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;

namespace WoWViewer.Terrain;

// Hover, pick, selection, debug/research info, per-object transforms and render filters.
public sealed partial class Pm4OverlayScene
{

    internal bool TryBuildHoveredPm4InfoByRay(Vector3 rayOrigin, Vector3 rayDir, out HoveredAssetInfo info, out float distance)
    {
        info = default;
        distance = float.MaxValue;

        if (!TryPickPm4ObjectByRay(rayOrigin, rayDir, out var objectKey, out _, out float hitDistance) || !objectKey.HasValue)
            return false;

        if (!IsHoverPickDistanceAllowed(hitDistance))
            return false;

        if (!_pm4ObjectLookup.TryGetValue(objectKey.Value, out Pm4OverlayObject? obj))
            return false;

        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;
        Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey.Value, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
        Vector3 center = applyObjectTransform ? ApplyPm4OverlayTransform(obj.Center, objectTransform) : obj.Center;

        info = BuildHoveredPm4Info(obj, center, objectKey.Value);
        distance = hitDistance;
        return true;
    }

    public bool SelectPm4ObjectByRay(Vector3 rayOrigin, Vector3 rayDir)
    {
        if (TryPickPm4ObjectByRay(rayOrigin, rayDir, out var bestKey, out var bestGroupKey, out _))
        {
            _selectedPm4ObjectKey = bestKey;
            _selectedPm4ObjectGroupKey = bestGroupKey;
            return true;
        }

        _selectedPm4ObjectKey = null;
        _selectedPm4ObjectGroupKey = null;
        return false;
    }

    public bool TryPickPm4ObjectByRay(
        Vector3 rayOrigin,
        Vector3 rayDir,
        out (int tileX, int tileY, uint ck24, int objectPart)? objectKey,
        out (int tileX, int tileY, uint ck24)? objectGroupKey,
        out float distance)
    {
        objectKey = null;
        objectGroupKey = null;
        distance = float.MaxValue;

        bool profile = Pm4Profiling.Enabled;
        System.Diagnostics.Stopwatch sw = profile ? s_pm4PickSw : null;
        long beforeTicks = profile ? sw.ElapsedTicks : 0;
        int aabbHits = 0;

        if (!_showPm4Overlay || _pm4TileObjects.Count == 0)
            return false;

        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;
        Vector3 padding = new(2f, 2f, 2f);
        float bestT = float.MaxValue;

        // Single pass: test every object's AABB directly (simple and reliable)
        foreach (var (tileKey, objects) in _pm4TileObjects)
        {
            if (!ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                continue;

            foreach (Pm4OverlayObject obj in objects)
            {
                if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                    continue;

                var candidateKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                Matrix4x4 objectTransform = BuildPm4ObjectTransform(candidateKey, applyPm4Transform, pm4Transform, out bool applyObjTransform);

                Vector3 bmin = obj.BoundsMin, bmax = obj.BoundsMax;
                if (applyObjTransform)
                    TransformBounds(bmin, bmax, objectTransform, out bmin, out bmax);

                float t = RayAABBIntersect(rayOrigin, rayDir, bmin - padding, bmax + padding);
                if (t >= 0f)
                    aabbHits++;
                if (t >= 0f && t < bestT && IsHoverPickDistanceAllowed(t))
                {
                    bestT = t;
                    objectKey = candidateKey;
                    objectGroupKey = ResolvePm4ObjectGroupKey(candidateKey);
                }
            }
        }

        distance = bestT;
        bool hit = objectKey.HasValue;

        if (profile)
        {
            long afterTicks = sw.ElapsedTicks;
            double elapsedMs = (afterTicks - beforeTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            s_pm4PickCallCount++;
            s_pm4PickAabbHitCount += aabbHits;
            s_pm4PickTotalMs += elapsedMs;
            if (elapsedMs > s_pm4PickMaxMs) s_pm4PickMaxMs = elapsedMs;
            s_pm4PickReportCount++;
            if (elapsedMs >= 50.0 || s_pm4PickReportCount >= 200)
            {
                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"[PM4-PROFILE] TryPickPm4ObjectByRay: call={s_pm4PickCallCount} last={elapsedMs:0.0}ms max={s_pm4PickMaxMs:0.0}ms avg={s_pm4PickTotalMs / s_pm4PickCallCount:0.0}ms aabbHits(last)={aabbHits} totalHits={s_pm4PickAabbHitCount} hit={hit}");
                s_pm4PickReportCount = 0;
            }
        }

        return hit;
    }

    public void ClearPm4ObjectSelection()
    {
        _selectedPm4ObjectKey = null;
        _selectedPm4ObjectGroupKey = null;
    }

    public bool SelectPm4ObjectGroupKey(uint regionId, ushort ck24ObjectId)
    {
        foreach (var kv in _pm4ObjectLookup)
        {
            var (tx, ty, ck24, part) = kv.Key;
            if (kv.Value.MshdRegionId == regionId && (ushort)(ck24 & 0xFFFF) == ck24ObjectId)
            {
                var key = (tx, ty, ck24, part);
                _selectedPm4ObjectKey = key;
                _selectedPm4ObjectGroupKey = ResolvePm4ObjectGroupKey(key);
                return true;
            }
        }
        return false;
    }

    public bool SelectPm4Object((int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        if (!_pm4ObjectLookup.ContainsKey(objectKey))
            return false;

        _selectedPm4ObjectKey = objectKey;
        _selectedPm4ObjectGroupKey = ResolvePm4ObjectGroupKey(objectKey);
        return true;
    }

    public bool TryGetPm4ObjectGroupKey(
        (int tileX, int tileY, uint ck24, int objectPart) objectKey,
        out (int tileX, int tileY, uint ck24) groupKey)
    {
        if (!_pm4ObjectLookup.ContainsKey(objectKey))
        {
            groupKey = default;
            return false;
        }

        groupKey = ResolvePm4ObjectGroupKey(objectKey);
        return true;
    }

    public void SetHighlightedPm4Objects(IEnumerable<(int tileX, int tileY, uint ck24, int objectPart)> objectKeys)
    {
        _highlightedPm4ObjectKeys.Clear();
        foreach (var objectKey in objectKeys)
        {
            if (_pm4ObjectLookup.ContainsKey(objectKey))
                _highlightedPm4ObjectKeys.Add(objectKey);
        }
    }

    public bool TryGetPm4ObjectDebugInfo((int tileX, int tileY, uint ck24, int objectPart) objectKey, out Pm4ObjectDebugInfo info)
    {
        info = default;
        if (!_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
            return false;

        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;
        Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);

        Vector3 center = applyObjectTransform ? ApplyPm4OverlayTransform(obj.Center, objectTransform) : obj.Center;
        Vector3 boundsMin = obj.BoundsMin;
        Vector3 boundsMax = obj.BoundsMax;
        if (applyObjectTransform)
            TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

        float nearestPositionRefDistance = float.NaN;
        if (_pm4TilePositionRefs.TryGetValue((objectKey.tileX, objectKey.tileY), out List<Vector3>? positionRefs)
            && positionRefs.Count > 0)
        {
            nearestPositionRefDistance = NearestPointDistance(center, positionRefs, applyPm4Transform, pm4Transform);
        }

        info = new Pm4ObjectDebugInfo(
            obj.Ck24,
            obj.Ck24Type,
            obj.Ck24ObjectId,
            obj.ObjectPartId,
            obj.LinkGroupObjectId,
            obj.LinkedPositionRefCount,
            obj.LinkedPositionRefSummary,
            objectKey.tileX,
            objectKey.tileY,
            obj.MshdField00,
            obj.MshdRegionId,
            obj.MshdField08,
            obj.SurfaceCount,
            obj.DominantGroupKey,
            obj.DominantAttributeMask,
            obj.DominantMscnRefIndex,
            obj.AverageSurfaceHeight,
            boundsMin,
            boundsMax,
            center,
            nearestPositionRefDistance,
            obj.PlanarTransform.SwapPlanarAxes,
            obj.PlanarTransform.InvertU,
            obj.PlanarTransform.InvertV,
            obj.PlanarTransform.InvertsWinding,
            obj.DistinctTypeFlags);

        return true;
    }

    public bool TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo info)
    {
        info = default;
        if (!_selectedPm4ObjectKey.HasValue)
            return false;

        return TryGetPm4ObjectDebugInfo(_selectedPm4ObjectKey.Value, out info);
    }

    public bool TryGetSelectedPm4ObjectResearchInfo(out Pm4SelectedObjectResearchInfo info)
    {
        info = default;
        if (!_selectedPm4ObjectKey.HasValue)
            return false;

        bool profile = Pm4Profiling.Enabled;
        long researchStartTicks = profile ? s_pm4ResearchSw.ElapsedTicks : 0;

        var objectKey = _selectedPm4ObjectKey.Value;
        if (!_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
            return false;

        if (string.IsNullOrWhiteSpace(obj.SourcePath))
            return false;

        if (!TryGetPm4ResearchContext(obj.SourcePath, out Pm4ResearchContext? context) || context == null)
            return false;

        List<Pm4ResearchHypothesisMatch> allMatches = context.HypothesisReport.Objects
            .Where(hypothesis => hypothesis.Ck24 == obj.Ck24)
            .Select(hypothesis => new Pm4ResearchHypothesisMatch(
                hypothesis.Family,
                hypothesis.FamilyObjectIndex,
                hypothesis.SurfaceCount,
                hypothesis.TotalIndexCount,
                hypothesis.MscnRefIndices.Count,
                hypothesis.GroupKeys.Count,
                hypothesis.MslkGroupObjectIds.Count,
                hypothesis.DominantLinkGroupObjectId,
                hypothesis.MprlFootprint.LinkedRefCount,
                hypothesis.MprlFootprint.LinkedInBoundsCount,
                hypothesis.PlacementComparison.CoordinateMode,
                hypothesis.PlacementComparison.PlanarTransform,
                hypothesis.PlacementComparison.FrameYawDegrees,
                hypothesis.PlacementComparison.MprlHeadingMeanDegrees,
                hypothesis.PlacementComparison.HeadingDeltaDegrees,
                ComputePm4ResearchMatchScore(obj, hypothesis)))
            .OrderBy(match => match.SimilarityScore)
            .ThenBy(match => match.Family)
            .ThenBy(match => match.FamilyObjectIndex)
            .ToList();

        int invalidRefIndexCount = context.DecodeAudit.ReferenceAudits
            .Where(static audit => audit.Name == "MSLK.RefIndex->MSUR")
            .Select(static audit => audit.InvalidCount)
            .FirstOrDefault();

        // Extract raw MSHD header fields
        string? mshdRawFields = null;
        IReadOnlyList<string>? mslkRawEntries = null;
        if (context.RawDocument != null)
        {
            var knownMshd = context.RawDocument.KnownChunks.Mshd;
            if (knownMshd is not null)
            {
                mshdRawFields = $"MSHD: F00={knownMshd.Field00} F04={knownMshd.Field04} F08={knownMshd.Field08} F0C={knownMshd.Field0C} F10={knownMshd.Field10} F14={knownMshd.Field14} F18={knownMshd.Field18} F1C={knownMshd.Field1C}";
            }

            // Collect MSLK entries referencing the selected object's surfaces by CK24
            var mslkLines = new List<string>();
            foreach (MslkEntry mslk in context.RawDocument.KnownChunks.Mslk)
            {
                if (mslk.RefIndex >= 0 && (uint)mslk.RefIndex < (uint)context.RawDocument.KnownChunks.Msur.Count
                    && context.RawDocument.KnownChunks.Msur[mslk.RefIndex].Ck24 == obj.Ck24)
                {
                    mslkLines.Add($"MSLK: TypeFlags=0x{mslk.TypeFlags:X2} Subtype=0x{mslk.Subtype:X2} Padding=0x{mslk.Padding:X4} GroupObjectId=0x{mslk.GroupObjectId:X8} MspiFirstIndex={mslk.MspiFirstIndex} MspiIndexCount={mslk.MspiIndexCount} LinkId=0x{mslk.LinkId:X8} RefIndex={mslk.RefIndex} SystemFlag=0x{mslk.SystemFlag:X4}");
                }
            }
            if (mslkLines.Count > 0)
                mslkRawEntries = mslkLines;
        }

        info = new Pm4SelectedObjectResearchInfo(
            obj.SourcePath,
            context.Snapshot.Version,
            context.Snapshot.MslkCount,
            context.Snapshot.MsurCount,
            context.Snapshot.MscnCount,
            context.Snapshot.MprlCount,
            invalidRefIndexCount,
            context.HypothesisReport.TotalHypothesisCount,
            allMatches.Count,
            context.Snapshot.Diagnostics.Count,
            context.Snapshot.Diagnostics.Take(3).ToList(),
            allMatches.Take(8).ToList(),
            mshdRawFields,
            mslkRawEntries);

        if (profile)
        {
            long afterTicks = s_pm4ResearchSw.ElapsedTicks;
            double elapsedMs = (afterTicks - researchStartTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            s_pm4ResearchCallCount++;
            s_pm4ResearchTotalMs += elapsedMs;
            if (elapsedMs > s_pm4ResearchMaxMs) s_pm4ResearchMaxMs = elapsedMs;
            s_pm4ResearchReportCount++;
            int mslkLines = mslkRawEntries?.Count ?? 0;
            int matchesCount = allMatches.Count;
            if (elapsedMs >= 50.0 || s_pm4ResearchReportCount >= 200)
            {
                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"[PM4-PROFILE] TryGetSelectedPm4ObjectResearchInfo: call={s_pm4ResearchCallCount} last={elapsedMs:0.0}ms max={s_pm4ResearchMaxMs:0.0}ms avg={s_pm4ResearchTotalMs / s_pm4ResearchCallCount:0.0}ms matches={matchesCount} mslkLines={mslkLines} mslkTotal={context.RawDocument?.KnownChunks.Mslk.Count ?? 0}");
                s_pm4ResearchReportCount = 0;
            }
        }

        return true;
    }

    /// <summary>
    /// Computes MSLK linking statistics across all loaded PM4 research contexts.
    /// Exposed as a plain record so the viewer never needs the internal context type.
    /// </summary>
    public Pm4MslkLinkingStats GetPm4MslkLinkingStats()
    {
        int totalFiles = 0;
        int totalMslkEntries = 0;
        int anchorOnlyLinks = 0;
        int pathWindowLinks = 0;
        int totalComponents = 0;
        int componentsWithLinks = 0;
        int componentsWithoutLinks = 0;
        int refIndexMismatches = 0;

        foreach ((string _, Pm4ResearchContext context) in _pm4ResearchBySourcePath)
        {
            if (context.RawDocument == null)
                continue;

            totalFiles++;
            var chunks = context.RawDocument.KnownChunks;
            totalMslkEntries += chunks.Mslk.Count;

            foreach (var link in chunks.Mslk)
            {
                if (link.MspiFirstIndex < 0)
                    anchorOnlyLinks++;
                else
                    pathWindowLinks++;
            }

            var linksBySurface = new Dictionary<int, List<MslkEntry>>();
            foreach (var link in chunks.Mslk)
            {
                if (link.RefIndex >= 0 && link.RefIndex < chunks.Msur.Count)
                {
                    if (!linksBySurface.TryGetValue(link.RefIndex, out var bucket))
                        linksBySurface[link.RefIndex] = bucket = new List<MslkEntry>();
                    bucket.Add(link);
                }
                else
                {
                    refIndexMismatches++;
                }
            }

            var surfacesByCk24 = new Dictionary<uint, List<int>>();
            for (int i = 0; i < chunks.Msur.Count; i++)
            {
                uint ck24 = chunks.Msur[i].Ck24;
                if (!surfacesByCk24.TryGetValue(ck24, out var bucket))
                    surfacesByCk24[ck24] = bucket = new List<int>();
                bucket.Add(i);
            }

            foreach ((uint _, List<int> surfaceIndices) in surfacesByCk24)
            {
                totalComponents++;
                bool hasLink = false;
                foreach (int si in surfaceIndices)
                {
                    if (linksBySurface.ContainsKey(si))
                    {
                        hasLink = true;
                        break;
                    }
                }
                if (hasLink) componentsWithLinks++;
                else componentsWithoutLinks++;
            }
        }

        return new Pm4MslkLinkingStats(
            totalFiles,
            totalMslkEntries,
            anchorOnlyLinks,
            pathWindowLinks,
            totalComponents,
            componentsWithLinks,
            componentsWithoutLinks,
            refIndexMismatches);
    }

    private bool TryGetPm4ResearchContext(string sourcePath, out Pm4ResearchContext? context)
    {
        if (_pm4ResearchBySourcePath.TryGetValue(sourcePath, out context))
            return true;

        if (_pm4ResearchUnavailablePaths.Contains(sourcePath) || _dataSource == null)
        {
            context = null;
            return false;
        }

        byte[]? bytes = _dataSource.ReadFile(sourcePath);
        if (bytes == null || bytes.Length == 0)
        {
            _pm4ResearchUnavailablePaths.Add(sourcePath);
            context = null;
            return false;
        }

        try
        {
            Pm4File researchFile = CorePm4DocumentReader.Read(bytes, sourcePath);
            context = new Pm4ResearchContext(
                sourcePath,
                CorePm4ResearchSnapshotBuilder.CreateSnapshot(researchFile),
                CorePm4ResearchAuditAnalyzer.Analyze(researchFile),
                CorePm4ResearchHierarchyAnalyzer.Analyze(researchFile),
                researchFile);
            _pm4ResearchBySourcePath[sourcePath] = context;
            return true;
        }
        catch (Exception ex)
        {
            _pm4ResearchUnavailablePaths.Add(sourcePath);
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4 Research] Failed to analyze '{sourcePath}': {ex.Message}");
            context = null;
            return false;
        }
    }

    private byte[]? ReadPm4FileForTile((int tileX, int tileY) tileKey)
    {
        if (_dataSource == null)
            return null;

        // Collect unique source paths for objects on this tile
        var seenPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (objectKey, obj) in _pm4ObjectLookup)
        {
            if (objectKey.tileX == tileKey.tileX && objectKey.tileY == tileKey.tileY
                && !string.IsNullOrWhiteSpace(obj.SourcePath)
                && seenPaths.Add(obj.SourcePath))
            {
                byte[]? bytes = _dataSource.ReadFile(obj.SourcePath);
                if (bytes != null && bytes.Length > 0)
                    return bytes;
            }
        }

        return null;
    }

    /// <summary>
    /// Lazily populate <see cref="_pm4TileMscnPoints"/> from the staged PM4 files.
    /// MSCN = scene-graph connector anchors. One Vector3 per MSUR surface (placed via MSUR.MscnRefIndex).
    /// </summary>
    internal void EnsurePm4MscnData()
    {
        if (_pm4TileObjects.Count == 0)
            return;
        foreach (var tileKey in _pm4TileObjects.Keys.ToList())
        {
            if (_pm4TileMscnPoints.ContainsKey(tileKey))
                continue;
            var bytes = ReadPm4FileForTile(tileKey);
            if (bytes == null) continue;
            var pm4 = CorePm4DocumentReader.Read(bytes, $"tile_{tileKey.tileX}_{tileKey.tileY}.pm4");
            if (pm4.KnownChunks.Mscn.Count == 0) continue;
            var pts = new List<Vector3>(pm4.KnownChunks.Mscn.Count);
            foreach (var p in pm4.KnownChunks.Mscn)
                pts.Add(new Vector3(WoWConstants.MapOrigin - p.X, WoWConstants.MapOrigin - p.Y, p.Z));
            _pm4TileMscnPoints[tileKey] = pts;
        }
    }

    /// <summary>
    /// Lazily populate <see cref="_pm4TileMspvPoints"/> from the staged PM4 files.
    /// MSPV = path-vertex positions reached via MSPI from MSLK link records. Only present when surfaces are connected.
    /// </summary>
    internal void EnsurePm4MspvData()
    {
        if (_pm4TileObjects.Count == 0)
            return;
        foreach (var tileKey in _pm4TileObjects.Keys.ToList())
        {
            if (_pm4TileMspvPoints.ContainsKey(tileKey))
                continue;
            var bytes = ReadPm4FileForTile(tileKey);
            if (bytes == null) continue;
            var pm4 = CorePm4DocumentReader.Read(bytes, $"tile_{tileKey.tileX}_{tileKey.tileY}.pm4");
            if (pm4.KnownChunks.Mspv.Count == 0) continue;
            var pts = new List<Vector3>(pm4.KnownChunks.Mspv.Count);
            foreach (var p in pm4.KnownChunks.Mspv)
                pts.Add(new Vector3(WoWConstants.MapOrigin - p.X, WoWConstants.MapOrigin - p.Y, p.Z));
            _pm4TileMspvPoints[tileKey] = pts;
        }
    }

    private static float ComputePm4ResearchMatchScore(Pm4OverlayObject obj, CorePm4ObjectHypothesis hypothesis)
    {
        float score = 0f;
        score += Math.Abs(hypothesis.SurfaceCount - obj.SurfaceCount) * 3f;
        score += Math.Abs(hypothesis.TotalIndexCount - obj.TotalIndexCount) * 0.125f;
        score += Math.Abs(hypothesis.MprlFootprint.LinkedRefCount - obj.LinkedPositionRefCount) * 4f;

        if (obj.LinkGroupObjectId != 0)
        {
            bool hasExactGroupObjectId = hypothesis.MslkGroupObjectIds.Contains(obj.LinkGroupObjectId);
            score += hasExactGroupObjectId ? -8f : 8f;
            if (hypothesis.DominantLinkGroupObjectId == obj.LinkGroupObjectId)
                score -= 4f;
        }

        return Math.Max(0f, score);
    }

    private static float NearestPointDistance(Vector3 point, IReadOnlyList<Vector3> candidates, bool applyPm4Transform, in Matrix4x4 pm4Transform)
    {
        float best = float.MaxValue;
        for (int i = 0; i < candidates.Count; i++)
        {
            Vector3 candidate = applyPm4Transform ? ApplyPm4OverlayTransform(candidates[i], pm4Transform) : candidates[i];
            float dist = Vector3.Distance(point, candidate);
            if (dist < best)
                best = dist;
        }

        return best;
    }

    private bool TryResolvePm4Asset(
        uint ck24,
        Vector3 boundsMin,
        Vector3 boundsMax,
        out string? assetName,
        out int uniqueId,
        out float placementZ,
        float zTolerance = 0.05f)
    {
        placementZ = BitConverter.UInt32BitsToSingle(ck24 << 8);
        assetName = null;
        uniqueId = 0;
        if (ck24 == 0)
            return false;

        float best = float.MaxValue;
        foreach (ObjectInstance inst in _wmoInstances)
        {
            Vector3 p = inst.PlacementPosition;
            if (p.X < boundsMin.X - 1f || p.X > boundsMax.X + 1f
                || p.Y < boundsMin.Y - 1f || p.Y > boundsMax.Y + 1f)
            {
                continue;
            }

            float delta = MathF.Abs(placementZ - p.Z);
            if (delta < best)
            {
                best = delta;
                assetName = string.IsNullOrEmpty(inst.ModelName) ? inst.ModelPath : inst.ModelName;
                uniqueId = inst.UniqueId;
            }
        }

        if (best <= zTolerance)
            return true;

        // Nothing in the scene placed this object, which on a tile whose ADT is gone is the normal
        // case rather than a failure. Fall back to the recovered-name side-car if one is present.
        // The name is a guess and is marked as one by the caller; a unique id is not invented.
        Pm4GeneratedPlacements.EnsureLoaded();
        if (Pm4GeneratedPlacements.TryResolve(ck24, boundsMin, boundsMax, out string? inferred, out double inferredScore))
        {
            assetName = $"{inferred} (inferred {inferredScore:F2})";
            uniqueId = 0;
            return true;
        }

        assetName = null;
        uniqueId = 0;
        return false;
    }

    private HoveredAssetInfo BuildHoveredPm4Info(Pm4OverlayObject obj, Vector3 worldPosition, (int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        bool resolved = TryResolvePm4Asset(
            obj.Ck24, obj.BoundsMin, obj.BoundsMax,
            out string? assetName, out int uniqueId, out float placementZ);

        // Identify the object by the placement that produced it where possible, and by its placement
        // height otherwise. The old label led with the raw 24-bit slice and a viewer-generated part
        // number, neither of which names anything: the slice is the top three bytes of a float and
        // the part id is an artefact of how the current overlay split the tile.
        // A hover tooltip answers "what is this", nothing more. Evidence and provenance belong in
        // the inspect panel, and a '%' in any of these strings would be eaten by ImGui's printf
        // formatting, so keep them short and symbol-free.
        string title;
        string detail;
        if (obj.Ck24 == 0)
        {
            title = "PM4 doodad collision";
            detail = $"tile ({objectKey.tileX}, {objectKey.tileY})   {obj.SurfaceCount} surfaces";
        }
        else if (resolved)
        {
            title = $"{System.IO.Path.GetFileName(assetName)}  #{uniqueId}";
            detail = $"tile ({objectKey.tileX}, {objectKey.tileY})   {obj.SurfaceCount} surfaces   Z {placementZ:F1}";
        }
        else
        {
            title = $"PM4 object   Z {placementZ:F1}";
            detail = $"tile ({objectKey.tileX}, {objectKey.tileY})   {obj.SurfaceCount} surfaces";
        }

        return new HoveredAssetInfo(
            "PM4",
            title,
            obj.SourcePath,
            detail,
            worldPosition,
            0,
                objectKey,
                ObjectType.None,
                -1,
                null);
    }

    internal bool TryBuildHoveredPm4Info(
        Matrix4x4 view,
        Matrix4x4 proj,
        float mouseViewportX,
        float mouseViewportY,
        float viewportWidth,
        float viewportHeight,
        out HoveredAssetInfo info,
        out int hitCount,
        out float bestDistanceSq,
        out float bestDepth)
    {
        info = default;
        hitCount = 0;
        bestDistanceSq = float.MaxValue;
        bestDepth = float.MaxValue;
        HoveredAssetInfo? bestInfo = null;
        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;

        foreach (KeyValuePair<(int tileX, int tileY), List<Pm4OverlayObject>> tileEntry in _pm4TileObjects)
        {
            List<Pm4OverlayObject> objects = tileEntry.Value;
            for (int i = 0; i < objects.Count; i++)
            {
                Pm4OverlayObject obj = objects[i];
                if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                    continue;

                var objectKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId);
                Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);

                Vector3 boundsMin = obj.BoundsMin;
                Vector3 boundsMax = obj.BoundsMax;
                Vector3 center = obj.Center;
                if (applyObjectTransform)
                {
                    TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);
                    center = ApplyPm4OverlayTransform(obj.Center, objectTransform);
                }

                if (!TryMeasureHoverInfoHit(boundsMin, boundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                if (!IsHoverPickPositionAllowed(center))
                    continue;

                hitCount++;
                const float distanceEpsilon = 0.01f;
                if (!bestInfo.HasValue
                    || distanceSq < bestDistanceSq - distanceEpsilon
                    || (MathF.Abs(distanceSq - bestDistanceSq) <= distanceEpsilon && depth < bestDepth))
                {
                    bestDistanceSq = distanceSq;
                    bestDepth = depth;
                    bestInfo = BuildHoveredPm4Info(obj, center, objectKey);
                }
            }
        }

        if (!bestInfo.HasValue)
            return false;

        info = bestInfo.Value;
        return true;
    }

    internal bool TryBuildPm4ObjectMatch((int tileX, int tileY, uint ck24, int objectPart) objectKey, int maxMatchesPerObject, out Pm4ObjectMatchObject objectMatch)
    {
        objectMatch = null!;

        EnsurePm4OverlayMatchesCameraWindow(GetPm4LoadAnchorCameraPosition());

        if (_instancesDirty)
            RebuildInstanceLists();

        if (!_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
            return false;

        Pm4ObjectMatchState pm4Object = BuildPm4ObjectMatchState(objectKey.tileX, objectKey.tileY, objectKey, obj);
        List<Pm4PlacementMatchState> placements = BuildPm4PlacementMatchStates();
        List<Pm4AssetProfileState> assetProfiles = BuildPm4AssetProfileStates(placements);
        objectMatch = BuildPm4ObjectMatchObject(pm4Object, placements, assetProfiles, Math.Max(1, maxMatchesPerObject));
        return true;
    }

    internal bool ShouldRenderPm4ObjectType(byte ck24Type)
    {
        return ck24Type switch
        {
            0x40 => _showPm4Type40,
            0x80 => _showPm4Type80,
            _ => _showPm4TypeOther
        };
    }

    public bool IsPm4SurfaceClassVisible(byte surfaceClass) => !_pm4HiddenSurfaceClasses.Contains(surfaceClass);

    public void SetPm4SurfaceClassVisible(byte surfaceClass, bool visible)
    {
        if (visible)
            _pm4HiddenSurfaceClasses.Remove(surfaceClass);
        else
            _pm4HiddenSurfaceClasses.Add(surfaceClass);
    }

    internal bool ShouldRenderPm4Object(Pm4OverlayObject obj)
        => ShouldRenderPm4ObjectType(obj.Ck24Type)
           && !_pm4HiddenSurfaceClasses.Contains(obj.DominantGroupKey);

    internal Matrix4x4 BuildPm4OverlayTransformMatrix()
    {
        float rotX = _pm4OverlayRotationDegrees.X * MathF.PI / 180f;
        float rotY = _pm4OverlayRotationDegrees.Y * MathF.PI / 180f;
        float rotZ = _pm4OverlayRotationDegrees.Z * MathF.PI / 180f;
        return Matrix4x4.CreateScale(_pm4OverlayScale)
            * Matrix4x4.CreateRotationX(rotX)
            * Matrix4x4.CreateRotationY(rotY)
            * Matrix4x4.CreateRotationZ(rotZ)
            * Matrix4x4.CreateTranslation(_pm4OverlayTranslation);
    }

    internal static Vector3 ApplyPm4OverlayTransform(Vector3 position, in Matrix4x4 transform)
    {
        return Vector3.Transform(position, transform);
    }

    internal static Matrix4x4 BuildPm4GeometryTransform(Pm4OverlayObject obj, in Matrix4x4 objectTransform, bool applyObjectTransform)
    {
        return applyObjectTransform
            ? obj.BaseTransform * objectTransform
            : obj.BaseTransform;
    }

    internal static Matrix4x4 BuildPm4BaseTransform(Vector3 placementAnchor, float baseRotationRadians)
    {
        Matrix4x4 transform = Matrix4x4.Identity;
        if (MathF.Abs(baseRotationRadians) > 1e-6f)
            transform *= Matrix4x4.CreateRotationZ(baseRotationRadians);

        transform *= Matrix4x4.CreateTranslation(placementAnchor);
        return transform;
    }

    private List<CorePm4CorrelationObjectState> BuildPm4CorrelationObjectStates()
    {
        bool applyPm4Transform = !IsNearZeroVector(_pm4OverlayTranslation)
            || !IsNearZeroVector(_pm4OverlayRotationDegrees)
            || !IsNearOneVector(_pm4OverlayScale);
        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        var inputs = new List<CorePm4CorrelationGeometryInput>(_pm4ObjectLookup.Count);

        foreach (var tileEntry in _pm4TileObjects.Where(tileEntry => IsTileWithinPm4MatchRadius(tileEntry.Key)))
        {
            foreach (Pm4OverlayObject obj in tileEntry.Value)
            {
                var objectKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId);
                var groupKey = ResolvePm4ObjectGroupKey(objectKey);
                Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                Matrix4x4 geometryTransform = BuildPm4GeometryTransform(obj, objectTransform, applyObjectTransform);
                inputs.Add(new CorePm4CorrelationGeometryInput(
                    tileEntry.Key.tileX,
                    tileEntry.Key.tileY,
                    new CorePm4ObjectGroupKey(groupKey.tileX, groupKey.tileY, groupKey.ck24),
                    new CorePm4CorrelationObjectDescriptor(
                        obj.Ck24,
                        obj.Ck24Type,
                        obj.ObjectPartId,
                        obj.LinkGroupObjectId,
                        obj.SurfaceCount,
                        obj.LinkedPositionRefCount,
                        obj.DominantGroupKey,
                        obj.DominantAttributeMask,
                        obj.DominantMscnRefIndex,
                        obj.AverageSurfaceHeight),
                    obj.Lines.Select(static line => new CorePm4GeometryLineSegment(line.From, line.To)).ToList(),
                    obj.Triangles.Select(static triangle => new CorePm4GeometryTriangle(triangle.A, triangle.B, triangle.C)).ToList(),
                    geometryTransform));
            }
        }

        return CorePm4CorrelationMath.BuildObjectStatesFromGeometry(inputs).ToList();
    }

    internal bool ShouldRenderPm4Object(
        Pm4OverlayObject obj,
        in Matrix4x4 objectTransform,
        bool applyObjectTransform,
        in Vector3 cameraPos,
        out Vector3 transformedCenter)
    {
        Vector3 boundsMin = obj.BoundsMin;
        Vector3 boundsMax = obj.BoundsMax;
        transformedCenter = obj.Center;

        if (applyObjectTransform)
        {
            TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);
            transformedCenter = ApplyPm4OverlayTransform(obj.Center, objectTransform);
        }

        float distSq = Vector3.DistanceSquared(cameraPos, transformedCenter);
        if (distSq > NoCullRadiusSq && !_frustumCuller.TestAABB(boundsMin, boundsMax))
            return false;

        return true;
    }

    private static bool IsNearZeroVector(Vector3 value)
    {
        return value.LengthSquared() < 0.0001f;
    }

    private static bool IsNearOneVector(Vector3 value)
    {
        return MathF.Abs(value.X - 1f) < 0.0001f
            && MathF.Abs(value.Y - 1f) < 0.0001f
            && MathF.Abs(value.Z - 1f) < 0.0001f;
    }

    private static Vector3 SanitizeScale(Vector3 scale)
    {
        const float minAbsScale = 0.0001f;

        float SanitizeComponent(float component)
        {
            if (MathF.Abs(component) >= minAbsScale)
                return component;

            return component < 0f ? -minAbsScale : minAbsScale;
        }

        return new Vector3(
            SanitizeComponent(scale.X),
            SanitizeComponent(scale.Y),
            SanitizeComponent(scale.Z));
    }

    private void RebuildPm4ObjectGroupBounds()
    {
        _pm4ObjectGroupBounds.Clear();

        foreach (var (objectKey, obj) in _pm4ObjectLookup)
        {
            var groupKey = ResolvePm4ObjectGroupKey(objectKey);
            if (_pm4ObjectGroupBounds.TryGetValue(groupKey, out var existingBounds))
            {
                _pm4ObjectGroupBounds[groupKey] = (
                    Vector3.Min(existingBounds.min, obj.BoundsMin),
                    Vector3.Max(existingBounds.max, obj.BoundsMax));
            }
            else
            {
                _pm4ObjectGroupBounds[groupKey] = (obj.BoundsMin, obj.BoundsMax);
            }
        }
    }

    public bool TryGetPm4ObjectGroupBounds((int tileX, int tileY, uint ck24) groupKey, out Vector3 min, out Vector3 max)
    {
        if (_pm4ObjectGroupBounds.TryGetValue(groupKey, out var b))
        {
            min = b.min;
            max = b.max;
            return true;
        }
        min = default;
        max = default;
        return false;
    }

    public IReadOnlyList<Pm4SurfaceGroupCluster> GetPm4SurfaceGroupClusters(int tileX, int tileY, uint ck24)
    {
        var clusterByGroupKey = new Dictionary<byte, (Vector3 min, Vector3 max, int count)>();

        foreach (var (objectKey, obj) in _pm4ObjectLookup)
        {
            if (objectKey.tileX != tileX || objectKey.tileY != tileY || objectKey.ck24 != ck24)
                continue;

            byte gk = obj.DominantGroupKey;
            if (clusterByGroupKey.TryGetValue(gk, out var existing))
            {
                clusterByGroupKey[gk] = (
                    Vector3.Min(existing.min, obj.BoundsMin),
                    Vector3.Max(existing.max, obj.BoundsMax),
                    existing.count + obj.SurfaceCount);
            }
            else
            {
                clusterByGroupKey[gk] = (obj.BoundsMin, obj.BoundsMax, obj.SurfaceCount);
            }
        }

        var results = new List<Pm4SurfaceGroupCluster>(clusterByGroupKey.Count);
        foreach (var kv in clusterByGroupKey.OrderBy(static kv => kv.Key))
        {
            results.Add(new Pm4SurfaceGroupCluster(kv.Key, kv.Value.min, kv.Value.max, kv.Value.count));
        }
        return results;
    }

    private void RebuildPm4TileCk24Bounds()
    {
        _pm4TileCk24Bounds.Clear();

        foreach (var (objectKey, obj) in _pm4ObjectLookup)
        {
            var tileCk24Key = (objectKey.tileX, objectKey.tileY, objectKey.ck24);
            if (_pm4TileCk24Bounds.TryGetValue(tileCk24Key, out var existingBounds))
            {
                _pm4TileCk24Bounds[tileCk24Key] = (
                    Vector3.Min(existingBounds.min, obj.BoundsMin),
                    Vector3.Max(existingBounds.max, obj.BoundsMax));
            }
            else
            {
                _pm4TileCk24Bounds[tileCk24Key] = (obj.BoundsMin, obj.BoundsMax);
            }
        }
    }

    private bool TryComputePm4ObjectGroupPivot(
        (int tileX, int tileY, uint ck24) groupKey,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform,
        out Vector3 pivot)
    {
        if (_pm4ObjectGroupBounds.TryGetValue(groupKey, out var groupBounds))
        {
            pivot = (groupBounds.min + groupBounds.max) * 0.5f;
            if (applyPm4Transform)
                pivot = ApplyPm4OverlayTransform(pivot, pm4Transform);
            return true;
        }

        pivot = Vector3.Zero;
        return false;
    }

    private bool TryComputePm4TileCk24Pivot(
        (int tileX, int tileY, uint ck24) tileCk24Key,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform,
        out Vector3 pivot)
    {
        if (_pm4TileCk24Bounds.TryGetValue(tileCk24Key, out var rawBounds))
        {
            pivot = (rawBounds.min + rawBounds.max) * 0.5f;
            if (applyPm4Transform)
                pivot = ApplyPm4OverlayTransform(pivot, pm4Transform);
            return true;
        }

        pivot = Vector3.Zero;
        return false;
    }

    internal Matrix4x4 BuildPm4ObjectTransform((int tileX, int tileY, uint ck24, int objectPart) objectKey,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform,
        out bool applyObjectTransform)
    {
        applyObjectTransform = false;
        Matrix4x4 transform = Matrix4x4.Identity;

        if (applyPm4Transform)
        {
            transform = pm4Transform;
            applyObjectTransform = true;
        }

        var objectGroupKey = ResolvePm4ObjectGroupKey(objectKey);
        var tileCk24Key = (objectKey.tileX, objectKey.tileY, objectKey.ck24);
        bool hasLayerTranslation = _pm4TileCk24Translations.TryGetValue(tileCk24Key, out Vector3 layerTranslation)
            && !IsNearZeroVector(layerTranslation);
        bool hasLayerRotation = _pm4TileCk24RotationsDegrees.TryGetValue(tileCk24Key, out Vector3 layerRotationDegrees)
            && !IsNearZeroVector(layerRotationDegrees);
        bool hasLayerScale = _pm4TileCk24Scales.TryGetValue(tileCk24Key, out Vector3 layerScale)
            && !IsNearOneVector(layerScale);

        bool hasGlobalFlip = _pm4FlipAllObjectsY;
        bool hasObjectTranslation = _pm4ObjectTranslations.TryGetValue(objectGroupKey, out Vector3 objectTranslation)
            && !IsNearZeroVector(objectTranslation);
        bool hasObjectRotation = _pm4ObjectRotationsDegrees.TryGetValue(objectGroupKey, out Vector3 objectRotationDegrees)
            && !IsNearZeroVector(objectRotationDegrees);
        bool hasObjectScale = _pm4ObjectScales.TryGetValue(objectGroupKey, out Vector3 objectScale)
            && !IsNearOneVector(objectScale);

        if (hasLayerRotation || hasLayerScale)
        {
            Vector3 pivot = Vector3.Zero;
            if (!TryComputePm4TileCk24Pivot(tileCk24Key, applyPm4Transform, pm4Transform, out pivot)
                && _pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? objectInfo))
            {
                pivot = objectInfo.Center;
                if (applyPm4Transform)
                    pivot = ApplyPm4OverlayTransform(pivot, pm4Transform);
            }

            Matrix4x4 layerRotationScale = Matrix4x4.Identity;
            if (hasLayerScale)
                layerRotationScale *= Matrix4x4.CreateScale(SanitizeScale(layerScale));

            if (hasLayerRotation)
            {
                float layerRotX = layerRotationDegrees.X * MathF.PI / 180f;
                float layerRotY = layerRotationDegrees.Y * MathF.PI / 180f;
                float layerRotZ = layerRotationDegrees.Z * MathF.PI / 180f;
                layerRotationScale *= Matrix4x4.CreateRotationX(layerRotX)
                    * Matrix4x4.CreateRotationY(layerRotY)
                    * Matrix4x4.CreateRotationZ(layerRotZ);
            }

            Matrix4x4 layerPivotTransform = Matrix4x4.CreateTranslation(-pivot)
                * layerRotationScale
                * Matrix4x4.CreateTranslation(pivot);
            transform = applyObjectTransform
                ? transform * layerPivotTransform
                : layerPivotTransform;
            applyObjectTransform = true;
        }

        if (hasLayerTranslation)
        {
            Matrix4x4 layerTranslationMatrix = Matrix4x4.CreateTranslation(layerTranslation);
            transform = applyObjectTransform
                ? transform * layerTranslationMatrix
                : layerTranslationMatrix;
            applyObjectTransform = true;
        }

        if (hasGlobalFlip || hasObjectRotation || hasObjectScale)
        {
            Vector3 pivot = Vector3.Zero;
            if (!TryComputePm4ObjectGroupPivot(objectGroupKey, applyPm4Transform, pm4Transform, out pivot)
                && _pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? objectInfo))
            {
                pivot = objectInfo.Center;
                if (applyPm4Transform)
                    pivot = ApplyPm4OverlayTransform(pivot, pm4Transform);
            }

            Matrix4x4 rotationScale = Matrix4x4.Identity;
            if (hasGlobalFlip)
            {
                rotationScale *= Matrix4x4.CreateScale(1f, -1f, 1f);
            }

            if (hasObjectScale)
                rotationScale *= Matrix4x4.CreateScale(SanitizeScale(objectScale));

            if (hasObjectRotation)
            {
                float objectRotX = objectRotationDegrees.X * MathF.PI / 180f;
                float objectRotY = objectRotationDegrees.Y * MathF.PI / 180f;
                float objectRotZ = objectRotationDegrees.Z * MathF.PI / 180f;
                rotationScale *= Matrix4x4.CreateRotationX(objectRotX)
                    * Matrix4x4.CreateRotationY(objectRotY)
                    * Matrix4x4.CreateRotationZ(objectRotZ);
            }

            Matrix4x4 objectPivotTransform = Matrix4x4.CreateTranslation(-pivot)
                * rotationScale
                * Matrix4x4.CreateTranslation(pivot);
            transform = applyObjectTransform
                ? transform * objectPivotTransform
                : objectPivotTransform;
            applyObjectTransform = true;
        }

        if (hasObjectTranslation)
        {
            Matrix4x4 objectTranslationMatrix = Matrix4x4.CreateTranslation(objectTranslation);
            transform = applyObjectTransform
                ? transform * objectTranslationMatrix
                : objectTranslationMatrix;
            applyObjectTransform = true;
        }

        return transform;
    }
}
