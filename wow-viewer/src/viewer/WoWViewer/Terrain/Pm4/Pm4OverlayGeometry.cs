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
using static WoWViewer.Terrain.Pm4OverlayMatching;
using static WoWViewer.Terrain.Pm4OverlayCacheCodec;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;

namespace WoWViewer.Terrain;

/// <summary>Pure static PM4 overlay helpers moved verbatim from <c>WorldScene</c> (Epic 251 U-01 E1).</summary>
internal static class Pm4OverlayGeometry
{

    internal static string BuildPm4ObjText(IReadOnlyList<Pm4OverlayObject> objects, int tileX, int tileY)
    {
        var builder = new StringBuilder();
        builder.AppendLine($"# PM4 tile {tileX:D2}_{tileY:D2}");
        builder.AppendLine($"# object_count {objects.Count}");

        int vertexIndex = 1;
        foreach (Pm4OverlayObject obj in objects)
        {
            string objectName = $"tile_{tileX:D2}_{tileY:D2}_ck24_{obj.Ck24:X6}_part_{obj.ObjectPartId:D4}";
            builder.AppendLine();
            builder.AppendLine($"o {objectName}");
            builder.AppendLine($"# source {obj.SourcePath}");
            builder.AppendLine($"# lines {obj.Lines.Count} triangles {obj.Triangles.Count} surfaces {obj.SurfaceCount} total_indices {obj.TotalIndexCount}");

            Matrix4x4 transform = obj.BaseTransform;
            for (int i = 0; i < obj.Triangles.Count; i++)
            {
                Pm4Triangle tri = obj.Triangles[i];
                Vector3 a = ApplyPm4OverlayTransform(tri.A, transform);
                Vector3 b = ApplyPm4OverlayTransform(tri.B, transform);
                Vector3 c = ApplyPm4OverlayTransform(tri.C, transform);
                AppendObjVertex(builder, a);
                AppendObjVertex(builder, b);
                AppendObjVertex(builder, c);
                builder.Append("f ")
                    .Append(vertexIndex)
                    .Append(' ')
                    .Append(vertexIndex + 1)
                    .Append(' ')
                    .Append(vertexIndex + 2)
                    .AppendLine();
                vertexIndex += 3;
            }

            for (int i = 0; i < obj.Lines.Count; i++)
            {
                Pm4LineSegment line = obj.Lines[i];
                Vector3 from = ApplyPm4OverlayTransform(line.From, transform);
                Vector3 to = ApplyPm4OverlayTransform(line.To, transform);
                AppendObjVertex(builder, from);
                AppendObjVertex(builder, to);
                builder.Append("l ")
                    .Append(vertexIndex)
                    .Append(' ')
                    .Append(vertexIndex + 1)
                    .AppendLine();
                vertexIndex += 2;
            }
        }

        return builder.ToString();
    }

    internal static void AppendObjVertex(StringBuilder builder, Vector3 vertex)
    {
        builder.Append("v ")
            .Append(vertex.X.ToString("G9", CultureInfo.InvariantCulture))
            .Append(' ')
            .Append(vertex.Y.ToString("G9", CultureInfo.InvariantCulture))
            .Append(' ')
            .Append(vertex.Z.ToString("G9", CultureInfo.InvariantCulture))
            .AppendLine();
    }

    internal static string SanitizePm4ExportPathSegment(string value)
    {
        char[] invalidChars = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(value.Length);
        for (int i = 0; i < value.Length; i++)
        {
            char current = value[i];
            builder.Append(invalidChars.Contains(current) ? '_' : current);
        }

        return builder.Length == 0 ? "pm4" : builder.ToString();
    }

    internal static bool TryMapPm4FileTileToTerrainTile(int fileTileX, int fileTileY, out int terrainTileX, out int terrainTileY)
    {
        // PM4 filename tiles are transposed relative to ADT terrain tile naming on the
        // development corpus. Map PM4 file XX_YY onto terrain tile YY_XX so camera-window
        // loads and tile-local placement land on the same ADT tile the user is viewing.
        terrainTileX = fileTileY;
        terrainTileY = fileTileX;

        return terrainTileX is >= 0 and <= 63
            && terrainTileY is >= 0 and <= 63;
    }

    internal static List<Pm4OverlayObject> RebasePm4ObjectParts(IReadOnlyList<Pm4OverlayObject> objects, int objectPartOffset)
    {
        if (objects.Count == 0 || objectPartOffset == 0)
            return objects.ToList();

        var rebased = new List<Pm4OverlayObject>(objects.Count);
        for (int i = 0; i < objects.Count; i++)
        {
            Pm4OverlayObject obj = objects[i];
            rebased.Add(Pm4OverlayObject.FromCachedLocalized(
                obj.SourcePath,
                obj.MshdField00,
                obj.MshdRegionId,
                obj.MshdField08,
                obj.Ck24,
                obj.Ck24Type,
                obj.ObjectPartId + objectPartOffset,
                obj.LinkGroupObjectId,
                obj.LinkedPositionRefCount,
                obj.LinkedPositionRefSummary,
                obj.Lines,
                obj.Triangles,
                obj.SurfaceCount,
                obj.TotalIndexCount,
                obj.DominantGroupKey,
                obj.DominantAttributeMask,
                obj.DominantMscnRefIndex,
                obj.AverageSurfaceHeight,
                obj.PlacementAnchor,
                obj.BaseRotationRadians,
                obj.PlanarTransform,
                obj.BoundsMin,
                obj.BoundsMax,
                obj.ConnectorKeys));
        }

        return rebased;
    }

    internal static List<Pm4OverlaySeedGroup> BuildPm4OverlaySeedGroups(Pm4File pm4)
    {
        List<Pm4IndexedSurface> indexedSurfaces = pm4.KnownChunks.Msur
            .Select((surface, surfaceIndex) => new Pm4IndexedSurface(surfaceIndex, surface))
            .Where(static indexedSurface => indexedSurface.Surface.IndexCount >= 3)
            .ToList();

        var groups = new List<Pm4OverlaySeedGroup>();
        foreach (IGrouping<uint, Pm4IndexedSurface> ck24Group in indexedSurfaces
            .Where(static indexedSurface => indexedSurface.Surface.Ck24 != 0)
            .GroupBy(static indexedSurface => indexedSurface.Surface.Ck24)
            .OrderBy(static group => group.Key))
        {
            groups.Add(new Pm4OverlaySeedGroup(
                ck24Group.Key,
                (byte)(ck24Group.Key >> 16),
                requiresConnectivitySeedSplit: false,
                ck24Group.ToList()));
        }

        foreach (IGrouping<(byte groupKey, byte attributeMask), Pm4IndexedSurface> zeroGroup in indexedSurfaces
            .Where(static indexedSurface => indexedSurface.Surface.Ck24 == 0)
            .GroupBy(static indexedSurface => (indexedSurface.Surface.GroupKey, indexedSurface.Surface.AttributeMask))
            .OrderBy(static group => group.Key.GroupKey)
            .ThenBy(static group => group.Key.AttributeMask))
        {
            groups.Add(new Pm4OverlaySeedGroup(
                0u,
                0,
                requiresConnectivitySeedSplit: true,
                zeroGroup.ToList()));
        }

        return groups;
    }

    internal static List<Pm4OverlayObject> BuildPm4TileObjects(
        Pm4File pm4,
        string sourcePath,
        int tileX,
        int tileY,
        bool splitCk24ByMscnRef,
        bool splitCk24ByConnectivity,
        bool includePathWalls,
        ref int remainingLineBudget,
        ref int remainingTriangleBudget,
        ref int rejectedLongEdges,
        out Pm4TileBuildDiagnostics diagnostics)
    {
        diagnostics = new Pm4TileBuildDiagnostics
        {
            TotalMsurCount = pm4.KnownChunks.Msur.Count,
        };

        var objects = new List<Pm4OverlayObject>();
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<MprlEntry> positionRefs = pm4.KnownChunks.Mprl;

        if (remainingLineBudget <= 0 || meshVertices.Count == 0)
            return objects;

        List<Pm4OverlaySeedGroup> seedGroups = BuildPm4OverlaySeedGroups(pm4);
        // The build also drops short-index surfaces at the seed-group stage; count them here.
        diagnostics.DroppedShortIndexCount = pm4.KnownChunks.Msur.Count(static s => s.IndexCount < 3);
        if (seedGroups.Count == 0)
            return objects;

        // MSLK path windows, indexed once per tile by the MSUR surface they reference.
        Dictionary<int, List<int>> mslkWindowsBySurface = includePathWalls
            ? BuildMslkWindowsBySurface(pm4)
            : [];

        var mshdGrouping = CorePm4MshdGroupingService.Describe(pm4.KnownChunks.Mshd);
        Pm4AxisConvention fileAxisConvention = DetectPm4AxisConvention(pm4);
        bool fallbackTileLocalCoordinates = IsLikelyTileLocal(meshVertices);
        int tileLineBudget = Math.Min(Pm4MaxLinesPerTile, remainingLineBudget);
        int tileTriangleBudget = Math.Min(Pm4MaxTrianglesPerTile, remainingTriangleBudget);

        // Viewer-generated split id used as a stable handle for this overlay build.
        // This is not a raw PM4 field from disk.
        int nextObjectPartId = 0;
        foreach (Pm4OverlaySeedGroup seedGroup in seedGroups)
        {
            if (tileLineBudget <= 0)
                break;

            uint ck24 = seedGroup.DisplayCk24;
            byte ck24Type = seedGroup.DisplayCk24Type;
            List<Pm4IndexedSurface> surfaceGroup = seedGroup.Surfaces;
            Pm4AxisConvention ck24AxisConvention = fileAxisConvention;
            List<MsurEntry> ck24Surfaces = surfaceGroup.Select(static entry => entry.Surface).ToList();
            List<MprlEntry> ck24PositionRefs = CollectLinkedPositionRefs(pm4, surfaceGroup);
            CorePm4CoordinateModeResolution seedCoordinateModeResolution = ResolveCk24CoordinateModeResolution(
                pm4,
                ck24Surfaces,
                ck24PositionRefs,
                tileX,
                tileY,
                ck24AxisConvention,
                fallbackTileLocalCoordinates);
            bool seedUseTileLocalCoordinates = seedCoordinateModeResolution.CoordinateMode == CorePm4CoordinateMode.TileLocal;
            // Keep one shared planar transform per CK24 so split linked/components stay on one coordinate plane.
            CorePm4PlacementSolution seedPlacement = ResolvePlacementSolution(
                pm4,
                ck24Surfaces,
                ck24PositionRefs,
                tileX,
                tileY,
                seedUseTileLocalCoordinates,
                ck24AxisConvention);
            Pm4PlanarTransform seedPlanarTransform = seedCoordinateModeResolution.PlanarTransform;
            Vector3 seedWorldPivot = seedPlacement.WorldPivot;
            float seedWorldYawCorrection = seedPlacement.WorldYawCorrectionRadians;
            float seedRendererFrameRotationRadians = ConvertWorldYawCorrectionToRendererRotationRadians(seedWorldYawCorrection);
            IReadOnlyList<Pm4ConnectorKey> seedConnectorKeys = BuildCk24ConnectorKeys(pm4, ck24Surfaces, seedPlacement);
            List<List<Pm4IndexedSurface>> linkedGroups = seedGroup.RequiresConnectivitySeedSplit
                ? SplitZeroCk24SeedGroup(pm4, surfaceGroup)
                : SplitSurfaceGroupByMslk(pm4, surfaceGroup);

            foreach (List<Pm4IndexedSurface> linkedGroup in linkedGroups)
            {
                if (linkedGroup.Count == 0 || tileLineBudget <= 0)
                    continue;

                uint dominantLinkGroupObjectId = SelectDominantMslkGroupObjectId(pm4, linkedGroup);
                List<MsurEntry> linkedSurfaces = linkedGroup.Select(static entry => entry.Surface).ToList();
                List<MprlEntry> linkedPositionRefs = CollectLinkedPositionRefs(pm4, linkedGroup);
                Pm4LinkedPositionRefSummary linkedPositionRefSummary = SummarizeLinkedPositionRefs(linkedPositionRefs);

                CorePm4CoordinateModeResolution linkedCoordinateModeResolution = ResolveCk24CoordinateModeResolution(
                    pm4,
                    linkedSurfaces,
                    linkedPositionRefs,
                    tileX,
                    tileY,
                    ck24AxisConvention,
                    fallbackTileLocalCoordinates);
                bool linkedUseTileLocalCoordinates = linkedCoordinateModeResolution.CoordinateMode == CorePm4CoordinateMode.TileLocal;

                CorePm4PlacementSolution linkedPlacement = ResolvePlacementSolution(
                    pm4,
                    linkedSurfaces,
                    linkedPositionRefs,
                    tileX,
                    tileY,
                    linkedUseTileLocalCoordinates,
                    ck24AxisConvention);

                Pm4PlanarTransform linkedPlanarTransform = linkedCoordinateModeResolution.PlanarTransform;
                Vector3 linkedWorldPivot = linkedPlacement.WorldPivot;
                float linkedWorldYawCorrection = linkedPlacement.WorldYawCorrectionRadians;
                float linkedRendererFrameRotationRadians = ConvertWorldYawCorrectionToRendererRotationRadians(linkedWorldYawCorrection);
                IReadOnlyList<Pm4ConnectorKey> linkedConnectorKeys = BuildCk24ConnectorKeys(pm4, linkedSurfaces, linkedPlacement);

                Vector3 linkedPlacementAnchor = ComputeSurfaceRendererCentroid(
                    pm4,
                    linkedSurfaces,
                    tileX,
                    tileY,
                    linkedUseTileLocalCoordinates,
                    ck24AxisConvention,
                    linkedPlanarTransform,
                    linkedWorldPivot,
                    linkedWorldYawCorrection);
                // MSUR.MsviFirstIndex is the surface's window start, so it recovers the surface
                // index after the split helpers have reduced indexed surfaces to bare entries.
                Dictionary<uint, int> surfaceIndexByMsviFirst = [];
                if (includePathWalls)
                {
                    foreach (Pm4IndexedSurface indexed in linkedGroup)
                        surfaceIndexByMsviFirst[indexed.Surface.MsviFirstIndex] = indexed.SurfaceIndex;
                }

                bool allowNestedSeedSplits = !seedGroup.RequiresConnectivitySeedSplit;
                List<List<MsurEntry>> anchorGroups = splitCk24ByMscnRef && allowNestedSeedSplits
                    ? SplitSurfaceGroupByMscnRef(linkedSurfaces)
                    : new List<List<MsurEntry>> { linkedSurfaces };

                foreach (List<MsurEntry> anchorGroup in anchorGroups)
                {
                    List<List<MsurEntry>> components = splitCk24ByConnectivity && allowNestedSeedSplits
                        ? SplitSurfaceGroupByConnectivity(pm4, anchorGroup)
                        : new List<List<MsurEntry>> { anchorGroup };

                    foreach (List<MsurEntry> component in components)
                    {
                        if (tileLineBudget <= 0)
                            break;

                        // Keep split components under one linked-group frame basis.
                        // MSUR 0x1C / CK24 is not sufficient to guarantee one shared object rotation
                        // across every linked sub-object in a seed group, especially on large WMO
                        // interiors where repeated carriers can appear under the same CK24 value.
                        List<Pm4LineSegment> lines = BuildCk24ObjectLines(pm4, component, tileX, tileY, linkedUseTileLocalCoordinates, ck24AxisConvention, linkedPlanarTransform, linkedWorldPivot, linkedWorldYawCorrection, tileLineBudget, ref rejectedLongEdges);
                        int componentRejectedOutOfRange = 0;
                        List<Pm4Triangle> triangles = tileTriangleBudget > 0
                            ? BuildCk24ObjectTriangles(pm4, component, tileX, tileY, linkedUseTileLocalCoordinates, ck24AxisConvention, linkedPlanarTransform, linkedWorldPivot, linkedWorldYawCorrection, tileTriangleBudget, out componentRejectedOutOfRange)
                            : new List<Pm4Triangle>();

                        diagnostics.DroppedOutOfRangeMsviCount += componentRejectedOutOfRange;

                        // Append this component's wall faces to the same mesh, so they render,
                        // pick and select as part of the object they stand on.
                        if (includePathWalls && mslkWindowsBySurface.Count > 0 && tileTriangleBudget > 0)
                        {
                            HashSet<int> componentSurfaceIndices = [];
                            foreach (MsurEntry componentSurface in component)
                            {
                                if (surfaceIndexByMsviFirst.TryGetValue(componentSurface.MsviFirstIndex, out int surfaceIndex))
                                    componentSurfaceIndices.Add(surfaceIndex);
                            }

                            if (componentSurfaceIndices.Count > 0)
                            {
                                var wallLines = new List<Pm4LineSegment>();
                                List<Pm4Triangle> wallTriangles = BuildMslkWallTriangles(
                                    pm4,
                                    componentSurfaceIndices,
                                    mslkWindowsBySurface,
                                    tileX,
                                    tileY,
                                    linkedUseTileLocalCoordinates,
                                    ck24AxisConvention,
                                    linkedPlanarTransform,
                                    Math.Max(0, tileTriangleBudget - triangles.Count),
                                    wallLines,
                                    Math.Max(0, tileLineBudget - lines.Count),
                                    out int componentWallFaces);

                                diagnostics.WallFaceCount += componentWallFaces;
                                triangles.AddRange(wallTriangles);
                                lines.AddRange(wallLines);
                            }
                        }

                        if (lines.Count == 0 && triangles.Count == 0)
                        {
                            diagnostics.DroppedEmptyComponentCount++;
                            continue;
                        }

                        byte dominantGroupKey = SelectDominantSurfaceValue(component, static surface => surface.GroupKey);
                        byte dominantAttributeMask = SelectDominantSurfaceValue(component, static surface => surface.AttributeMask);
                        uint dominantMscnRefIndex = SelectDominantSurfaceValue(component, static surface => surface.MscnRefIndex);
                        float averageSurfaceHeight = component.Count > 0 ? component.Average(static surface => surface.Height) : 0f;
                        int totalIndexCount = component.Sum(static surface => surface.IndexCount);

                        objects.Add(new Pm4OverlayObject(
                            sourcePath,
                            mshdGrouping.Field00,
                            mshdGrouping.RegionId,
                            mshdGrouping.Field08,
                            ck24,
                            ck24Type,
                            nextObjectPartId++,
                            dominantLinkGroupObjectId,
                            linkedPositionRefs.Count,
                            linkedPositionRefSummary,
                            lines,
                            triangles,
                            component.Count,
                            totalIndexCount,
                            dominantGroupKey,
                            dominantAttributeMask,
                            dominantMscnRefIndex,
                            averageSurfaceHeight,
                            linkedPlacementAnchor,
                            linkedRendererFrameRotationRadians,
                            linkedPlanarTransform,
                            linkedConnectorKeys));

                        // Collect MSLK.TypeFlags from surfaces in this component.
                        // Match MSLK.RefIndex against the MSUR entry at that index position.
                        uint typeFlagsMask = 0;
                        if (pm4.KnownChunks.Mslk.Count > 0)
                        {
                            foreach (MslkEntry mslk in pm4.KnownChunks.Mslk)
                            {
                                if (mslk.TypeFlags == 0)
                                    continue;
                                if ((uint)mslk.RefIndex < (uint)pm4.KnownChunks.Msur.Count &&
                                    component.Contains(pm4.KnownChunks.Msur[mslk.RefIndex]))
                                {
                                    typeFlagsMask |= 1u << mslk.TypeFlags;
                                }
                            }
                        }
                        if (typeFlagsMask != 0)
                            objects[^1].DistinctTypeFlags = typeFlagsMask;

                        tileLineBudget -= lines.Count;
                        tileTriangleBudget -= triangles.Count;
                    }
                }
            }
        }

        int linesUsed = objects.Sum(obj => obj.Lines.Count);
        int trianglesUsed = objects.Sum(obj => obj.Triangles.Count);
        remainingLineBudget -= linesUsed;
        remainingTriangleBudget -= trianglesUsed;
        diagnostics.DroppedLongEdgeLines = rejectedLongEdges;
        return objects;
    }

    /// <summary>
    /// Returns the canonical frame instead of fitting one per object.
    /// </summary>
    /// <remarks>
    /// This used to call <c>CorePm4PlacementMath.ResolveCoordinateMode</c>, which scored candidate
    /// coordinate modes and planar transforms against MPRL. Measured over the whole development
    /// corpus, that fitter was wrong in both directions:
    ///
    /// - it selected <c>TileLocal</c> for 18 objects whose coordinates are absolute, then added tile
    ///   offsets to them. The human tents in <c>development_01_00.pm4</c> were one, thrown from tile
    ///   (0,1) to (1,-1) while all three of that tile's real ADT placements sit inside the canonical
    ///   footprint;
    /// - the yaw correction it produced rotated 974 of 1,895 objects by 15-45 degrees. Scored
    ///   against MODF world bounding boxes over the 127 objects whose box can actually see a
    ///   rotation, containment fell from 93.3% to 88.2%, against 79.0% for a deliberately wrong
    ///   45-degree control. It hurt 96 objects and helped 3.
    ///
    /// `pm4 bounds-audit --by-region` and `pm4 yaw-evidence` reproduce both numbers.
    /// <c>CorePm4PlacementMath</c> keeps the fitter for callers that still want to explore it; the
    /// render path simply no longer asks.
    /// </remarks>
    internal static CorePm4CoordinateModeResolution ResolveCk24CoordinateModeResolution(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        IReadOnlyList<MprlEntry> anchorPositionRefs,
        int tileX,
        int tileY,
        Pm4AxisConvention axisConvention,
        bool fallbackTileLocalCoordinates)
    {
        return CanonicalCoordinateModeResolution;
    }

    internal static List<List<Pm4IndexedSurface>> SplitSurfaceGroupByMslk(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        var groups = new List<List<Pm4IndexedSurface>>();
        if (surfaces.Count == 0)
            return groups;

        if (!TryPartitionSurfaceGroupByMslk(pm4, surfaces, out List<List<Pm4IndexedSurface>> linkedComponents, out List<Pm4IndexedSurface> unlinked))
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        if (linkedComponents.Count <= 1)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        foreach (List<Pm4IndexedSurface> component in linkedComponents.OrderBy(component => component.Min(entry => entry.SurfaceIndex)))
            groups.Add(component);

        if (unlinked.Count > 0)
            groups.Add(unlinked);

        return groups;
    }

    internal static List<List<Pm4IndexedSurface>> SplitZeroCk24SeedGroup(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        if (!TryPartitionSurfaceGroupByMslk(pm4, surfaces, out List<List<Pm4IndexedSurface>> linkedComponents, out List<Pm4IndexedSurface> unlinked))
            return SplitIndexedSurfaceGroupByConnectivity(pm4, surfaces);

        if (linkedComponents.Count == 0)
            return SplitIndexedSurfaceGroupByConnectivity(pm4, surfaces);

        var groups = new List<List<Pm4IndexedSurface>>();
        foreach (List<Pm4IndexedSurface> component in linkedComponents.OrderBy(component => component.Min(entry => entry.SurfaceIndex)))
            groups.Add(component);

        if (unlinked.Count > 0)
            groups.AddRange(SplitIndexedSurfaceGroupByConnectivity(pm4, unlinked));

        return groups;
    }

    internal static bool TryPartitionSurfaceGroupByMslk(
        Pm4File pm4,
        IReadOnlyList<Pm4IndexedSurface> surfaces,
        out List<List<Pm4IndexedSurface>> linkedComponents,
        out List<Pm4IndexedSurface> unlinked)
    {
        linkedComponents = new List<List<Pm4IndexedSurface>>();
        unlinked = new List<Pm4IndexedSurface>();

        IReadOnlyList<CorePm4MslkEntry> linkEntries = pm4.KnownChunks.Mslk;
        int surfaceCount = pm4.KnownChunks.Msur.Count;
        if (surfaces.Count <= 1 || linkEntries.Count == 0)
            return false;

        var surfaceIndexToLocal = new Dictionary<int, int>(surfaces.Count);
        for (int i = 0; i < surfaces.Count; i++)
            surfaceIndexToLocal[surfaces[i].SurfaceIndex] = i;

        var groupToMembers = new Dictionary<uint, HashSet<int>>();
        for (int i = 0; i < linkEntries.Count; i++)
        {
            CorePm4MslkEntry link = linkEntries[i];
            if (link.GroupObjectId == 0)
                continue;

            if (link.RefIndex >= surfaceCount || !surfaceIndexToLocal.TryGetValue(link.RefIndex, out int localRefIndex))
                continue;

            if (!groupToMembers.TryGetValue(link.GroupObjectId, out HashSet<int>? members))
            {
                members = new HashSet<int>();
                groupToMembers[link.GroupObjectId] = members;
            }

            members.Add(localRefIndex);
        }

        if (groupToMembers.Count == 0)
            return false;

        int[] parent = new int[surfaces.Count];
        for (int i = 0; i < parent.Length; i++)
            parent[i] = i;

        static int Find(int[] parentArray, int index)
        {
            while (parentArray[index] != index)
            {
                parentArray[index] = parentArray[parentArray[index]];
                index = parentArray[index];
            }

            return index;
        }

        static void Union(int[] parentArray, int a, int b)
        {
            int rootA = Find(parentArray, a);
            int rootB = Find(parentArray, b);
            if (rootA != rootB)
                parentArray[rootB] = rootA;
        }

        var linkedLocalIndices = new HashSet<int>();
        foreach (HashSet<int> members in groupToMembers.Values)
        {
            if (members.Count < 2)
                continue;

            int first = members.First();
            linkedLocalIndices.Add(first);
            foreach (int member in members)
            {
                linkedLocalIndices.Add(member);
                Union(parent, first, member);
            }
        }

        if (linkedLocalIndices.Count < 2)
            return false;

        var linkedByRoot = new Dictionary<int, List<Pm4IndexedSurface>>();
        for (int i = 0; i < surfaces.Count; i++)
        {
            if (!linkedLocalIndices.Contains(i))
            {
                unlinked.Add(surfaces[i]);
                continue;
            }

            int root = Find(parent, i);
            if (!linkedByRoot.TryGetValue(root, out List<Pm4IndexedSurface>? component))
            {
                component = new List<Pm4IndexedSurface>();
                linkedByRoot[root] = component;
            }

            component.Add(surfaces[i]);
        }

        if (linkedByRoot.Count == 0)
            return false;

        linkedComponents = linkedByRoot.Values.ToList();
        return true;
    }

    internal static List<List<Pm4IndexedSurface>> SplitIndexedSurfaceGroupByConnectivity(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var components = new List<List<Pm4IndexedSurface>>();
        if (surfaces.Count == 0)
            return components;
        if (surfaces.Count == 1)
        {
            components.Add(new List<Pm4IndexedSurface> { surfaces[0] });
            return components;
        }

        var surfaceVertices = new List<List<int>>(surfaces.Count);
        var vertexToSurfaceIndices = new Dictionary<int, List<int>>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s].Surface;
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            var vertices = new List<int>();
            var unique = new HashSet<int>();

            if (surface.IndexCount > 0 && firstIndex >= 0 && endExclusive > firstIndex)
            {
                for (int idx = firstIndex; idx < endExclusive; idx++)
                {
                    int vertexIndex = (int)meshIndices[idx];
                    if ((uint)vertexIndex >= (uint)meshVertices.Count)
                        continue;
                    if (!unique.Add(vertexIndex))
                        continue;

                    vertices.Add(vertexIndex);
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? owners))
                    {
                        owners = new List<int>();
                        vertexToSurfaceIndices[vertexIndex] = owners;
                    }

                    owners.Add(s);
                }
            }

            surfaceVertices.Add(vertices);
        }

        var visited = new bool[surfaces.Count];
        var queue = new Queue<int>();
        for (int start = 0; start < surfaces.Count; start++)
        {
            if (visited[start])
                continue;

            visited[start] = true;
            queue.Enqueue(start);
            var component = new List<Pm4IndexedSurface>();

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                component.Add(surfaces[current]);

                List<int> vertices = surfaceVertices[current];
                for (int v = 0; v < vertices.Count; v++)
                {
                    int vertexIndex = vertices[v];
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? neighbors))
                        continue;

                    for (int n = 0; n < neighbors.Count; n++)
                    {
                        int neighborSurface = neighbors[n];
                        if (visited[neighborSurface])
                            continue;

                        visited[neighborSurface] = true;
                        queue.Enqueue(neighborSurface);
                    }
                }
            }

            components.Add(component);
        }

        return components;
    }

    internal static uint SelectDominantMslkGroupObjectId(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        IReadOnlyList<CorePm4MslkEntry> linkEntries = pm4.KnownChunks.Mslk;
        if (surfaces.Count == 0 || linkEntries.Count == 0)
            return 0;

        int surfaceCount = pm4.KnownChunks.Msur.Count;
        var surfaceIndices = new HashSet<int>(surfaces.Select(static surface => surface.SurfaceIndex));
        var counts = new Dictionary<uint, int>();

        uint bestGroupObjectId = 0;
        int bestCount = 0;
        for (int i = 0; i < linkEntries.Count; i++)
        {
            CorePm4MslkEntry link = linkEntries[i];
            if (link.GroupObjectId == 0)
                continue;

            if (!LinkReferencesSurface(link, surfaceIndices, surfaceCount))
                continue;

            int nextCount = 1;
            if (counts.TryGetValue(link.GroupObjectId, out int existingCount))
                nextCount = existingCount + 1;
            counts[link.GroupObjectId] = nextCount;

            if (nextCount > bestCount)
            {
                bestCount = nextCount;
                bestGroupObjectId = link.GroupObjectId;
            }
        }

        return bestGroupObjectId;
    }

    internal static List<MprlEntry> CollectLinkedPositionRefs(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        var refs = new List<MprlEntry>();
        IReadOnlyList<CorePm4MslkEntry> linkEntries = pm4.KnownChunks.Mslk;
        IReadOnlyList<MprlEntry> positionRefs = pm4.KnownChunks.Mprl;
        if (surfaces.Count == 0 || linkEntries.Count == 0 || positionRefs.Count == 0)
            return refs;

        int surfaceCount = pm4.KnownChunks.Msur.Count;
        var surfaceIndices = new HashSet<int>(surfaces.Select(static surface => surface.SurfaceIndex));
        var seenRefIndices = new HashSet<int>();
        HashSet<uint> groupObjectIds = CollectMslkGroupObjectIds(linkEntries, surfaceIndices, surfaceCount);

        if (groupObjectIds.Count > 0)
        {
            for (int i = 0; i < linkEntries.Count; i++)
            {
                CorePm4MslkEntry link = linkEntries[i];
                if (link.GroupObjectId == 0 || !groupObjectIds.Contains(link.GroupObjectId))
                    continue;
                if ((uint)link.RefIndex >= (uint)positionRefs.Count)
                    continue;
                if (!seenRefIndices.Add(link.RefIndex))
                    continue;

                refs.Add(positionRefs[link.RefIndex]);
            }

            if (refs.Count > 0)
                return refs;
        }

        for (int i = 0; i < linkEntries.Count; i++)
        {
            CorePm4MslkEntry link = linkEntries[i];
            if ((uint)link.RefIndex >= (uint)positionRefs.Count)
                continue;

            if (!LinkReferencesSurface(link, surfaceIndices, surfaceCount))
                continue;

            if (!seenRefIndices.Add(link.RefIndex))
                continue;

            refs.Add(positionRefs[link.RefIndex]);
        }

        return refs;
    }

    internal static HashSet<uint> CollectMslkGroupObjectIds(
        IReadOnlyList<CorePm4MslkEntry> linkEntries,
        HashSet<int> surfaceIndices,
        int surfaceCount)
    {
        var groupObjectIds = new HashSet<uint>();
        for (int i = 0; i < linkEntries.Count; i++)
        {
            CorePm4MslkEntry link = linkEntries[i];
            if (link.GroupObjectId == 0)
                continue;
            if (!LinkReferencesSurface(link, surfaceIndices, surfaceCount))
                continue;

            groupObjectIds.Add(link.GroupObjectId);
        }

        return groupObjectIds;
    }

    internal static bool LinkReferencesSurface(MslkEntry link, HashSet<int> surfaceIndices, int surfaceCount)
    {
        // The current shared PM4 reader exposes surface linkage through RefIndex.
        if (link.RefIndex < surfaceCount && surfaceIndices.Contains(link.RefIndex))
            return true;

        return false;
    }

    internal static Pm4LinkedPositionRefSummary SummarizeLinkedPositionRefs(IReadOnlyList<MprlEntry> positionRefs)
    {
        return FromCorePm4LinkedPositionRefSummary(
            CorePm4PlacementMath.SummarizeLinkedPositionRefs(ConvertToCorePm4PositionRefs(positionRefs)));
    }

    internal static bool TryComputePlanarPrincipalYaw(
        IReadOnlyList<Vector3> objectVertices,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        out float yawRadians)
    {
        yawRadians = 0f;
        if (objectVertices.Count < 3)
            return false;

        int sampleCount = Math.Min(512, objectVertices.Count);
        int stride = Math.Max(1, objectVertices.Count / sampleCount);
        double meanX = 0d;
        double meanY = 0d;
        int used = 0;

        for (int i = 0; i < objectVertices.Count; i += stride)
        {
            Vector3 world = ConvertPm4VertexToWorld(objectVertices[i], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            meanX += world.X;
            meanY += world.Y;
            used++;
        }

        if (used < 3)
            return false;

        meanX /= used;
        meanY /= used;

        double covXX = 0d;
        double covYY = 0d;
        double covXY = 0d;
        for (int i = 0; i < objectVertices.Count; i += stride)
        {
            Vector3 world = ConvertPm4VertexToWorld(objectVertices[i], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            double dx = world.X - meanX;
            double dy = world.Y - meanY;
            covXX += dx * dx;
            covYY += dy * dy;
            covXY += dx * dy;
        }

        if (covXX + covYY < 1e-4)
            return false;

        yawRadians = 0.5f * (float)Math.Atan2(2.0 * covXY, covXX - covYY);
        return true;
    }

    internal static Vector3 ComputeSurfaceRendererCentroid(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3 worldPivot,
        float worldYawCorrectionRadians)
    {
        List<Vector3> objectVertices = CollectSurfaceVertices(pm4, surfaces);
        if (objectVertices.Count == 0)
            return Vector3.Zero;

        Vector3 centroid = Vector3.Zero;
        for (int i = 0; i < objectVertices.Count; i++)
            centroid += objectVertices[i];
        centroid /= objectVertices.Count;

        return ConvertPm4VertexToRenderer(
            centroid,
            tileX,
            tileY,
            useTileLocalCoordinates,
            axisConvention,
            planarTransform,
            worldPivot,
            worldYawCorrectionRadians);
    }

    internal static List<Pm4LineSegment> BuildCk24ObjectLines(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3 worldPivot,
        float worldYawCorrectionRadians,
        int lineBudget,
        ref int rejectedLongEdges)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var lines = new List<Pm4LineSegment>();
        var uniqueEdges = new HashSet<ulong>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            if (lines.Count >= lineBudget)
                break;

            int firstIndex = (int)surface.MsviFirstIndex;
            int surfaceIndexCount = surface.IndexCount;
            if (surfaceIndexCount < 2 || firstIndex < 0 || firstIndex >= meshIndices.Count)
                continue;

            int endExclusive = Math.Min(firstIndex + surfaceIndexCount, meshIndices.Count);
            if (endExclusive - firstIndex < 2)
                continue;

            int prevVertex = (int)meshIndices[firstIndex];
            if ((uint)prevVertex >= (uint)meshVertices.Count)
                continue;

            for (int idx = firstIndex + 1; idx < endExclusive && lines.Count < lineBudget; idx++)
            {
                int nextVertex = (int)meshIndices[idx];
                if ((uint)nextVertex >= (uint)meshVertices.Count)
                    continue;

                AddUniqueEdge(pm4, prevVertex, nextVertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
                prevVertex = nextVertex;
            }

            // Close each surface loop so CK24 objects stay visually self-contained.
            if (lines.Count < lineBudget)
            {
                int firstVertex = (int)meshIndices[firstIndex];
                int lastVertex = (int)meshIndices[endExclusive - 1];
                if ((uint)firstVertex < (uint)meshVertices.Count && (uint)lastVertex < (uint)meshVertices.Count)
                    AddUniqueEdge(pm4, lastVertex, firstVertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            }
        }

        return lines;
    }

    internal static List<Pm4Triangle> BuildCk24ObjectTriangles(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3 worldPivot,
        float worldYawCorrectionRadians,
        int triangleBudget,
        out int rejectedOutOfRange)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var triangles = new List<Pm4Triangle>();
        rejectedOutOfRange = 0;

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            if (triangles.Count >= triangleBudget)
                break;

            int firstIndex = (int)surface.MsviFirstIndex;
            int surfaceIndexCount = surface.IndexCount;
            if (surfaceIndexCount < 3 || firstIndex < 0 || firstIndex >= meshIndices.Count)
            {
                rejectedOutOfRange++;
                continue;
            }

            int endExclusive = Math.Min(firstIndex + surfaceIndexCount, meshIndices.Count);
            int indexCount = endExclusive - firstIndex;
            if (indexCount < 3)
            {
                rejectedOutOfRange++;
                continue;
            }

            // Most PM4 surfaces are listed as loops; use a fan from the first vertex.
            int i0 = (int)meshIndices[firstIndex];
            if ((uint)i0 >= (uint)meshVertices.Count)
            {
                rejectedOutOfRange++;
                continue;
            }

            Vector3 v0 = ConvertPm4VertexToRenderer(meshVertices[i0], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            for (int idx = firstIndex + 1; idx + 1 < endExclusive && triangles.Count < triangleBudget; idx++)
            {
                int i1 = (int)meshIndices[idx];
                int i2 = (int)meshIndices[idx + 1];
                if ((uint)i1 >= (uint)meshVertices.Count || (uint)i2 >= (uint)meshVertices.Count)
                    continue;

                Vector3 v1 = ConvertPm4VertexToRenderer(meshVertices[i1], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
                Vector3 v2 = ConvertPm4VertexToRenderer(meshVertices[i2], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
                triangles.Add(planarTransform.InvertsWinding
                    ? new Pm4Triangle(v0, v2, v1)
                    : new Pm4Triangle(v0, v1, v2));
            }
        }

        return triangles;
    }

    /// <summary>
    /// Builds the vertical wall faces that belong to a set of surfaces, from the MSLK path windows
    /// that reference them (MSLK.RefIndex -> MSUR) through MSPI into MSPV.
    /// </summary>
    /// <remarks>
    /// Each window is one planar polygon, emitted as a fan. Measured over the 616-file development
    /// corpus: 98.05% of windows hold exactly 4 indices, 1.84% hold 6, 99.6% are coplanar, and zero
    /// of 598,790 faces have Z as their dominant normal component. They are walls; MSUR is floors.
    /// This is the geometry the viewer has never drawn.
    /// </remarks>
    internal static List<Pm4Triangle> BuildMslkWallTriangles(
        Pm4File pm4,
        IReadOnlyCollection<int> surfaceIndices,
        IReadOnlyDictionary<int, List<int>> mslkWindowsBySurface,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int triangleBudget,
        List<Pm4LineSegment> wallLines,
        int lineBudget,
        out int wallFaceCount)
    {
        var triangles = new List<Pm4Triangle>();
        wallFaceCount = 0;

        IReadOnlyList<Vector3> pathVertices = pm4.KnownChunks.Mspv;
        IReadOnlyList<uint> pathIndices = pm4.KnownChunks.Mspi;
        if (pathVertices.Count == 0 || pathIndices.Count == 0 || mslkWindowsBySurface.Count == 0)
            return triangles;

        var scratch = new List<Vector3>(8);

        foreach (int surfaceIndex in surfaceIndices)
        {
            if (!mslkWindowsBySurface.TryGetValue(surfaceIndex, out List<int>? linkIndices))
                continue;

            foreach (int linkIndex in linkIndices)
            {
                if (triangles.Count >= triangleBudget)
                    return triangles;

                MslkEntry link = pm4.KnownChunks.Mslk[linkIndex];
                int first = link.MspiFirstIndex;
                int count = link.MspiIndexCount;
                if (first < 0 || count < 3 || first + count > pathIndices.Count)
                    continue;

                scratch.Clear();
                for (int i = 0; i < count; i++)
                {
                    uint vertexIndex = pathIndices[first + i];
                    if (vertexIndex < (uint)pathVertices.Count)
                        scratch.Add(ConvertPm4VertexToRenderer(pathVertices[(int)vertexIndex], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform));
                }

                if (scratch.Count < 3)
                    continue;

                wallFaceCount++;
                for (int i = 1; i + 1 < scratch.Count && triangles.Count < triangleBudget; i++)
                {
                    triangles.Add(planarTransform.InvertsWinding
                        ? new Pm4Triangle(scratch[0], scratch[i + 1], scratch[i])
                        : new Pm4Triangle(scratch[0], scratch[i], scratch[i + 1]));
                }

                // The overlay draws lines unless "PM4 Solid Fill" is on, so a triangle-only wall
                // would be invisible in the default wireframe view. Emit the quad outline too.
                for (int i = 0; i < scratch.Count && wallLines.Count < lineBudget; i++)
                    wallLines.Add(new Pm4LineSegment(scratch[i], scratch[(i + 1) % scratch.Count]));
            }
        }

        return triangles;
    }

    /// <summary>
    /// Indexes MSLK path windows by the MSUR surface they reference, once per tile.
    /// </summary>
    /// <remarks>
    /// Entries with a negative MspiFirstIndex carry no path at all — 53% of the corpus — and are
    /// skipped here rather than being treated as empty geometry. Prior art reads them as doodad
    /// placements; that is a separate question from wall rendering.
    /// </remarks>
    internal static Dictionary<int, List<int>> BuildMslkWindowsBySurface(Pm4File pm4)
    {
        var windowsBySurface = new Dictionary<int, List<int>>();
        int surfaceCount = pm4.KnownChunks.Msur.Count;
        IReadOnlyList<MslkEntry> links = pm4.KnownChunks.Mslk;

        for (int i = 0; i < links.Count; i++)
        {
            MslkEntry link = links[i];
            if (link.MspiFirstIndex < 0 || link.MspiIndexCount < 3 || link.RefIndex >= surfaceCount)
                continue;

            if (!windowsBySurface.TryGetValue(link.RefIndex, out List<int>? linkIndices))
            {
                linkIndices = [];
                windowsBySurface[link.RefIndex] = linkIndices;
            }

            linkIndices.Add(i);
        }

        return windowsBySurface;
    }

    internal static List<Pm4LineSegment> BuildFallbackMeshLines(
        Pm4File pm4,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int lineBudget,
        ref int rejectedLongEdges)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var lines = new List<Pm4LineSegment>();
        var uniqueEdges = new HashSet<ulong>();

        for (int i = 0; i + 2 < meshIndices.Count && lines.Count < lineBudget; i += 3)
        {
            int i0 = (int)meshIndices[i];
            int i1 = (int)meshIndices[i + 1];
            int i2 = (int)meshIndices[i + 2];

            if ((uint)i0 >= (uint)meshVertices.Count ||
                (uint)i1 >= (uint)meshVertices.Count ||
                (uint)i2 >= (uint)meshVertices.Count)
                continue;

            AddUniqueEdge(pm4, i0, i1, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            AddUniqueEdge(pm4, i1, i2, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            AddUniqueEdge(pm4, i2, i0, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
        }

        return lines;
    }

    internal static List<List<MsurEntry>> SplitSurfaceGroupByConnectivity(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var components = new List<List<MsurEntry>>();
        if (surfaces.Count == 0)
            return components;
        if (surfaces.Count == 1)
        {
            components.Add(new List<MsurEntry> { surfaces[0] });
            return components;
        }

        var surfaceVertices = new List<List<int>>(surfaces.Count);
        var vertexToSurfaceIndices = new Dictionary<int, List<int>>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            var vertices = new List<int>();
            var unique = new HashSet<int>();

            if (surface.IndexCount > 0 && firstIndex >= 0 && endExclusive > firstIndex)
            {
                for (int idx = firstIndex; idx < endExclusive; idx++)
                {
                    int vertexIndex = (int)meshIndices[idx];
                    if ((uint)vertexIndex >= (uint)meshVertices.Count)
                        continue;
                    if (!unique.Add(vertexIndex))
                        continue;

                    vertices.Add(vertexIndex);
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? owners))
                    {
                        owners = new List<int>();
                        vertexToSurfaceIndices[vertexIndex] = owners;
                    }
                    owners.Add(s);
                }
            }

            surfaceVertices.Add(vertices);
        }

        var visited = new bool[surfaces.Count];
        var queue = new Queue<int>();
        for (int start = 0; start < surfaces.Count; start++)
        {
            if (visited[start])
                continue;

            visited[start] = true;
            queue.Enqueue(start);
            var component = new List<MsurEntry>();

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                component.Add(surfaces[current]);

                List<int> vertices = surfaceVertices[current];
                for (int v = 0; v < vertices.Count; v++)
                {
                    int vertexIndex = vertices[v];
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? neighbors))
                        continue;

                    for (int n = 0; n < neighbors.Count; n++)
                    {
                        int neighborSurface = neighbors[n];
                        if (visited[neighborSurface])
                            continue;

                        visited[neighborSurface] = true;
                        queue.Enqueue(neighborSurface);
                    }
                }
            }

            components.Add(component);
        }

        return components;
    }

    internal static List<List<MsurEntry>> SplitSurfaceGroupByMscnRef(IReadOnlyList<MsurEntry> surfaces)
    {
        if (surfaces.Count <= 1)
            return new List<List<MsurEntry>> { surfaces.ToList() };

        var groups = surfaces
            .GroupBy(static surface => surface.MscnRefIndex)
            .Select(static group => group.ToList())
            .Where(static group => group.Count > 0)
            .ToList();

        return groups.Count > 0 ? groups : new List<List<MsurEntry>> { surfaces.ToList() };
    }

    internal static IReadOnlyList<Pm4ConnectorKey> BuildCk24ConnectorKeys(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        CorePm4PlacementSolution placement)
    {
        if (surfaces.Count == 0 || pm4.KnownChunks.Mscn.Count == 0)
            return Array.Empty<Pm4ConnectorKey>();

        return CorePm4PlacementMath.BuildConnectorKeys(
                pm4.KnownChunks.Mscn,
                ConvertToCorePm4Surfaces(surfaces),
                placement)
            .Select(FromCorePm4ConnectorKey)
            .ToList();
    }

    internal static CorePm4ConnectorKey ToCorePm4ConnectorKey(Pm4ConnectorKey key) => new(key.X, key.Y, key.Z);

    internal static Pm4ConnectorKey FromCorePm4ConnectorKey(CorePm4ConnectorKey key) => new(key.X, key.Y, key.Z);

    internal static void IncludePointInBounds(Vector3 point, ref Vector3 boundsMin, ref Vector3 boundsMax, ref bool hasBounds)
    {
        if (!hasBounds)
        {
            boundsMin = point;
            boundsMax = point;
            hasBounds = true;
            return;
        }

        boundsMin = Vector3.Min(boundsMin, point);
        boundsMax = Vector3.Max(boundsMax, point);
    }

    internal static byte SelectDominantSurfaceValue(IReadOnlyList<MsurEntry> surfaces, Func<MsurEntry, byte> selector)
    {
        if (surfaces.Count == 0)
            return 0;

        Span<int> counts = stackalloc int[256];
        for (int i = 0; i < surfaces.Count; i++)
            counts[selector(surfaces[i])]++;

        int bestCount = -1;
        byte bestValue = 0;
        for (int i = 0; i < counts.Length; i++)
        {
            int count = counts[i];
            if (count <= bestCount)
                continue;

            bestCount = count;
            bestValue = (byte)i;
        }

        return bestValue;
    }

    internal static uint SelectDominantSurfaceValue(IReadOnlyList<MsurEntry> surfaces, Func<MsurEntry, uint> selector)
    {
        if (surfaces.Count == 0)
            return 0;

        var counts = new Dictionary<uint, int>();
        uint bestValue = 0;
        int bestCount = -1;

        for (int i = 0; i < surfaces.Count; i++)
        {
            uint value = selector(surfaces[i]);
            int count = 1;
            if (counts.TryGetValue(value, out int existing))
                count = existing + 1;
            counts[value] = count;

            if (count > bestCount)
            {
                bestCount = count;
                bestValue = value;
            }
        }

        return bestValue;
    }

    internal static List<Vector3> BuildPm4PositionRefMarkers(Pm4File pm4, int limit)
    {
        var markers = new List<Vector3>();
        IReadOnlyList<MprlEntry> positionRefs = pm4.KnownChunks.Mprl;
        int count = Math.Min(limit, positionRefs.Count);
        for (int i = 0; i < count; i++)
        {
            Vector3 world = ConvertMprlPositionToWorld(positionRefs[i].Position);
            markers.Add(new Vector3(
                WoWConstants.MapOrigin - world.Y,
                WoWConstants.MapOrigin - world.X,
                world.Z + 0.5f));
        }

        return markers;
    }

    internal static List<Pm4Triangle> BuildFallbackMeshTriangles(
        Pm4File pm4,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int triangleBudget)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var triangles = new List<Pm4Triangle>();

        for (int i = 0; i + 2 < meshIndices.Count && triangles.Count < triangleBudget; i += 3)
        {
            int i0 = (int)meshIndices[i];
            int i1 = (int)meshIndices[i + 1];
            int i2 = (int)meshIndices[i + 2];

            if ((uint)i0 >= (uint)meshVertices.Count ||
                (uint)i1 >= (uint)meshVertices.Count ||
                (uint)i2 >= (uint)meshVertices.Count)
                continue;

            Vector3 v0 = ConvertPm4VertexToRenderer(meshVertices[i0], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            Vector3 v1 = ConvertPm4VertexToRenderer(meshVertices[i1], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            Vector3 v2 = ConvertPm4VertexToRenderer(meshVertices[i2], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            triangles.Add(planarTransform.InvertsWinding
                ? new Pm4Triangle(v0, v2, v1)
                : new Pm4Triangle(v0, v1, v2));
        }

        return triangles;
    }

    internal static void AddUniqueEdge(Pm4File pm4, int ia, int ib,
        int tileX, int tileY, bool useTileLocalCoordinates, Pm4AxisConvention axisConvention, Pm4PlanarTransform planarTransform,
        HashSet<ulong> uniqueEdges, List<Pm4LineSegment> lines, int tileLineBudget,
        ref int rejectedLongEdges,
        Vector3? worldPivot = null,
        float worldYawCorrectionRadians = 0f)
    {
        if (ia == ib || lines.Count >= tileLineBudget)
            return;

        ulong key = PackEdgeKey(ia, ib);
        if (!uniqueEdges.Add(key))
            return;

        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        Vector3 from = ConvertPm4VertexToRenderer(meshVertices[ia], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, worldPivot, worldYawCorrectionRadians);
        Vector3 to = ConvertPm4VertexToRenderer(meshVertices[ib], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, worldPivot, worldYawCorrectionRadians);

        if (Vector3.DistanceSquared(from, to) > Pm4MaxEdgeLength * Pm4MaxEdgeLength)
        {
            rejectedLongEdges++;
            return;
        }

        lines.Add(new Pm4LineSegment(from, to));
    }

    internal static ulong PackEdgeKey(int ia, int ib)
    {
        uint lo = ia < ib ? (uint)ia : (uint)ib;
        uint hi = ia < ib ? (uint)ib : (uint)ia;
        return ((ulong)lo << 32) | hi;
    }
}
