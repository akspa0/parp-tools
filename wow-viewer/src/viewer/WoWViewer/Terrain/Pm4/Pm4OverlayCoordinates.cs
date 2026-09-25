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
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayColors;

namespace WoWViewer.Terrain;

/// <summary>Pure static PM4 overlay helpers moved verbatim from <c>WorldScene</c> (Epic 251 U-01 E1).</summary>
internal static class Pm4OverlayCoordinates
{

    internal static Pm4AxisConvention DetectPm4AxisConvention(Pm4File pm4)
    {
        // Pick the basis that yields the most horizontal (floor-like) triangles.
        // This avoids forcing users to manually undo a 90-degree wall orientation.
        var candidates = new[]
        {
            Pm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4AxisConvention.YZPlaneXUp
        };

        Pm4AxisConvention bestConvention = Pm4AxisConvention.XYPlaneZUp;
        float bestScore = float.MinValue;
        foreach (Pm4AxisConvention candidate in candidates)
        {
            float score = ScoreAxisConventionByTriangleNormals(pm4, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestConvention = candidate;
            }
        }

        if (bestScore > 0f)
            return bestConvention;

        return DetectAxisConventionByRanges(pm4.KnownChunks.Msvt);
    }

    internal static Pm4AxisConvention DetectPm4AxisConvention(Pm4File pm4, IEnumerable<MsurEntry> surfaces)
    {
        var surfaceList = surfaces as List<MsurEntry> ?? surfaces.ToList();
        if (surfaceList.Count == 0)
            return DetectPm4AxisConvention(pm4);

        var candidates = new[]
        {
            Pm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4AxisConvention.YZPlaneXUp
        };

        Pm4AxisConvention bestConvention = Pm4AxisConvention.XYPlaneZUp;
        float bestScore = float.MinValue;
        foreach (Pm4AxisConvention candidate in candidates)
        {
            float score = ScoreAxisConventionBySurfaceNormals(pm4, surfaceList, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestConvention = candidate;
            }
        }

        if (bestScore > 0f)
            return bestConvention;

        List<Vector3> groupVertices = CollectSurfaceVertices(pm4, surfaceList);
        return groupVertices.Count > 0
            ? DetectAxisConventionByRanges(groupVertices)
            : DetectPm4AxisConvention(pm4);
    }

    internal static CorePm4AxisConvention ToCoreAxisConvention(Pm4AxisConvention convention)
    {
        return convention switch
        {
            Pm4AxisConvention.XZPlaneYUp => CorePm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.YZPlaneXUp => CorePm4AxisConvention.YZPlaneXUp,
            _ => CorePm4AxisConvention.XYPlaneZUp
        };
    }

    internal static List<CorePm4MsurEntry> ConvertToCorePm4Surfaces(IReadOnlyList<MsurEntry> surfaces)
    {
        return surfaces as List<CorePm4MsurEntry> ?? surfaces.ToList();
    }

    internal static List<CorePm4MprlEntry> ConvertToCorePm4PositionRefs(IReadOnlyList<MprlEntry> positionRefs)
    {
        return positionRefs as List<CorePm4MprlEntry> ?? positionRefs.ToList();
    }

    internal static Pm4LinkedPositionRefSummary FromCorePm4LinkedPositionRefSummary(CorePm4LinkedPositionRefSummary summary)
    {
        return new Pm4LinkedPositionRefSummary(
            summary.TotalCount,
            summary.NormalCount,
            summary.TerminatorCount,
            summary.FloorMin,
            summary.FloorMax,
            summary.HeadingMinDegrees,
            summary.HeadingMaxDegrees,
            summary.HeadingMeanDegrees);
    }

    /// <summary>
    /// Builds the placement for a surface group in the canonical frame, with no yaw correction.
    /// </summary>
    /// <remarks>
    /// The pivot is still the group's real world centroid, because selection and connector merging
    /// use it. Only the fitted rotation is dropped — see
    /// <see cref="ResolveCk24CoordinateModeResolution"/> for the evidence that it was wrong.
    /// </remarks>
    internal static CorePm4PlacementSolution ResolvePlacementSolution(
        Pm4File pm4,
        IEnumerable<MsurEntry> surfaces,
        IReadOnlyList<MprlEntry>? anchorPositionRefs,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention)
    {
        var surfaceList = surfaces as List<MsurEntry> ?? surfaces.ToList();
        CorePm4CoordinateMode coordinateMode = CanonicalCoordinateModeResolution.CoordinateMode;
        Pm4PlanarTransform planarTransform = CanonicalCoordinateModeResolution.PlanarTransform;

        Vector3 worldPivot = CorePm4PlacementMath.ComputeSurfaceWorldCentroid(
            pm4.KnownChunks.Msvt,
            pm4.KnownChunks.Msvi,
            ConvertToCorePm4Surfaces(surfaceList),
            tileX,
            tileY,
            coordinateMode,
            ToCoreAxisConvention(axisConvention),
            planarTransform);

        return new CorePm4PlacementSolution(
            tileX,
            tileY,
            coordinateMode,
            ToCoreAxisConvention(axisConvention),
            planarTransform,
            worldPivot,
            WorldYawCorrectionRadians: 0f);
    }

    internal static float ScoreAxisConventionByTriangleNormals(Pm4File pm4, Pm4AxisConvention convention)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        if (meshVertices.Count == 0 || meshIndices.Count < 3)
            return 0f;

        float sum = 0f;
        int samples = 0;
        const int maxSamples = 1024;

        for (int i = 0; i + 2 < meshIndices.Count && samples < maxSamples; i += 3)
        {
            int i0 = (int)meshIndices[i];
            int i1 = (int)meshIndices[i + 1];
            int i2 = (int)meshIndices[i + 2];
            if ((uint)i0 >= (uint)meshVertices.Count ||
                (uint)i1 >= (uint)meshVertices.Count ||
                (uint)i2 >= (uint)meshVertices.Count)
                continue;

            Vector3 a = ConvertPm4VertexToWorld(meshVertices[i0], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));
            Vector3 b = ConvertPm4VertexToWorld(meshVertices[i1], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));
            Vector3 c = ConvertPm4VertexToWorld(meshVertices[i2], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));

            Vector3 normal = Vector3.Cross(b - a, c - a);
            float length = normal.Length();
            if (length < 1e-5f)
                continue;

            // Higher |normal.Z| means more floor-like orientation in this renderer.
            sum += MathF.Abs(normal.Z / length);
            samples++;
        }

        return samples > 0 ? sum / samples : 0f;
    }

    internal static float ScoreAxisConventionBySurfaceNormals(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces, Pm4AxisConvention convention)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        if (meshVertices.Count == 0 || meshIndices.Count < 3 || surfaces.Count == 0)
            return 0f;

        float sum = 0f;
        int samples = 0;
        const int maxSamples = 1024;

        for (int s = 0; s < surfaces.Count && samples < maxSamples; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            if (surface.IndexCount < 3 || firstIndex < 0 || endExclusive - firstIndex < 3)
                continue;

            int i0 = (int)meshIndices[firstIndex];
            if ((uint)i0 >= (uint)meshVertices.Count)
                continue;

            Vector3 a = ConvertPm4VertexToWorld(meshVertices[i0], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));
            for (int idx = firstIndex + 1; idx + 1 < endExclusive && samples < maxSamples; idx++)
            {
                int i1 = (int)meshIndices[idx];
                int i2 = (int)meshIndices[idx + 1];
                if ((uint)i1 >= (uint)meshVertices.Count || (uint)i2 >= (uint)meshVertices.Count)
                    continue;

                Vector3 b = ConvertPm4VertexToWorld(meshVertices[i1], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));
                Vector3 c = ConvertPm4VertexToWorld(meshVertices[i2], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));

                Vector3 normal = Vector3.Cross(b - a, c - a);
                float length = normal.Length();
                if (length < 1e-5f)
                    continue;

                sum += MathF.Abs(normal.Z / length);
                samples++;
            }
        }

        return samples > 0 ? sum / samples : 0f;
    }

    internal static List<Vector3> CollectSurfaceVertices(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var vertices = new List<Vector3>();
        var seen = new HashSet<int>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            if (surface.IndexCount <= 0 || firstIndex < 0 || endExclusive <= firstIndex)
                continue;

            for (int idx = firstIndex; idx < endExclusive; idx++)
            {
                int vertexIndex = (int)meshIndices[idx];
                if ((uint)vertexIndex >= (uint)meshVertices.Count)
                    continue;
                if (!seen.Add(vertexIndex))
                    continue;

                vertices.Add(meshVertices[vertexIndex]);
            }
        }

        return vertices;
    }

    internal static Pm4AxisConvention DetectAxisConventionByRanges(IReadOnlyList<Vector3> vertices)
    {
        if (vertices.Count == 0)
            return Pm4AxisConvention.XYPlaneZUp;

        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float minZ = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float maxZ = float.MinValue;

        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 v = vertices[i];
            if (v.X < minX) minX = v.X;
            if (v.Y < minY) minY = v.Y;
            if (v.Z < minZ) minZ = v.Z;
            if (v.X > maxX) maxX = v.X;
            if (v.Y > maxY) maxY = v.Y;
            if (v.Z > maxZ) maxZ = v.Z;
        }

        float rangeX = maxX - minX;
        float rangeY = maxY - minY;
        float rangeZ = maxZ - minZ;
        const float tieTolerance = 8f;

        if (rangeY + tieTolerance < rangeX && rangeY + tieTolerance < rangeZ)
            return Pm4AxisConvention.XZPlaneYUp;
        if (rangeZ + tieTolerance < rangeX && rangeZ + tieTolerance < rangeY)
            return Pm4AxisConvention.XYPlaneZUp;
        if (rangeX + tieTolerance < rangeY && rangeX + tieTolerance < rangeZ)
            return Pm4AxisConvention.YZPlaneXUp;

        // Ambiguous ranges: default to WoW-style XY plane with Z up.
        return Pm4AxisConvention.XYPlaneZUp;
    }

    internal static bool IsLikelyTileLocal(IReadOnlyList<Vector3> vertices)
    {
        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float minZ = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float maxZ = float.MinValue;

        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 v = vertices[i];
            if (v.X < minX) minX = v.X;
            if (v.Y < minY) minY = v.Y;
            if (v.Z < minZ) minZ = v.Z;
            if (v.X > maxX) maxX = v.X;
            if (v.Y > maxY) maxY = v.Y;
            if (v.Z > maxZ) maxZ = v.Z;
        }

        const float tolerance = 64f;
        float tileSpan = Pm4CoordinateService.TileSize;

        bool xyLocal = minX >= -tolerance && minY >= -tolerance &&
                       maxX <= tileSpan + tolerance && maxY <= tileSpan + tolerance;
        bool xzLocal = minX >= -tolerance && minZ >= -tolerance &&
                       maxX <= tileSpan + tolerance && maxZ <= tileSpan + tolerance;
        bool yzLocal = minY >= -tolerance && minZ >= -tolerance &&
                       maxY <= tileSpan + tolerance && maxZ <= tileSpan + tolerance;

        return xyLocal || xzLocal || yzLocal;
    }

    internal static Vector3 ConvertPm4VertexToWorld(Vector3 pm4Vertex, int tileX, int tileY, bool useTileLocalCoordinates, Pm4AxisConvention axisConvention, Pm4PlanarTransform planarTransform)
    {
        float localU;
        float localV;
        float localUp;

        switch (axisConvention)
        {
            case Pm4AxisConvention.XZPlaneYUp:
                localU = pm4Vertex.X;
                localV = pm4Vertex.Z;
                localUp = pm4Vertex.Y;
                break;
            case Pm4AxisConvention.YZPlaneXUp:
                localU = pm4Vertex.Y;
                localV = pm4Vertex.Z;
                localUp = pm4Vertex.X;
                break;
            case Pm4AxisConvention.XYPlaneZUp:
            default:
                // The older PM4 R&D exporter that matched placed WMO/M2 assets used
                // a fixed MSVT planar order of (Y, X, Z), not raw (X, Y, Z).
                // Keep Z-up, but preserve that planar basis here so the viewer stops
                // trying to approximate it with per-object swap/invert heuristics.
                localU = pm4Vertex.Y;
                localV = pm4Vertex.X;
                localUp = pm4Vertex.Z;
                break;
        }

        if (planarTransform.SwapPlanarAxes)
            (localU, localV) = (localV, localU);

        float tileSpan = Pm4CoordinateService.TileSize;
        float worldX;
        float worldY;

        if (useTileLocalCoordinates)
        {
            float mappedU = planarTransform.InvertU ? tileSpan - localU : localU;
            float mappedV = planarTransform.InvertV ? tileSpan - localV : localV;

            // Viewer world uses the standard WoW tile convention where file tile X advances along
            // world Y and file tile Y advances along world X. Keeping these unswapped only happens
            // to look correct on origin tiles and shifts non-origin tile-local PM4 onto the wrong grid.
            worldX = tileY * tileSpan + mappedU;
            worldY = tileX * tileSpan + mappedV;
        }
        else
        {
            if (planarTransform.InvertU)
                localU = -localU;
            if (planarTransform.InvertV)
                localV = -localV;

            worldX = localU;
            worldY = localV;
        }

        return new Vector3(worldX, worldY, localUp);
    }

    internal static Vector3 RotateWorldAroundPivot(Vector3 world, Vector3 pivot, float yawRadians)
    {
        if (MathF.Abs(yawRadians) < 1e-6f)
            return world;

        float sin = MathF.Sin(yawRadians);
        float cos = MathF.Cos(yawRadians);
        float dx = world.X - pivot.X;
        float dy = world.Y - pivot.Y;

        float rx = dx * cos - dy * sin;
        float ry = dx * sin + dy * cos;
        return new Vector3(pivot.X + rx, pivot.Y + ry, world.Z);
    }

    internal static Vector3 ConvertWorldToRenderer(Vector3 world)
    {
        return new Vector3(
            WoWConstants.MapOrigin - world.Y,
            WoWConstants.MapOrigin - world.X,
            world.Z + 0.5f);
    }

    internal static float ConvertWorldYawCorrectionToRendererRotationRadians(float worldYawCorrectionRadians)
    {
        return -worldYawCorrectionRadians;
    }

    internal static Vector3 ConvertPm4VertexToRenderer(
        Vector3 pm4Vertex,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3? worldPivot = null,
        float worldYawCorrectionRadians = 0f)
    {
        Vector3 world = ConvertPm4VertexToWorld(pm4Vertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
        if (worldPivot.HasValue && MathF.Abs(worldYawCorrectionRadians) > 1e-6f)
            world = RotateWorldAroundPivot(world, worldPivot.Value, worldYawCorrectionRadians);

        // Canonical world->renderer transform used across terrain/object pipelines.
        // rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX, rendererZ = wowZ
        return ConvertWorldToRenderer(world);
    }
}
