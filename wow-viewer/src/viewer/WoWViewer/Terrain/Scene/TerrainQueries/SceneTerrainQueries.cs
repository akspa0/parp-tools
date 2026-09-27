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
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;
using WowViewer.Core.Runtime.World.Selection;

namespace WoWViewer.Terrain;

/// <summary>
/// Queries against the loaded terrain and placed objects: camera-path collision, loaded-terrain height sampling, and MDX terrain occlusion.
/// Moved verbatim from <see cref="WorldScene"/> (Spec 255). Scene state it still needs comes
/// only through <see cref="IWorldSceneHost"/>; the bridge members below keep the names the moved
/// code used inside WorldScene, so no moved body was edited.
/// </summary>
public sealed class SceneTerrainQueries
{
    private readonly IWorldSceneHost _host;

    internal SceneTerrainQueries(IWorldSceneHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the WorldScene members).
    private ref bool _instancesDirty => ref _host.InstancesDirty;
    private TerrainManager _terrainManager => _host.TerrainManager;
    private ref List<ObjectInstance> _wmoInstances => ref _host.WmoInstances;
    private void RebuildInstanceLists() => _host.RebuildInstanceLists();
    private static bool AreFiniteOrderedBounds(Vector3 min, Vector3 max) => WorldScene.AreFiniteOrderedBounds(min, max);

    internal bool IsMdxFullyOccludedByTerrain(in ObjectInstance inst)
    {
        if (!TrySampleLoadedTerrainHeight(inst.PlacementPosition.X, inst.PlacementPosition.Y, out float terrainHeight))
            return false;

        float objectTop = MathF.Max(inst.BoundsMin.Z, inst.BoundsMax.Z);
        if (!float.IsFinite(objectTop))
            return false;

        const float terrainOcclusionMargin = 1.0f;
        return terrainHeight >= objectTop + terrainOcclusionMargin;
    }

    /// <summary>
    /// Resolves a camera-path sample against the loaded world. Terrain collision is
    /// heightfield-only; WMO collision uses the resident placement bounds as a
    /// conservative sweep volume. Both are deliberately opt-in because the viewer
    /// also supports free-fly inspection through geometry.
    /// </summary>
    public bool TryResolveCameraPathCollision(
        Vector3 previousPosition,
        Vector3 desiredPosition,
        float clearance,
        bool terrainCollision,
        bool wmoCollision,
        out Vector3 resolvedPosition)
    {
        resolvedPosition = desiredPosition;
        float safeClearance = float.IsFinite(clearance) ? Math.Clamp(clearance, 0f, 32f) : 0f;
        bool collided = false;

        if (terrainCollision && TrySampleLoadedTerrainHeight(desiredPosition.X, desiredPosition.Y, out float terrainHeight))
        {
            float minimumCameraZ = terrainHeight + safeClearance;
            if (resolvedPosition.Z < minimumCameraZ)
            {
                resolvedPosition.Z = minimumCameraZ;
                collided = true;
            }
        }

        if (wmoCollision)
        {
            if (_instancesDirty)
                RebuildInstanceLists();

            Vector3 segmentStart = previousPosition;
            Vector3 segmentEnd = resolvedPosition;
            foreach (ObjectInstance instance in _wmoInstances)
            {
                if (!AreFiniteOrderedBounds(instance.BoundsMin, instance.BoundsMax))
                    continue;

                Vector3 boundsMin = instance.BoundsMin - new Vector3(safeClearance);
                Vector3 boundsMax = instance.BoundsMax + new Vector3(safeClearance);
                if (!TrySegmentAabb(segmentStart, segmentEnd, boundsMin, boundsMax, out float entryT))
                    continue;

                bool startInside = IsPointInsideAabb(segmentStart, boundsMin, boundsMax);
                // A placement AABB is an exterior shell, not an indoor collision mesh.
                // Preserve paths that start inside a WMO instead of ejecting them from
                // the entire building; only stop an outside-to-inside sweep here.
                if (startInside)
                    continue;

                if (entryT > 0f)
                {
                    float stopT = Math.Clamp(entryT - 0.0025f, 0f, 1f);
                    resolvedPosition = Vector3.Lerp(segmentStart, segmentEnd, stopT);
                }
                else if (IsPointInsideAabb(segmentEnd, boundsMin, boundsMax))
                    resolvedPosition = segmentStart;

                collided = true;
                segmentEnd = resolvedPosition;
            }
        }

        return collided;
    }

    private static bool IsPointInsideAabb(Vector3 point, Vector3 min, Vector3 max)
        => point.X >= min.X && point.X <= max.X
            && point.Y >= min.Y && point.Y <= max.Y
            && point.Z >= min.Z && point.Z <= max.Z;

    private static bool TrySegmentAabb(Vector3 start, Vector3 end, Vector3 min, Vector3 max, out float entryT)
    {
        entryT = 0f;
        float exitT = 1f;
        Vector3 delta = end - start;
        for (int axis = 0; axis < 3; axis++)
        {
            float origin = start[axis];
            float direction = delta[axis];
            float axisMin = min[axis];
            float axisMax = max[axis];
            if (MathF.Abs(direction) < 0.000001f)
            {
                if (origin < axisMin || origin > axisMax)
                    return false;
                continue;
            }

            float inverse = 1f / direction;
            float near = (axisMin - origin) * inverse;
            float far = (axisMax - origin) * inverse;
            if (near > far)
                (near, far) = (far, near);
            entryT = MathF.Max(entryT, near);
            exitT = MathF.Min(exitT, far);
            if (entryT > exitT)
                return false;
        }

        return entryT >= 0f && entryT <= 1f;
    }

    private bool TrySampleLoadedTerrainHeight(float worldX, float worldY, out float height)
    {
        height = 0f;

        return TrySampleLoadedTerrainHeight(_terrainManager, _terrainManager.Renderer, worldX, worldY, out height);
    }

    private static bool TrySampleLoadedTerrainHeight(TerrainManager terrainManager, TerrainRenderer renderer, float worldX, float worldY, out float height)
    {
        height = 0f;

        TerrainRenderer.TerrainChunkInfo? chunkInfo = renderer.GetChunkInfoAt(worldX, worldY);
        if (!chunkInfo.HasValue)
            return false;

        if (!terrainManager.TryGetTileLoadResult(chunkInfo.Value.TileX, chunkInfo.Value.TileY, out TileLoadResult tile))
            return false;

        TerrainChunkData? chunk = tile.Chunks.FirstOrDefault(c => c.ChunkX == chunkInfo.Value.ChunkX && c.ChunkY == chunkInfo.Value.ChunkY);
        if (chunk == null || chunk.Heights == null || chunk.Heights.Length < 145)
            return false;

        float localX = chunk.WorldPosition.Y - worldY;
        float localY = chunk.WorldPosition.X - worldX;
        localX = Math.Clamp(localX, 0f, WoWConstants.ChunkSize);
        localY = Math.Clamp(localY, 0f, WoWConstants.ChunkSize);
        height = SampleHeightOuterGrid(chunk, localX, localY);
        return true;
    }

    private static float SampleHeightOuterGrid(TerrainChunkData chunk, float localX, float localY)
    {
        if (chunk.Heights == null || chunk.Heights.Length < 145)
            return chunk.WorldPosition.Z;

        float cellSize = WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        Span<float> grid = stackalloc float[9 * 9];
        grid.Clear();

        for (int i = 0; i < 145; i++)
        {
            GetChunkVertexPosition(i, out int row, out int col, out bool isInner);
            if (isInner)
                continue;

            int gridY = row / 2;
            if ((uint)gridY >= 9u || (uint)col >= 9u)
                continue;

            grid[(gridY * 9) + col] = chunk.Heights[i];
        }

        float gridX = localX / subCellSize;
        float gridYFloat = localY / subCellSize;
        int ix = Math.Clamp((int)MathF.Floor(gridX), 0, 7);
        int iy = Math.Clamp((int)MathF.Floor(gridYFloat), 0, 7);
        float fx = Math.Clamp(gridX - ix, 0f, 1f);
        float fy = Math.Clamp(gridYFloat - iy, 0f, 1f);

        float h00 = grid[(iy * 9) + ix];
        float h10 = grid[(iy * 9) + (ix + 1)];
        float h01 = grid[((iy + 1) * 9) + ix];
        float h11 = grid[((iy + 1) * 9) + (ix + 1)];

        float h0 = h00 + ((h10 - h00) * fx);
        float h1 = h01 + ((h11 - h01) * fx);
        return h0 + ((h1 - h0) * fy);
    }

    private static void GetChunkVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int currentRow = 0; currentRow < 17; currentRow++)
        {
            int rowSize = (currentRow % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = currentRow;
                col = remaining;
                isInner = (currentRow % 2 == 1);
                return;
            }

            remaining -= rowSize;
        }
    }
}
