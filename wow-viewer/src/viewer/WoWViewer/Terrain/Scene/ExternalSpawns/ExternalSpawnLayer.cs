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
/// External (SQL/population) spawns layered onto the world scene: building their instances, clearing them, and their counts and scale multiplier.
/// Moved verbatim from <see cref="WorldScene"/> (Spec 255). Scene state it still needs comes
/// only through <see cref="IWorldSceneHost"/>; the bridge members below keep the names the moved
/// code used inside WorldScene, so no moved body was edited.
/// </summary>
public sealed class ExternalSpawnLayer
{
    private readonly IWorldSceneHost _host;

    internal ExternalSpawnLayer(IWorldSceneHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the WorldScene members).
    private WorldAssetManager _assets => _host.Assets;
    private List<ObjectInstance> _externalMdxInstances => _host.ExternalMdxInstances;
    private List<ObjectInstance> _externalSkyboxInstances => _host.ExternalSkyboxInstances;
    private List<ObjectInstance> _externalWmoInstances => _host.ExternalWmoInstances;
    private ref bool _instancesDirty => ref _host.InstancesDirty;
    private static void TransformBounds(Vector3 boundsMin, Vector3 boundsMax, in Matrix4x4 transform, out Vector3 transformedMin, out Vector3 transformedMax) => WorldScene.TransformBounds(boundsMin, boundsMax, in transform, out transformedMin, out transformedMax);
    private static void TransformBounds(Vector3 min, Vector3 max, Matrix4x4 m, out Vector3 outMin, out Vector3 outMax) => WorldScene.TransformBounds(min, max, m, out outMin, out outMax);

    public int ExternalSpawnMdxCount => _externalMdxInstances.Count;
    public int ExternalSpawnWmoCount => _externalWmoInstances.Count;
    public int ExternalSpawnInstanceCount => ExternalSpawnMdxCount + ExternalSpawnWmoCount;
    public float SqlGameObjectMdxScaleMultiplier { get; set; } = 1.0f;

    public void ClearExternalSpawns()
    {
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();
        _instancesDirty = true;
    }

    public void SetExternalSpawns(IEnumerable<WorldSpawnRecord> spawns)
    {
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();

        foreach (var spawn in spawns)
        {
            if (string.IsNullOrWhiteSpace(spawn.ModelPath))
                continue;

            string modelPath = spawn.ModelPath.Replace('/', '\\');
            bool isWmo = modelPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);

            string key = WorldAssetManager.NormalizeKey(modelPath);
            float orientationRadians = spawn.OrientationWowRadians;
            float yawOffsetRadians = spawn.SpawnType == WorldSpawnType.Creature ? MathF.PI : 0f;
            float finalYawRadians = orientationRadians + yawOffsetRadians;
            float finalYawDegrees = finalYawRadians * (180f / MathF.PI);
            float baseScale = spawn.EffectiveScale > 0 ? spawn.EffectiveScale : 1.0f;
            float mdxScale = baseScale;
            if (spawn.SpawnType == WorldSpawnType.GameObject)
                mdxScale *= SqlGameObjectMdxScaleMultiplier > 0 ? SqlGameObjectMdxScaleMultiplier : 1.0f;

            var pos = SqlSpawnCoordinateConverter.ToRendererPosition(spawn.PositionWow);
            var (tileX, tileY) = ComputeTileCoordinates(pos);

            if (isWmo)
            {
                var transform = Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 localMin, localMax, worldMin, worldMax;
                if (_assets.TryGetWmoPlacementBounds(key, out localMin, out localMax))
                {
                    TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
                }
                else
                {
                    localMin = localMax = Vector3.Zero;
                    worldMin = pos - new Vector3(2f);
                    worldMax = pos + new Vector3(2f);
                }

                _externalWmoInstances.Add(new ObjectInstance
                {
                    ModelKey = key,
                    Transform = transform,
                    BoundsMin = worldMin,
                    BoundsMax = worldMax,
                    LocalBoundsMin = localMin,
                    LocalBoundsMax = localMax,
                    BoundsResolved = localMin != Vector3.Zero || localMax != Vector3.Zero,
                    ModelName = Path.GetFileName(modelPath),
                    ModelPath = modelPath,
                    PlacementPosition = pos,
                    PlacementRotation = new Vector3(0f, 0f, finalYawDegrees),
                    PlacementScale = 1.0f,
                    UniqueId = spawn.SpawnId,
                    PlacementEntryIndex = -1,
                    TileX = tileX,
                    TileY = tileY,
                    HasTileCoordinate = true
                });
            }
            else
            {
                var transform = Matrix4x4.CreateScale(mdxScale)
                    * Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 bbMin, bbMax;
                Vector3 localMin = Vector3.Zero;
                Vector3 localMax = Vector3.Zero;
                bool boundsResolved = false;
                if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
                {
                    localMin = modelMin;
                    localMax = modelMax;
                    boundsResolved = true;
                    TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
                }
                else
                {
                    bbMin = pos - new Vector3(2f);
                    bbMax = pos + new Vector3(2f);
                }

                var instance = new ObjectInstance
                {
                    ModelKey = key,
                    Transform = transform,
                    BoundsMin = bbMin,
                    BoundsMax = bbMax,
                    LocalBoundsMin = localMin,
                    LocalBoundsMax = localMax,
                    BoundsResolved = boundsResolved,
                    ModelName = Path.GetFileName(modelPath),
                    ModelPath = modelPath,
                    PlacementPosition = pos,
                    PlacementRotation = new Vector3(0f, 0f, finalYawDegrees),
                    PlacementScale = mdxScale,
                    UniqueId = spawn.SpawnId,
                    PlacementEntryIndex = -1,
                    TileX = tileX,
                    TileY = tileY,
                    HasTileCoordinate = true
                };

                if (SceneAtmosphere.IsSkyboxModelPath(modelPath))
                    _externalSkyboxInstances.Add(instance);
                else
                    _externalMdxInstances.Add(instance);
            }
        }

        ViewerLog.Info(ViewerLog.Category.Terrain,
            $"SQL spawns injected: {_externalMdxInstances.Count} MDX, {_externalSkyboxInstances.Count} skybox, {_externalWmoInstances.Count} WMO");

        _instancesDirty = true;
    }

    private static (int tileX, int tileY) ComputeTileCoordinates(Vector3 rendererPosition)
    {
        int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - rendererPosition.X) / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - rendererPosition.Y) / WoWConstants.ChunkSize);
        return (tileX, tileY);
    }
}
