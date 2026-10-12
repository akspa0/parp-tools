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
/// Selected world object: selection by ray or index, re-resolution after instance rebuilds, WMO-doodad selection, and moving the selected placement.
/// Moved verbatim from <see cref="WorldScene"/> (Spec 255). Scene state it still needs comes
/// only through <see cref="IWorldSceneHost"/>; the bridge members below keep the names the moved
/// code used inside WorldScene, so no moved body was edited.
/// </summary>
public sealed class SceneSelectionState
{
    private readonly IWorldSceneHost _host;

    internal SceneSelectionState(IWorldSceneHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the WorldScene members).
    private WorldAssetManager _assets => _host.Assets;
    private SceneHoverPickController _hoverPick => _host.HoverPick;
    private ref bool _instancesDirty => ref _host.InstancesDirty;
    private ref List<ObjectInstance> _mdxInstances => ref _host.MdxInstances;
    private TerrainManager _terrainManager => _host.TerrainManager;
    private Dictionary<(int, int), List<ObjectInstance>> _tileMdxInstances => _host.TileMdxInstances;
    private Dictionary<(int, int), List<ObjectInstance>> _tileWmoInstances => _host.TileWmoInstances;
    private ref List<ObjectInstance> _wmoInstances => ref _host.WmoInstances;
    private void RebuildInstanceLists() => _host.RebuildInstanceLists();
    private static void TransformBounds(Vector3 min, Vector3 max, Matrix4x4 m, out Vector3 outMin, out Vector3 outMax) => WorldScene.TransformBounds(min, max, m, out outMin, out outMax);
    private static void TransformBounds(Vector3 boundsMin, Vector3 boundsMax, in Matrix4x4 transform, out Vector3 transformedMin, out Vector3 transformedMax) => WorldScene.TransformBounds(boundsMin, boundsMax, in transform, out transformedMin, out transformedMax);

    // Object selection
    private ObjectType _selectedObjectType = ObjectType.None;
    private int _selectedObjectIndex = -1;
    private int _selectedWmoParentIndex = -1;
    private SelectedSceneObjectKey? _selectedSceneObjectKey;
    public ObjectType SelectedObjectType => _selectedObjectType;
    public int SelectedObjectIndex => _selectedObjectIndex;
    public int SelectedWmoParentIndex => _selectedWmoParentIndex;

    /// <summary>Get the currently selected object instance, or null if nothing selected.</summary>
    public ObjectInstance? SelectedInstance => TryGetSelectedSceneInstance(out ObjectInstance instance) ? instance : null;

    public bool TryGetSelectedPlacementSourceData(out string sourcePath, out byte[] sourceBytes)
    {
        sourcePath = string.Empty;
        sourceBytes = Array.Empty<byte>();

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
            return false;

        ObjectInstance instance = selected.Value;
        if (!instance.HasTileCoordinate || instance.PlacementEntryIndex < 0)
            return false;

        return _terrainManager.Adapter.TryGetPlacementSourceData(instance.TileX, instance.TileY, out sourcePath, out sourceBytes);
    }

    public bool TryGetSelectedPlacementWritablePath(out string? fullPath)
    {
        fullPath = null;

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
            return false;

        ObjectInstance instance = selected.Value;
        if (!instance.HasTileCoordinate || instance.PlacementEntryIndex < 0)
            return false;

        return _terrainManager.Adapter.TryGetPlacementWritablePath(instance.TileX, instance.TileY, out fullPath);
    }

    public bool TryUpdateSelectedPlacementPosition(Vector3 newPosition, out string error)
    {
        error = string.Empty;

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
        {
            error = "No world object is selected.";
            return false;
        }

        ObjectInstance current = selected.Value;
        if (!current.HasTileCoordinate || current.PlacementEntryIndex < 0)
        {
            error = "The selected object is not backed by a writable ADT placement entry.";
            return false;
        }

        if (_selectedObjectType is not (ObjectType.Mdx or ObjectType.Wmo))
        {
            error = "Only ADT MDDF and MODF placements are supported by the current save seam.";
            return false;
        }

        Dictionary<(int, int), List<ObjectInstance>> tileInstances = _selectedObjectType == ObjectType.Mdx
            ? _tileMdxInstances
            : _tileWmoInstances;

        if (!tileInstances.TryGetValue((current.TileX, current.TileY), out List<ObjectInstance>? instances))
        {
            error = $"Tile ({current.TileX}, {current.TileY}) is not currently loaded.";
            return false;
        }

        int instanceIndex = FindPlacementInstanceIndex(instances, current);
        if (instanceIndex < 0)
        {
            error = "The selected placement could not be matched back to the loaded tile instance list.";
            return false;
        }

        ObjectInstance updated = MovePlacementInstance(current, newPosition, _selectedObjectType);
        instances[instanceIndex] = updated;

        _terrainManager.TryUpdateCachedPlacementPosition(_selectedObjectType, current.TileX, current.TileY, current.PlacementEntryIndex, newPosition);
        UpdateAdapterPlacementPosition(_selectedObjectType, current, newPosition);

        _instancesDirty = true;
        RebuildInstanceLists();
        return true;
    }

    private bool TryGetSelectedSceneInstance(out ObjectInstance instance)
    {
        // Asset promotion can resolve bounds after the frame-maintenance pass and mark the
        // placement lists dirty. Never rebuild the full world synchronously from a read/query
        // accessor (the render-time selected-bounds overlay is one such accessor); let the next
        // frame's scene-maintenance pass perform the rebuild once.
        if (_instancesDirty)
        {
            instance = default;
            return false;
        }

        if (TryGetSceneObjectByIndex(_selectedObjectType, _selectedObjectIndex, out instance))
        {
            if (!_selectedSceneObjectKey.HasValue || !IsSameSceneObject(instance, _selectedSceneObjectKey.Value))
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(_selectedObjectType, instance);

            return true;
        }

        if (_selectedSceneObjectKey.HasValue
            && TryResolveSelectedSceneObject(_selectedSceneObjectKey.Value, out ObjectType resolvedType, out int resolvedIndex, out instance))
        {
            _selectedObjectType = resolvedType;
            _selectedObjectIndex = resolvedIndex;
            return true;
        }

        instance = default;
        return false;
    }

    internal void RestoreSelectedSceneObjectAfterRebuild()
    {
        if (!_selectedSceneObjectKey.HasValue)
            return;

        if (TryResolveSelectedSceneObject(_selectedSceneObjectKey.Value, out ObjectType resolvedType, out int resolvedIndex, out _))
        {
            _selectedObjectType = resolvedType;
            _selectedObjectIndex = resolvedIndex;
            return;
        }

        _selectedObjectIndex = -1;
    }

    private bool TryResolveSelectedSceneObject(SelectedSceneObjectKey key, out ObjectType objectType, out int objectIndex, out ObjectInstance instance)
    {
        List<ObjectInstance> instances = key.ObjectType switch
        {
            ObjectType.Wmo => _wmoInstances,
            ObjectType.Mdx => _mdxInstances,
            _ => []
        };

        for (int index = 0; index < instances.Count; index++)
        {
            ObjectInstance candidate = instances[index];
            if (!IsSameSceneObject(candidate, key))
                continue;

            objectType = key.ObjectType;
            objectIndex = index;
            instance = candidate;
            return true;
        }

        objectType = ObjectType.None;
        objectIndex = -1;
        instance = default;
        return false;
    }

    public bool TryGetSelectedWmoDoodad(out WmoDoodadInfo doodadInfo, out Vector3 worldPosition, out ObjectInstance parentWmo)
    {
        doodadInfo = default;
        worldPosition = Vector3.Zero;
        parentWmo = default;

        if (_selectedObjectType != ObjectType.WmoDoodad || _selectedWmoParentIndex < 0 || _selectedWmoParentIndex >= _wmoInstances.Count)
            return false;

        parentWmo = _wmoInstances[_selectedWmoParentIndex];
        if (!_assets.TryGetLoadedWmo(parentWmo.ModelKey, out WmoRenderer? wmo) || wmo == null)
            return false;

        if (!wmo.TryGetDoodadInfo(_selectedObjectIndex, out doodadInfo))
            return false;

        worldPosition = Vector3.Transform(doodadInfo.LocalPosition, parentWmo.Transform);
        return true;
    }

    /// <summary>
    /// Build the selected WMO doodad as an <see cref="ObjectInstance"/> for the inspector and the
    /// selection-bounds overlay.
    /// </summary>
    /// <remarks>
    /// This used to synthesise a placeholder: a hard-coded <c>position ± 1</c> cube for bounds, no
    /// model name or path, no rotation, no scale, and the MODD table index copied into
    /// <c>UniqueId</c>. Three separate defects followed from that.
    ///
    /// The bounds were the visible one. <see cref="WmoRenderer.TryGetDoodadBounds"/> already
    /// computes the doodad's real transformed AABB and the picker already carried it, but this
    /// method threw it away and substituted a 2-yard cube. The selection overlay then inflates
    /// whatever it is given by a further 0.75 yd or more for its accent box, so a scroll a third of
    /// a yard across was drawn inside about 3.8 yards of wireframe. The minimum-half-extent clamp
    /// added alongside the overlay could never have helped: 1.0 already exceeds the 0.75 floor, so
    /// the clamp was a no-op on exactly the objects it was meant to fix.
    ///
    /// <c>UniqueId</c> was the subtle one. MODD has no uniqueId field at all — uniqueId identifies
    /// an MDDF/MODF placement in an ADT. Copying the MODD index into it did not merely mislabel the
    /// inspector; <c>ShouldHideObjectInstanceByUniqueId</c> keys on that field, so a doodad could be
    /// hidden by an unrelated placement's id colliding with its table index. The def index is kept
    /// in <see cref="ObjectInstance.PlacementEntryIndex"/>, which is what it actually is, and
    /// UniqueId is left at 0 to mean "this kind of object does not have one".
    /// </remarks>
    private bool TryBuildSelectedWmoDoodadInstance(out ObjectInstance instance)
    {
        instance = default;

        if (!TryGetSelectedWmoDoodad(out WmoDoodadInfo dInfo, out Vector3 worldPosition, out ObjectInstance parentWmo))
            return false;

        if (!_assets.TryGetLoadedWmo(parentWmo.ModelKey, out WmoRenderer? wmoRenderer) || wmoRenderer == null)
            return false;

        // Real geometry bounds when the doodad's model is loaded. When it is not, say so rather
        // than presenting the placeholder cube as the object's extent.
        bool boundsResolved = false;
        Vector3 boundsMin = worldPosition;
        Vector3 boundsMax = worldPosition;
        if (wmoRenderer.TryGetDoodadBounds(dInfo.Index, parentWmo.Transform, out Vector3 bMin, out Vector3 bMax, out boundsResolved))
        {
            boundsMin = bMin;
            boundsMax = bMax;
        }

        // Local (model-space) geometry bounds drive the oriented tight selection box. When the
        // doodad's model has not streamed in yet, request it so real bounds resolve within a few
        // frames instead of the selection sitting on a centroid cube indefinitely.
        bool hasLocalBounds = wmoRenderer.TryGetDoodadLocalBounds(dInfo.Index, out Vector3 localMin, out Vector3 localMax);
        if (!boundsResolved)
            wmoRenderer.RequestDoodadModelLoad(dInfo.Index);

        if (!wmoRenderer.TryGetDoodadWorldTransform(dInfo.Index, parentWmo.Transform, out Matrix4x4 doodadTransform))
            doodadTransform = Matrix4x4.CreateTranslation(worldPosition);

        instance = new ObjectInstance
        {
            ModelKey = dInfo.ModelPath,
            ModelPath = dInfo.ModelPath,
            ModelName = System.IO.Path.GetFileName(dInfo.ModelPath),
            AssetKind = "WMO Doodad",
            PlacementPosition = worldPosition,
            PlacementRotation = QuaternionToEulerDegrees(dInfo.Orientation),
            PlacementScale = dInfo.Scale,
            BoundsMin = boundsMin,
            BoundsMax = boundsMax,
            BoundsResolved = boundsResolved,
            // MODD has no uniqueId. The def index is a MODD table index, meaningful only inside this
            // WMO; it belongs here, not in UniqueId. See the remarks above.
            PlacementEntryIndex = dInfo.DoodadDefIndex,
            UniqueId = 0,
            LocalBoundsMin = hasLocalBounds ? localMin : Vector3.Zero,
            LocalBoundsMax = hasLocalBounds ? localMax : Vector3.Zero,
            SelectionLocalBoundsMin = hasLocalBounds ? localMin : Vector3.Zero,
            SelectionLocalBoundsMax = hasLocalBounds ? localMax : Vector3.Zero,
            SelectionBoundsResolved = hasLocalBounds,
            Transform = doodadTransform
        };
        return true;
    }

    private static Vector3 QuaternionToEulerDegrees(Quaternion q)
    {
        float sinR = 2f * (q.W * q.X + q.Y * q.Z);
        float cosR = 1f - 2f * (q.X * q.X + q.Y * q.Y);
        float roll = MathF.Atan2(sinR, cosR);

        float sinP = Math.Clamp(2f * (q.W * q.Y - q.Z * q.X), -1f, 1f);
        float pitch = MathF.Asin(sinP);

        float sinY = 2f * (q.W * q.Z + q.X * q.Y);
        float cosY = 1f - 2f * (q.Y * q.Y + q.Z * q.Z);
        float yaw = MathF.Atan2(sinY, cosY);

        const float ToDegrees = 180f / MathF.PI;
        return new Vector3(roll * ToDegrees, pitch * ToDegrees, yaw * ToDegrees);
    }

    private bool TryGetSceneObjectByIndex(ObjectType objectType, int objectIndex, out ObjectInstance instance)
    {
        switch (objectType)
        {
            case ObjectType.Wmo when objectIndex >= 0 && objectIndex < _wmoInstances.Count:
                instance = _wmoInstances[objectIndex];
                return true;
            case ObjectType.Mdx when objectIndex >= 0 && objectIndex < _mdxInstances.Count:
                instance = _mdxInstances[objectIndex];
                return true;
            case ObjectType.WmoDoodad when TryBuildSelectedWmoDoodadInstance(out instance):
                return true;
            default:
                instance = default;
                return false;
        }
    }

    private static SelectedSceneObjectKey CreateSelectedSceneObjectKey(ObjectType objectType, ObjectInstance instance)
    {
        return new SelectedSceneObjectKey(
            objectType,
            instance.UniqueId,
            instance.PlacementEntryIndex,
            instance.TileX,
            instance.TileY,
            instance.HasTileCoordinate,
            instance.ModelKey,
            instance.PlacementPosition);
    }

    private static bool IsSameSceneObject(ObjectInstance candidate, SelectedSceneObjectKey key)
    {
        if (candidate.UniqueId != key.UniqueId || candidate.PlacementEntryIndex != key.PlacementEntryIndex)
            return false;

        if (candidate.HasTileCoordinate != key.HasTileCoordinate)
            return false;

        if (candidate.HasTileCoordinate)
            return candidate.TileX == key.TileX && candidate.TileY == key.TileY;

        if (!string.Equals(candidate.ModelKey, key.ModelKey, StringComparison.OrdinalIgnoreCase))
            return false;

        return Vector3.DistanceSquared(candidate.PlacementPosition, key.PlacementPosition) < 0.0001f;
    }

    private static int FindPlacementInstanceIndex(List<ObjectInstance> instances, ObjectInstance current)
    {
        for (int index = 0; index < instances.Count; index++)
        {
            ObjectInstance candidate = instances[index];
            if (candidate.UniqueId == current.UniqueId
                && candidate.PlacementEntryIndex == current.PlacementEntryIndex
                && candidate.TileX == current.TileX
                && candidate.TileY == current.TileY)
            {
                return index;
            }
        }

        return -1;
    }

    private ObjectInstance MovePlacementInstance(ObjectInstance current, Vector3 newPosition, ObjectType objectType)
    {
        Vector3 delta = newPosition - current.PlacementPosition;
        current.PlacementPosition = newPosition;

        switch (objectType)
        {
            case ObjectType.Mdx:
            {
                var transform = WorldPlacementTransform.Build(
                    newPosition,
                    current.PlacementRotation,
                    current.PlacementScale);

                current.Transform = transform;
                if (_assets.TryGetMdxBounds(current.ModelKey, out Vector3 localMin, out Vector3 localMax))
                {
                    current.LocalBoundsMin = localMin;
                    current.LocalBoundsMax = localMax;
                    current.BoundsResolved = true;
                    TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);
                    current.BoundsMin = worldMin;
                    current.BoundsMax = worldMax;
                }
                else
                {
                    current.BoundsMin += delta;
                    current.BoundsMax += delta;
                    current.BoundsResolved = false;
                }

                return current;
            }

            case ObjectType.Wmo:
            {
                var transform = WorldPlacementTransform.Build(
                    newPosition,
                    current.PlacementRotation);

                current.Transform = transform;
                if (_assets.TryGetWmoPlacementBounds(current.ModelKey, out Vector3 localMin, out Vector3 localMax))
                {
                    current.LocalBoundsMin = localMin;
                    current.LocalBoundsMax = localMax;
                    current.BoundsResolved = true;
                    TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);
                    current.BoundsMin = worldMin;
                    current.BoundsMax = worldMax;
                }
                else
                {
                    current.BoundsMin += delta;
                    current.BoundsMax += delta;
                    current.BoundsResolved = false;
                }

                return current;
            }

            default:
                return current;
        }
    }

    private void UpdateAdapterPlacementPosition(ObjectType objectType, ObjectInstance current, Vector3 newPosition)
    {
        switch (objectType)
        {
            case ObjectType.Mdx:
                UpdateMddfPlacementPosition(current, newPosition);
                break;

            case ObjectType.Wmo:
                UpdateModfPlacementPosition(current, newPosition);
                break;
        }
    }

    private void UpdateMddfPlacementPosition(ObjectInstance current, Vector3 newPosition)
    {
        List<MddfPlacement> placements = _terrainManager.Adapter.MddfPlacements;
        int index = FindPlacementIndexByUniqueIdAndPosition(placements, current.UniqueId, current.PlacementPosition);
        if (index < 0)
            index = FindPlacementIndexByUniqueId(placements, current.UniqueId);
        if (index < 0)
            return;

        MddfPlacement updated = placements[index];
        updated.Position = newPosition;
        placements[index] = updated;
    }

    private void UpdateModfPlacementPosition(ObjectInstance current, Vector3 newPosition)
    {
        List<ModfPlacement> placements = _terrainManager.Adapter.ModfPlacements;
        int index = FindPlacementIndexByUniqueIdAndPosition(placements, current.UniqueId, current.PlacementPosition);
        if (index < 0)
            index = FindPlacementIndexByUniqueId(placements, current.UniqueId);
        if (index < 0)
            return;

        ModfPlacement updated = placements[index];
        Vector3 delta = newPosition - updated.Position;
        updated.Position = newPosition;
        updated.BoundsMin += delta;
        updated.BoundsMax += delta;
        placements[index] = updated;
    }

    private static int FindPlacementIndexByUniqueIdAndPosition(List<MddfPlacement> placements, int uniqueId, Vector3 position)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId && Vector3.DistanceSquared(placements[index].Position, position) < 0.0001f)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueId(List<MddfPlacement> placements, int uniqueId)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueIdAndPosition(List<ModfPlacement> placements, int uniqueId, Vector3 position)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId && Vector3.DistanceSquared(placements[index].Position, position) < 0.0001f)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueId(List<ModfPlacement> placements, int uniqueId)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId)
                return index;
        }

        return -1;
    }

    /// <summary>
    /// Select the nearest object whose AABB is hit by a ray from camera.
    /// Call with screen-space mouse coords to pick objects.
    /// </summary>
    public void SelectObjectByRay(Vector3 rayOrigin, Vector3 rayDir)
    {
        if (_hoverPick.TryPickSceneObjectByRay(rayOrigin, rayDir, out ObjectType bestType, out int bestIndex, out _))
        {
            _selectedObjectType = bestType;
            _selectedObjectIndex = bestIndex;
            if (TryGetSceneObjectByIndex(bestType, bestIndex, out ObjectInstance selectedInstance))
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(bestType, selectedInstance);
            return;
        }

        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
        _selectedSceneObjectKey = null;
    }

    public bool SelectSceneObject(ObjectType objectType, int objectIndex, int parentWmoIndex = -1)
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        switch (objectType)
        {
            case ObjectType.Wmo when objectIndex >= 0 && objectIndex < _wmoInstances.Count:
                _selectedObjectType = objectType;
                _selectedObjectIndex = objectIndex;
                _selectedWmoParentIndex = -1;
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(objectType, _wmoInstances[objectIndex]);
                return true;
            case ObjectType.Mdx when objectIndex >= 0 && objectIndex < _mdxInstances.Count:
                _selectedObjectType = objectType;
                _selectedObjectIndex = objectIndex;
                _selectedWmoParentIndex = -1;
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(objectType, _mdxInstances[objectIndex]);
                return true;
            case ObjectType.WmoDoodad when parentWmoIndex >= 0 && parentWmoIndex < _wmoInstances.Count:
                _selectedObjectType = objectType;
                _selectedObjectIndex = objectIndex;
                _selectedWmoParentIndex = parentWmoIndex;
                _selectedSceneObjectKey = null;
                return true;
            default:
                return false;
        }
    }

    public void ClearSelection()
    {
        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
        _selectedWmoParentIndex = -1;
        _selectedSceneObjectKey = null;
    }
}
