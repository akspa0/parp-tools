using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.Runtime.World;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Pm4;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

/// <summary>
/// Service providing live PM4 doodad collision matching, duplication, human-in-the-loop teaching,
/// and undo in the viewer (Specs 287, 289, 291).
/// </summary>
internal sealed class Pm4DoodadDuplicationService
{
    private readonly IViewerAppHost _host;
    private readonly Stack<IPm4UndoOperation> _undoStack = new();
    private readonly List<TaughtPairing> _taughtPairings = new();

    public IReadOnlyList<TaughtPairing> TaughtPairings => _taughtPairings;
    public string? LastOperationMessage { get; set; }

    public static bool IsWmoAsset(string? modelPath, string? assetKind = null)
        => Pm4DoodadDuplicationEngine.IsWmoAsset(modelPath, assetKind);

    public static string ResolveAssetKind(string? modelPath, string? assetKind = null)
        => Pm4DoodadDuplicationEngine.ResolveAssetKind(modelPath, assetKind);

    private interface IPm4UndoOperation
    {
        bool Execute(IViewerAppHost host);
    }

    private sealed class MultiTileDuplicationUndo(Dictionary<(int tileX, int tileY), List<int>> addedByTile) : IPm4UndoOperation
    {
        public bool Execute(IViewerAppHost host)
        {
            var worldScene = host.WorldScene;
            if (worldScene == null)
                return false;

            IPm4OverlayHost pm4Host = worldScene;
            int totalRemoved = 0;

            foreach (var ((tileX, tileY), addedIds) in addedByTile)
            {
                var idSet = new HashSet<int>(addedIds);
                if (pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx))
                {
                    totalRemoved += tileMdx.RemoveAll(inst => idSet.Contains(inst.UniqueId));
                }
                if (pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var tileWmo))
                {
                    totalRemoved += tileWmo.RemoveAll(inst => idSet.Contains(inst.UniqueId));
                }
            }

            if (totalRemoved > 0)
            {
                pm4Host.RebuildInstanceLists();
                string msg = $"Undid population of {totalRemoved} instances across {addedByTile.Count} tiles.";
                host.StatusMessage = msg;
                host.Pm4DoodadDuplication.LastOperationMessage = msg;
                return true;
            }

            return false;
        }
    }

    private sealed class DuplicationUndo(int tileX, int tileY, List<int> addedIds) : IPm4UndoOperation
    {
        public bool Execute(IViewerAppHost host)
        {
            var worldScene = host.WorldScene;
            if (worldScene == null)
                return false;

            IPm4OverlayHost pm4Host = worldScene;
            int removed = 0;
            var idSet = new HashSet<int>(addedIds);

            if (pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out List<ObjectInstance>? tileMdx))
            {
                removed += tileMdx.RemoveAll(inst => idSet.Contains(inst.UniqueId));
            }
            if (pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out List<ObjectInstance>? tileWmo))
            {
                removed += tileWmo.RemoveAll(inst => idSet.Contains(inst.UniqueId));
            }

            if (removed > 0)
            {
                pm4Host.RebuildInstanceLists();
                string msg = $"Undid duplication of {removed} instances on tile ({tileX}, {tileY}).";
                host.StatusMessage = msg;
                host.Pm4DoodadDuplication.LastOperationMessage = msg;
                return true;
            }

            return false;
        }
    }

    private sealed class SnapUndo(int tileX, int tileY, List<(int UniqueId, Vector3 Position, Vector3 Rotation, float Scale)> previousStates) : IPm4UndoOperation
    {
        public bool Execute(IViewerAppHost host)
        {
            var worldScene = host.WorldScene;
            if (worldScene == null)
                return false;

            IPm4OverlayHost pm4Host = worldScene;
            var stateMap = new Dictionary<int, (Vector3 Position, Vector3 Rotation, float Scale)>();
            foreach (var s in previousStates)
                stateMap[s.UniqueId] = (s.Position, s.Rotation, s.Scale);

            int restoredCount = 0;
            if (pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out List<ObjectInstance>? tileMdx))
            {
                for (int i = 0; i < tileMdx.Count; i++)
                {
                    var inst = tileMdx[i];
                    if (stateMap.TryGetValue(inst.UniqueId, out var state))
                    {
                        inst.PlacementPosition = state.Position;
                        inst.PlacementRotation = state.Rotation;
                        inst.PlacementScale = state.Scale;
                        inst.Transform = WorldPlacementTransform.Build(inst.PlacementPosition, inst.PlacementRotation, inst.PlacementScale);
                        WorldScene.TransformBounds(inst.LocalBoundsMin, inst.LocalBoundsMax, inst.Transform, out inst.BoundsMin, out inst.BoundsMax);
                        WorldScene.TransformBounds(inst.SelectionLocalBoundsMin, inst.SelectionLocalBoundsMax, inst.Transform, out inst.SelectionLocalBoundsMin, out inst.SelectionLocalBoundsMax);
                        tileMdx[i] = inst;
                        restoredCount++;
                    }
                }
            }

            if (pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out List<ObjectInstance>? tileWmo))
            {
                for (int i = 0; i < tileWmo.Count; i++)
                {
                    var inst = tileWmo[i];
                    if (stateMap.TryGetValue(inst.UniqueId, out var state))
                    {
                        inst.PlacementPosition = state.Position;
                        inst.PlacementRotation = state.Rotation;
                        inst.PlacementScale = state.Scale;
                        inst.Transform = WorldPlacementTransform.Build(inst.PlacementPosition, inst.PlacementRotation, inst.PlacementScale);
                        WorldScene.TransformBounds(inst.LocalBoundsMin, inst.LocalBoundsMax, inst.Transform, out inst.BoundsMin, out inst.BoundsMax);
                        WorldScene.TransformBounds(inst.SelectionLocalBoundsMin, inst.SelectionLocalBoundsMax, inst.Transform, out inst.SelectionLocalBoundsMin, out inst.SelectionLocalBoundsMax);
                        tileWmo[i] = inst;
                        restoredCount++;
                    }
                }
            }

            if (restoredCount > 0)
            {
                pm4Host.RebuildInstanceLists();
                string msg = $"Restored {restoredCount} placements to pre-magnetized state on tile ({tileX}, {tileY}).";
                host.StatusMessage = msg;
                host.Pm4DoodadDuplication.LastOperationMessage = msg;
                return true;
            }

            return false;
        }
    }

    public Pm4CurationAndSpatialMatchingService Curation { get; }

    public Pm4DoodadDuplicationService(IViewerAppHost host)
    {
        _host = host;
        Curation = new Pm4CurationAndSpatialMatchingService(host, this);
        LoadTaughtPairingsFromDisk();
    }

    public bool CanUndo => _undoStack.Count > 0;

    /// <summary>
    /// Collects world-space bounds for every placement on a tile, including the transposed tile key
    /// (PM4 tiles are sometimes keyed with swapped coordinates). Used to detect hulls that already
    /// have an assigned/paired model so matching does not spawn duplicate copies.
    /// </summary>
    private static List<PlacementBounds> BuildPlacementBounds(IPm4OverlayHost pm4Host, int tileX, int tileY)
    {
        var result = new List<PlacementBounds>();
        Collect(tileX, tileY);
        if (tileX != tileY)
            Collect(tileY, tileX);
        return result;

        void Collect(int tx, int ty)
        {
            if (pm4Host.TileMdxInstances.TryGetValue((tx, ty), out var mdx) && mdx != null)
            {
                for (int i = 0; i < mdx.Count; i++)
                    result.Add(new PlacementBounds(mdx[i].BoundsMin, mdx[i].BoundsMax, mdx[i].PlacementPosition, mdx[i].BoundsResolved));
            }

            if (pm4Host.TileWmoInstances.TryGetValue((tx, ty), out var wmo) && wmo != null)
            {
                for (int i = 0; i < wmo.Count; i++)
                    result.Add(new PlacementBounds(wmo[i].BoundsMin, wmo[i].BoundsMax, wmo[i].PlacementPosition, wmo[i].BoundsResolved));
            }
        }
    }

    /// <summary>
    /// Builds a world-space <see cref="Pm4DoodadTarget"/> from a PM4 overlay object.
    /// </summary>
    internal static Pm4DoodadTarget BuildTarget(Pm4OverlayObject obj, int tileX, int tileY)
    {
        WorldScene.TransformBounds(obj.BoundsMin, obj.BoundsMax, obj.BaseTransform, out Vector3 hullMin, out Vector3 hullMax);
        return new Pm4DoodadTarget(
            tileX,
            tileY,
            obj.Ck24,
            obj.ObjectPartId,
            Vector3.Transform(obj.Center, obj.BaseTransform),
            hullMin,
            hullMax,
            obj.BaseRotationRadians,
            obj.DominantGroupKey);
    }

    /// <summary>
    /// True when the hull already contains a placement of the same model path. Blocks duplicate
    /// explicit spawns while still allowing different models in a multi-model hull.
    /// </summary>
    private static bool HullHasSameModel(IPm4OverlayHost pm4Host, Pm4OverlayObject obj, int tileX, int tileY, string? modelPath)
    {
        if (string.IsNullOrEmpty(modelPath))
            return false;

        var target = BuildTarget(obj, tileX, tileY);
        bool sameModel = false;

        void Check(List<ObjectInstance>? list)
        {
            if (list == null || sameModel) return;
            for (int i = 0; i < list.Count; i++)
            {
                var inst = list[i];
                if (!string.Equals(inst.ModelPath, modelPath, StringComparison.OrdinalIgnoreCase))
                    continue;
                var pb = new PlacementBounds(inst.BoundsMin, inst.BoundsMax, inst.PlacementPosition, inst.BoundsResolved);
                if (Pm4DoodadDuplicationEngine.IsTargetOccupied(target, [pb]))
                {
                    sameModel = true;
                    return;
                }
            }
        }

        pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var mdx);
        pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var wmo);
        Check(mdx);
        Check(wmo);
        if (!sameModel && tileX != tileY)
        {
            pm4Host.TileMdxInstances.TryGetValue((tileY, tileX), out var mdxT);
            pm4Host.TileWmoInstances.TryGetValue((tileY, tileX), out var wmoT);
            Check(mdxT);
            Check(wmoT);
        }

        return sameModel;
    }

    /// <summary>
    /// Duplicates a template model instance into all unmatched PM4 doodad collision hulls on the specified tile.
    /// </summary>
    public int DuplicateModelToUnmatchedPm4Doodads(ObjectInstance template, int tileX, int tileY, float searchRadius = 150f)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null)
            return 0;

        IPm4OverlayHost pm4Host = worldScene;

        // Resolve tile coordinates
        int targetTileX = template.HasTileCoordinate ? template.TileX : tileX;
        int targetTileY = template.HasTileCoordinate ? template.TileY : tileY;

        // Retrieve loaded PM4 overlay objects for the tile
        if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((targetTileX, targetTileY), out var pm4Objects) || pm4Objects.Count == 0)
        {
            // Fallback: check transposed tile coordinates in case map uses flipped index convention
            if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((targetTileY, targetTileX), out pm4Objects) || pm4Objects.Count == 0)
            {
                _host.StatusMessage = $"No PM4 objects loaded on tile ({targetTileX}, {targetTileY}).";
                return 0;
            }
        }

        // Collect candidate PM4 doodad collision targets
        var candidateTargets = new List<Pm4DoodadTarget>();
        for (int i = 0; i < pm4Objects.Count; i++)
        {
            var obj = pm4Objects[i];
            // Match PM4 doodad collision components (Ck24 == 0, dominant group 0x03, or explicit M2 ck24 types)
            if (obj.Ck24 != 0 && obj.DominantGroupKey != 0x03 && obj.Ck24Type != 0x40 && obj.Ck24Type != 0x41)
                continue;

            Vector3 worldCentroid = Vector3.Transform(obj.Center, obj.BaseTransform);
            if (searchRadius > 0f && Vector3.Distance(worldCentroid, template.PlacementPosition) > searchRadius)
                continue;

            // Target bounds must be world-space: FitModelToTarget solves Z against
            // BoundsMax.Z (hull top) / BoundsMin.Z (hull floor), and its PCA yaw path
            // consumes world vertices.
            WorldScene.TransformBounds(obj.BoundsMin, obj.BoundsMax, obj.BaseTransform, out Vector3 worldMin, out Vector3 worldMax);
            var worldVerts = new List<Vector3>(obj.Triangles.Count * 3);
            for (int t = 0; t < obj.Triangles.Count; t++)
            {
                worldVerts.Add(Vector3.Transform(obj.Triangles[t].A, obj.BaseTransform));
                worldVerts.Add(Vector3.Transform(obj.Triangles[t].B, obj.BaseTransform));
                worldVerts.Add(Vector3.Transform(obj.Triangles[t].C, obj.BaseTransform));
            }

            candidateTargets.Add(new Pm4DoodadTarget(
                targetTileX,
                targetTileY,
                obj.Ck24,
                obj.ObjectPartId,
                worldCentroid,
                worldMin,
                worldMax,
                obj.BaseRotationRadians,
                obj.DominantGroupKey,
                worldVerts));
        }

        if (candidateTargets.Count == 0)
        {
            _host.StatusMessage = $"No PM4 doodad collision hulls found within {searchRadius:F0} yds of selection.";
            return 0;
        }

        bool isWmo = IsWmoAsset(template.ModelPath, template.AssetKind);
        float templateScale = isWmo ? 1.0f : template.PlacementScale;

        // Get existing placements on this tile to reject occupied targets. Bounds-based occupancy
        // (not origin distance) is required: WMO/M2 origins sit at the model base, so an
        // origin-distance test misses already-paired hulls and spawns duplicate copies.
        pm4Host.TileMdxInstances.TryGetValue((targetTileX, targetTileY), out List<ObjectInstance>? tileMdx);
        pm4Host.TileWmoInstances.TryGetValue((targetTileX, targetTileY), out List<ObjectInstance>? tileWmo);

        var placementBounds = BuildPlacementBounds(pm4Host, targetTileX, targetTileY);
        int maxExistingUniqueId = 0;
        if (tileMdx != null)
        {
            for (int i = 0; i < tileMdx.Count; i++)
                if (tileMdx[i].UniqueId > maxExistingUniqueId) maxExistingUniqueId = tileMdx[i].UniqueId;
        }
        if (tileWmo != null)
        {
            for (int i = 0; i < tileWmo.Count; i++)
                if (tileWmo[i].UniqueId > maxExistingUniqueId) maxExistingUniqueId = tileWmo[i].UniqueId;
        }

        var templateInfo = new ModelTemplateInfo(
            template.ModelPath,
            template.ModelKey,
            template.ModelName,
            template.PlacementPosition,
            template.PlacementRotation,
            templateScale,
            template.LocalBoundsMin,
            template.LocalBoundsMax,
            template.BoundsMin,
            template.BoundsMax);

        int startId = Math.Max(Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase, maxExistingUniqueId + 1);
        List<DuplicatedDoodadResult> duplicated = Pm4DoodadDuplicationEngine.DuplicateAcrossTargets(
            templateInfo,
            candidateTargets,
            placementBounds,
            startId,
            originFallbackRadius: isWmo ? 16.0f : 8.0f);

        if (duplicated.Count == 0)
        {
            string failMsg = "All nearby PM4 doodad collision hulls are already occupied by existing models.";
            _host.StatusMessage = failMsg;
            LastOperationMessage = failMsg;
            return 0;
        }

        var addedIds = new List<int>(duplicated.Count);
        for (int i = 0; i < duplicated.Count; i++)
        {
            DuplicatedDoodadResult d = duplicated[i];
            float placementScale = isWmo ? 1.0f : d.Scale;
            Matrix4x4 transform = WorldPlacementTransform.Build(d.Position, d.RotationDegrees, placementScale);
            WorldScene.TransformBounds(template.LocalBoundsMin, template.LocalBoundsMax, transform, out Vector3 bbMin, out Vector3 bbMax);
            WorldScene.TransformBounds(template.SelectionLocalBoundsMin, template.SelectionLocalBoundsMax, transform, out Vector3 selMin, out Vector3 selMax);

            var newInst = new ObjectInstance
            {
                ModelKey = template.ModelKey,
                Transform = transform,
                BoundsMin = bbMin,
                BoundsMax = bbMax,
                LocalBoundsMin = template.LocalBoundsMin,
                LocalBoundsMax = template.LocalBoundsMax,
                BoundsResolved = template.BoundsResolved,
                SelectionLocalBoundsMin = selMin,
                SelectionLocalBoundsMax = selMax,
                SelectionBoundsResolved = template.SelectionBoundsResolved,
                ModelName = template.ModelName,
                ModelPath = template.ModelPath,
                AssetKind = isWmo ? "WMO" : ResolveAssetKind(template.ModelPath, template.AssetKind),
                PlacementPosition = d.Position,
                PlacementRotation = d.RotationDegrees,
                PlacementScale = placementScale,
                UniqueId = d.UniqueId,
                PlacementEntryIndex = isWmo ? (tileWmo?.Count ?? 0) : (tileMdx?.Count ?? 0),
                TileX = targetTileX,
                TileY = targetTileY,
                HasTileCoordinate = true
            };

            if (isWmo)
            {
                if (tileWmo == null)
                {
                    tileWmo = new List<ObjectInstance>();
                    pm4Host.TileWmoInstances[(targetTileX, targetTileY)] = tileWmo;
                }
                tileWmo.Add(newInst);
                pm4Host.Assets.QueueWmoLoad(newInst.ModelKey);
                pm4Host.Assets.PrioritizeWmoLoad(newInst.ModelKey);
            }
            else
            {
                if (tileMdx == null)
                {
                    tileMdx = new List<ObjectInstance>();
                    pm4Host.TileMdxInstances[(targetTileX, targetTileY)] = tileMdx;
                }
                tileMdx.Add(newInst);
                pm4Host.Assets.QueueMdxLoad(newInst.ModelKey);
                pm4Host.Assets.PrioritizeMdxLoad(newInst.ModelKey);
            }
            addedIds.Add(d.UniqueId);
        }

        _undoStack.Push(new DuplicationUndo(targetTileX, targetTileY, addedIds));
        pm4Host.RebuildInstanceLists();
        pm4Host.Assets.ProcessPendingLoads(maxLoads: 64, maxBudgetMs: 100.0);
        string successMsg = $"✓ Duplicated {duplicated.Count} instances of {template.ModelName} into PM4 doodad collision hulls.";
        _host.StatusMessage = successMsg;
        LastOperationMessage = successMsg;
        return duplicated.Count;
    }

    private static List<(Vector3 Centroid, Vector3 Min, Vector3 Max, Pm4OverlayObject Rep, int TileX, int TileY)> BuildUnifiedPm4Candidates(
        IReadOnlyList<Pm4OverlayObject> pm4Objects,
        int tileX = 0,
        int tileY = 0)
    {
        var candidates = new List<(Vector3 Centroid, Vector3 Min, Vector3 Max, Pm4OverlayObject Rep, int TileX, int TileY)>();
        var ck24Groups = new Dictionary<uint, List<Pm4OverlayObject>>();
        var zeroParts = new List<Pm4OverlayObject>();

        for (int p = 0; p < pm4Objects.Count; p++)
        {
            var obj = pm4Objects[p];
            if (obj.Ck24 != 0)
            {
                if (!ck24Groups.TryGetValue(obj.Ck24, out var list))
                {
                    list = new List<Pm4OverlayObject>();
                    ck24Groups[obj.Ck24] = list;
                }
                list.Add(obj);
            }
            else
            {
                zeroParts.Add(obj);
            }
        }

        foreach (var (_, parts) in ck24Groups)
        {
            Vector3 uMin = new(float.MaxValue);
            Vector3 uMax = new(float.MinValue);
            for (int i = 0; i < parts.Count; i++)
            {
                WorldScene.TransformBounds(parts[i].BoundsMin, parts[i].BoundsMax, parts[i].BaseTransform, out Vector3 bMin, out Vector3 bMax);
                uMin = Vector3.Min(uMin, bMin);
                uMax = Vector3.Max(uMax, bMax);
            }
            Vector3 uCentroid = (uMin + uMax) * 0.5f;
            candidates.Add((uCentroid, uMin, uMax, parts[0], tileX, tileY));
        }

        for (int i = 0; i < zeroParts.Count; i++)
        {
            var zp = zeroParts[i];
            WorldScene.TransformBounds(zp.BoundsMin, zp.BoundsMax, zp.BaseTransform, out Vector3 bMin, out Vector3 bMax);
            Vector3 uCentroid = (bMin + bMax) * 0.5f;
            candidates.Add((uCentroid, bMin, bMax, zp, tileX, tileY));
        }

        return candidates;
    }

    /// <summary>
    /// Extracts all surface vertices for a PM4 hull. If the hull belongs to a composite object
    /// <summary>
    /// Extracts transformed world-space surface vertices for a PM4 object.
    /// When pm4Obj has a non-zero LinkGroupObjectId, constituent parts sharing that exact LinkGroupObjectId
    /// are gathered. Otherwise, extracts only pm4Obj itself to prevent merging unrelated objects across the tile.
    /// </summary>
    public static List<Vector3> ExtractCompositeSurfaceVertices(
        WorldScene? worldScene,
        Pm4OverlayObject pm4Obj,
        int pm4TileX,
        int pm4TileY,
        out float minWorldZ)
    {
        var surfaceVerts = new List<Vector3>();
        minWorldZ = float.MaxValue;

        if (pm4Obj.LinkGroupObjectId != 0 && worldScene != null &&
            (worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((pm4TileX, pm4TileY), out var tileObjs) ||
             worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((pm4TileY, pm4TileX), out tileObjs)))
        {
            for (int i = 0; i < tileObjs.Count; i++)
            {
                var part = tileObjs[i];
                if (part.LinkGroupObjectId == pm4Obj.LinkGroupObjectId && part.Ck24 == pm4Obj.Ck24)
                {
                    WorldScene.TransformBounds(part.BoundsMin, part.BoundsMax, part.BaseTransform, out Vector3 bMin, out _);
                    if (bMin.Z < minWorldZ) minWorldZ = bMin.Z;
                    for (int t = 0; t < part.Triangles.Count; t++)
                    {
                        surfaceVerts.Add(Vector3.Transform(part.Triangles[t].A, part.BaseTransform));
                        surfaceVerts.Add(Vector3.Transform(part.Triangles[t].B, part.BaseTransform));
                        surfaceVerts.Add(Vector3.Transform(part.Triangles[t].C, part.BaseTransform));
                    }
                }
            }
        }

        if (surfaceVerts.Count == 0)
        {
            WorldScene.TransformBounds(pm4Obj.BoundsMin, pm4Obj.BoundsMax, pm4Obj.BaseTransform, out Vector3 bMin, out _);
            minWorldZ = bMin.Z;
            for (int t = 0; t < pm4Obj.Triangles.Count; t++)
            {
                surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].A, pm4Obj.BaseTransform));
                surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].B, pm4Obj.BaseTransform));
                surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].C, pm4Obj.BaseTransform));
            }
        }

        return surfaceVerts;
    }

    /// <summary>
    /// Accurately aligns a placed object to a specific PM4 collision hull, aligning centroid,
    /// top-surface height, and PCA principal Yaw rotation.
    /// </summary>
    public bool AlignPlacementToPm4Hull(
        ObjectInstance inst,
        Pm4OverlayObject pm4Obj,
        int pm4TileX,
        int pm4TileY,
        bool fitScale,
        out string statusMessage)
    {
        statusMessage = string.Empty;
        var worldScene = _host.WorldScene;
        if (worldScene == null)
            return false;

        IPm4OverlayHost pm4Host = worldScene;
        int tileX = inst.TileX;
        int tileY = inst.TileY;

        bool isWmo = IsWmoAsset(inst.ModelPath, inst.AssetKind);
        List<ObjectInstance>? targetList = null;
        int idx = -1;

        if (isWmo)
        {
            if (pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var tileWmo))
            {
                idx = tileWmo.FindIndex(m => m.UniqueId == inst.UniqueId);
                if (idx >= 0) targetList = tileWmo;
            }
            if (targetList == null || idx < 0)
            {
                foreach (var pair in pm4Host.TileWmoInstances)
                {
                    idx = pair.Value.FindIndex(m => m.UniqueId == inst.UniqueId);
                    if (idx >= 0)
                    {
                        targetList = pair.Value;
                        tileX = pair.Key.Item1;
                        tileY = pair.Key.Item2;
                        break;
                    }
                }
            }
        }

        if (targetList == null && pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx))
        {
            idx = tileMdx.FindIndex(m => m.UniqueId == inst.UniqueId);
            if (idx >= 0) targetList = tileMdx;
        }

        if (targetList == null || idx < 0)
        {
            foreach (var pair in pm4Host.TileMdxInstances)
            {
                idx = pair.Value.FindIndex(m => m.UniqueId == inst.UniqueId);
                if (idx >= 0)
                {
                    targetList = pair.Value;
                    tileX = pair.Key.Item1;
                    tileY = pair.Key.Item2;
                    break;
                }
            }
        }

        if (targetList == null || idx < 0)
        {
            statusMessage = $"Could not locate placement #{inst.UniqueId} in resident scene lists.";
            _host.StatusMessage = statusMessage;
            LastOperationMessage = statusMessage;
            return false;
        }

        var prev = targetList[idx];
        var undoList = new List<(int, Vector3, Vector3, float)>
        {
            (prev.UniqueId, prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale)
        };
        _undoStack.Push(new SnapUndo(tileX, tileY, undoList));

        var surfaceVerts = ExtractCompositeSurfaceVertices(worldScene, pm4Obj, pm4TileX, pm4TileY, out float minWorldZ);
        worldScene.Pm4Overlay._pm4TilePositionRefs.TryGetValue((pm4TileX, pm4TileY), out var mprlList);
        float? placementZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : null;

        Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, minWorldZ);

        Vector3 localMin = prev.SelectionBoundsResolved && prev.SelectionLocalBoundsMax != Vector3.Zero
            ? prev.SelectionLocalBoundsMin : prev.LocalBoundsMin;
        Vector3 localMax = prev.SelectionBoundsResolved && prev.SelectionLocalBoundsMax != Vector3.Zero
            ? prev.SelectionLocalBoundsMax : prev.LocalBoundsMax;

        float? knownYawOffset = null;
        var taught = _taughtPairings.Find(t =>
            string.Equals(t.ModelPath, prev.ModelPath, StringComparison.OrdinalIgnoreCase) ||
            (t.Ck24 != 0 && t.Ck24 == pm4Obj.Ck24));
        if (taught != null)
        {
            knownYawOffset = taught.YawOffset;
        }
        else if (Pm4PriorKnowledgeDatabase.Instance.TryGetAsset(prev.ModelPath, out var dbAsset) && dbAsset != null && MathF.Abs(dbAsset.YawOffset) > 1e-4f)
        {
            knownYawOffset = dbAsset.YawOffset;
        }

        float candDx = MathF.Abs(localMax.X - localMin.X);
        float candDy = MathF.Abs(localMax.Y - localMin.Y);
        float solvedYaw;
        if (knownYawOffset.HasValue)
        {
            solvedYaw = prism.YawDegrees + knownYawOffset.Value;
        }
        else
        {
            solvedYaw = prism.YawDegrees + (candDy > candDx * 1.15f ? 90f : 0f);
            // If the resident model already has an authored rotation, pick the 180° hemisphere
            // closest to its current heading so alignment doesn't invert the model bow-to-stern.
            float currentYaw = prev.PlacementRotation.Z;
            float delta = MathF.IEEERemainder(currentYaw - solvedYaw, 360f);
            if (MathF.Abs(delta) > 90f)
            {
                solvedYaw += 180f;
            }
        }
        while (solvedYaw > 180f) solvedYaw -= 360f;
        while (solvedYaw < -180f) solvedYaw += 360f;

        var cand = Pm4ModelCandidate.Create(prev.ModelPath, prev.AssetKind ?? "M2", localMin, localMax);
        var matchResults = Pm4DirectModelMatcher.MatchPrism(prism, [cand], 1);

        float finalScale = isWmo ? 1.0f : (fitScale && matchResults.Count > 0 ? matchResults[0].SolvedScale : prev.PlacementScale);
        Matrix4x4 unpositioned = WorldPlacementTransform.Build(Vector3.Zero, new Vector3(0f, 0f, solvedYaw), finalScale);
        WorldScene.TransformBounds(localMin, localMax, unpositioned, out Vector3 rotMin, out Vector3 rotMax);
        Vector3 rotCenter = (rotMin + rotMax) * 0.5f;

        float targetPlacementZ;
        if (isWmo)
        {
            if (placementZ.HasValue && MathF.Abs(placementZ.Value) > 0.01f)
            {
                // MSUR._0x1C (Ck24 << 8) is the authentic producing MODF Position.Z.
                // Do NOT subtract rotMin.Z; WMO local origin (0,0,0) is authored to align with Position.Z.
                targetPlacementZ = placementZ.Value;
            }
            else
            {
                targetPlacementZ = prism.BaseZ;
            }
        }
        else
        {
            targetPlacementZ = prism.TopZ - (rotMax.Z > 0.01f ? rotMax.Z : MathF.Max(0.01f, rotMax.Z - rotMin.Z));
        }

        float dist = Vector3.Distance(prev.PlacementPosition, new Vector3(prism.Centroid.X - rotCenter.X, prism.Centroid.Y - rotCenter.Y, targetPlacementZ));
        prev.PlacementPosition = new Vector3(prism.Centroid.X - rotCenter.X, prism.Centroid.Y - rotCenter.Y, targetPlacementZ);
        prev.PlacementRotation = new Vector3(0f, 0f, solvedYaw);
        prev.PlacementScale = finalScale;

        prev.Transform = WorldPlacementTransform.Build(prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale);
        WorldScene.TransformBounds(prev.LocalBoundsMin, prev.LocalBoundsMax, prev.Transform, out prev.BoundsMin, out prev.BoundsMax);
        WorldScene.TransformBounds(prev.SelectionLocalBoundsMin, prev.SelectionLocalBoundsMax, prev.Transform, out prev.SelectionLocalBoundsMin, out prev.SelectionLocalBoundsMax);
        targetList[idx] = prev;

        pm4Host.RebuildInstanceLists();

        // Automatically detect and log spatial relationship with updated orientation
        Curation.TryDetectSpatialRelationship(pm4Obj, pm4TileX, pm4TileY, worldScene, out _, prev);

        float hullZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : 0f;
        string hullZStr = pm4Obj.Ck24 != 0 ? $"Object Z {hullZ:F3}" : $"part {pm4Obj.ObjectPartId}";
        statusMessage = $"✓ Aligned {prev.ModelName} (#{prev.UniqueId}) to PM4 hull {hullZStr} (delta: {dist:F1} yds, yaw: {prev.PlacementRotation.Z:F1}°).";
        _host.StatusMessage = statusMessage;
        LastOperationMessage = statusMessage;
        return true;
    }

    /// <summary>
    /// Rotates a placed scene object in-place around its world centroid by a specified yaw angle delta.
    /// If an active PM4 hull is selected, updates the taught pairing and spatial relationship.
    /// </summary>
    public bool RotatePlacement(ObjectInstance inst, float deltaYawDegrees)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null) return false;
        IPm4OverlayHost pm4Host = worldScene;

        int tileX = inst.TileX;
        int tileY = inst.TileY;
        bool isWmo = IsWmoAsset(inst.ModelPath, inst.AssetKind);
        List<ObjectInstance>? targetList = null;
        int idx = -1;

        if (isWmo && pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var tileWmo))
        {
            idx = tileWmo.FindIndex(m => m.UniqueId == inst.UniqueId);
            if (idx >= 0) targetList = tileWmo;
        }
        if (targetList == null && pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx))
        {
            idx = tileMdx.FindIndex(m => m.UniqueId == inst.UniqueId);
            if (idx >= 0) targetList = tileMdx;
        }
        if (targetList == null || idx < 0)
        {
            foreach (var pair in pm4Host.TileWmoInstances)
            {
                idx = pair.Value.FindIndex(m => m.UniqueId == inst.UniqueId);
                if (idx >= 0) { targetList = pair.Value; tileX = pair.Key.Item1; tileY = pair.Key.Item2; break; }
            }
        }
        if (targetList == null || idx < 0)
        {
            foreach (var pair in pm4Host.TileMdxInstances)
            {
                idx = pair.Value.FindIndex(m => m.UniqueId == inst.UniqueId);
                if (idx >= 0) { targetList = pair.Value; tileX = pair.Key.Item1; tileY = pair.Key.Item2; break; }
            }
        }

        if (targetList == null || idx < 0) return false;

        var prev = targetList[idx];
        _undoStack.Push(new SnapUndo(tileX, tileY, new List<(int, Vector3, Vector3, float)>
        {
            (prev.UniqueId, prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale)
        }));

        float newYaw = prev.PlacementRotation.Z + deltaYawDegrees;
        while (newYaw > 180f) newYaw -= 360f;
        while (newYaw < -180f) newYaw += 360f;

        Vector3 currentCentroid = (prev.BoundsMin + prev.BoundsMax) * 0.5f;
        Vector3 localMin = prev.SelectionBoundsResolved && prev.SelectionLocalBoundsMax != Vector3.Zero ? prev.SelectionLocalBoundsMin : prev.LocalBoundsMin;
        Vector3 localMax = prev.SelectionBoundsResolved && prev.SelectionLocalBoundsMax != Vector3.Zero ? prev.SelectionLocalBoundsMax : prev.LocalBoundsMax;

        Matrix4x4 unpositioned = WorldPlacementTransform.Build(Vector3.Zero, new Vector3(prev.PlacementRotation.X, prev.PlacementRotation.Y, newYaw), prev.PlacementScale);
        WorldScene.TransformBounds(localMin, localMax, unpositioned, out Vector3 rotMin, out Vector3 rotMax);
        Vector3 rotCenter = (rotMin + rotMax) * 0.5f;

        prev.PlacementPosition = new Vector3(currentCentroid.X - rotCenter.X, currentCentroid.Y - rotCenter.Y, prev.PlacementPosition.Z);
        prev.PlacementRotation = new Vector3(prev.PlacementRotation.X, prev.PlacementRotation.Y, newYaw);
        prev.Transform = WorldPlacementTransform.Build(prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale);
        WorldScene.TransformBounds(prev.LocalBoundsMin, prev.LocalBoundsMax, prev.Transform, out prev.BoundsMin, out prev.BoundsMax);
        WorldScene.TransformBounds(prev.SelectionLocalBoundsMin, prev.SelectionLocalBoundsMax, prev.Transform, out prev.SelectionLocalBoundsMin, out prev.SelectionLocalBoundsMax);

        targetList[idx] = prev;
        pm4Host.RebuildInstanceLists();

        // Update taught pairing if paired with PM4 object
        if (worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue)
        {
            var selKey = worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value;
            if (worldScene.Pm4Overlay._pm4ObjectLookup.TryGetValue(selKey, out var pm4Obj))
            {
                TeachPairing(pm4Obj, prev, selKey.tileX, selKey.tileY);
            }
        }

        string msg = $"✓ Rotated {prev.ModelName} (#{prev.UniqueId}) by {deltaYawDegrees:+0;-0}° (Yaw: {newYaw:F1}°).";
        _host.StatusMessage = msg;
        LastOperationMessage = msg;
        return true;
    }
    /// <summary>
    /// Nudges a placed scene object along the vertical Z axis by a specified delta in yards.
    /// Updates the transform, bounding boxes, and propagates to taught pairings.
    /// </summary>
    public bool NudgePlacementZ(ObjectInstance inst, float deltaZ)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null) return false;
        IPm4OverlayHost pm4Host = worldScene;

        int tileX = inst.TileX;
        int tileY = inst.TileY;
        bool isWmo = IsWmoAsset(inst.ModelPath, inst.AssetKind);
        List<ObjectInstance>? targetList = null;
        int idx = -1;

        if (isWmo && pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var tileWmo))
        {
            idx = tileWmo.FindIndex(m => m.UniqueId == inst.UniqueId);
            if (idx >= 0) targetList = tileWmo;
        }
        if (targetList == null && pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx))
        {
            idx = tileMdx.FindIndex(m => m.UniqueId == inst.UniqueId);
            if (idx >= 0) targetList = tileMdx;
        }
        if (targetList == null || idx < 0)
        {
            foreach (var pair in pm4Host.TileWmoInstances)
            {
                idx = pair.Value.FindIndex(m => m.UniqueId == inst.UniqueId);
                if (idx >= 0) { targetList = pair.Value; tileX = pair.Key.Item1; tileY = pair.Key.Item2; break; }
            }
        }
        if (targetList == null || idx < 0)
        {
            foreach (var pair in pm4Host.TileMdxInstances)
            {
                idx = pair.Value.FindIndex(m => m.UniqueId == inst.UniqueId);
                if (idx >= 0) { targetList = pair.Value; tileX = pair.Key.Item1; tileY = pair.Key.Item2; break; }
            }
        }

        if (targetList == null || idx < 0) return false;

        var prev = targetList[idx];
        _undoStack.Push(new SnapUndo(tileX, tileY, new List<(int, Vector3, Vector3, float)>
        {
            (prev.UniqueId, prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale)
        }));

        prev.PlacementPosition = new Vector3(prev.PlacementPosition.X, prev.PlacementPosition.Y, prev.PlacementPosition.Z + deltaZ);
        prev.Transform = WorldPlacementTransform.Build(prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale);
        WorldScene.TransformBounds(prev.LocalBoundsMin, prev.LocalBoundsMax, prev.Transform, out prev.BoundsMin, out prev.BoundsMax);
        WorldScene.TransformBounds(prev.SelectionLocalBoundsMin, prev.SelectionLocalBoundsMax, prev.Transform, out prev.SelectionLocalBoundsMin, out prev.SelectionLocalBoundsMax);

        targetList[idx] = prev;
        pm4Host.RebuildInstanceLists();

        if (worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue)
        {
            var selKey = worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value;
            if (worldScene.Pm4Overlay._pm4ObjectLookup.TryGetValue(selKey, out var pm4Obj))
            {
                TeachPairing(pm4Obj, prev, selKey.tileX, selKey.tileY);
            }
        }

        string msg = $"✓ Nudged {prev.ModelName} (#{prev.UniqueId}) Z by {deltaZ:+0.00;-0.00} yd (Z: {prev.PlacementPosition.Z:F3}).";
        _host.StatusMessage = msg;
        LastOperationMessage = msg;
        return true;
    }

    /// <summary>
    /// Teaches a verified human pairing between a PM4 collision hull and a scene model, saving
    /// the learned geometry signature and solved parameters for automation across the map.
    /// </summary>
    public TaughtPairing TeachPairing(
        Pm4OverlayObject pm4Obj,
        ObjectInstance inst,
        int pm4TileX,
        int pm4TileY)
    {
        var worldScene = _host.WorldScene;
        var surfaceVerts = ExtractCompositeSurfaceVertices(worldScene, pm4Obj, pm4TileX, pm4TileY, out float minWorldZ);
        List<Vector3>? mprlList = null;
        if (worldScene != null)
            worldScene.Pm4Overlay._pm4TilePositionRefs.TryGetValue((pm4TileX, pm4TileY), out mprlList);
        float? placementZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : null;
        Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, minWorldZ);

        Vector3 localMin = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
            ? inst.SelectionLocalBoundsMin : inst.LocalBoundsMin;
        Vector3 localMax = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
            ? inst.SelectionLocalBoundsMax : inst.LocalBoundsMax;
        bool isWmo = IsWmoAsset(inst.ModelPath, inst.AssetKind);
        string assetKind = isWmo ? "WMO" : ResolveAssetKind(inst.ModelPath, inst.AssetKind);
        var cand = Pm4ModelCandidate.Create(inst.ModelPath, assetKind, localMin, localMax);
        var matchResults = Pm4DirectModelMatcher.MatchPrism(prism, [cand], 1);
        float scale = isWmo ? 1.0f : (matchResults.Count > 0 ? matchResults[0].SolvedScale : inst.PlacementScale);

        float yawOffset = inst.PlacementRotation.Z - prism.YawDegrees;
        while (yawOffset > 180f) yawOffset -= 360f;
        while (yawOffset < -180f) yawOffset += 360f;

        var shape3D = pm4Obj.Shape3D;
        Vector3 hullCentroid = Vector3.Transform(pm4Obj.Center, pm4Obj.BaseTransform);
        WorldScene.TransformBounds(pm4Obj.BoundsMin, pm4Obj.BoundsMax, pm4Obj.BaseTransform, out Vector3 hullMin, out Vector3 hullMax);

        var taught = new TaughtPairing(
            pm4Obj.Ck24,
            pm4Obj.DominantGroupKey,
            inst.ModelName,
            inst.ModelPath,
            assetKind,
            scale,
            prism.YawDegrees,
            prism.UnrotatedLength,
            prism.UnrotatedWidth,
            prism.Height,
            DateTime.UtcNow,
            localMin,
            localMax,
            yawOffset,
            shape3D.VerticalSliceRadii,
            shape3D.InertiaTensor,
            shape3D.NormalDistribution,
            // Spec 298: Authentic Placed Object & PM4 Hull Telemetry
            PlacementPosition: inst.PlacementPosition,
            PlacementRotation: inst.PlacementRotation,
            PlacementScale: inst.PlacementScale,
            UniqueId: inst.UniqueId,
            TileX: pm4TileX,
            TileY: pm4TileY,
            ObjectPartId: pm4Obj.ObjectPartId,
            LinkGroupObjectId: pm4Obj.LinkGroupObjectId,
            HullCentroid: hullCentroid,
            HullBoundsMin: hullMin,
            HullBoundsMax: hullMax);

        _taughtPairings.RemoveAll(t =>
            string.Equals(t.ModelPath, taught.ModelPath, StringComparison.OrdinalIgnoreCase)
            || (t.Ck24 != 0 && t.Ck24 == taught.Ck24));
        _taughtPairings.Add(taught);

        SaveTaughtPairingsToDisk();

        string msg = $"✓ Saved teaching rule: {inst.ModelName} ({assetKind}) ↔ PM4 {prism.UnrotatedLength:F1}x{prism.UnrotatedWidth:F1}x{prism.Height:F1} yds (scale {scale:F2}, yaw offset {yawOffset:F1}°).";
        _host.StatusMessage = msg;
        LastOperationMessage = msg;
        return taught;
    }

    /// <summary>
    /// Explicitly registers or updates a verified human-taught pairing rule and saves it to disk.
    /// </summary>
    public void TeachPairingExplicit(TaughtPairing taught)
    {
        bool isWmo = IsWmoAsset(taught.ModelPath, taught.AssetKind);
        if (isWmo && (taught.SolvedScale != 1.0f || !string.Equals(taught.AssetKind, "WMO", StringComparison.OrdinalIgnoreCase)))
        {
            taught = taught with { AssetKind = "WMO", SolvedScale = 1.0f };
        }

        _taughtPairings.RemoveAll(t =>
            string.Equals(t.ModelPath, taught.ModelPath, StringComparison.OrdinalIgnoreCase)
            || (t.Ck24 != 0 && t.Ck24 == taught.Ck24));
        _taughtPairings.Add(taught);
        SaveTaughtPairingsToDisk();
    }

    private string ResolveMapName(string? mapName = null)
    {
        if (!string.IsNullOrWhiteSpace(mapName)) return Pm4PairingProjectStore.SanitizeMapName(mapName);
        if (!string.IsNullOrWhiteSpace(_host.LoadedFileName))
            return Pm4PairingProjectStore.SanitizeMapName(_host.LoadedFileName);
        return "development";
    }

    public string SaveTaughtProject(string? mapName = null)
    {
        string map = ResolveMapName(mapName);
        string path = Pm4PairingProjectStore.SaveProject(map, _taughtPairings);
        string msg = $"✓ Saved {_taughtPairings.Count} taught pairings to {path}";
        _host.StatusMessage = msg;
        LastOperationMessage = msg;
        return path;
    }

    public bool LoadTaughtProject(string? mapName, out int count)
    {
        string map = ResolveMapName(mapName);
        if (Pm4PairingProjectStore.LoadProject(map, out var loaded))
        {
            _taughtPairings.Clear();
            _taughtPairings.AddRange(loaded);
            count = _taughtPairings.Count;
            string msg = $"✓ Loaded {count} taught pairings for '{map}'";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
            return true;
        }
        count = 0;
        return false;
    }

    public string ExportTaughtPairingsManifest(string? mapName = null, string? exportDir = null)
    {
        string map = ResolveMapName(mapName);
        string path = Pm4PairingProjectStore.SaveManifest(map, _taughtPairings, exportDir);
        string msg = $"✓ Exported taught pairing manifest to {path}";
        _host.StatusMessage = msg;
        LastOperationMessage = msg;
        return path;
    }

    private void SaveTaughtPairingsToDisk()
    {
        SaveTaughtProject();
    }

    private void LoadTaughtPairingsFromDisk()
    {
        LoadTaughtProject(null, out _);
    }

    public bool DeleteTaughtPairing(TaughtPairing pairing)
    {
        if (_taughtPairings.Remove(pairing))
        {
            SaveTaughtPairingsToDisk();
            string msg = $"Deleted taught rule for {pairing.ModelName}.";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Automatically applies a human-taught pairing to all unoccupied PM4 hulls on a tile.
    /// </summary>
    public int ApplyTaughtPairingToTile(TaughtPairing taught, int tileX, int tileY)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null) return 0;
        IPm4OverlayHost pm4Host = worldScene;

        if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileX, tileY), out var pm4Objects) || pm4Objects.Count == 0)
        {
            if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileY, tileX), out pm4Objects) || pm4Objects.Count == 0)
                return 0;
        }

        var added = PopulateTileFromTaughtPairing(pm4Host, worldScene, taught, tileX, tileY, pm4Objects);
        if (added.Count > 0)
        {
            var addedByTile = new Dictionary<(int, int), List<int>> { [(tileX, tileY)] = added };
            _undoStack.Push(new MultiTileDuplicationUndo(addedByTile));
            pm4Host.RebuildInstanceLists();
            pm4Host.Assets.ProcessPendingLoads(maxLoads: 128, maxBudgetMs: 150.0);
            string msg = $"✓ Populated {added.Count} PM4 hulls matching shape of {taught.ModelName} on tile ({tileX}, {tileY}).";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
            return added.Count;
        }

        string failMsg = $"No unoccupied PM4 hulls matching shape of {taught.ModelName} on tile ({tileX}, {tileY}).";
        _host.StatusMessage = failMsg;
        LastOperationMessage = failMsg;
        return 0;
    }

    /// <summary>
    /// Applies a human-taught pairing to all matching unoccupied PM4 hulls across ALL loaded tiles.
    /// </summary>
    public int ApplyTaughtPairingAcrossAllTiles(TaughtPairing taught)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null) return 0;
        IPm4OverlayHost pm4Host = worldScene;

        var addedByTile = new Dictionary<(int tileX, int tileY), List<int>>();

        foreach (var (tileCoord, pm4Objects) in worldScene.Pm4Overlay._pm4TileObjects)
        {
            if (pm4Objects == null || pm4Objects.Count == 0) continue;
            var tileAdded = PopulateTileFromTaughtPairing(pm4Host, worldScene, taught, tileCoord.tileX, tileCoord.tileY, pm4Objects);
            if (tileAdded.Count > 0)
            {
                addedByTile[tileCoord] = tileAdded;
            }
        }

        int totalAdded = addedByTile.Values.Sum(v => v.Count);
        if (totalAdded > 0)
        {
            _undoStack.Push(new MultiTileDuplicationUndo(addedByTile));
            pm4Host.RebuildInstanceLists();
            pm4Host.Assets.ProcessPendingLoads(maxLoads: 128, maxBudgetMs: 150.0);
            string msg = $"✓ Populated {totalAdded} PM4 hulls matching shape of {taught.ModelName} across {addedByTile.Count} loaded tiles.";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
            return totalAdded;
        }

        string failMsg = $"No unoccupied PM4 hulls matching shape of {taught.ModelName} found across {worldScene.Pm4Overlay._pm4TileObjects.Count} loaded tiles.";
        _host.StatusMessage = failMsg;
        LastOperationMessage = failMsg;
        return 0;
    }

    /// <summary>
    /// Applies ALL saved human-taught pairings to all matching unoccupied PM4 hulls across ALL loaded tiles.
    /// </summary>
    public int ApplyAllTaughtPairingsAcrossAllTiles()
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null || _taughtPairings.Count == 0)
        {
            string emptyMsg = _taughtPairings.Count == 0
                ? "No taught pairings saved yet. Pair a PM4 hull and model to teach one first."
                : "No active world scene.";
            _host.StatusMessage = emptyMsg;
            LastOperationMessage = emptyMsg;
            return 0;
        }

        IPm4OverlayHost pm4Host = worldScene;
        var addedByTile = new Dictionary<(int tileX, int tileY), List<int>>();
        int totalRulesApplied = 0;

        for (int r = 0; r < _taughtPairings.Count; r++)
        {
            var taught = _taughtPairings[r];
            int ruleAddedCount = 0;

            foreach (var (tileCoord, pm4Objects) in worldScene.Pm4Overlay._pm4TileObjects)
            {
                if (pm4Objects == null || pm4Objects.Count == 0) continue;
                var tileAdded = PopulateTileFromTaughtPairing(pm4Host, worldScene, taught, tileCoord.tileX, tileCoord.tileY, pm4Objects);
                if (tileAdded.Count > 0)
                {
                    if (!addedByTile.TryGetValue(tileCoord, out var list))
                    {
                        list = new List<int>();
                        addedByTile[tileCoord] = list;
                    }
                    list.AddRange(tileAdded);
                    ruleAddedCount += tileAdded.Count;
                }
            }

            if (ruleAddedCount > 0)
                totalRulesApplied++;
        }

        int totalAdded = addedByTile.Values.Sum(v => v.Count);
        if (totalAdded > 0)
        {
            _undoStack.Push(new MultiTileDuplicationUndo(addedByTile));
            pm4Host.RebuildInstanceLists();
            pm4Host.Assets.ProcessPendingLoads(maxLoads: 128, maxBudgetMs: 150.0);
            string msg = $"✓ Automation complete: Populated {totalAdded} placements using {totalRulesApplied} taught rules across {addedByTile.Count} tiles.";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
            return totalAdded;
        }

        string failMsg = $"No matching unoccupied PM4 hulls found for any of the {_taughtPairings.Count} taught rules across loaded tiles.";
        _host.StatusMessage = failMsg;
        LastOperationMessage = failMsg;
        return 0;
    }

    /// <summary>
    /// Checks whether an arbitrary PM4 hull on any tile matches any taught pairing rule.
    /// </summary>
    public TaughtPairing? FindMatchingTaughtRule(
        Pm4OverlayObject pm4Obj,
        int pm4TileX,
        int pm4TileY,
        out float solvedScale,
        out Vector3 solvedPos,
        out Vector3 solvedRot)
    {
        solvedScale = 1f;
        solvedPos = default;
        solvedRot = default;

        if (_taughtPairings.Count == 0) return null;

        var worldScene = _host.WorldScene;
        var surfaceVerts = ExtractCompositeSurfaceVertices(worldScene, pm4Obj, pm4TileX, pm4TileY, out float minWorldZ);
        List<Vector3>? mprlList = null;
        if (worldScene != null)
            worldScene.Pm4Overlay._pm4TilePositionRefs.TryGetValue((pm4TileX, pm4TileY), out mprlList);
        float? placementZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : null;
        Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, minWorldZ);

        for (int i = 0; i < _taughtPairings.Count; i++)
        {
            var taught = _taughtPairings[i];
            Vector3 localMin = taught.LocalBoundsMin != Vector3.Zero || taught.LocalBoundsMax != Vector3.Zero
                ? taught.LocalBoundsMin : -new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f;
            Vector3 localMax = taught.LocalBoundsMin != Vector3.Zero || taught.LocalBoundsMax != Vector3.Zero
                ? taught.LocalBoundsMax : new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f;
            var cand = Pm4ModelCandidate.Create(taught.ModelPath, taught.AssetKind ?? "M2", localMin, localMax);

            if (MatchesTaughtShape(taught, cand, prism, pm4Obj.Ck24, pm4Obj.DominantGroupKey, out solvedScale, out solvedPos, out solvedRot, pm4Obj.Shape3D))
            {
                return taught;
            }
        }

        return null;
    }

    /// <summary>
    /// Instantiates a taught pairing directly into a PM4 hull.
    /// </summary>
    public bool InstantiateTaughtPairingIntoHull(
        Pm4OverlayObject pm4Obj,
        int pm4TileX,
        int pm4TileY,
        TaughtPairing taught,
        float solvedScale,
        Vector3 solvedPos,
        Vector3 solvedRot,
        bool force = false)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null) return false;
        IPm4OverlayHost pm4Host = worldScene;

        if (!force && HullHasSameModel(pm4Host, pm4Obj, pm4TileX, pm4TileY, taught.ModelPath))
        {
            string dupMsg = $"PM4 hull [{pm4TileX}_{pm4TileY}] Obj #{pm4Obj.ObjectPartId} already has {taught.ModelName} assigned; skipped duplicate spawn.";
            _host.StatusMessage = dupMsg;
            LastOperationMessage = dupMsg;
            return false;
        }

        bool isWmo = IsWmoAsset(taught.ModelPath, taught.AssetKind);
        float actualScale = isWmo ? 1.0f : solvedScale;
        string assetKind = isWmo ? "WMO" : ResolveAssetKind(taught.ModelPath, taught.AssetKind);

        pm4Host.TileMdxInstances.TryGetValue((pm4TileX, pm4TileY), out var tileMdx);
        pm4Host.TileWmoInstances.TryGetValue((pm4TileX, pm4TileY), out var tileWmo);

        int maxId = 0;
        if (tileMdx != null)
            for (int i = 0; i < tileMdx.Count; i++) if (tileMdx[i].UniqueId > maxId) maxId = tileMdx[i].UniqueId;
        if (tileWmo != null)
            for (int i = 0; i < tileWmo.Count; i++) if (tileWmo[i].UniqueId > maxId) maxId = tileWmo[i].UniqueId;

        int newId = Math.Max(Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase, maxId + 1);
        Matrix4x4 transform = WorldPlacementTransform.Build(solvedPos, solvedRot, actualScale);

        Vector3 localMin = taught.LocalBoundsMin != Vector3.Zero || taught.LocalBoundsMax != Vector3.Zero
            ? taught.LocalBoundsMin : -new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f;
        Vector3 localMax = taught.LocalBoundsMin != Vector3.Zero || taught.LocalBoundsMax != Vector3.Zero
            ? taught.LocalBoundsMax : new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f;

        WorldScene.TransformBounds(localMin, localMax, transform, out Vector3 bMin, out Vector3 bMax);

        var newInst = new ObjectInstance
        {
            ModelKey = WorldAssetManager.NormalizeKey(taught.ModelPath),
            Transform = transform,
            BoundsMin = bMin,
            BoundsMax = bMax,
            LocalBoundsMin = localMin,
            LocalBoundsMax = localMax,
            BoundsResolved = true,
            ModelName = taught.ModelName,
            ModelPath = taught.ModelPath,
            AssetKind = assetKind,
            PlacementPosition = solvedPos,
            PlacementRotation = solvedRot,
            PlacementScale = actualScale,
            UniqueId = newId,
            PlacementEntryIndex = isWmo ? (tileWmo?.Count ?? 0) : (tileMdx?.Count ?? 0),
            TileX = pm4TileX,
            TileY = pm4TileY,
            HasTileCoordinate = true
        };

        if (isWmo)
        {
            if (tileWmo == null)
            {
                tileWmo = new List<ObjectInstance>();
                pm4Host.TileWmoInstances[(pm4TileX, pm4TileY)] = tileWmo;
            }
            tileWmo.Add(newInst);
            pm4Host.Assets.QueueWmoLoad(newInst.ModelKey);
            pm4Host.Assets.PrioritizeWmoLoad(newInst.ModelKey);
        }
        else
        {
            if (tileMdx == null)
            {
                tileMdx = new List<ObjectInstance>();
                pm4Host.TileMdxInstances[(pm4TileX, pm4TileY)] = tileMdx;
            }
            tileMdx.Add(newInst);
            pm4Host.Assets.QueueMdxLoad(newInst.ModelKey);
            pm4Host.Assets.PrioritizeMdxLoad(newInst.ModelKey);
        }

        var addedByTile = new Dictionary<(int, int), List<int>> { [(pm4TileX, pm4TileY)] = [newId] };
        _undoStack.Push(new MultiTileDuplicationUndo(addedByTile));
        pm4Host.RebuildInstanceLists();
        pm4Host.Assets.ProcessPendingLoads(maxLoads: 64, maxBudgetMs: 100.0);

        string msg = $"✓ Spawned taught {taught.ModelName} (#{newId}) into PM4 hull [{pm4TileX}_{pm4TileY}] Obj #{pm4Obj.ObjectPartId}.";
        _host.StatusMessage = msg;
        LastOperationMessage = msg;
        return true;
    }

    private List<int> PopulateTileFromTaughtPairing(
        IPm4OverlayHost pm4Host,
        WorldScene worldScene,
        TaughtPairing taught,
        int tileX,
        int tileY,
        IReadOnlyList<Pm4OverlayObject> pm4Objects)
    {
        bool isWmo = IsWmoAsset(taught.ModelPath, taught.AssetKind);
        string assetKind = isWmo ? "WMO" : ResolveAssetKind(taught.ModelPath, taught.AssetKind);
        pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx);
        pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var tileWmo);

        int maxExistingUniqueId = 0;
        var placementBounds = BuildPlacementBounds(pm4Host, tileX, tileY);

        if (tileMdx != null)
        {
            for (int i = 0; i < tileMdx.Count; i++)
                if (tileMdx[i].UniqueId > maxExistingUniqueId) maxExistingUniqueId = tileMdx[i].UniqueId;
        }
        if (tileWmo != null)
        {
            for (int i = 0; i < tileWmo.Count; i++)
                if (tileWmo[i].UniqueId > maxExistingUniqueId) maxExistingUniqueId = tileWmo[i].UniqueId;
        }

        int nextId = Math.Max(Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase, maxExistingUniqueId + 1);
        var addedIds = new List<int>();

        float occRadius = Math.Max(2.5f, Math.Min(taught.Width, taught.Length) * 0.75f);

        Vector3 localMin = taught.LocalBoundsMin != Vector3.Zero || taught.LocalBoundsMax != Vector3.Zero
            ? taught.LocalBoundsMin : -new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f;
        Vector3 localMax = taught.LocalBoundsMin != Vector3.Zero || taught.LocalBoundsMax != Vector3.Zero
            ? taught.LocalBoundsMax : new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f;
        var cand = Pm4ModelCandidate.Create(taught.ModelPath, assetKind, localMin, localMax);

        worldScene.Pm4Overlay._pm4TilePositionRefs.TryGetValue((tileX, tileY), out var mprlList);

        for (int i = 0; i < pm4Objects.Count; i++)
        {
            var obj = pm4Objects[i];

            // 1. Occupancy check: skip hulls that already have an assigned/paired model.
            // Compare world-space bounds, not hull centroid vs model origin — WMO/M2 origins sit
            // at the base and are routinely far from the hull centroid, which caused duplicate
            // spawns on hulls that were already paired.
            var target = BuildTarget(obj, tileX, tileY);
            if (Pm4DoodadDuplicationEngine.IsTargetOccupied(target, placementBounds, occRadius))
                continue;

            // 2. Extrude candidate footprint prism
            var surfaceVerts = new List<Vector3>(obj.Triangles.Count * 3);
            for (int t = 0; t < obj.Triangles.Count; t++)
            {
                surfaceVerts.Add(Vector3.Transform(obj.Triangles[t].A, obj.BaseTransform));
                surfaceVerts.Add(Vector3.Transform(obj.Triangles[t].B, obj.BaseTransform));
                surfaceVerts.Add(Vector3.Transform(obj.Triangles[t].C, obj.BaseTransform));
            }
            float? placementZ = obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(obj.Ck24 << 8) : null;
            Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, target.BoundsMin.Z);

            // 3. Shape match evaluation
            if (!MatchesTaughtShape(taught, cand, prism, obj.Ck24, obj.DominantGroupKey, out float solvedScale, out Vector3 solvedPos, out Vector3 solvedRot, obj.Shape3D))
                continue;

            float actualScale = isWmo ? 1.0f : solvedScale;
            Matrix4x4 transform = WorldPlacementTransform.Build(solvedPos, solvedRot, actualScale);
            WorldScene.TransformBounds(cand.LocalBoundsMin, cand.LocalBoundsMax, transform, out Vector3 bMin, out Vector3 bMax);

            int assignedId = nextId++;
            var newInst = new ObjectInstance
            {
                ModelKey = WorldAssetManager.NormalizeKey(taught.ModelPath),
                Transform = transform,
                BoundsMin = bMin,
                BoundsMax = bMax,
                LocalBoundsMin = cand.LocalBoundsMin,
                LocalBoundsMax = cand.LocalBoundsMax,
                BoundsResolved = true,
                ModelName = taught.ModelName,
                ModelPath = taught.ModelPath,
                AssetKind = assetKind,
                PlacementPosition = solvedPos,
                PlacementRotation = solvedRot,
                PlacementScale = actualScale,
                UniqueId = assignedId,
                PlacementEntryIndex = isWmo ? (tileWmo?.Count ?? 0) : (tileMdx?.Count ?? 0),
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true
            };

            if (isWmo)
            {
                if (tileWmo == null)
                {
                    tileWmo = new List<ObjectInstance>();
                    pm4Host.TileWmoInstances[(tileX, tileY)] = tileWmo;
                }
                tileWmo.Add(newInst);
                pm4Host.Assets.QueueWmoLoad(newInst.ModelKey);
                pm4Host.Assets.PrioritizeWmoLoad(newInst.ModelKey);
            }
            else
            {
                if (tileMdx == null)
                {
                    tileMdx = new List<ObjectInstance>();
                    pm4Host.TileMdxInstances[(tileX, tileY)] = tileMdx;
                }
                tileMdx.Add(newInst);
                pm4Host.Assets.QueueMdxLoad(newInst.ModelKey);
                pm4Host.Assets.PrioritizeMdxLoad(newInst.ModelKey);
            }

            placementBounds.Add(new PlacementBounds(bMin, bMax, solvedPos, true));
            addedIds.Add(assignedId);
        }

        return addedIds;
    }

    internal static bool MatchesTaughtShape(
        TaughtPairing taught,
        Pm4ModelCandidate cand,
        Pm4FootprintPrism prism,
        uint objCk24,
        byte dominantGroupKey,
        out float solvedScale,
        out Vector3 solvedPos,
        out Vector3 solvedRot,
        Pm4Geometry3DShapeSignature? objShape = null)
    {
        bool isWmo = IsWmoAsset(taught.ModelPath, taught.AssetKind);
        solvedScale = isWmo ? 1.0f : 1.0f;
        solvedPos = default;
        solvedRot = default;

        // M2 doodads generally don't match building floor classes (0x10..0x15) unless taught that way
        if (!isWmo && (dominantGroupKey >= 0x10 && dominantGroupKey <= 0x15) && taught.DominantGroupKey < 0x10)
            return false;

        bool exactCkMatch = taught.Ck24 != 0 && taught.Ck24 == objCk24;

        float prismLen = MathF.Max(prism.UnrotatedLength, prism.UnrotatedWidth);
        float prismWid = MathF.Min(prism.UnrotatedLength, prism.UnrotatedWidth);
        float taughtLen = MathF.Max(taught.Length, taught.Width);
        float taughtWid = MathF.Min(taught.Length, taught.Width);

        float prismPlanar = MathF.Sqrt(prismLen * prismLen + prismWid * prismWid);
        float taughtPlanar = MathF.Sqrt(taughtLen * taughtLen + taughtWid * taughtWid);

        // WMOs have fixed 1:1 scale: reject if PM4 hull footprint size diverges significantly from authentic building
        if (isWmo && !exactCkMatch)
        {
            float sizeRatio = taughtPlanar > 0.01f ? prismPlanar / taughtPlanar : 1.0f;
            if (sizeRatio < 0.60f || sizeRatio > 1.50f)
                return false;
        }

        // 3D Geometry Shape Signature Comparison:
        // If taught rule includes 3D shape signature (vertical slice profile, inertia moments, normal distribution),
        // verify that the candidate 3D shape actually conforms to the taught 3D shape.
        if (taught.VerticalSliceRadii != null && objShape != null && !exactCkMatch)
        {
            float dProfile = 0f;
            for (int b = 0; b < Pm4Geometry3DShapeSignature.VerticalSliceBands; b++)
            {
                dProfile += MathF.Abs(taught.VerticalSliceRadii[b] - objShape.VerticalSliceRadii[b]);
            }
            dProfile /= Pm4Geometry3DShapeSignature.VerticalSliceBands;

            float dInertia = Vector3.Distance(taught.InertiaTensor, objShape.InertiaTensor);
            float dNormals = Vector3.Distance(taught.NormalDistribution, objShape.NormalDistribution);

            // Reject if 3D vertical profile, mass distribution, or surface normal angles differ significantly
            if (dProfile > 0.40f || dInertia > 0.35f || dNormals > 0.40f)
                return false;
        }

        float prismHeight = MathF.Max(0.01f, prism.Height);
        float taughtHeight = MathF.Max(0.01f, taught.Height);

        float prismAspectXY = prismLen / MathF.Max(0.01f, prismWid);
        float taughtAspectXY = taughtLen / MathF.Max(0.01f, taughtWid);
        float diffAspectXY = MathF.Abs(prismAspectXY - taughtAspectXY) / MathF.Max(prismAspectXY, taughtAspectXY);

        float prismAspectZH = prismHeight / MathF.Max(0.01f, prismLen);
        float taughtAspectZH = taughtHeight / MathF.Max(0.01f, taughtLen);
        float diffAspectZH = MathF.Abs(prismAspectZH - taughtAspectZH) / MathF.Max(prismAspectZH, taughtAspectZH);

        float sizeDiff = MathF.Abs(prismPlanar - taughtPlanar) / MathF.Max(prismPlanar, taughtPlanar);

        double shapeScore = Math.Max(0.0, 1.0 - (0.5 * diffAspectXY + 0.3 * diffAspectZH + 0.2 * sizeDiff));

        if (!exactCkMatch)
        {
            if (diffAspectXY > 0.25f || (!isWmo && (sizeDiff > 0.45f || shapeScore < 0.70)))
                return false;
        }

        var matchResults = Pm4DirectModelMatcher.MatchPrism(prism, [cand], 1);
        if (matchResults.Count > 0)
        {
            var match = matchResults[0];
            solvedScale = isWmo ? 1.0f : match.SolvedScale;

            // MatchPrism solves SolvedPosition for its +90° heuristic yaw, but the applied
            // rotation is the taught yaw offset. Re-solve the position for the final taught
            // yaw so the rotated model center lands exactly on the hull centroid (same math
            // as AlignPlacementToPm4Hull); otherwise taught placements drift off-hull.
            float yaw = prism.YawDegrees + taught.YawOffset;
            while (yaw > 180f) yaw -= 360f;
            while (yaw < -180f) yaw += 360f;

            Matrix4x4 unpositioned = WorldPlacementTransform.Build(Vector3.Zero, new Vector3(0f, 0f, yaw), solvedScale);
            WorldScene.TransformBounds(cand.LocalBoundsMin, cand.LocalBoundsMax, unpositioned, out Vector3 rotMin, out Vector3 rotMax);
            Vector3 rotCenter = (rotMin + rotMax) * 0.5f;

            float posZ;
            if (isWmo)
            {
                float? hullPlacementZ = objCk24 != 0 ? BitConverter.UInt32BitsToSingle(objCk24 << 8) : null;
                posZ = hullPlacementZ.HasValue && MathF.Abs(hullPlacementZ.Value) > 0.01f ? hullPlacementZ.Value : prism.BaseZ;
            }
            else
            {
                posZ = prism.TopZ - (rotMax.Z > 0.01f ? rotMax.Z : MathF.Max(0.01f, rotMax.Z - rotMin.Z));
            }

            solvedPos = new Vector3(prism.Centroid.X - rotCenter.X, prism.Centroid.Y - rotCenter.Y, posZ);
            solvedRot = new Vector3(0f, 0f, yaw);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Instantiates a candidate match directly into an unplaced PM4 hull.
    /// </summary>
    public bool SpawnCandidateIntoPm4Hull(
        Pm4OverlayObject pm4Obj,
        int pm4TileX,
        int pm4TileY,
        Pm4DirectMatchResult match,
        out string statusMessage,
        bool force = false)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null)
        {
            statusMessage = "No active world scene.";
            return false;
        }

        IPm4OverlayHost pm4Host = worldScene;

        if (!force && HullHasSameModel(pm4Host, pm4Obj, pm4TileX, pm4TileY, match.Candidate.ModelPath))
        {
            statusMessage = $"PM4 hull [{pm4TileX}_{pm4TileY}] Obj #{pm4Obj.ObjectPartId} already has {match.Candidate.ModelName} assigned; skipped duplicate spawn.";
            _host.StatusMessage = statusMessage;
            LastOperationMessage = statusMessage;
            return false;
        }

        pm4Host.TileMdxInstances.TryGetValue((pm4TileX, pm4TileY), out var tileMdx);
        pm4Host.TileWmoInstances.TryGetValue((pm4TileX, pm4TileY), out var tileWmo);

        int maxId = 0;
        if (tileMdx != null)
        {
            for (int i = 0; i < tileMdx.Count; i++)
                if (tileMdx[i].UniqueId > maxId) maxId = tileMdx[i].UniqueId;
        }
        if (tileWmo != null)
        {
            for (int i = 0; i < tileWmo.Count; i++)
                if (tileWmo[i].UniqueId > maxId) maxId = tileWmo[i].UniqueId;
        }

        bool isWmo = IsWmoAsset(match.Candidate.ModelPath, match.Candidate.AssetKind);
        float actualScale = isWmo ? 1.0f : match.SolvedScale;
        string assetKind = isWmo ? "WMO" : ResolveAssetKind(match.Candidate.ModelPath, match.Candidate.AssetKind);

        int newId = Math.Max(Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase, maxId + 1);
        Matrix4x4 transform = WorldPlacementTransform.Build(match.SolvedPosition, match.SolvedRotation, actualScale);
        WorldScene.TransformBounds(match.Candidate.LocalBoundsMin, match.Candidate.LocalBoundsMax, transform, out Vector3 bMin, out Vector3 bMax);

        var newInst = new ObjectInstance
        {
            ModelKey = WorldAssetManager.NormalizeKey(match.Candidate.ModelPath),
            Transform = transform,
            BoundsMin = bMin,
            BoundsMax = bMax,
            LocalBoundsMin = match.Candidate.LocalBoundsMin,
            LocalBoundsMax = match.Candidate.LocalBoundsMax,
            BoundsResolved = true,
            ModelName = match.Candidate.ModelName,
            ModelPath = match.Candidate.ModelPath,
            AssetKind = assetKind,
            PlacementPosition = match.SolvedPosition,
            PlacementRotation = match.SolvedRotation,
            PlacementScale = actualScale,
            UniqueId = newId,
            PlacementEntryIndex = isWmo ? (tileWmo?.Count ?? 0) : (tileMdx?.Count ?? 0),
            TileX = pm4TileX,
            TileY = pm4TileY,
            HasTileCoordinate = true
        };

        if (isWmo)
        {
            if (tileWmo == null)
            {
                tileWmo = new List<ObjectInstance>();
                pm4Host.TileWmoInstances[(pm4TileX, pm4TileY)] = tileWmo;
            }
            tileWmo.Add(newInst);
            pm4Host.Assets.QueueWmoLoad(newInst.ModelKey);
            pm4Host.Assets.PrioritizeWmoLoad(newInst.ModelKey);
        }
        else
        {
            if (tileMdx == null)
            {
                tileMdx = new List<ObjectInstance>();
                pm4Host.TileMdxInstances[(pm4TileX, pm4TileY)] = tileMdx;
            }
            tileMdx.Add(newInst);
            pm4Host.Assets.QueueMdxLoad(newInst.ModelKey);
            pm4Host.Assets.PrioritizeMdxLoad(newInst.ModelKey);
        }

        _undoStack.Push(new DuplicationUndo(pm4TileX, pm4TileY, [newId]));
        pm4Host.RebuildInstanceLists();
        pm4Host.Assets.ProcessPendingLoads(maxLoads: 64, maxBudgetMs: 100.0);

        statusMessage = $"✓ Spawned {match.Candidate.ModelName} (#{newId}) into PM4 hull [{pm4TileX}_{pm4TileY}] Obj #{pm4Obj.ObjectPartId} (Score: {match.Score:P0}, Scale: {actualScale:F2}, Yaw: {match.SolvedRotation.Z:F1}°).";
        _host.StatusMessage = statusMessage;
        LastOperationMessage = statusMessage;
        return true;
    }

    /// <summary>
    /// Magnetically snaps a single placed M2 or WMO object to its nearest PM4 navmesh collision hull.
    /// </summary>
    public bool MagnetizePlacementToPm4(ObjectInstance inst, float maxSnapDistance = 60.0f)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null)
            return false;

        var allCandidates = new List<(Vector3 Centroid, Vector3 Min, Vector3 Max, Pm4OverlayObject Rep, int TileX, int TileY)>();
        foreach (var (tileCoord, objList) in worldScene.Pm4Overlay._pm4TileObjects)
        {
            var tileCands = BuildUnifiedPm4Candidates(objList, tileCoord.tileX, tileCoord.tileY);
            allCandidates.AddRange(tileCands);
        }

        if (allCandidates.Count == 0)
        {
            string msg = $"No PM4 objects loaded in scene to magnetize to.";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
            return false;
        }

        (Vector3 Centroid, Vector3 Min, Vector3 Max, Pm4OverlayObject Rep, int TileX, int TileY)? bestCandidate = null;
        float bestDist = maxSnapDistance;

        for (int i = 0; i < allCandidates.Count; i++)
        {
            var c = allCandidates[i];
            float dist = Vector2.Distance(new Vector2(c.Centroid.X, c.Centroid.Y), new Vector2(inst.PlacementPosition.X, inst.PlacementPosition.Y));
            if (dist < bestDist)
            {
                bestDist = dist;
                bestCandidate = c;
            }
        }

        if (bestCandidate == null)
        {
            string msg = $"No PM4 collision hull found within {maxSnapDistance:F0} yds of {inst.ModelName}.";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
            return false;
        }

        return AlignPlacementToPm4Hull(inst, bestCandidate.Value.Rep, bestCandidate.Value.TileX, bestCandidate.Value.TileY, fitScale: false, out _);
    }

    /// <summary>
    /// Scans all resident / donor placements across all loaded tiles, correlates each placed object
    /// with its closest matching PM4 navmesh hull, and automatically registers verified TaughtPairing rules.
    /// </summary>
    public (int Scanned, int NewlyHarvested, int AlreadyKnown, int Ambiguous) HarvestPairingsFromPlacedObjects(
        IViewerAppHost host,
        float maxCentroidDistance = 15.0f)
    {
        var worldScene = host.WorldScene;
        if (worldScene == null)
            return (0, 0, 0, 0);

        IPm4OverlayHost pm4Host = worldScene;
        var loadedTiles = new HashSet<(int TileX, int TileY)>();
        foreach (var key in worldScene.Pm4Overlay._pm4TileObjects.Keys)
            loadedTiles.Add(key);
        foreach (var key in pm4Host.TileMdxInstances.Keys)
            loadedTiles.Add(key);
        foreach (var key in pm4Host.TileWmoInstances.Keys)
            loadedTiles.Add(key);

        if (loadedTiles.Count == 0)
        {
            string emptyMsg = "No tiles loaded with resident placements and PM4 collision to harvest.";
            host.StatusMessage = emptyMsg;
            LastOperationMessage = emptyMsg;
            return (0, 0, 0, 0);
        }

        int scanned = 0;
        int newlyHarvested = 0;
        int alreadyKnown = 0;
        int ambiguous = 0;

        foreach (var (tileX, tileY) in loadedTiles)
        {
            if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileX, tileY), out var pm4Objects) || pm4Objects.Count == 0)
            {
                if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileY, tileX), out pm4Objects) || pm4Objects.Count == 0)
                    continue;
            }

            pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx);
            pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var tileWmo);

            var placements = new List<ObjectInstance>();
            if (tileWmo != null) placements.AddRange(tileWmo);
            if (tileMdx != null) placements.AddRange(tileMdx);

            if (placements.Count == 0)
                continue;

            // Build list of PM4 candidate objects with world-space bounds and centroids
            var pm4Candidates = new List<(Vector3 WorldCenter, Vector3 WorldMin, Vector3 WorldMax, Pm4OverlayObject Obj)>();
            var candBoxes = new List<(Vector3 WorldCenter, Vector3 WorldMin, Vector3 WorldMax)>();
            for (int p = 0; p < pm4Objects.Count; p++)
            {
                var obj = pm4Objects[p];
                WorldScene.TransformBounds(obj.BoundsMin, obj.BoundsMax, obj.BaseTransform, out Vector3 wMin, out Vector3 wMax);
                Vector3 wCenter = (wMin + wMax) * 0.5f;
                pm4Candidates.Add((wCenter, wMin, wMax, obj));
                candBoxes.Add((wCenter, wMin, wMax));
            }

            foreach (var inst in placements)
            {
                if (string.IsNullOrWhiteSpace(inst.ModelPath))
                    continue;

                scanned++;

                Vector3 instPos = inst.PlacementPosition;
                Vector3 modelSpan = inst.BoundsResolved
                    ? (inst.BoundsMax - inst.BoundsMin)
                    : (inst.LocalBoundsMax - inst.LocalBoundsMin);

                int bestIdx = Pm4DoodadDuplicationEngine.FindBestMatchingCandidateHull(
                    instPos, modelSpan, candBoxes, maxCentroidDistance: maxCentroidDistance);

                if (bestIdx < 0)
                    continue;

                var bestCandidate = pm4Candidates[bestIdx];

                // Check if a rule for this exact model already exists
                bool isKnown = _taughtPairings.Any(t =>
                    string.Equals(t.ModelPath, inst.ModelPath, StringComparison.OrdinalIgnoreCase));

                if (isKnown)
                {
                    alreadyKnown++;
                    continue;
                }

                try
                {
                    TeachPairing(bestCandidate.Obj, inst, tileX, tileY);
                    newlyHarvested++;
                }
                catch
                {
                    ambiguous++;
                }
            }
        }

        SaveTaughtPairingsToDisk();

        string summaryMsg = newlyHarvested > 0
            ? $"✓ Harvested {newlyHarvested} new pairing rule(s) ({alreadyKnown} already known) from {scanned} placed model(s) across {loadedTiles.Count} loaded tile(s)."
            : $"Harvest scan complete: {alreadyKnown} model(s) already taught, {scanned} scanned across {loadedTiles.Count} loaded tile(s).";

        host.StatusMessage = summaryMsg;
        LastOperationMessage = summaryMsg;
        return (scanned, newlyHarvested, alreadyKnown, ambiguous);
    }

    /// <summary>
    /// Magnetically snaps all resident ADT placements (M2 and WMO) on a tile to their nearest PM4 collision hulls.
    /// </summary>
    public int MagnetizeTilePlacementsToPm4(int tileX, int tileY, float maxSnapDistance = 60.0f)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null)
            return 0;

        IPm4OverlayHost pm4Host = worldScene;
        if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileX, tileY), out var pm4Objects) || pm4Objects.Count == 0)
        {
            if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileY, tileX), out pm4Objects) || pm4Objects.Count == 0)
            {
                _host.StatusMessage = $"No PM4 objects loaded on tile ({tileX}, {tileY}) to magnetize to.";
                return 0;
            }
        }

        pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx);
        pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var tileWmo);

        int totalPlacements = (tileMdx?.Count ?? 0) + (tileWmo?.Count ?? 0);
        if (totalPlacements == 0)
        {
            _host.StatusMessage = $"No resident placements found on tile ({tileX}, {tileY}).";
            return 0;
        }

        var candidates = BuildUnifiedPm4Candidates(pm4Objects);
        var undoList = new List<(int, Vector3, Vector3, float)>();
        int snappedCount = 0;

        void SnapList(List<ObjectInstance>? list, bool isWmo)
        {
            if (list == null) return;
            for (int i = 0; i < list.Count; i++)
            {
                var inst = list[i];
                (Vector3 Centroid, Vector3 Min, Vector3 Max, Pm4OverlayObject Rep, int TileX, int TileY)? bestMatch = null;
                float bestDist = maxSnapDistance;

                for (int p = 0; p < candidates.Count; p++)
                {
                    var c = candidates[p];
                    float dist = Vector2.Distance(new Vector2(c.Centroid.X, c.Centroid.Y), new Vector2(inst.PlacementPosition.X, inst.PlacementPosition.Y));
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestMatch = c;
                    }
                }

                if (bestMatch != null && bestDist > 0.05f)
                {
                    Vector3 localMin = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
                        ? inst.SelectionLocalBoundsMin
                        : inst.LocalBoundsMin;
                    Vector3 localMax = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
                        ? inst.SelectionLocalBoundsMax
                        : inst.LocalBoundsMax;

                    float candDx = MathF.Abs(localMax.X - localMin.X);
                    float candDy = MathF.Abs(localMax.Y - localMin.Y);

                    float? knownYawOffset = null;
                    var taught = _taughtPairings.Find(t =>
                        string.Equals(t.ModelPath, inst.ModelPath, StringComparison.OrdinalIgnoreCase) ||
                        (t.Ck24 != 0 && t.Ck24 == bestMatch.Value.Rep.Ck24));
                    if (taught != null)
                    {
                        knownYawOffset = taught.YawOffset;
                    }
                    else if (Pm4PriorKnowledgeDatabase.Instance.TryGetAsset(inst.ModelPath, out var dbAsset) && dbAsset != null && MathF.Abs(dbAsset.YawOffset) > 1e-4f)
                    {
                        knownYawOffset = dbAsset.YawOffset;
                    }

                    var surfVerts = ExtractCompositeSurfaceVertices(worldScene, bestMatch.Value.Rep, bestMatch.Value.TileX, bestMatch.Value.TileY, out float _);
                    Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfVerts, null, null, bestMatch.Value.Min.Z);

                    float solvedYaw;
                    if (knownYawOffset.HasValue)
                    {
                        solvedYaw = prism.YawDegrees + knownYawOffset.Value;
                    }
                    else
                    {
                        solvedYaw = prism.YawDegrees + (candDy > candDx * 1.15f ? 90f : 0f);
                        float delta = MathF.IEEERemainder(inst.PlacementRotation.Z - solvedYaw, 360f);
                        if (MathF.Abs(delta) > 90f)
                        {
                            solvedYaw += 180f;
                        }
                    }
                    while (solvedYaw > 180f) solvedYaw -= 360f;
                    while (solvedYaw < -180f) solvedYaw += 360f;

                    Matrix4x4 unpositioned = WorldPlacementTransform.Build(Vector3.Zero, new Vector3(0f, 0f, solvedYaw), inst.PlacementScale);
                    WorldScene.TransformBounds(localMin, localMax, unpositioned, out Vector3 rotMin, out Vector3 rotMax);
                    Vector3 rotCenter = (rotMin + rotMax) * 0.5f;

                    float targetPlacementZ;
                    if (isWmo)
                    {
                        float? placementZ = bestMatch.Value.Rep.Ck24 != 0
                            ? BitConverter.UInt32BitsToSingle(bestMatch.Value.Rep.Ck24 << 8)
                            : null;

                        if (placementZ.HasValue && MathF.Abs(placementZ.Value) > 0.01f)
                        {
                            targetPlacementZ = placementZ.Value;
                        }
                        else
                        {
                            targetPlacementZ = bestMatch.Value.Min.Z;
                        }
                    }
                    else
                    {
                        float modelTopZ = rotMax.Z > 0.01f ? rotMax.Z : MathF.Max(0.01f, rotMax.Z - rotMin.Z);
                        targetPlacementZ = bestMatch.Value.Max.Z - modelTopZ;
                    }

                    undoList.Add((inst.UniqueId, inst.PlacementPosition, inst.PlacementRotation, inst.PlacementScale));
                    inst.PlacementPosition = new Vector3(bestMatch.Value.Centroid.X - rotCenter.X, bestMatch.Value.Centroid.Y - rotCenter.Y, targetPlacementZ);
                    inst.PlacementRotation = new Vector3(0f, 0f, solvedYaw);
                    inst.Transform = WorldPlacementTransform.Build(inst.PlacementPosition, inst.PlacementRotation, inst.PlacementScale);
                    WorldScene.TransformBounds(inst.LocalBoundsMin, inst.LocalBoundsMax, inst.Transform, out inst.BoundsMin, out inst.BoundsMax);
                    WorldScene.TransformBounds(inst.SelectionLocalBoundsMin, inst.SelectionLocalBoundsMax, inst.Transform, out inst.SelectionLocalBoundsMin, out inst.SelectionLocalBoundsMax);
                    list[i] = inst;
                    snappedCount++;
                }
            }
        }

        SnapList(tileMdx, isWmo: false);
        SnapList(tileWmo, isWmo: true);

        if (snappedCount > 0)
        {
            _undoStack.Push(new SnapUndo(tileX, tileY, undoList));
            pm4Host.RebuildInstanceLists();
            string msg = $"✓ Magnetized {snappedCount} placements to PM4 collision geometry on tile ({tileX}, {tileY}).";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
        }
        else
        {
            string msg = $"No placements within snap threshold ({maxSnapDistance:F0} yds) on tile ({tileX}, {tileY}).";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
        }

        return snappedCount;
    }

    /// <summary>
    /// Undoes the last operation (duplication or magnetization snap).
    /// </summary>
    public bool UndoLastOperation()
    {
        if (_undoStack.Count == 0)
            return false;

        return _undoStack.Pop().Execute(_host);
    }

    /// <summary>
    /// Legacy alias for UndoLastOperation.
    /// </summary>
    public bool UndoLastDuplication() => UndoLastOperation();

    /// <summary>
    /// Exports all resident placements to a Zarr v60 dataset archive in the project output folder (T006).
    /// </summary>
    public string SaveMergedPlacementsToZarr(string projectOutputFolder, string mapName)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null)
            throw new InvalidOperationException("No active world scene to export.");

        IPm4OverlayHost pm4Host = worldScene;
        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        string storeName = $"{mapName}_curated_{timestamp}.zarr";
        string storePath = Path.Combine(projectOutputFolder, storeName);
        Directory.CreateDirectory(storePath);

        // 1. Write root zarr.json
        var rootZarrJson = new
        {
            zarr_format = 3,
            node_type = "group",
            attributes = new
            {
                version = "v60",
                map = mapName,
                created_utc = DateTime.UtcNow.ToString("o"),
                generator = "WoWViewer PM4 Curation Workbench"
            }
        };
        File.WriteAllText(Path.Combine(storePath, "zarr.json"), JsonSerializer.Serialize(rootZarrJson, new JsonSerializerOptions { WriteIndented = true }));

        // 2. Export placement records across all tiles
        var tileRecords = new List<object>();
        foreach (var (tileCoord, mdxList) in pm4Host.TileMdxInstances)
        {
            var placements = new List<object>(mdxList.Count);
            for (int i = 0; i < mdxList.Count; i++)
            {
                ObjectInstance inst = mdxList[i];
                placements.Add(new
                {
                    unique_id = inst.UniqueId,
                    model_path = inst.ModelPath,
                    position = new[] { inst.PlacementPosition.X, inst.PlacementPosition.Y, inst.PlacementPosition.Z },
                    rotation = new[] { inst.PlacementRotation.X, inst.PlacementRotation.Y, inst.PlacementRotation.Z },
                    scale = inst.PlacementScale
                });
            }

            tileRecords.Add(new
            {
                tile_x = tileCoord.Item1,
                tile_y = tileCoord.Item2,
                placement_count = placements.Count,
                placements
            });
        }

        string placementsJson = JsonSerializer.Serialize(tileRecords, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(storePath, "curated_placements.json"), placementsJson);

        _host.StatusMessage = $"Saved curated placements to Zarr v60 store: {storeName}";
        return storePath;
    }

    /// <summary>
    /// Patches modified ADT tiles with the current curated placements and saves them to an output folder.
    /// Preserves existing terrain/textures where source ADTs exist via AdtPlacementEditor,
    /// or synthesizes clean monolithic LK ADTs for blank/unauthored development tiles via BlankAdtFactory and LkAdtWriter.
    /// </summary>
    public (int tilesWritten, int placementsAdded, string outDir) SavePatchedAdts(string? targetFolder = null, string? mapName = null)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null)
            throw new InvalidOperationException("No active world scene to export.");

        IPm4OverlayHost pm4Host = worldScene;
        string resolvedMap = !string.IsNullOrEmpty(mapName)
            ? mapName
            : (!string.IsNullOrEmpty(_host.LoadedFileName) ? Path.GetFileNameWithoutExtension(_host.LoadedFileName) : "development");

        string outDir = targetFolder ?? Path.Combine(Directory.GetCurrentDirectory(), "output", "patched_adts", resolvedMap);
        Directory.CreateDirectory(outDir);

        var tilesToPatch = new HashSet<(int, int)>();
        foreach (var key in pm4Host.TileMdxInstances.Keys) tilesToPatch.Add(key);
        foreach (var key in pm4Host.TileWmoInstances.Keys) tilesToPatch.Add(key);

        int totalTilesWritten = 0;
        int totalPlacementsAdded = 0;

        foreach ((int tx, int ty) in tilesToPatch)
        {
            pm4Host.TileMdxInstances.TryGetValue((tx, ty), out var mdxList);
            pm4Host.TileWmoInstances.TryGetValue((tx, ty), out var wmoList);

            int mdxCount = mdxList?.Count ?? 0;
            int wmoCount = wmoList?.Count ?? 0;
            if (mdxCount == 0 && wmoCount == 0) continue;

            string tileStem = $"{resolvedMap}_{tx}_{ty}";
            string outAdtPath = Path.Combine(outDir, $"{tileStem}.adt");
            string virtualAdtPath = $"World\\Maps\\{resolvedMap}\\{tileStem}.adt";

            byte[]? baseBytes = null;
            if (_host.DataSource != null)
            {
                baseBytes = _host.DataSource.ReadFile(virtualAdtPath);
                if (baseBytes == null || baseBytes.Length == 0)
                {
                    string altPath = $"World/Maps/{resolvedMap}/{tileStem}.adt";
                    baseBytes = _host.DataSource.ReadFile(altPath);
                }
            }

            if (baseBytes != null && baseBytes.Length > 0)
            {
                var edits = new List<AdtPlacementEdit>();
                if (mdxList != null)
                {
                    for (int i = 0; i < mdxList.Count; i++)
                    {
                        var inst = mdxList[i];
                        if (inst.UniqueId >= Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase)
                        {
                            edits.Add(new AdtPlacementAddEdit(
                                AdtPlacementKind.Model,
                                inst.ModelPath,
                                inst.PlacementPosition,
                                inst.PlacementRotation,
                                inst.PlacementScale));
                            totalPlacementsAdded++;
                        }
                        else
                        {
                            edits.Add(new AdtPlacementMoveEdit(AdtPlacementKind.Model, -1, inst.UniqueId, inst.PlacementPosition));
                            edits.Add(new AdtPlacementRotateEdit(AdtPlacementKind.Model, -1, inst.UniqueId, inst.PlacementRotation));
                        }
                    }
                }
                if (wmoList != null)
                {
                    for (int i = 0; i < wmoList.Count; i++)
                    {
                        var inst = wmoList[i];
                        if (inst.UniqueId >= Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase)
                        {
                            edits.Add(new AdtPlacementAddEdit(
                                AdtPlacementKind.WorldModel,
                                inst.ModelPath,
                                inst.PlacementPosition,
                                inst.PlacementRotation,
                                inst.PlacementScale));
                            totalPlacementsAdded++;
                        }
                        else
                        {
                            edits.Add(new AdtPlacementMoveEdit(AdtPlacementKind.WorldModel, -1, inst.UniqueId, inst.PlacementPosition));
                            edits.Add(new AdtPlacementRotateEdit(AdtPlacementKind.WorldModel, -1, inst.UniqueId, inst.PlacementRotation));
                        }
                    }
                }

                if (edits.Count > 0)
                {
                    var result = AdtPlacementEditor.Apply(baseBytes, virtualAdtPath, edits);
                    File.WriteAllBytes(outAdtPath, result.Bytes);
                    totalTilesWritten++;
                }
            }
            else
            {
                var modelNames = new List<string>();
                var wmoNames = new List<string>();
                var modelPlacements = new List<AdtModelPlacement>();
                var wmoPlacements = new List<AdtWorldModelPlacement>();

                int nextUid = 1;
                if (mdxList != null)
                {
                    foreach (var inst in mdxList)
                    {
                        int nameIdx = modelNames.IndexOf(inst.ModelPath);
                        if (nameIdx < 0) { nameIdx = modelNames.Count; modelNames.Add(inst.ModelPath); }
                        modelPlacements.Add(new AdtModelPlacement(
                            nameIdx,
                            inst.ModelPath,
                            inst.UniqueId > 0 ? inst.UniqueId : nextUid++,
                            inst.PlacementPosition,
                            inst.PlacementRotation,
                            inst.PlacementScale));
                        totalPlacementsAdded++;
                    }
                }

                if (wmoList != null)
                {
                    foreach (var inst in wmoList)
                    {
                        int nameIdx = wmoNames.IndexOf(inst.ModelPath);
                        if (nameIdx < 0) { nameIdx = wmoNames.Count; wmoNames.Add(inst.ModelPath); }
                        ushort scaleUshort = (ushort)Math.Clamp((int)MathF.Round(inst.PlacementScale * 1024f), 0, 65535);
                        wmoPlacements.Add(new AdtWorldModelPlacement(
                            nameIdx,
                            inst.ModelPath,
                            inst.UniqueId > 0 ? inst.UniqueId : nextUid++,
                            inst.PlacementPosition,
                            inst.PlacementRotation,
                            inst.BoundsMin,
                            inst.BoundsMax,
                            Flags: 0,
                            DoodadSet: 0,
                            NameSet: 0,
                            Scale: scaleUshort));
                        totalPlacementsAdded++;
                    }
                }

                var catalog = new AdtPlacementCatalog(outAdtPath, MapFileKind.Adt, modelNames, wmoNames, modelPlacements, wmoPlacements);
                var blank = BlankAdtFactory.CreateBlank(resolvedMap, tx, ty);
                var populated = BlankAdtFactory.WithPlacements(blank, catalog);
                LkAdtWriter.Write(outAdtPath, populated);
                totalTilesWritten++;
            }
        }

        string msg = $"✓ Saved {totalTilesWritten} patched ADT files to {outDir} ({totalPlacementsAdded} placements).";
        _host.StatusMessage = msg;
        LastOperationMessage = msg;
        return (totalTilesWritten, totalPlacementsAdded, outDir);
    }
}
