using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text.Json;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.Runtime.World;
using WoWViewer.Terrain;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

/// <summary>
/// Service providing live PM4 doodad collision matching, duplication, and undo in the viewer (Spec 287 Phase 1).
/// Allows selecting a model (e.g. DRAGONBLIGHT_CLIFF_01.M2) and magnetically populating nearby unmatched
/// PM4 collision hulls, with curation saving to Zarr v60 datasets.
/// </summary>
internal sealed class Pm4DoodadDuplicationService
{
    private readonly IViewerAppHost _host;
    private readonly Stack<IPm4UndoOperation> _undoStack = new();

    private interface IPm4UndoOperation
    {
        bool Execute(IViewerAppHost host);
    }

    private sealed class DuplicationUndo(int tileX, int tileY, List<int> addedIds) : IPm4UndoOperation
    {
        public bool Execute(IViewerAppHost host)
        {
            var worldScene = host.WorldScene;
            if (worldScene == null)
                return false;

            IPm4OverlayHost pm4Host = worldScene;
            if (pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out List<ObjectInstance>? tileMdx))
            {
                var idSet = new HashSet<int>(addedIds);
                int removed = tileMdx.RemoveAll(inst => idSet.Contains(inst.UniqueId));
                pm4Host.RebuildInstanceLists();
                host.StatusMessage = $"Undid duplication of {removed} instances on tile ({tileX}, {tileY}).";
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
            if (pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out List<ObjectInstance>? tileMdx))
            {
                var stateMap = new Dictionary<int, (Vector3 Position, Vector3 Rotation, float Scale)>();
                foreach (var s in previousStates)
                    stateMap[s.UniqueId] = (s.Position, s.Rotation, s.Scale);

                int restoredCount = 0;
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

                pm4Host.RebuildInstanceLists();
                host.StatusMessage = $"Restored {restoredCount} placements to pre-magnetized state on tile ({tileX}, {tileY}).";
                return true;
            }

            return false;
        }
    }

    public Pm4DoodadDuplicationService(IViewerAppHost host)
    {
        _host = host;
    }

    public bool CanUndo => _undoStack.Count > 0;

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

            candidateTargets.Add(new Pm4DoodadTarget(
                targetTileX,
                targetTileY,
                obj.Ck24,
                obj.ObjectPartId,
                worldCentroid,
                obj.BoundsMin,
                obj.BoundsMax,
                obj.BaseRotationRadians,
                obj.DominantGroupKey));
        }

        if (candidateTargets.Count == 0)
        {
            _host.StatusMessage = $"No PM4 doodad collision hulls found within {searchRadius:F0} yds of selection.";
            return 0;
        }

        // Get existing M2 placements on this tile to reject occupied targets
        if (!pm4Host.TileMdxInstances.TryGetValue((targetTileX, targetTileY), out List<ObjectInstance>? tileMdx))
        {
            tileMdx = new List<ObjectInstance>();
            pm4Host.TileMdxInstances[(targetTileX, targetTileY)] = tileMdx;
        }

        var existingPositions = new List<Vector3>(tileMdx.Count);
        int maxExistingUniqueId = 0;
        for (int i = 0; i < tileMdx.Count; i++)
        {
            existingPositions.Add(tileMdx[i].PlacementPosition);
            if (tileMdx[i].UniqueId > maxExistingUniqueId)
                maxExistingUniqueId = tileMdx[i].UniqueId;
        }

        var templateInfo = new ModelTemplateInfo(
            template.ModelPath,
            template.ModelKey,
            template.ModelName,
            template.PlacementPosition,
            template.PlacementRotation,
            template.PlacementScale,
            template.LocalBoundsMin,
            template.LocalBoundsMax,
            template.BoundsMin,
            template.BoundsMax);

        int startId = Math.Max(Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase, maxExistingUniqueId + 1);
        List<DuplicatedDoodadResult> duplicated = Pm4DoodadDuplicationEngine.DuplicateAcrossTargets(
            templateInfo,
            candidateTargets,
            existingPositions,
            startId,
            occupancyRadius: 8.0f);

        if (duplicated.Count == 0)
        {
            _host.StatusMessage = "All nearby PM4 doodad collision hulls are already occupied by existing models.";
            return 0;
        }

        var addedIds = new List<int>(duplicated.Count);
        for (int i = 0; i < duplicated.Count; i++)
        {
            DuplicatedDoodadResult d = duplicated[i];
            Matrix4x4 transform = WorldPlacementTransform.Build(d.Position, d.RotationDegrees, d.Scale);
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
                PlacementPosition = d.Position,
                PlacementRotation = d.RotationDegrees,
                PlacementScale = d.Scale,
                UniqueId = d.UniqueId,
                PlacementEntryIndex = tileMdx.Count,
                TileX = targetTileX,
                TileY = targetTileY,
                HasTileCoordinate = true
            };

            tileMdx.Add(newInst);
            addedIds.Add(d.UniqueId);
        }

        _undoStack.Push(new DuplicationUndo(targetTileX, targetTileY, addedIds));
        pm4Host.RebuildInstanceLists();
        _host.StatusMessage = $"Duplicated {duplicated.Count} instances of {template.ModelName} into PM4 doodad collision hulls.";
        return duplicated.Count;
    }

    /// <summary>
    /// Magnetically snaps a single placed M2 object to its nearest PM4 navmesh collision hull.
    /// </summary>
    public bool MagnetizePlacementToPm4(ObjectInstance inst, float maxSnapDistance = 35.0f)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null)
            return false;

        IPm4OverlayHost pm4Host = worldScene;
        int tileX = inst.TileX;
        int tileY = inst.TileY;

        if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileX, tileY), out var pm4Objects) || pm4Objects.Count == 0)
        {
            if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileY, tileX), out pm4Objects) || pm4Objects.Count == 0)
            {
                _host.StatusMessage = $"No PM4 objects loaded on tile ({tileX}, {tileY}) to magnetize to.";
                return false;
            }
        }

        Pm4OverlayObject? bestMatch = null;
        float bestDist = maxSnapDistance;
        Vector3 bestCentroid = Vector3.Zero;
        float bestZ = 0f;

        for (int i = 0; i < pm4Objects.Count; i++)
        {
            var obj = pm4Objects[i];
            Vector3 centroid = Vector3.Transform(obj.Center, obj.BaseTransform);
            float dist = Vector3.Distance(centroid, inst.PlacementPosition);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestMatch = obj;
                bestCentroid = centroid;
                bestZ = obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(obj.Ck24 << 8) : centroid.Z;
            }
        }

        if (bestMatch == null)
        {
            _host.StatusMessage = $"No PM4 collision hull found within {maxSnapDistance:F0} yds of {inst.ModelName}.";
            return false;
        }

        if (Math.Abs(bestZ) < 0.001f || float.IsNaN(bestZ) || float.IsInfinity(bestZ))
            bestZ = bestCentroid.Z;

        if (!pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx))
            return false;

        int idx = tileMdx.FindIndex(m => m.UniqueId == inst.UniqueId);
        if (idx < 0)
            return false;

        var prev = tileMdx[idx];
        var undoList = new List<(int, Vector3, Vector3, float)>
        {
            (prev.UniqueId, prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale)
        };
        _undoStack.Push(new SnapUndo(tileX, tileY, undoList));

        Matrix4x4 unpositioned = WorldPlacementTransform.Build(Vector3.Zero, prev.PlacementRotation, prev.PlacementScale);
        Vector3 localMin = prev.SelectionBoundsResolved && prev.SelectionLocalBoundsMax != Vector3.Zero
            ? prev.SelectionLocalBoundsMin
            : prev.LocalBoundsMin;
        Vector3 localMax = prev.SelectionBoundsResolved && prev.SelectionLocalBoundsMax != Vector3.Zero
            ? prev.SelectionLocalBoundsMax
            : prev.LocalBoundsMax;
        WorldScene.TransformBounds(localMin, localMax, unpositioned, out Vector3 rotMin, out Vector3 rotMax);

        float modelTopZ = rotMax.Z > 0.01f ? rotMax.Z : MathF.Max(0.01f, rotMax.Z - rotMin.Z);
        float targetPlacementZ = bestCentroid.Z - modelTopZ;
        Vector3 rotCenter = (rotMin + rotMax) * 0.5f;

        prev.PlacementPosition = new Vector3(bestCentroid.X - rotCenter.X, bestCentroid.Y - rotCenter.Y, targetPlacementZ);
        prev.Transform = WorldPlacementTransform.Build(prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale);
        WorldScene.TransformBounds(prev.LocalBoundsMin, prev.LocalBoundsMax, prev.Transform, out prev.BoundsMin, out prev.BoundsMax);
        WorldScene.TransformBounds(prev.SelectionLocalBoundsMin, prev.SelectionLocalBoundsMax, prev.Transform, out prev.SelectionLocalBoundsMin, out prev.SelectionLocalBoundsMax);
        tileMdx[idx] = prev;

        pm4Host.RebuildInstanceLists();
        _host.StatusMessage = $"Magnetized {inst.ModelName} (#{inst.UniqueId}) to PM4 template (delta: {bestDist:F2} yds).";
        return true;
    }

    /// <summary>
    /// Magnetically snaps all resident ADT placements on a tile to their nearest PM4 collision hulls.
    /// </summary>
    public int MagnetizeTilePlacementsToPm4(int tileX, int tileY, float maxSnapDistance = 25.0f)
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

        if (!pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx) || tileMdx.Count == 0)
        {
            _host.StatusMessage = $"No M2 placements resident on tile ({tileX}, {tileY}).";
            return 0;
        }

        var undoList = new List<(int, Vector3, Vector3, float)>();
        int snappedCount = 0;

        for (int i = 0; i < tileMdx.Count; i++)
        {
            var inst = tileMdx[i];
            Pm4OverlayObject? bestMatch = null;
            float bestDist = maxSnapDistance;
            Vector3 bestCentroid = Vector3.Zero;
            float bestZ = 0f;

            for (int p = 0; p < pm4Objects.Count; p++)
            {
                var obj = pm4Objects[p];
                Vector3 centroid = Vector3.Transform(obj.Center, obj.BaseTransform);
                float dist = Vector3.Distance(centroid, inst.PlacementPosition);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestMatch = obj;
                    bestCentroid = centroid;
                    bestZ = obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(obj.Ck24 << 8) : centroid.Z;
                }
            }

            if (bestMatch != null && bestDist > 0.05f)
            {
                if (Math.Abs(bestZ) < 0.001f || float.IsNaN(bestZ) || float.IsInfinity(bestZ))
                    bestZ = bestCentroid.Z;

                Matrix4x4 unpositioned = WorldPlacementTransform.Build(Vector3.Zero, inst.PlacementRotation, inst.PlacementScale);
                Vector3 localMin = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
                    ? inst.SelectionLocalBoundsMin
                    : inst.LocalBoundsMin;
                Vector3 localMax = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
                    ? inst.SelectionLocalBoundsMax
                    : inst.LocalBoundsMax;
                WorldScene.TransformBounds(localMin, localMax, unpositioned, out Vector3 rotMin, out Vector3 rotMax);

                float modelTopZ = rotMax.Z > 0.01f ? rotMax.Z : MathF.Max(0.01f, rotMax.Z - rotMin.Z);
                float targetPlacementZ = bestCentroid.Z - modelTopZ;
                Vector3 rotCenter = (rotMin + rotMax) * 0.5f;

                undoList.Add((inst.UniqueId, inst.PlacementPosition, inst.PlacementRotation, inst.PlacementScale));
                inst.PlacementPosition = new Vector3(bestCentroid.X - rotCenter.X, bestCentroid.Y - rotCenter.Y, targetPlacementZ);
                inst.Transform = WorldPlacementTransform.Build(inst.PlacementPosition, inst.PlacementRotation, inst.PlacementScale);
                WorldScene.TransformBounds(inst.LocalBoundsMin, inst.LocalBoundsMax, inst.Transform, out inst.BoundsMin, out inst.BoundsMax);
                WorldScene.TransformBounds(inst.SelectionLocalBoundsMin, inst.SelectionLocalBoundsMax, inst.Transform, out inst.SelectionLocalBoundsMin, out inst.SelectionLocalBoundsMax);
                tileMdx[i] = inst;
                snappedCount++;
            }
        }

        if (snappedCount > 0)
        {
            _undoStack.Push(new SnapUndo(tileX, tileY, undoList));
            pm4Host.RebuildInstanceLists();
            _host.StatusMessage = $"Magnetized {snappedCount} placements to PM4 collision geometry on tile ({tileX}, {tileY}).";
        }
        else
        {
            _host.StatusMessage = $"No placements within snap threshold ({maxSnapDistance:F0} yds) on tile ({tileX}, {tileY}).";
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
}
