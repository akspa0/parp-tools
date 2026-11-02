using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.Runtime.World;
using WoWViewer.Terrain;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

/// <summary>
/// A human-taught pairing between a PM4 collision hull signature and an authentic scene model.
/// </summary>
internal sealed record TaughtPairing(
    uint Ck24,
    byte DominantGroupKey,
    string ModelName,
    string ModelPath,
    string AssetKind,
    float SolvedScale,
    float YawDegrees,
    float Length,
    float Width,
    float Height,
    DateTime TaughtUtc);

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
            string failMsg = "All nearby PM4 doodad collision hulls are already occupied by existing models.";
            _host.StatusMessage = failMsg;
            LastOperationMessage = failMsg;
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

        bool isWmo = string.Equals(inst.AssetKind, "wmo", StringComparison.OrdinalIgnoreCase);
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

        var surfaceVerts = new List<Vector3>(pm4Obj.Triangles.Count * 3);
        for (int t = 0; t < pm4Obj.Triangles.Count; t++)
        {
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].A, pm4Obj.BaseTransform));
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].B, pm4Obj.BaseTransform));
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].C, pm4Obj.BaseTransform));
        }

        worldScene.Pm4Overlay._pm4TilePositionRefs.TryGetValue((pm4TileX, pm4TileY), out var mprlList);
        float? placementZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : null;
        WorldScene.TransformBounds(pm4Obj.BoundsMin, pm4Obj.BoundsMax, pm4Obj.BaseTransform, out Vector3 worldMin, out _);

        Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, worldMin.Z);

        Vector3 localMin = prev.SelectionBoundsResolved && prev.SelectionLocalBoundsMax != Vector3.Zero
            ? prev.SelectionLocalBoundsMin : prev.LocalBoundsMin;
        Vector3 localMax = prev.SelectionBoundsResolved && prev.SelectionLocalBoundsMax != Vector3.Zero
            ? prev.SelectionLocalBoundsMax : prev.LocalBoundsMax;

        var cand = Pm4ModelCandidate.Create(prev.ModelPath, prev.AssetKind ?? "M2", localMin, localMax);
        var matchResults = Pm4DirectModelMatcher.MatchPrism(prism, [cand], 1);

        float dist;
        if (matchResults.Count > 0)
        {
            var match = matchResults[0];
            dist = Vector3.Distance(prev.PlacementPosition, match.SolvedPosition);
            prev.PlacementPosition = match.SolvedPosition;
            prev.PlacementRotation = match.SolvedRotation;
            if (fitScale) prev.PlacementScale = match.SolvedScale;
        }
        else
        {
            Matrix4x4 unpositioned = WorldPlacementTransform.Build(Vector3.Zero, prev.PlacementRotation, prev.PlacementScale);
            WorldScene.TransformBounds(localMin, localMax, unpositioned, out Vector3 rotMin, out Vector3 rotMax);
            Vector3 rotCenter = (rotMin + rotMax) * 0.5f;
            float modelTopZ = rotMax.Z > 0.01f ? rotMax.Z : MathF.Max(0.01f, rotMax.Z - rotMin.Z);
            float targetPlacementZ = isWmo ? prism.BaseZ - rotMin.Z : prism.TopZ - modelTopZ;
            prev.PlacementPosition = new Vector3(prism.Centroid.X - rotCenter.X, prism.Centroid.Y - rotCenter.Y, targetPlacementZ);
            prev.PlacementRotation = new Vector3(prev.PlacementRotation.X, prev.PlacementRotation.Y, prism.YawDegrees);
            dist = Vector3.Distance(inst.PlacementPosition, prev.PlacementPosition);
        }

        prev.Transform = WorldPlacementTransform.Build(prev.PlacementPosition, prev.PlacementRotation, prev.PlacementScale);
        WorldScene.TransformBounds(prev.LocalBoundsMin, prev.LocalBoundsMax, prev.Transform, out prev.BoundsMin, out prev.BoundsMax);
        WorldScene.TransformBounds(prev.SelectionLocalBoundsMin, prev.SelectionLocalBoundsMax, prev.Transform, out prev.SelectionLocalBoundsMin, out prev.SelectionLocalBoundsMax);
        targetList[idx] = prev;

        pm4Host.RebuildInstanceLists();
        statusMessage = $"✓ Aligned {prev.ModelName} (#{prev.UniqueId}) to PM4 hull 0x{pm4Obj.Ck24:X6} (delta: {dist:F1} yds, yaw: {prev.PlacementRotation.Z:F1}°).";
        _host.StatusMessage = statusMessage;
        LastOperationMessage = statusMessage;
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
        var surfaceVerts = new List<Vector3>(pm4Obj.Triangles.Count * 3);
        for (int t = 0; t < pm4Obj.Triangles.Count; t++)
        {
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].A, pm4Obj.BaseTransform));
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].B, pm4Obj.BaseTransform));
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].C, pm4Obj.BaseTransform));
        }

        List<Vector3>? mprlList = null;
        if (worldScene != null)
            worldScene.Pm4Overlay._pm4TilePositionRefs.TryGetValue((pm4TileX, pm4TileY), out mprlList);
        float? placementZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : null;
        WorldScene.TransformBounds(pm4Obj.BoundsMin, pm4Obj.BoundsMax, pm4Obj.BaseTransform, out Vector3 worldMin, out _);
        Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, worldMin.Z);

        Vector3 localMin = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
            ? inst.SelectionLocalBoundsMin : inst.LocalBoundsMin;
        Vector3 localMax = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
            ? inst.SelectionLocalBoundsMax : inst.LocalBoundsMax;
        var cand = Pm4ModelCandidate.Create(inst.ModelPath, inst.AssetKind ?? "M2", localMin, localMax);
        var matchResults = Pm4DirectModelMatcher.MatchPrism(prism, [cand], 1);
        float scale = matchResults.Count > 0 ? matchResults[0].SolvedScale : inst.PlacementScale;

        var taught = new TaughtPairing(
            pm4Obj.Ck24,
            pm4Obj.DominantGroupKey,
            inst.ModelName,
            inst.ModelPath,
            inst.AssetKind ?? "M2",
            scale,
            prism.YawDegrees,
            prism.UnrotatedLength,
            prism.UnrotatedWidth,
            prism.Height,
            DateTime.UtcNow);

        _taughtPairings.RemoveAll(t => t.Ck24 != 0 && t.Ck24 == taught.Ck24);
        _taughtPairings.Add(taught);

        try
        {
            string outDir = Path.Combine(Directory.GetCurrentDirectory(), "output", "curation");
            Directory.CreateDirectory(outDir);
            string json = JsonSerializer.Serialize(_taughtPairings, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(Path.Combine(outDir, "pm4_taught_pairings.json"), json);
        }
        catch { }

        string msg = $"✓ Saved teaching rule: PM4 0x{pm4Obj.Ck24:X6} ({pm4Obj.DominantGroupKey:X2}) -> {inst.ModelName} (scale {scale:F2}, yaw {prism.YawDegrees:F1}°).";
        _host.StatusMessage = msg;
        LastOperationMessage = msg;
        return taught;
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

        if (!pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var tileMdx))
        {
            tileMdx = new List<ObjectInstance>();
            pm4Host.TileMdxInstances[(tileX, tileY)] = tileMdx;
        }

        int maxExistingUniqueId = 0;
        var existingPositions = new List<Vector3>(tileMdx.Count);
        for (int i = 0; i < tileMdx.Count; i++)
        {
            existingPositions.Add(tileMdx[i].PlacementPosition);
            if (tileMdx[i].UniqueId > maxExistingUniqueId) maxExistingUniqueId = tileMdx[i].UniqueId;
        }

        int nextId = Math.Max(Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase, maxExistingUniqueId + 1);
        var addedIds = new List<int>();

        for (int i = 0; i < pm4Objects.Count; i++)
        {
            var obj = pm4Objects[i];
            bool matches = (taught.Ck24 != 0 && obj.Ck24 == taught.Ck24)
                || (taught.Ck24 == 0 && obj.Ck24 == 0 && obj.DominantGroupKey == taught.DominantGroupKey);
            if (!matches) continue;

            Vector3 centroid = Vector3.Transform(obj.Center, obj.BaseTransform);
            bool occupied = false;
            for (int e = 0; e < existingPositions.Count; e++)
            {
                if (Vector3.DistanceSquared(centroid, existingPositions[e]) < 8.0f * 8.0f)
                {
                    occupied = true;
                    break;
                }
            }
            if (occupied) continue;

            var surfaceVerts = new List<Vector3>(obj.Triangles.Count * 3);
            for (int t = 0; t < obj.Triangles.Count; t++)
            {
                surfaceVerts.Add(Vector3.Transform(obj.Triangles[t].A, obj.BaseTransform));
                surfaceVerts.Add(Vector3.Transform(obj.Triangles[t].B, obj.BaseTransform));
                surfaceVerts.Add(Vector3.Transform(obj.Triangles[t].C, obj.BaseTransform));
            }
            worldScene.Pm4Overlay._pm4TilePositionRefs.TryGetValue((tileX, tileY), out var mprlList);
            float? placementZ = obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(obj.Ck24 << 8) : null;
            WorldScene.TransformBounds(obj.BoundsMin, obj.BoundsMax, obj.BaseTransform, out Vector3 worldMin, out _);
            Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, worldMin.Z);

            var cand = Pm4ModelCandidate.Create(taught.ModelPath, taught.AssetKind, -new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f, new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f);
            var matchResults = Pm4DirectModelMatcher.MatchPrism(prism, [cand], 1);
            if (matchResults.Count == 0) continue;

            var match = matchResults[0];
            Matrix4x4 transform = WorldPlacementTransform.Build(match.SolvedPosition, match.SolvedRotation, match.SolvedScale);
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
                PlacementPosition = match.SolvedPosition,
                PlacementRotation = match.SolvedRotation,
                PlacementScale = match.SolvedScale,
                UniqueId = assignedId,
                PlacementEntryIndex = tileMdx.Count,
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true
            };
            tileMdx.Add(newInst);
            existingPositions.Add(match.SolvedPosition);
            addedIds.Add(assignedId);
        }

        if (addedIds.Count > 0)
        {
            _undoStack.Push(new DuplicationUndo(tileX, tileY, addedIds));
            pm4Host.RebuildInstanceLists();
            string msg = $"✓ Populated {addedIds.Count} PM4 hulls matching 0x{taught.Ck24:X6} with {taught.ModelName} on tile ({tileX}, {tileY}).";
            _host.StatusMessage = msg;
            LastOperationMessage = msg;
            return addedIds.Count;
        }

        string failMsg = $"No unoccupied PM4 hulls matching 0x{taught.Ck24:X6} on tile ({tileX}, {tileY}).";
        _host.StatusMessage = failMsg;
        LastOperationMessage = failMsg;
        return 0;
    }

    /// <summary>
    /// Instantiates a candidate match directly into an unplaced PM4 hull.
    /// </summary>
    public bool SpawnCandidateIntoPm4Hull(
        Pm4OverlayObject pm4Obj,
        int pm4TileX,
        int pm4TileY,
        Pm4DirectMatchResult match,
        out string statusMessage)
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null)
        {
            statusMessage = "No active world scene.";
            return false;
        }

        IPm4OverlayHost pm4Host = worldScene;
        if (!pm4Host.TileMdxInstances.TryGetValue((pm4TileX, pm4TileY), out var tileMdx))
        {
            tileMdx = new List<ObjectInstance>();
            pm4Host.TileMdxInstances[(pm4TileX, pm4TileY)] = tileMdx;
        }

        int maxId = 0;
        for (int i = 0; i < tileMdx.Count; i++)
            if (tileMdx[i].UniqueId > maxId) maxId = tileMdx[i].UniqueId;

        int newId = Math.Max(Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase, maxId + 1);
        Matrix4x4 transform = WorldPlacementTransform.Build(match.SolvedPosition, match.SolvedRotation, match.SolvedScale);
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
            PlacementPosition = match.SolvedPosition,
            PlacementRotation = match.SolvedRotation,
            PlacementScale = match.SolvedScale,
            UniqueId = newId,
            PlacementEntryIndex = tileMdx.Count,
            TileX = pm4TileX,
            TileY = pm4TileY,
            HasTileCoordinate = true
        };

        tileMdx.Add(newInst);
        _undoStack.Push(new DuplicationUndo(pm4TileX, pm4TileY, [newId]));
        pm4Host.RebuildInstanceLists();

        statusMessage = $"✓ Spawned {match.Candidate.ModelName} (#{newId}) into PM4 hull 0x{pm4Obj.Ck24:X6} (Score: {match.Score:P0}, Scale: {match.SolvedScale:F2}, Yaw: {match.SolvedRotation.Z:F1}°).";
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
                    Matrix4x4 unpositioned = WorldPlacementTransform.Build(Vector3.Zero, inst.PlacementRotation, inst.PlacementScale);
                    Vector3 localMin = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
                        ? inst.SelectionLocalBoundsMin
                        : inst.LocalBoundsMin;
                    Vector3 localMax = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
                        ? inst.SelectionLocalBoundsMax
                        : inst.LocalBoundsMax;
                    WorldScene.TransformBounds(localMin, localMax, unpositioned, out Vector3 rotMin, out Vector3 rotMax);
                    Vector3 rotCenter = (rotMin + rotMax) * 0.5f;

                    float targetPlacementZ;
                    if (isWmo)
                    {
                        targetPlacementZ = bestMatch.Value.Min.Z - rotMin.Z;
                    }
                    else
                    {
                        float modelTopZ = rotMax.Z > 0.01f ? rotMax.Z : MathF.Max(0.01f, rotMax.Z - rotMin.Z);
                        targetPlacementZ = bestMatch.Value.Max.Z - modelTopZ;
                    }

                    undoList.Add((inst.UniqueId, inst.PlacementPosition, inst.PlacementRotation, inst.PlacementScale));
                    inst.PlacementPosition = new Vector3(bestMatch.Value.Centroid.X - rotCenter.X, bestMatch.Value.Centroid.Y - rotCenter.Y, targetPlacementZ);
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
}
