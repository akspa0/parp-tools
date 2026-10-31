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
    private readonly Stack<DuplicationUndoEntry> _undoStack = new();

    public record struct DuplicationUndoEntry(int TileX, int TileY, List<int> AddedUniqueIds);

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

        _undoStack.Push(new DuplicationUndoEntry(targetTileX, targetTileY, addedIds));
        pm4Host.RebuildInstanceLists();
        _host.StatusMessage = $"Duplicated {duplicated.Count} instances of {template.ModelName} into PM4 doodad collision hulls.";
        return duplicated.Count;
    }

    /// <summary>
    /// Undoes the last duplication operation, removing newly added instances and rebuilding scene instance lists.
    /// </summary>
    public bool UndoLastDuplication()
    {
        if (_undoStack.Count == 0)
            return false;

        var entry = _undoStack.Pop();
        var worldScene = _host.WorldScene;
        if (worldScene == null)
            return false;

        IPm4OverlayHost pm4Host = worldScene;
        if (pm4Host.TileMdxInstances.TryGetValue((entry.TileX, entry.TileY), out List<ObjectInstance>? tileMdx))
        {
            var idSet = new HashSet<int>(entry.AddedUniqueIds);
            int removed = tileMdx.RemoveAll(inst => idSet.Contains(inst.UniqueId));
            pm4Host.RebuildInstanceLists();
            _host.StatusMessage = $"Undid duplication of {removed} instances on tile ({entry.TileX}, {entry.TileY}).";
            return true;
        }

        return false;
    }

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
