using System;
using System.Collections.Generic;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Terrain;
using WoWViewer.Rendering;

namespace WoWViewer;

public partial class ViewerApp
{
    private const int MaxSceneClickSelectionHits = 10;

    private readonly List<ClickSelectionCandidate> _clickSelectionCandidates = new();
    private readonly List<SceneObjectPickHit> _sceneClickSelectionHits = new();
    private Vector2 _clickSelectionOverlayPosition;
    private int _clickSelectionSceneHitOverflowCount;

    private sealed record ClickSelectionCandidate(
        string DedupKey,
        string Title,
        string Detail,
        string SecondaryDetail,
        float? Distance,
        Action Apply);

    private bool TryHandleSceneClickSelection(
        float mouseX,
        float mouseY,
        float localX,
        float localY,
        float viewportWidth,
        float viewportHeight,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 rayOrigin,
        Vector3 rayDir)
    {
        if (_worldScene == null)
            return false;

        // If cluster selector is already open, close it so new clicks can select other objects or clear
        if (_sceneClusterSelector3D != null && _sceneClusterSelector3D.IsActive)
        {
            _sceneClusterSelector3D.Close();
            ClearPendingClickSelection();
        }

        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey = null;
        TerrainRenderer.TerrainChunkInfo? clickedTerrainChunk = null;
        Vector3? clickedWorldPoint = null;
        TerrainRenderer? terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (terrainRenderer != null
            && TryRaycastTerrain(terrainRenderer, rayOrigin, rayDir, GetSceneFarPlane(), out TerrainRenderer.TerrainChunkInfo terrainHit, out Vector3 terrainHitPoint))
        {
            clickedTerrainChunk = terrainHit;
            clickedChunkKey = (terrainHit.TileX, terrainHit.TileY, terrainHit.ChunkX, terrainHit.ChunkY);
            clickedWorldPoint = terrainHitPoint;
        }

        _clickSelectionCandidates.Clear();
        _clickSelectionSceneHitOverflowCount = 0;

        var addedKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (TryPickTaxiNodeByRay(rayOrigin, rayDir, out int taxiNodeIdByRay, out _))
            AddTaxiNodeClickSelectionCandidate(addedKeys, taxiNodeIdByRay, "Ray hit");

        if (TryPickTaxiRouteByRay(rayOrigin, rayDir, out int taxiRouteIdByRay, out _))
            AddTaxiRouteClickSelectionCandidate(addedKeys, taxiRouteIdByRay, "Ray hit");

        if (TryPickTaxiNodeAtMouse(localX, localY, viewportWidth, viewportHeight, view, proj, out int taxiNodeId))
            AddTaxiNodeClickSelectionCandidate(addedKeys, taxiNodeId, "Screen-space pick");

        if (TryPickTaxiRouteAtMouse(localX, localY, viewportWidth, viewportHeight, view, proj, out int taxiRouteId))
            AddTaxiRouteClickSelectionCandidate(addedKeys, taxiRouteId, "Screen-space pick");

        if (TryPickAreaPoiAtMouse(localX, localY, viewportWidth, viewportHeight, view, proj, out int areaPoiId))
            AddAreaPoiClickSelectionCandidate(addedKeys, areaPoiId);

        HoveredAssetInfo? hoveredInfo = _worldScene.HoveredAssetInfo;
        if (hoveredInfo.HasValue
            && string.Equals(hoveredInfo.Value.AssetKind, "WL liquid", StringComparison.OrdinalIgnoreCase)
            && TryResolveHoveredWlLiquidBody(hoveredInfo.Value, out WlLiquidBody? hoveredWlBody)
            && hoveredWlBody != null)
        {
            AddWlLiquidClickSelectionCandidate(addedKeys, hoveredWlBody);
        }

        var hoveredPm4Key = _worldScene.ShowPm4Overlay ? _worldScene.HoveredAssetInfo?.Pm4ObjectKey : null;
        if (hoveredPm4Key.HasValue)
            AddPm4ClickSelectionCandidate(addedKeys, hoveredPm4Key.Value, null, "Hovered PM4 object");

        bool pm4Hit = _worldScene.TryPickPm4ObjectByRay(rayOrigin, rayDir, out var pm4HitKey, out _, out float pm4HitDistance) && pm4HitKey.HasValue;
        if (pm4Hit)
            AddPm4ClickSelectionCandidate(addedKeys, pm4HitKey.Value, pm4HitDistance, "Ray hit");

        // If PM4 overlay is on and we hit a PM4 object, skip scene object picking
        // (PM4 objects are behind scene WMO/M2 visually, so the ray hits both)
        if (!pm4Hit || !_worldScene.ShowPm4Overlay)
        {
            if (_worldScene.TryPickSceneObjectsByRay(rayOrigin, rayDir, _sceneClickSelectionHits, clickedChunkKey, clickedWorldPoint))
            {
                // 1. Cull any hits that are behind the clicked terrain point (occluded by terrain)
                float? terrainDist = clickedWorldPoint.HasValue ? (clickedWorldPoint.Value - rayOrigin).Length() : null;
                var validHits = new List<SceneObjectPickHit>();
                for (int i = 0; i < _sceneClickSelectionHits.Count; i++)
                {
                    var hit = _sceneClickSelectionHits[i];
                    if (terrainDist.HasValue && hit.Distance > terrainDist.Value + 1.5f)
                        continue;
                    validHits.Add(hit);
                }

                // 2. WMO Container Fall-Through:
                // When clicking within the confines of a WMO's bounding box, clicks must fall through
                // to interior MDX/M2 objects, WMO doodads, or nested smaller WMOs.
                // If an interior object is hit along the ray inside a WMO's bounding box,
                // the enclosing WMO must not occlude or capture the click.
                var containerWmoIndices = new HashSet<int>();
                foreach (var hit in validHits)
                {
                    if (hit.ObjectType != ObjectType.Wmo)
                        continue;

                    bool hasInteriorHit = validHits.Any(other =>
                    {
                        if (other.ObjectType == ObjectType.Wmo && other.ObjectIndex == hit.ObjectIndex)
                            return false;

                        // Test if other object's selection point is within this WMO's bounding box (with slight margin)
                        return other.SelectionPoint.X >= hit.BoundsMin.X - 0.5f && other.SelectionPoint.X <= hit.BoundsMax.X + 0.5f
                            && other.SelectionPoint.Y >= hit.BoundsMin.Y - 0.5f && other.SelectionPoint.Y <= hit.BoundsMax.Y + 0.5f
                            && other.SelectionPoint.Z >= hit.BoundsMin.Z - 0.5f && other.SelectionPoint.Z <= hit.BoundsMax.Z + 0.5f;
                    });

                    if (hasInteriorHit)
                        containerWmoIndices.Add(hit.ObjectIndex);
                }

                if (containerWmoIndices.Count > 0)
                {
                    var nonContainerHits = validHits.Where(h => h.ObjectType != ObjectType.Wmo || !containerWmoIndices.Contains(h.ObjectIndex)).ToList();
                    if (nonContainerHits.Count > 0)
                        validHits = nonContainerHits;
                }

                if (validHits.Count > 0)
                {
                    // Sort candidate hits strictly by ray distance to camera
                    validHits.Sort(static (a, b) => a.Distance.CompareTo(b.Distance));
                    float minHitDist = validHits[0].Distance;

                    // Spatial proximity threshold: only cluster objects if literally on top of each other (<= 2.0 yards)
                    const float clusterThreshold = 2.0f;
                    var clusteredHits = new List<SceneObjectPickHit>();
                    for (int i = 0; i < validHits.Count; i++)
                    {
                        if (validHits[i].Distance <= minHitDist + clusterThreshold)
                            clusteredHits.Add(validHits[i]);
                    }

                    int sceneHitCount = Math.Min(clusteredHits.Count, MaxSceneClickSelectionHits);
                    _clickSelectionSceneHitOverflowCount = Math.Max(0, clusteredHits.Count - sceneHitCount);

                    for (int i = 0; i < sceneHitCount; i++)
                        AddSceneObjectClickSelectionCandidate(addedKeys, clusteredHits[i]);
                }
            }
        }

        // Global cluster proximity filter:
        // If multiple candidates were added, cull any candidate that is far behind the closest candidate
        if (_clickSelectionCandidates.Count > 1)
        {
            float minDistance = float.MaxValue;
            foreach (var c in _clickSelectionCandidates)
            {
                if (c.Distance.HasValue && c.Distance.Value < minDistance)
                    minDistance = c.Distance.Value;
            }

            if (minDistance < float.MaxValue)
            {
                const float globalClusterThreshold = 2.0f;
                _clickSelectionCandidates.RemoveAll(c => c.Distance.HasValue && c.Distance.Value > minDistance + globalClusterThreshold);
            }
        }

        if (_clickSelectionCandidates.Count == 0)
        {
            // A terrain click with no competing selectable target pins the MCNK for the Inspector.
            // Object/taxi/PM4/POI candidates return above and retain their existing selection path.
            if (clickedTerrainChunk is TerrainRenderer.TerrainChunkInfo terrainChunk)
            {
                SelectTerrainChunkFromClick(terrainChunk);
                ClearPendingClickSelection();
                return true;
            }

            return false;
        }

        if (_clickSelectionCandidates.Count == 1)
        {
            Action apply = _clickSelectionCandidates[0].Apply;
            ClearPendingClickSelection();
            apply();
            return true;
        }

        Vector3 clusterCenter = clickedWorldPoint ?? (rayOrigin + rayDir * 10f);
        if (_sceneClusterSelector3D != null)
        {
            var clusterItems = new List<ClusterItem>();
            foreach (var c in _clickSelectionCandidates)
            {
                clusterItems.Add(new ClusterItem(
                    c.DedupKey,
                    c.Title,
                    c.Detail,
                    c.SecondaryDetail,
                    clusterCenter,
                    c.Distance,
                    c.Apply));
            }
            _sceneClusterSelector3D.Open(clusterCenter, clusterItems);
        }

        _clickSelectionOverlayPosition = new Vector2(mouseX + 18f, mouseY + 18f);
        _statusMessage = $"Ambiguous click: {_clickSelectionCandidates.Count} candidates under the cursor.";
        return true;
    }

    private void DrawClickSelectionOverlay()
    {
        if (_sceneClusterSelector3D != null && _sceneClusterSelector3D.IsActive)
        {
            if (_sceneClusterSelector3D.HandleInput())
            {
                ClearPendingClickSelection();
                return;
            }

            if (TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            {
                var view = _camera.GetViewMatrix();
                float aspect = vpW / Math.Max(vpH, 1f);
                var proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovDegrees * MathF.PI / 180f, aspect, 0.1f, GetSceneFarPlane());
                _sceneClusterSelector3D.RenderScreenOverlay(view, proj, vpX, vpY, vpW, vpH);
            }

            if (!_sceneClusterSelector3D.IsActive)
            {
                ClearPendingClickSelection();
            }
        }
        else if (_clickSelectionCandidates.Count > 0)
        {
            ClearPendingClickSelection();
        }
    }

    private void ClearPendingClickSelection()
    {
        _clickSelectionCandidates.Clear();
        _clickSelectionSceneHitOverflowCount = 0;
        _sceneClusterSelector3D?.Close();
    }

    private void AddTaxiNodeClickSelectionCandidate(HashSet<string> addedKeys, int nodeId, string source)
    {
        if (_worldScene?.TaxiLoader == null)
            return;

        var node = _worldScene.GetTaxiNode(nodeId);
        if (node == null)
            return;

        TryAddClickSelectionCandidate(
            addedKeys,
            new ClickSelectionCandidate(
                $"taxi-node:{nodeId}",
                $"Taxi Node [{node.Id}] {node.Name}",
                source,
                string.IsNullOrWhiteSpace(node.MountModelPath) ? "No resolved mount model." : node.MountModelPath,
                null,
                () =>
                {
                    ClearSelectedWlLiquidBody(clearListIsolation: true);
                    SelectTaxiNode(nodeId, toggle: true);
                }));
    }

    private void AddTaxiRouteClickSelectionCandidate(HashSet<string> addedKeys, int routeId, string source)
    {
        if (_worldScene?.TaxiLoader == null)
            return;

        var route = _worldScene.GetTaxiRoute(routeId);
        if (route == null)
            return;

        string fromName = _worldScene.GetTaxiNode(route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
        string toName = _worldScene.GetTaxiNode(route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";

        TryAddClickSelectionCandidate(
            addedKeys,
            new ClickSelectionCandidate(
                $"taxi-route:{routeId}",
                $"Taxi Route [{route.PathId}]",
                $"{fromName} -> {toName}",
                source,
                null,
                () =>
                {
                    ClearSelectedWlLiquidBody(clearListIsolation: true);
                    SelectTaxiRoute(routeId, toggle: false);
                }));
    }

    private void AddAreaPoiClickSelectionCandidate(HashSet<string> addedKeys, int poiId)
    {
        if (_worldScene?.PoiLoader == null)
            return;

        var poi = _worldScene.PoiLoader.Entries.Find(entry => entry.Id == poiId);
        if (poi == null)
            return;

        TryAddClickSelectionCandidate(
            addedKeys,
            new ClickSelectionCandidate(
                $"area-poi:{poiId}",
                $"Area POI [{poi.Id}] {poi.Name}",
                $"WoW: ({poi.WoWPosition.X:F1}, {poi.WoWPosition.Y:F1}, {poi.WoWPosition.Z:F1})",
                $"Icon: {poi.Icon}  Importance: {poi.Importance}",
                null,
                () =>
                {
                    ClearSelectedWlLiquidBody(clearListIsolation: true);
                    SelectAreaPoi(poiId, toggle: true);
                }));
    }

    private void AddWlLiquidClickSelectionCandidate(HashSet<string> addedKeys, WlLiquidBody body)
    {
        TryAddClickSelectionCandidate(
            addedKeys,
            new ClickSelectionCandidate(
                $"wl:{body.BodyKey}",
                $"WL liquid {body.Name}",
                $"Blocks: {body.BlockCount}  Verts: {body.Vertices.Length}",
                body.SourcePath,
                null,
                () =>
                {
                    if (!TryFindWlLiquidBodyByKey(body.BodyKey, out WlLiquidBody? selectedBody) || selectedBody == null)
                        return;

                    _worldScene?.ClearSelection();
                    _worldScene?.ClearTaxiSelection();
                    _worldScene?.ClearPm4ObjectSelection();
                    ClearSelectedAreaPoiInfo();
                    SetSelectedWlLiquidBody(
                        selectedBody,
                        isolateInList: true,
                        focusInspectWorkspace: true,
                        statusMessage: $"WL inspect: selected '{selectedBody.Name}' and isolated it in the inspect list.");
                }));
    }

    private void AddPm4ClickSelectionCandidate(
        HashSet<string> addedKeys,
        (int tileX, int tileY, uint ck24, int objectPart) objectKey,
        float? distance,
        string source)
    {
        TryAddClickSelectionCandidate(
            addedKeys,
            new ClickSelectionCandidate(
                $"pm4:{objectKey.tileX}:{objectKey.tileY}:{objectKey.ck24}:{objectKey.objectPart}",
                $"PM4 0x{objectKey.ck24:X6} part {objectKey.objectPart}",
                $"Tile: ({objectKey.tileY}, {objectKey.tileX})",
                source,
                distance,
                () =>
                {
                    if (_worldScene == null || !_worldScene.SelectPm4Object(objectKey))
                        return;

                    ClearSelectedWlLiquidBody(clearListIsolation: true);
                    _worldScene.ClearTaxiSelection();
                    _worldScene.ClearSelection();
                    ClearSelectedAreaPoiInfo();
                    UpdateSelectedPm4ObjectInfo(objectKey);
                }));
    }

    private void AddSceneObjectClickSelectionCandidate(HashSet<string> addedKeys, SceneObjectPickHit hit)
    {
        string dedupKey = hit.ObjectType == ObjectType.WmoDoodad
            ? $"scene:{hit.ObjectType}:{hit.ParentWmoIndex}:{hit.ObjectIndex}"
            : $"scene:{hit.ObjectType}:{hit.ObjectIndex}";
        // WMO doodads have no uniqueId to report (MODD carries none), so the position is the whole
        // identity here. Printing "UniqueId: 0" would read as a real id.
        string position = $"Pos: ({hit.PlacementPosition.X:F1}, {hit.PlacementPosition.Y:F1}, {hit.PlacementPosition.Z:F1})";
        string detail = hit.ObjectType == ObjectType.WmoDoodad
            ? $"in WMO [{hit.ParentWmoIndex}]  {position}"
            : $"UniqueId: {hit.UniqueId}  {position}";

        TryAddClickSelectionCandidate(
            addedKeys,
            new ClickSelectionCandidate(
                dedupKey,
                $"{hit.KindLabel} {hit.ModelName}",
                detail,
                hit.ModelPath,
                hit.Distance,
                () =>
                {
                    if (_worldScene == null)
                        return;

                    // Toggle off / deselect if already selected
                    if (_worldScene.SelectedObjectType == hit.ObjectType && _worldScene.SelectedObjectIndex == hit.ObjectIndex)
                    {
                        _worldScene.ClearSelection();
                        _selectedObjectIndex = -1;
                        _selectedObjectType = "";
                        _selectedObjectInfo = "";
                        return;
                    }

                    if (!_worldScene.SelectSceneObject(hit.ObjectType, hit.ObjectIndex, hit.ParentWmoIndex))
                        return;

                    ClearSelectedWlLiquidBody(clearListIsolation: true);
                    _worldScene.ClearTaxiSelection();
                    _worldScene.ClearPm4ObjectSelection();
                    ClearSelectedAreaPoiInfo();
                    RefreshSelectedWorldObjectInfo();
                }));
    }

    private void TryAddClickSelectionCandidate(HashSet<string> addedKeys, ClickSelectionCandidate candidate)
    {
        if (!addedKeys.Add(candidate.DedupKey))
            return;

        _clickSelectionCandidates.Add(candidate);
    }
}
