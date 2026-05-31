using System;
using System.Collections.Generic;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Terrain;

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

        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey = null;
        Vector3? clickedWorldPoint = null;
        TerrainRenderer? terrainRenderer = _terrainManager?.Renderer;
        if (terrainRenderer != null
            && TryRaycastTerrain(terrainRenderer, rayOrigin, rayDir, GetSceneFarPlane(), out TerrainRenderer.TerrainChunkInfo terrainHit, out Vector3 terrainHitPoint))
        {
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

        if (_worldScene.TryPickPm4ObjectByRay(rayOrigin, rayDir, out var pm4HitKey, out _, out float pm4HitDistance) && pm4HitKey.HasValue)
            AddPm4ClickSelectionCandidate(addedKeys, pm4HitKey.Value, pm4HitDistance, "Ray hit");

        if (_worldScene.TryPickSceneObjectsByRay(rayOrigin, rayDir, _sceneClickSelectionHits, clickedChunkKey, clickedWorldPoint))
        {
            int sceneHitCount = Math.Min(_sceneClickSelectionHits.Count, MaxSceneClickSelectionHits);
            _clickSelectionSceneHitOverflowCount = Math.Max(0, _sceneClickSelectionHits.Count - sceneHitCount);

            for (int i = 0; i < sceneHitCount; i++)
                AddSceneObjectClickSelectionCandidate(addedKeys, _sceneClickSelectionHits[i]);
        }

        if (_clickSelectionCandidates.Count == 0)
            return false;

        if (_clickSelectionCandidates.Count == 1)
        {
            Action apply = _clickSelectionCandidates[0].Apply;
            ClearPendingClickSelection();
            apply();
            return true;
        }

        _clickSelectionOverlayPosition = new Vector2(mouseX + 18f, mouseY + 18f);
        _statusMessage = $"Ambiguous click: {_clickSelectionCandidates.Count} candidates under the cursor.";
        return true;
    }

    private void DrawClickSelectionOverlay()
    {
        if (_clickSelectionCandidates.Count <= 1)
            return;

        if (ImGui.IsKeyPressed(ImGuiKey.Escape))
        {
            ClearPendingClickSelection();
            return;
        }

        Vector2 displaySize = ImGui.GetIO().DisplaySize;
        Vector2 overlayPos = new(
            MathF.Min(_clickSelectionOverlayPosition.X, MathF.Max(8f, displaySize.X - 430f)),
            MathF.Min(_clickSelectionOverlayPosition.Y, MathF.Max(8f, displaySize.Y - 420f)));

        ImGui.SetNextWindowPos(overlayPos, ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(16f, 14f));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, 2f);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowRounding, 4f);
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0.04f, 0.05f, 0.09f, 0.985f));
        ImGui.PushStyleColor(ImGuiCol.Border, new Vector4(0.95f, 0.79f, 0.28f, 0.98f));
        ImGui.PushStyleColor(ImGuiCol.Separator, new Vector4(0.88f, 0.73f, 0.22f, 0.82f));

        ImGuiWindowFlags flags = ImGuiWindowFlags.NoDecoration
            | ImGuiWindowFlags.AlwaysAutoResize
            | ImGuiWindowFlags.NoDocking
            | ImGuiWindowFlags.NoSavedSettings
            | ImGuiWindowFlags.NoMove;

        Action? pendingAction = null;
        bool shouldClose = false;

        if (!ImGui.Begin("##ClickSelectionOverlay", flags))
        {
            ImGui.End();
            ImGui.PopStyleColor(3);
            ImGui.PopStyleVar(3);
            return;
        }

        ImGui.SetWindowFontScale(1.16f);
        ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.38f, 1.0f), "Choose Target");
        ImGui.SetWindowFontScale(1.0f);
        ImGui.TextColored(new Vector4(0.82f, 0.86f, 0.94f, 1.0f), "Multiple objects overlap under this click.");
        ImGui.TextDisabled("Pick the exact thing you meant to select.");
        ImGui.Separator();

        for (int i = 0; i < _clickSelectionCandidates.Count; i++)
        {
            ClickSelectionCandidate candidate = _clickSelectionCandidates[i];
            if (ImGui.Selectable($"{candidate.Title}##click_pick_{i}", false, ImGuiSelectableFlags.None, new Vector2(360f, 0f)))
            {
                pendingAction = candidate.Apply;
                break;
            }

            if (!string.IsNullOrWhiteSpace(candidate.Detail))
                ImGui.TextColored(new Vector4(0.92f, 0.94f, 0.98f, 1.0f), candidate.Detail);

            if (!string.IsNullOrWhiteSpace(candidate.SecondaryDetail))
            {
                ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + 360f);
                ImGui.TextColored(new Vector4(0.54f, 0.84f, 0.52f, 1.0f), candidate.SecondaryDetail);
                ImGui.PopTextWrapPos();
            }

            if (candidate.Distance.HasValue)
                ImGui.TextDisabled($"Hit distance: {candidate.Distance.Value:F1}");

            if (i + 1 < _clickSelectionCandidates.Count)
                ImGui.Separator();
        }

        if (_clickSelectionSceneHitOverflowCount > 0)
        {
            ImGui.Separator();
            ImGui.TextDisabled($"+{_clickSelectionSceneHitOverflowCount} farther scene hits omitted from this list.");
        }

        ImGui.Spacing();
        if (ImGui.Button("Cancel"))
            shouldClose = true;

        ImGui.SameLine();
        ImGui.TextDisabled("Esc closes this chooser.");

        ImGui.End();
        ImGui.PopStyleColor(3);
        ImGui.PopStyleVar(3);

        if (pendingAction != null)
        {
            ClearPendingClickSelection();
            pendingAction();
            return;
        }

        if (shouldClose)
            ClearPendingClickSelection();
    }

    private void ClearPendingClickSelection()
    {
        _clickSelectionCandidates.Clear();
        _clickSelectionSceneHitOverflowCount = 0;
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
                $"Tile: ({objectKey.tileX}, {objectKey.tileY})",
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
        string dedupKey = $"scene:{hit.ObjectType}:{hit.ObjectIndex}";
        string detail = $"UniqueId: {hit.UniqueId}  Pos: ({hit.PlacementPosition.X:F1}, {hit.PlacementPosition.Y:F1}, {hit.PlacementPosition.Z:F1})";

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
                    if (_worldScene == null || !_worldScene.SelectSceneObject(hit.ObjectType, hit.ObjectIndex))
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
