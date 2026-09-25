using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Taxi node/route and area-POI selection: mouse/ray picking, selection info, focus, and taxi-actor model overrides.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class TaxiAndAreaPoiSelectionService
{
    private readonly IViewerAppHost _host;

    internal TaxiAndAreaPoiSelectionService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private ref string? _lastVirtualPath => ref _host.LastVirtualPath;
    private Dictionary<string, Dictionary<int, string>> _savedTaxiActorModelOverridesByMap => _host.SavedTaxiActorModelOverridesByMap;
    private ref int _selectedAreaPoiId => ref _host.SelectedAreaPoiId;
    private ref int _selectedObjectIndex => ref _host.SelectedObjectIndex;
    private ref string _selectedObjectInfo => ref _host.SelectedObjectInfo;
    private ref string _selectedObjectType => ref _host.SelectedObjectType;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref string _taxiActorModelOverrideInput => ref _host.TaxiActorModelOverrideInput;
    private ref int _taxiActorModelOverrideInputRouteId => ref _host.TaxiActorModelOverrideInputRouteId;
    private ref int _taxiActorModelOverrideTargetRouteId => ref _host.TaxiActorModelOverrideTargetRouteId;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private string? GetCurrentSessionMapName() => _host.GetCurrentSessionMapName();
    private void SaveViewerSettings() => _host.SaveViewerSettings();
    private bool TryGetSelectedBrowserModelPath(out string assetPath) => _host.TryGetSelectedBrowserModelPath(out assetPath);

    private const float TaxiNodePickHalfWidth = 42f;
    private const float TaxiNodePickBottomPadding = 18f;
    private const float TaxiNodePickTopPadding = 96f;
    private const float TaxiRouteHandlePickHalfWidth = 40f;
    private const float TaxiRouteHandlePickBottomPadding = 20f;
    private const float TaxiRouteHandlePickTopPadding = 72f;
    private const float TaxiRouteSegmentPickHalfWidth = 28f;

    internal void ApplyTaxiActorModelOverride(int routeId, string? modelPath)
    {
        if (_worldScene == null || routeId < 0)
            return;

        string? currentMapName = GetCurrentSessionMapName();
        if (!string.IsNullOrWhiteSpace(currentMapName))
        {
            if (!_savedTaxiActorModelOverridesByMap.TryGetValue(currentMapName, out Dictionary<int, string>? overridesByRoute))
            {
                overridesByRoute = new Dictionary<int, string>();
                _savedTaxiActorModelOverridesByMap[currentMapName] = overridesByRoute;
            }

            if (string.IsNullOrWhiteSpace(modelPath))
            {
                overridesByRoute.Remove(routeId);
                if (overridesByRoute.Count == 0)
                    _savedTaxiActorModelOverridesByMap.Remove(currentMapName);
            }
            else
            {
                overridesByRoute[routeId] = modelPath.Trim().Replace('/', '\\');
            }
        }

        _worldScene.SetTaxiActorModelOverride(routeId, modelPath);
        SaveViewerSettings();
    }

    private void ApplySavedTaxiActorModelOverridesForCurrentMap()
    {
        if (_worldScene == null)
            return;

        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return;

        if (!_savedTaxiActorModelOverridesByMap.TryGetValue(currentMapName, out Dictionary<int, string>? overridesByRoute))
            return;

        foreach ((int routeId, string modelPath) in overridesByRoute)
            _worldScene.SetTaxiActorModelOverride(routeId, modelPath);
    }

    internal bool TryApplySelectedBrowserAssetToTaxiOverride()
    {
        if (!TryGetTaxiActorOverrideRouteId(out int routeId))
        {
            _statusMessage = "Select a taxi node or route first.";
            return false;
        }

        if (!TryGetSelectedBrowserModelPath(out string assetPath))
        {
            _statusMessage = "Select an .mdx, .mdl, or .m2 asset in the file browser first.";
            return false;
        }

        _taxiActorModelOverrideTargetRouteId = routeId;
        _taxiActorModelOverrideInput = assetPath.Replace('/', '\\');
        _taxiActorModelOverrideInputRouteId = routeId;
        ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
        RefreshSelectedTaxiInfo();
        _statusMessage = $"Applied taxi actor override from browser asset to route {routeId}.";
        return true;
    }

    internal void SelectTaxiNode(int nodeId, bool toggle)
    {
        if (_worldScene?.TaxiLoader == null)
            return;

        int nextNodeId = toggle && _worldScene.SelectedTaxiNodeId == nodeId ? -1 : nodeId;
        _worldScene.SelectedTaxiNodeId = nextNodeId;
        _worldScene.ClearSelection();
        _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
        ClearSelectedAreaPoiInfo();

        if (nextNodeId < 0)
        {
            ClearSelectedTaxiInfo();
            return;
        }

        RefreshSelectedTaxiInfo();
    }

    internal void SelectTaxiRoute(int pathId, bool toggle)
    {
        if (_worldScene?.TaxiLoader == null)
            return;

        int nextRouteId = toggle && _worldScene.SelectedTaxiRouteId == pathId ? -1 : pathId;
        _worldScene.SelectedTaxiRouteId = nextRouteId;
        _worldScene.ClearSelection();
        _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
        ClearSelectedAreaPoiInfo();

        if (nextRouteId < 0)
        {
            ClearSelectedTaxiInfo();
            return;
        }

        RefreshSelectedTaxiInfo();
    }

    internal void RefreshSelectedTaxiInfo()
    {
        if (_worldScene?.TaxiLoader == null)
            return;

        _selectedObjectIndex = -1;

        if (_worldScene.SelectedTaxiNodeId >= 0)
        {
            var node = _worldScene.GetTaxiNode(_worldScene.SelectedTaxiNodeId);
            if (node == null)
            {
                ClearSelectedTaxiInfo();
                return;
            }

            int routeCount = _worldScene.TaxiLoader.Routes.Count(route => route.FromNodeId == node.Id || route.ToNodeId == node.Id);
            string mountCreatureIds = node.MountCreatureIds.Length > 0
                ? string.Join(", ", node.MountCreatureIds.Where(id => id > 0))
                : "none";

            _selectedObjectType = "Taxi Node";
            _selectedObjectInfo =
                $"Taxi Node [{node.Id}] {node.Name}\n" +
                $"Position: ({node.Position.X:F1}, {node.Position.Y:F1}, {node.Position.Z:F1})\n" +
                $"Routes: {routeCount}\n" +
                $"Mount Creature IDs: {mountCreatureIds}\n" +
                $"Resolved Mount Creature: {node.MountCreatureId}\n" +
                $"Resolved Display ID: {node.MountDisplayId}\n" +
                $"Resolved Model: {node.MountModelPath ?? "not found"}";
            return;
        }

        if (_worldScene.SelectedTaxiRouteId >= 0)
        {
            var route = _worldScene.GetTaxiRoute(_worldScene.SelectedTaxiRouteId);
            if (route == null)
            {
                ClearSelectedTaxiInfo();
                return;
            }

            var fromNode = _worldScene.GetTaxiNode(route.FromNodeId);
            var toNode = _worldScene.GetTaxiNode(route.ToNodeId);
            TaxiPathLoader.TaxiNode? mountNode = fromNode;
            if (mountNode == null || string.IsNullOrWhiteSpace(mountNode.MountModelPath))
                mountNode = toNode;

            string fromName = fromNode?.Name ?? $"#{route.FromNodeId}";
            string toName = toNode?.Name ?? $"#{route.ToNodeId}";
            string? actorOverridePath = _worldScene.GetTaxiActorModelOverride(route.PathId);
            string resolvedActorModelPath = _worldScene.GetResolvedTaxiActorModelPath(route.PathId) ?? "not found";

            _selectedObjectType = "Taxi Route";
            _selectedObjectInfo =
                $"Taxi Route [{route.PathId}]\n" +
                $"From: {fromName}\n" +
                $"To: {toName}\n" +
                $"Cost: {route.Cost}\n" +
                $"Waypoints: {route.Waypoints.Count}\n" +
                $"Actor Override: {actorOverridePath ?? "auto"}\n" +
                $"Resolved Actor Model: {resolvedActorModelPath}";
            return;
        }

        ClearSelectedTaxiInfo();
    }

    internal void SelectAreaPoi(int poiId, bool toggle)
    {
        if (_worldScene?.PoiLoader == null)
            return;

        int nextPoiId = toggle && _selectedAreaPoiId == poiId ? -1 : poiId;
        _selectedAreaPoiId = nextPoiId;
        _worldScene.ClearSelection();
        _worldScene.ClearTaxiSelection();
        _worldScene.Pm4Overlay.ClearPm4ObjectSelection();

        if (nextPoiId < 0)
        {
            ClearSelectedAreaPoiInfo();
            return;
        }

        RefreshSelectedAreaPoiInfo();
    }

    private void RefreshSelectedAreaPoiInfo()
    {
        if (_worldScene?.PoiLoader == null || _selectedAreaPoiId < 0)
        {
            ClearSelectedAreaPoiInfo();
            return;
        }

        AreaPoiLoader.AreaPoiEntry? poi = _worldScene.PoiLoader.Entries
            .FirstOrDefault(entry => entry.Id == _selectedAreaPoiId);
        if (poi == null)
        {
            ClearSelectedAreaPoiInfo();
            return;
        }

        _selectedObjectIndex = -1;
        _selectedObjectType = "Area POI";
        _selectedObjectInfo =
            $"Area POI [{poi.Id}] {poi.Name}\n" +
            $"Position: ({poi.Position.X:F1}, {poi.Position.Y:F1}, {poi.Position.Z:F1})\n" +
            $"WoW Position: ({poi.WoWPosition.X:F1}, {poi.WoWPosition.Y:F1}, {poi.WoWPosition.Z:F1})\n" +
            $"Icon: {poi.Icon}\n" +
            $"Importance: {poi.Importance}\n" +
            $"Flags: 0x{poi.Flags:X}\n" +
            $"Continent ID: {poi.ContinentId}";
    }

    internal void ClearSelectedTaxiInfo()
    {
        if (!_selectedObjectType.StartsWith("Taxi", StringComparison.OrdinalIgnoreCase))
            return;

        _selectedObjectIndex = -1;
        _selectedObjectType = "";
        _selectedObjectInfo = "";
        _taxiActorModelOverrideInput = "";
        _taxiActorModelOverrideInputRouteId = -1;
        _taxiActorModelOverrideTargetRouteId = -1;
    }

    internal void ClearSelectedAreaPoiInfo()
    {
        _selectedAreaPoiId = -1;
        if (!string.Equals(_selectedObjectType, "Area POI", StringComparison.OrdinalIgnoreCase))
            return;

        _selectedObjectIndex = -1;
        _selectedObjectType = "";
        _selectedObjectInfo = "";
    }

    internal bool TryPickTaxiNodeAtMouse(float localX, float localY, float viewportWidth, float viewportHeight, Matrix4x4 view, Matrix4x4 proj, out int nodeId)
    {
        nodeId = -1;
        if (_worldScene?.TaxiLoader == null || !_worldScene.ShowTaxi)
            return false;

        const float pickRadiusPixels = 48f;
        float bestDistanceSq = pickRadiusPixels * pickRadiusPixels;

        foreach (var node in _worldScene.TaxiLoader.Nodes)
        {
            if (!_worldScene.IsTaxiNodeVisible(node))
                continue;

            if (!SceneViewportMath.TryProjectWorldToViewport(node.Position + new Vector3(0f, 0f, 50f), view, proj, viewportWidth, viewportHeight, out Vector2 projected))
                continue;

            float dx = projected.X - localX;
            float dy = projected.Y - localY;
            float distSq = dx * dx + dy * dy;
            if (distSq > bestDistanceSq)
                continue;

            bestDistanceSq = distSq;
            nodeId = node.Id;
        }

        return nodeId >= 0;
    }

    internal bool TryPickTaxiRouteAtMouse(float localX, float localY, float viewportWidth, float viewportHeight, Matrix4x4 view, Matrix4x4 proj, out int pathId)
    {
        pathId = -1;
        if (_worldScene?.TaxiLoader == null || !_worldScene.ShowTaxi)
            return false;

        Vector2 pointer = new(localX, localY);

        const float handlePickRadiusPixels = 72f;
        float bestHandleDistSq = handlePickRadiusPixels * handlePickRadiusPixels;

        foreach (var route in _worldScene.TaxiLoader.Routes)
        {
            if (!_worldScene.IsTaxiRouteVisible(route))
                continue;

            if (!_worldScene.TryGetTaxiRouteSelectionPoint(route.PathId, out Vector3 selectionPoint))
                continue;

            if (!SceneViewportMath.TryProjectWorldToViewport(selectionPoint + new Vector3(0f, 0f, 30f), view, proj, viewportWidth, viewportHeight, out Vector2 projected))
                continue;

            float distSq = Vector2.DistanceSquared(projected, pointer);
            if (distSq > bestHandleDistSq)
                continue;

            bestHandleDistSq = distSq;
            pathId = route.PathId;
        }

        if (pathId >= 0)
            return true;

        const float linePickRadiusPixels = 56f;
        float bestLineDistSq = linePickRadiusPixels * linePickRadiusPixels;

        foreach (var route in _worldScene.TaxiLoader.Routes)
        {
            if (!_worldScene.IsTaxiRouteVisible(route) || route.Waypoints.Count < 2)
                continue;

            for (int i = 0; i < route.Waypoints.Count - 1; i++)
            {
                if (!SceneViewportMath.TryProjectWorldToViewport(route.Waypoints[i], view, proj, viewportWidth, viewportHeight, out Vector2 a)
                    || !SceneViewportMath.TryProjectWorldToViewport(route.Waypoints[i + 1], view, proj, viewportWidth, viewportHeight, out Vector2 b))
                {
                    continue;
                }

                float distSq = SceneViewportMath.DistanceSquaredPointToSegment(pointer, a, b);
                if (distSq > bestLineDistSq)
                    continue;

                bestLineDistSq = distSq;
                pathId = route.PathId;
            }
        }

        return pathId >= 0;
    }

    internal bool TryPickTaxiNodeByRay(Vector3 rayOrigin, Vector3 rayDir, out int nodeId, out float hitDistance)
    {
        nodeId = -1;
        hitDistance = float.MaxValue;
        if (_worldScene?.TaxiLoader == null || !_worldScene.ShowTaxi)
            return false;

        foreach (TaxiPathLoader.TaxiNode node in _worldScene.TaxiLoader.Nodes)
        {
            if (!_worldScene.IsTaxiNodeVisible(node))
                continue;

            float localDistance = SceneViewportMath.RayAabbIntersect(
                rayOrigin,
                rayDir,
                node.Position - new Vector3(TaxiNodePickHalfWidth, TaxiNodePickHalfWidth, TaxiNodePickBottomPadding),
                node.Position + new Vector3(TaxiNodePickHalfWidth, TaxiNodePickHalfWidth, TaxiNodePickTopPadding));
            if (localDistance < 0f || localDistance >= hitDistance)
                continue;

            hitDistance = localDistance;
            nodeId = node.Id;
        }

        return nodeId >= 0;
    }

    internal bool TryPickTaxiRouteByRay(Vector3 rayOrigin, Vector3 rayDir, out int pathId, out float hitDistance)
    {
        pathId = -1;
        hitDistance = float.MaxValue;
        if (_worldScene?.TaxiLoader == null || !_worldScene.ShowTaxi)
            return false;

        foreach (TaxiPathLoader.TaxiRoute route in _worldScene.TaxiLoader.Routes)
        {
            if (!_worldScene.IsTaxiRouteVisible(route))
                continue;

            if (_worldScene.TryGetTaxiRouteSelectionPoint(route.PathId, out Vector3 selectionPoint))
            {
                float handleDistance = SceneViewportMath.RayAabbIntersect(
                    rayOrigin,
                    rayDir,
                    selectionPoint - new Vector3(TaxiRouteHandlePickHalfWidth, TaxiRouteHandlePickHalfWidth, TaxiRouteHandlePickBottomPadding),
                    selectionPoint + new Vector3(TaxiRouteHandlePickHalfWidth, TaxiRouteHandlePickHalfWidth, TaxiRouteHandlePickTopPadding));
                if (handleDistance >= 0f && handleDistance < hitDistance)
                {
                    hitDistance = handleDistance;
                    pathId = route.PathId;
                }
            }

            if (route.Waypoints.Count < 2)
                continue;

            for (int index = 0; index < route.Waypoints.Count - 1; index++)
            {
                Vector3 segmentMin = Vector3.Min(route.Waypoints[index], route.Waypoints[index + 1])
                    - new Vector3(TaxiRouteSegmentPickHalfWidth, TaxiRouteSegmentPickHalfWidth, TaxiRouteSegmentPickHalfWidth);
                Vector3 segmentMax = Vector3.Max(route.Waypoints[index], route.Waypoints[index + 1])
                    + new Vector3(TaxiRouteSegmentPickHalfWidth, TaxiRouteSegmentPickHalfWidth, TaxiRouteSegmentPickHalfWidth);
                float segmentDistance = SceneViewportMath.RayAabbIntersect(rayOrigin, rayDir, segmentMin, segmentMax);
                if (segmentDistance < 0f || segmentDistance >= hitDistance)
                    continue;

                hitDistance = segmentDistance;
                pathId = route.PathId;
            }
        }

        return pathId >= 0;
    }

    internal bool TryPickAreaPoiAtMouse(float localX, float localY, float viewportWidth, float viewportHeight, Matrix4x4 view, Matrix4x4 proj, out int poiId)
    {
        poiId = -1;
        if (_worldScene?.PoiLoader == null || !_worldScene.ShowPoi)
            return false;

        const float pickRadiusPixels = 36f;
        float bestDistanceSq = pickRadiusPixels * pickRadiusPixels;
        Vector2 pointer = new(localX, localY);

        foreach (AreaPoiLoader.AreaPoiEntry poi in _worldScene.PoiLoader.Entries)
        {
            if (!SceneViewportMath.TryProjectWorldToViewport(poi.Position + new Vector3(0f, 0f, 56f), view, proj, viewportWidth, viewportHeight, out Vector2 projected))
                continue;

            float distSq = Vector2.DistanceSquared(projected, pointer);
            if (distSq > bestDistanceSq)
                continue;

            bestDistanceSq = distSq;
            poiId = poi.Id;
        }

        return poiId >= 0;
    }

    internal void FocusSelectedTaxi()
    {
        if (_worldScene == null)
            return;

        if (_worldScene.SelectedTaxiRouteId >= 0)
        {
            int routeId = _worldScene.SelectedTaxiRouteId;
            if (_worldScene.TryGetTaxiRouteSelectionPoint(routeId, out Vector3 routePoint))
            {
                _camera.Position = routePoint + new Vector3(0f, 0f, 100f);
                _camera.Pitch = -30f;
                _statusMessage = $"Focused taxi route {routeId}.";
            }
            return;
        }

        if (_worldScene.SelectedTaxiNodeId >= 0)
        {
            TaxiPathLoader.TaxiNode? node = _worldScene.GetTaxiNode(_worldScene.SelectedTaxiNodeId);
            if (node != null)
            {
                _camera.Position = node.Position + new Vector3(0f, 0f, 50f);
                _camera.Pitch = -30f;
                _statusMessage = $"Focused taxi node {node.Id}.";
            }
        }
    }

    internal IReadOnlyList<TaxiPathLoader.TaxiRoute> GetTaxiActorOverrideCandidateRoutes()
    {
        if (_worldScene?.TaxiLoader == null)
            return Array.Empty<TaxiPathLoader.TaxiRoute>();

        if (_worldScene.SelectedTaxiRouteId >= 0)
        {
            TaxiPathLoader.TaxiRoute? selectedRoute = _worldScene.GetTaxiRoute(_worldScene.SelectedTaxiRouteId);
            return selectedRoute != null
                ? new[] { selectedRoute }
                : Array.Empty<TaxiPathLoader.TaxiRoute>();
        }

        if (_worldScene.SelectedTaxiNodeId >= 0)
        {
            int nodeId = _worldScene.SelectedTaxiNodeId;
            return _worldScene.TaxiLoader.Routes
                .Where(route => route.FromNodeId == nodeId || route.ToNodeId == nodeId)
                .OrderBy(route => route.PathId)
                .ToList();
        }

        return Array.Empty<TaxiPathLoader.TaxiRoute>();
    }

    internal bool TryGetTaxiActorOverrideRouteId(out int routeId)
    {
        routeId = -1;
        IReadOnlyList<TaxiPathLoader.TaxiRoute> candidateRoutes = GetTaxiActorOverrideCandidateRoutes();
        if (candidateRoutes.Count == 0)
        {
            _taxiActorModelOverrideTargetRouteId = -1;
            return false;
        }

        int preferredRouteId = _worldScene?.SelectedTaxiRouteId >= 0
            ? _worldScene.SelectedTaxiRouteId
            : _taxiActorModelOverrideTargetRouteId;

        TaxiPathLoader.TaxiRoute? activeRoute = candidateRoutes.FirstOrDefault(route => route.PathId == preferredRouteId)
            ?? candidateRoutes[0];

        _taxiActorModelOverrideTargetRouteId = activeRoute.PathId;
        routeId = activeRoute.PathId;
        return true;
    }

    internal string GetTaxiRouteDisplayLabel(int pathId)
    {
        if (_worldScene == null)
            return $"Route #{pathId}";

        TaxiPathLoader.TaxiRoute? route = _worldScene.GetTaxiRoute(pathId);
        if (route == null)
            return $"Route #{pathId}";

        string fromName = _worldScene.GetTaxiNode(route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
        string toName = _worldScene.GetTaxiNode(route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";
        return $"[{route.PathId}] {fromName} -> {toName}";
    }

    internal void SyncTaxiActorModelOverrideInput(int routeId)
    {
        if (_worldScene == null || routeId < 0)
        {
            _taxiActorModelOverrideInputRouteId = -1;
            _taxiActorModelOverrideInput = "";
            return;
        }

        if (_taxiActorModelOverrideInputRouteId == routeId)
            return;

        _taxiActorModelOverrideInputRouteId = routeId;
        _taxiActorModelOverrideInput = _worldScene.GetTaxiActorModelOverride(routeId)
            ?? _worldScene.GetResolvedTaxiActorModelPath(routeId)
            ?? "";
    }

    internal bool TryGetLoadedTaxiActorModelPath(out string modelPath)
    {
        modelPath = string.Empty;

        string? candidatePath = _lastVirtualPath;
        if (string.IsNullOrWhiteSpace(candidatePath) || !IsTaxiActorModelPath(candidatePath))
            return false;

        modelPath = candidatePath.Replace('/', '\\');
        return true;
    }

    internal static bool IsTaxiActorModelPath(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".m2", StringComparison.OrdinalIgnoreCase);
    }
}
