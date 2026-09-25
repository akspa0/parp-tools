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
/// Area context: resolves the camera's current area (AreaTable), the area overlay and its on-screen labels, plus area-lookup diagnostics.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class AreaContextService
{
    private readonly IViewerAppHost _host;

    internal AreaContextService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref AreaTableService? _areaTableService => ref _host.AreaTableService;
    private ref Camera _camera => ref _host.Camera;
    private ref WowViewer.Core.World.AreaLookupResult? _currentAreaLookup => ref _host.CurrentAreaLookup;
    private ref string _currentAreaName => ref _host.CurrentAreaName;
    private ref int _currentMapId => ref _host.CurrentMapId;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private HashSet<string> _reportedAreaDiagnostics => _host.ReportedAreaDiagnostics;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref WorldScene? _worldScene => ref _host.WorldScene;

    private string _currentZoneName = "";
    private Vector3 _lastAreaLookupCameraPosition = new(float.NaN);
    private int _areaLookupTick;
    private int _lastAreaLookupLoadedTileCount = -1;
    private int _lastAreaLookupMapId = int.MinValue;
    private TerrainRenderer? _areaOverlayRenderer;
    private AreaTableService? _areaOverlayAreaTableService;
    private int _areaOverlayRevision = int.MinValue;
    private int _areaOverlayMapId = int.MinValue;

    private void ReportAreaLookupDiagnostic(int areaId)
    {
        if (_areaTableService == null)
            return;

        string diagnostic = _areaTableService.DescribeLookup(areaId, _currentMapId);
        if (_reportedAreaDiagnostics.Add(diagnostic))
            ViewerLog.Important(ViewerLog.Category.General, diagnostic);
    }

    internal void UpdateCurrentAreaContext(TerrainRenderer? renderer)
    {
        if (_areaTableService == null)
        {
            _currentAreaLookup = null;
            _currentAreaName = string.Empty;
            _currentZoneName = string.Empty;
            return;
        }

        int loadedTileCount = _terrainManager?.LoadedTileCount ?? _vlmTerrainManager?.LoadedTileCount ?? 0;
        bool cameraMoved = float.IsNaN(_lastAreaLookupCameraPosition.X)
            || Vector3.DistanceSquared(_camera.Position, _lastAreaLookupCameraPosition) >= 16f;
        bool mapChanged = _currentMapId != _lastAreaLookupMapId;
        bool residencyChanged = loadedTileCount != _lastAreaLookupLoadedTileCount;

        if (++_areaLookupTick < 10 && !cameraMoved && !mapChanged && !residencyChanged)
            return;

        _areaLookupTick = 0;
        _lastAreaLookupCameraPosition = _camera.Position;
        _lastAreaLookupLoadedTileCount = loadedTileCount;
        _lastAreaLookupMapId = _currentMapId;

        // 1. Check if camera is inside a placed WMO group in the world scene
        if (_worldScene != null && _worldScene.TryGetWmoGroupAt(_camera.Position, out var wmoInst, out var wmoR, out int renderGroupIndex))
        {
            uint wmoGroupId = wmoR.GetRenderGroupAreaId(renderGroupIndex);
            string? rawGroupName = wmoR.GetRenderGroupRawName(renderGroupIndex);
            var wmoArea = _areaTableService.ResolveWmoArea(wmoR.WmoId, renderGroupIndex, wmoGroupId, _currentMapId, rawGroupName);
            if (wmoArea.Reason == WowViewer.Core.World.AreaResolutionReason.Resolved)
            {
                _currentAreaLookup = wmoArea;
                _currentZoneName = _currentAreaLookup.ZoneText ?? string.Empty;
                _currentAreaName = _currentAreaLookup.SubzoneText ?? _currentAreaLookup.ZoneText ?? string.Empty;
                return;
            }
        }
        else if (_renderer is WmoRenderer standaloneWmo)
        {
            int standaloneGroupIndex = standaloneWmo.FindGroupContainingPoint(_camera.Position);
            if (standaloneGroupIndex >= 0)
            {
                uint wmoGroupId = standaloneWmo.GetRenderGroupAreaId(standaloneGroupIndex);
                string? rawGroupName = standaloneWmo.GetRenderGroupRawName(standaloneGroupIndex);
                var wmoArea = _areaTableService.ResolveWmoArea(standaloneWmo.WmoId, standaloneGroupIndex, wmoGroupId, _currentMapId, rawGroupName);
                if (wmoArea.Reason == WowViewer.Core.World.AreaResolutionReason.Resolved)
                {
                    _currentAreaLookup = wmoArea;
                    _currentZoneName = _currentAreaLookup.ZoneText ?? string.Empty;
                    _currentAreaName = _currentAreaLookup.SubzoneText ?? _currentAreaLookup.ZoneText ?? string.Empty;
                    return;
                }
            }
        }

        // 2. Fall back to terrain chunk under camera
        if (renderer == null)
        {
            _currentAreaLookup = WowViewer.Core.World.AreaLookupResult.Unresolved(0, _currentMapId, WowViewer.Core.World.AreaResolutionReason.NoTerrainChunk);
            _currentAreaName = string.Empty;
            _currentZoneName = string.Empty;
            return;
        }

        var chunk = renderer.GetChunkInfoAt(_camera.Position.X, _camera.Position.Y);
        _currentAreaLookup = chunk is null
            ? WowViewer.Core.World.AreaLookupResult.Unresolved(0, _currentMapId, WowViewer.Core.World.AreaResolutionReason.NoTerrainChunk)
            : _areaTableService.ResolveArea(chunk.Value.AreaId, _currentMapId);

        _currentZoneName = _currentAreaLookup.ZoneText ?? string.Empty;
        _currentAreaName = _currentAreaLookup.SubzoneText ?? _currentAreaLookup.ZoneText ?? string.Empty;

        if (_currentAreaLookup.Reason != WowViewer.Core.World.AreaResolutionReason.Resolved)
            ReportAreaLookupDiagnostic(_currentAreaLookup.RawAreaId);
    }

    internal void UpdateAreaOverlay(TerrainRenderer? renderer)
    {
        if (_worldScene == null || !_worldScene.ShowAreaRegionOverlay)
            return;

        if (_areaTableService == null || renderer == null)
        {
            _worldScene.SetAreaOverlay(new AreaOverlayBuildResult(
                Array.Empty<AreaOverlayRegion>(),
                0,
                0));
            _areaOverlayRenderer = renderer;
            _areaOverlayAreaTableService = _areaTableService;
            _areaOverlayRevision = int.MinValue;
            _areaOverlayMapId = _currentMapId;
            return;
        }

        if (ReferenceEquals(_areaOverlayRenderer, renderer)
            && ReferenceEquals(_areaOverlayAreaTableService, _areaTableService)
            && _areaOverlayRevision == renderer.ResidentChunkRevision
            && _areaOverlayMapId == _currentMapId)
        {
            return;
        }

        AreaOverlayBuildResult result = AreaOverlayRegionBuilder.Build(
            renderer.EnumerateResidentChunkInfos(),
            _areaTableService,
            _currentMapId);
        _worldScene.SetAreaOverlay(result);
        _areaOverlayRenderer = renderer;
        _areaOverlayAreaTableService = _areaTableService;
        _areaOverlayRevision = renderer.ResidentChunkRevision;
        _areaOverlayMapId = _currentMapId;
    }

    internal void DrawAreaOverlayLabels(
        Matrix4x4 view,
        Matrix4x4 proj,
        float viewportX,
        float viewportY,
        float viewportWidth,
        float viewportHeight)
    {
        if (_worldScene is not { ShowAreaRegionOverlay: true } scene || scene.AreaOverlayRegions.Count == 0)
            return;

        var drawList = ImGui.GetForegroundDrawList();
        foreach (AreaOverlayRegion region in scene.AreaOverlayRegions)
        {
            if (!SceneViewportMath.TryProjectWorldToViewport(
                    region.LabelPosition,
                    view,
                    proj,
                    viewportWidth,
                    viewportHeight,
                    out Vector2 projected))
            {
                continue;
            }

            if (projected.X < -80f || projected.X > viewportWidth + 80f
                || projected.Y < -40f || projected.Y > viewportHeight + 40f)
            {
                continue;
            }

            string label = $"{(region.Kind == AreaOverlayRegionKind.Zone ? "Zone" : "Subzone")}: {region.Name}";
            Vector2 textSize = ImGui.CalcTextSize(label);
            Vector2 textPos = new(
                viewportX + projected.X - textSize.X * 0.5f,
                viewportY + projected.Y - textSize.Y - 16f);
            Vector2 rectMin = textPos - new Vector2(8f, 5f);
            Vector2 rectMax = textPos + textSize + new Vector2(8f, 5f);
            Vector4 color = new(region.Color, 1f);
            Vector4 background = new(region.Color * 0.32f + new Vector3(0.05f), 0.92f);

            drawList.AddCircleFilled(
                new(viewportX + projected.X, viewportY + projected.Y),
                4f,
                ImGui.ColorConvertFloat4ToU32(color));
            drawList.AddRectFilled(rectMin, rectMax, ImGui.ColorConvertFloat4ToU32(background), 4f);
            drawList.AddRect(rectMin, rectMax, ImGui.ColorConvertFloat4ToU32(color), 4f, ImDrawFlags.None, 1.5f);
            drawList.AddText(textPos, ImGui.ColorConvertFloat4ToU32(new Vector4(0.98f, 0.99f, 1f, 1f)), label);
        }
    }
}
