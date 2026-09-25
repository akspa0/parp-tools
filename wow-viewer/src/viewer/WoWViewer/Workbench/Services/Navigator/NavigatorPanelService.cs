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
/// Navigator panel: left sidebar, world overview, map discovery and export controls, file browser and asset-path actions.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class NavigatorPanelService
{
    private readonly IViewerAppHost _host;

    internal NavigatorPanelService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see NavigatorPanelService.Host.cs.


    internal bool TryGetSelectedBrowserAssetPath(out string assetPath)
    {
        assetPath = string.Empty;
        if (_selectedFileIndex < 0 || _selectedFileIndex >= _filteredFiles.Count)
            return false;

        assetPath = _filteredFiles[_selectedFileIndex];
        return !string.IsNullOrWhiteSpace(assetPath);
    }

    internal bool TryGetSelectedBrowserModelPath(out string assetPath)
    {
        if (TryGetSelectedBrowserAssetPath(out assetPath) && TaxiAndAreaPoiSelectionService.IsTaxiActorModelPath(assetPath))
            return true;

        assetPath = string.Empty;
        return false;
    }

    internal void CopyTextToClipboard(string text, string description)
    {
        if (string.IsNullOrWhiteSpace(text))
            return;

        ImGui.SetClipboardText(text);
        _statusMessage = $"Copied {description} to clipboard.";
    }

    internal static string NormalizeAssetPathForUi(string assetPath)
        => string.IsNullOrWhiteSpace(assetPath)
            ? string.Empty
            : assetPath.Trim().Replace('/', '\\');

    private bool CanLoadAssetFromDataSource(string assetPath)
        => _dataSource != null
            && !string.IsNullOrWhiteSpace(assetPath)
            && !Path.IsPathRooted(assetPath);

    internal void FramePoint(Vector3 target, float radius = 2f)
    {
        float effectiveRadius = MathF.Max(radius, 1f);
        float distance = MathF.Max(effectiveRadius * 4f, 12f);
        Vector3 cameraPosition = target + new Vector3(-distance, 0f, effectiveRadius * 1.2f);
        Vector3 lookDirection = Vector3.Normalize(target - cameraPosition);

        _camera.Position = cameraPosition;
        _camera.Yaw = MathF.Atan2(lookDirection.Y, lookDirection.X) * (180f / MathF.PI);
        _camera.Pitch = MathF.Asin(Math.Clamp(lookDirection.Z, -1f, 1f)) * (180f / MathF.PI);
    }

    internal void DrawAssetPathActions(string label, string assetPath, string idSuffix)
    {
        string normalizedPath = NormalizeAssetPathForUi(assetPath);
        if (string.IsNullOrWhiteSpace(normalizedPath))
        {
            ImGui.TextDisabled($"{label}: unavailable");
            return;
        }

        ImGui.Text(label);
        if (ImGui.SmallButton($"Copy Path##{idSuffix}"))
            CopyTextToClipboard(normalizedPath, "asset path");

        ImGui.SameLine();
        bool canLoad = CanLoadAssetFromDataSource(normalizedPath);
        if (!canLoad)
            ImGui.BeginDisabled();
        if (ImGui.SmallButton($"Load Asset##{idSuffix}"))
            _modelLoader.LoadFileFromDataSource(normalizedPath);
        if (!canLoad)
            ImGui.EndDisabled();

        ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + 520f);
        ImGui.TextDisabled(normalizedPath);
        ImGui.PopTextWrapPos();
    }

    internal bool TryInspectHoveredSceneAssetInSelection()
    {
        if (_worldScene?.HoveredAssetInfo is not HoveredAssetInfo info || !info.HasSceneObject)
            return false;

        if (!_worldScene.SelectSceneObject(info.SceneObjectType, info.SceneObjectIndex, info.ParentWmoIndex))
            return false;

        ClearSelectedWlLiquidBody(clearListIsolation: true);
        _worldScene.ClearTaxiSelection();
        _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
        _taxiAndAreaPoi.ClearSelectedAreaPoiInfo();
        RefreshSelectedWorldObjectInfo();
        return true;
    }
}
