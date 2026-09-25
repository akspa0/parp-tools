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
using static WoWViewer.ImGuiListLayout;
using static WoWViewer.InvestigationService;

namespace WoWViewer;

/// <summary>
/// World objects panel: placement lists, the world-objects inspector content, and per-map object path filters.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class WorldObjectsPanelService
{
    private readonly IViewerAppHost _host;

    internal WorldObjectsPanelService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private PlacementEditService _placementEditing => _host.PlacementEditing;
    private Dictionary<string, SavedObjectPathFilterMap> _savedObjectPathFiltersByMap => _host.SavedObjectPathFiltersByMap;
    private ref int _selectedAreaPoiId => ref _host.SelectedAreaPoiId;
    private SqlSpawnStreamingService _sqlSpawnStreaming => _host.SqlSpawnStreaming;
    private ref string _statusMessage => ref _host.StatusMessage;
    private TaxiAndAreaPoiSelectionService _taxiAndAreaPoi => _host.TaxiAndAreaPoi;
    private ref bool _taxiRideCameraEnabled => ref _host.TaxiRideCameraEnabled;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VisualInvestigationMode _visualInvestigationMode => ref _host.VisualInvestigationMode;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref bool _wlLayerListIsolationEnabled => ref _host.WlLayerListIsolationEnabled;
    private ref string _wlLayerSelectedBodyKey => ref _host.WlLayerSelectedBodyKey;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void DrawTerrainChunkInvestigationPanel(bool defaultOpen) => _host.DrawTerrainChunkInvestigationPanel(defaultOpen);
    private void DrawToolbarPopupButton(string label, string summary, string popupId, Action drawContent) => _host.DrawToolbarPopupButton(label, summary, popupId, drawContent);
    private void DrawVisualInvestigationToolbox(bool showWorldObjectRangeControls) => _host.DrawVisualInvestigationToolbox(showWorldObjectRangeControls);
    private void OpenPm4Workbench(Pm4WorkbenchTab tab) => _host.OpenPm4Workbench(tab);
    private void SaveViewerSettings() => _host.SaveViewerSettings();
    private void SetSelectedWlLiquidBody(WlLiquidBody body, bool isolateInList, bool focusInspectWorkspace, string? statusMessage = null) => _host.SetSelectedWlLiquidBody(body, isolateInList, focusInspectWorkspace, statusMessage);
    private bool ShouldIncludeWlBodyInUiList(WlLiquidBody body) => _host.ShouldIncludeWlBodyInUiList(body);
    private bool IsWlListIsolationActive => _host.IsWlListIsolationActive;

    private string _objectPathFilterInput = "";
    private bool _objectPathFilterInputAppliesToWmo = true;
    private bool _objectPathFilterInputAppliesToMdx = true;

    /// <summary>
    /// Canonical Scene > Placements body. Keep this list-only so scene
    /// navigation does not also become the owner for diagnostics or tools.
    /// </summary>
    private void DrawPlacementListsContent()
    {
        if (_worldScene == null)
            return;

        if (_worldScene.ModfPlacements.Count > 0 && ImGui.TreeNode($"WMO Placements ({_worldScene.ModfPlacements.Count})"))
        {
            if (ImGui.BeginChild("##CanonicalWmoPlacements", new Vector2(0, 220f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(_worldScene.ModfPlacements.Count, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var p = _worldScene.ModfPlacements[i];
                    string name = p.NameIndex < _worldScene.WmoModelNames.Count
                        ? Path.GetFileName(_worldScene.WmoModelNames[p.NameIndex]) : "?";
                    string label = $"[{i}] {name}";
                    if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick)
                        && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    {
                        _camera.Position = p.Position + new Vector3(0, 0, 50);
                        _camera.Pitch = -30f;
                    }

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})");
                        ImGui.Text($"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})");
                        ImGui.Text($"Flags: 0x{p.Flags:X4}");
                        ImGui.Text($"Bounds: ({p.BoundsMin.X:F0},{p.BoundsMin.Y:F0},{p.BoundsMin.Z:F0}) - ({p.BoundsMax.X:F0},{p.BoundsMax.Y:F0},{p.BoundsMax.Z:F0})");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < _worldScene.ModfPlacements.Count)
                    ImGui.Dummy(new Vector2(0, (_worldScene.ModfPlacements.Count - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        int mddfCount = _worldScene.MddfPlacements.Count;
        int mddfShow = Math.Min(mddfCount, 200);
        if (mddfCount > 0 && ImGui.TreeNode($"MDX Placements ({mddfCount}{(mddfCount > mddfShow ? $", showing {mddfShow}" : "")})"))
        {
            if (ImGui.BeginChild("##CanonicalMdxPlacements", new Vector2(0, 220f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(mddfShow, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var p = _worldScene.MddfPlacements[i];
                    string name = p.NameIndex < _worldScene.MdxModelNames.Count
                        ? Path.GetFileName(_worldScene.MdxModelNames[p.NameIndex]) : "?";
                    string label = $"[{i}] {name} s={p.Scale:F2}";
                    if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick)
                        && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    {
                        _camera.Position = p.Position + new Vector3(0, 0, 20);
                        _camera.Pitch = -30f;
                    }

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})");
                        ImGui.Text($"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})");
                        ImGui.Text($"Scale: {p.Scale:F3}");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < mddfShow)
                    ImGui.Dummy(new Vector2(0, (mddfShow - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        if (mddfCount == 0 && _worldScene.ModfPlacements.Count == 0)
            ImGui.TextDisabled("No WMO or MDX placements are loaded for the current world.");
    }

    internal void DrawWorldObjectsContentCore()
    {
        if (_worldScene == null) return;

        _placementEditing.DrawSelectedPlacementEditControls();
        DrawVisualInvestigationToolbox(showWorldObjectRangeControls: true);
        ImGui.Separator();
        DrawTerrainChunkInvestigationPanel(defaultOpen: _visualInvestigationMode == VisualInvestigationMode.Adt);
        ImGui.Separator();

        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

        ImGui.Separator();
        _sqlSpawnStreaming.DrawPopulationSubTabContent();

        bool showPm4Overlay = _worldScene.Pm4Overlay.ShowPm4Overlay;
        if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
            _worldScene.Pm4Overlay.ShowPm4Overlay = showPm4Overlay;
        if (ImGui.IsItemHovered() && _worldScene.Pm4Overlay.ShowPm4Overlay)
            ImGui.SetTooltip(_worldScene.Pm4Overlay.Pm4Status);

        DrawToolbarPopupButton("PM4 Actions", string.Empty, "##Pm4OverlayActionsPopup", () =>
        {
            if (ImGui.Button("PM4 Workbench"))
            {
                OpenPm4Workbench(_worldScene.Pm4Overlay.HasSelectedPm4Object ? Pm4WorkbenchTab.Selection : Pm4WorkbenchTab.Overlay);
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button("Reload PM4"))
            {
                _worldScene.Pm4Overlay.ReloadPm4Overlay();
                ImGui.CloseCurrentPopup();
            }
        });

        if (_worldScene.Pm4Overlay.IsPm4Loading)
            ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), $"PM4 loading... {_worldScene.Pm4Overlay.Pm4Status}");
        else if (_worldScene.Pm4Overlay.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4Overlay.Pm4VisibleObjectCount}/{_worldScene.Pm4Overlay.Pm4ObjectCount} visible objects, {_worldScene.Pm4Overlay.Pm4VisibleLineCount}/{_worldScene.Pm4Overlay.Pm4LineCount} lines, {_worldScene.Pm4Overlay.Pm4VisibleTriangleCount}/{_worldScene.Pm4Overlay.Pm4TriangleCount} tris");
        else
            ImGui.TextDisabled("PM4 stays lightweight here. Use the inspector workbench for overlay tuning, object matches, and correlation.");
        if (_worldScene.Pm4Overlay.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4 status: {_worldScene.Pm4Overlay.Pm4Status}");
        ImGui.TextDisabled("PM4 settings and deep analysis live in Inspector > PM4 Workbench.");

        ImGui.Separator();

        // POI toggle — lazy-loaded on first request
        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0)
        {
            bool showPoi = _worldScene.ShowPoi;
            if (ImGui.Checkbox($"Area POIs ({_worldScene.PoiLoader.Entries.Count})", ref showPoi))
                _worldScene.ShowPoi = showPoi;
        }
        else if (!_worldScene.PoiLoadAttempted)
        {
            DrawToolbarPopupButton("POI Actions", "load", "##PoiActionsPopup", () =>
            {
                if (ImGui.Button("Load Area POIs"))
                {
                    _worldScene.ShowPoi = true;
                    ImGui.CloseCurrentPopup();
                }
            });
        }
        else if (_worldScene.PoiLoadAttempted && (_worldScene.PoiLoader == null || _worldScene.PoiLoader.Entries.Count == 0))
        {
            ImGui.TextDisabled("Area POIs: none found");
        }

        ImGui.Separator();

        string taxiSummary = _worldScene.SelectedTaxiRouteId >= 0
            ? $"route {_worldScene.SelectedTaxiRouteId}"
            : _worldScene.SelectedTaxiNodeId >= 0
                ? $"node {_worldScene.SelectedTaxiNodeId}"
                : _taxiRideCameraEnabled
                    ? "ride active"
                    : _worldScene.ShowTaxi
                        ? "visible"
                        : string.Empty;
        // Taxi panel is accessed via the Utilities workbench tab only.
        // The toolbar popup was removed because ImGui popups have no title bar and
        // dismiss on any outside click, making route selection impossible.

        // WL loose liquid files (WLW/WLQ/WLM) — lazy-loaded on first toggle
        if (_worldScene.WlLoader != null && _worldScene.WlLoader.HasData)
        {
            bool showWl = _worldScene.ShowWlLiquids;
            if (ImGui.Checkbox($"WL Liquids ({_worldScene.WlLoader.Bodies.Count})", ref showWl))
                _worldScene.ShowWlLiquids = showWl;
            if (_worldScene.ShowWlLiquids && ImGui.IsItemHovered())
                ImGui.SetTooltip("Loose WLW/WLQ/WLM liquid project files.\nContains water data for deleted/missing tiles.");

            if (liquidRenderer != null && ImGui.TreeNode("WL Bodies"))
            {
                int visibleCount = 0;
                foreach (var b in _worldScene.WlLoader.Bodies)
                {
                    if (liquidRenderer.IsWlBodyVisible(b.BodyKey))
                        visibleCount++;
                }

                bool hasSelected = !string.IsNullOrWhiteSpace(_wlLayerSelectedBodyKey);
                DrawToolbarPopupButton("WL Body Actions", string.Empty, "##WlBodyActionsPopup", () =>
                {
                    if (ImGui.Button("Show All"))
                    {
                        liquidRenderer.SetAllWlBodiesVisible(true);
                        ImGui.CloseCurrentPopup();
                    }

                    if (ImGui.Button("Hide All"))
                    {
                        liquidRenderer.SetAllWlBodiesVisible(false);
                        ImGui.CloseCurrentPopup();
                    }

                    if (!hasSelected)
                        ImGui.BeginDisabled();
                    if (ImGui.Button("Solo Selected"))
                    {
                        liquidRenderer.SetAllWlBodiesVisible(false);
                        liquidRenderer.SetWlBodyVisible(_wlLayerSelectedBodyKey, true);
                        ImGui.CloseCurrentPopup();
                    }
                    if (!hasSelected)
                        ImGui.EndDisabled();

                    if (IsWlListIsolationActive && ImGui.Button("Clear List Isolation"))
                    {
                        _wlLayerListIsolationEnabled = false;
                        ImGui.CloseCurrentPopup();
                    }
                });

                ImGui.TextDisabled($"Visible: {visibleCount}/{_worldScene.WlLoader.Bodies.Count}");

                if (ImGui.BeginTable("##wl_layers", 4, ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp))
                {
                    ImGui.TableSetupColumn("V", ImGuiTableColumnFlags.WidthFixed, 24f);
                    ImGui.TableSetupColumn("Type", ImGuiTableColumnFlags.WidthFixed, 48f);
                    ImGui.TableSetupColumn("Group", ImGuiTableColumnFlags.WidthFixed, 72f);
                    ImGui.TableSetupColumn("Layer", ImGuiTableColumnFlags.WidthStretch);
                    ImGui.TableHeadersRow();

                    for (int i = 0; i < _worldScene.WlLoader.Bodies.Count; i++)
                    {
                        var body = _worldScene.WlLoader.Bodies[i];
                        if (!ShouldIncludeWlBodyInUiList(body))
                            continue;

                        ImGui.TableNextRow();

                        ImGui.TableSetColumnIndex(0);
                        bool visible = liquidRenderer.IsWlBodyVisible(body.BodyKey);
                        if (ImGui.Checkbox($"##wl_vis_{i}", ref visible))
                            liquidRenderer.SetWlBodyVisible(body.BodyKey, visible);

                        ImGui.TableSetColumnIndex(1);
                        ImGui.TextUnformatted(body.FileType.ToString());

                        ImGui.TableSetColumnIndex(2);
                        ImGui.TextUnformatted(body.GroupLabel);

                        ImGui.TableSetColumnIndex(3);
                        bool isSelected = string.Equals(_wlLayerSelectedBodyKey, body.BodyKey, StringComparison.OrdinalIgnoreCase);
                        string label = $"{body.Name}##wl_layer_{i}";
                        if (ImGui.Selectable(label, isSelected, ImGuiSelectableFlags.SpanAllColumns))
                            SetSelectedWlLiquidBody(body, isolateInList: false, focusInspectWorkspace: false);
                        if (ImGui.IsItemHovered())
                        {
                            ImGui.BeginTooltip();
                            ImGui.TextUnformatted(body.SourcePath);
                            ImGui.Text($"Blocks: {body.BlockCount}  Verts: {body.Vertices.Length}");
                            ImGui.Text($"Mode: {body.GroupLabel}  Z: {body.MinHeight:F1}..{body.MaxHeight:F1}");
                            ImGui.EndTooltip();
                        }
                    }

                    ImGui.EndTable();
                }

                ImGui.TreePop();
            }

            if (ImGui.TreeNode("WL Transform Tuning"))
            {
                var ts = WlLiquidLoader.TransformSettings;

                bool enabled = ts.Enabled;
                if (ImGui.Checkbox("Enable Transform", ref enabled))
                    ts.Enabled = enabled;

                bool swapXY = ts.SwapXYBeforeRotation;
                if (ImGui.Checkbox("Swap XY Before Rotation", ref swapXY))
                    ts.SwapXYBeforeRotation = swapXY;

                var rot = ts.RotationDegrees;
                if (ImGui.InputFloat3("Rotation (deg)", ref rot, "%.3f"))
                    ts.RotationDegrees = rot;

                var tr = ts.Translation;
                if (ImGui.InputFloat3("Translation", ref tr, "%.3f"))
                    ts.Translation = tr;

                WlLiquidLoader.WlBodyGroupingMode groupingMode = ts.GroupingMode;
                if (ImGui.BeginCombo("Grouping", GetWlLiquidGroupingModeLabel(groupingMode)))
                {
                    foreach (WlLiquidLoader.WlBodyGroupingMode option in Enum.GetValues<WlLiquidLoader.WlBodyGroupingMode>())
                    {
                        bool isSelected = option == groupingMode;
                        if (ImGui.Selectable(GetWlLiquidGroupingModeLabel(option), isSelected))
                            ts.GroupingMode = option;
                        if (isSelected)
                            ImGui.SetItemDefaultFocus();
                    }

                    ImGui.EndCombo();
                }

                float planeHeightTolerance = ts.PlaneHeightTolerance;
                if (ImGui.SliderFloat("Plane Weld Tolerance", ref planeHeightTolerance, 0.05f, 4.00f, "%.2f"))
                    ts.PlaneHeightTolerance = planeHeightTolerance;

                DrawToolbarPopupButton("WL Transform Actions", string.Empty, "##WlTransformActionsPopup", () =>
                {
                    if (ImGui.Button("Apply + Reload WL"))
                    {
                        _worldScene.ReloadWlLiquids();
                        ImGui.CloseCurrentPopup();
                    }

                    if (ImGui.Button("Print Current WL Transform"))
                    {
                        ViewerLog.Important(ViewerLog.Category.Terrain,
                            $"[WL Transform] Enabled={ts.Enabled} SwapXY={ts.SwapXYBeforeRotation} " +
                            $"Rot=({ts.RotationDegrees.X:F1},{ts.RotationDegrees.Y:F1},{ts.RotationDegrees.Z:F1}) " +
                            $"Trans=({ts.Translation.X:F1},{ts.Translation.Y:F1},{ts.Translation.Z:F1})");
                        ImGui.CloseCurrentPopup();
                    }
                });

                ImGui.TextDisabled("Tune here, then share the printed values to hard-wire final config.");
                ImGui.TreePop();
            }
        }

        if (_worldScene.LitLoader != null && _worldScene.LitLoader.HasData)
        {
            bool showLitLights = _worldScene.ShowLitLights;
            if (ImGui.Checkbox($"LIT Lights ({_worldScene.LitLoader.Lights.Count})", ref showLitLights))
                _worldScene.ShowLitLights = showLitLights;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Alpha-era lights.lit placement overlay. Pins show light origins; boxes show approximate influence radius.");

            bool useLitFogOverride = _worldScene.UseLitFogOverride;
            if (ImGui.Checkbox("Use LIT Lighting Override", ref useLitFogOverride))
                _worldScene.UseLitFogOverride = useLitFogOverride;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Experimental: apply the selected LIT profile over the viewer's always-present global lighting path.");

            if (_worldScene.LastLitSample != null)
                ImGui.TextDisabled($"LIT sample: {_worldScene.LastLitSample.DominantLightName}  fogEnd={_worldScene.LastLitSample.FogEnd:F1}");
            else
                ImGui.TextDisabled(_worldScene.LitStatus);
        }
        else if (!_worldScene.LitLoadAttempted)
        {
            DrawToolbarPopupButton("LIT Actions", "load", "##LitActionsPopup", () =>
            {
                if (ImGui.Button("Load LIT Lights"))
                {
                    _worldScene.ShowLitLights = true;
                    ImGui.CloseCurrentPopup();
                }
            });
        }
        else
        {
            ImGui.TextDisabled(_worldScene.LitStatus);
        }

        bool objectFogEnabled = _worldScene.ObjectFogEnabled;
        if (ImGui.Checkbox("Fog Objects", ref objectFogEnabled))
            _worldScene.ObjectFogEnabled = objectFogEnabled;

        bool showHoverTooltips = _worldScene.ShowHoveredAssetTooltips;
        if (ImGui.Checkbox("Hover Tooltips", ref showHoverTooltips))
            _worldScene.ShowHoveredAssetTooltips = showHoverTooltips;

        bool limitHoverPickRange = _worldScene.LimitHoveredAssetRange;
        if (ImGui.Checkbox("Limit Hover/Pick Range", ref limitHoverPickRange))
            _worldScene.LimitHoveredAssetRange = limitHoverPickRange;

        if (_worldScene.LimitHoveredAssetRange)
        {
            bool useDynamicHoverRange = _worldScene.UseDynamicHoveredAssetRange;
            if (ImGui.Checkbox("Dynamic Hover Range", ref useDynamicHoverRange))
                _worldScene.UseDynamicHoveredAssetRange = useDynamicHoverRange;

            float hoverPickRange = _worldScene.HoveredAssetMaxDistance;
            if (ImGui.SliderFloat("Hover/Pick Range", ref hoverPickRange, 100f, MaxTerrainFogDistance, "%.2f yd"))
                _worldScene.HoveredAssetMaxDistance = hoverPickRange;

            ImGui.TextDisabled($"Effective range: {_worldScene.EffectiveHoveredAssetMaxDistance:F2} yd");
        }

        bool showSelectedObjectBounds = _worldScene.ShowSelectedObjectBounds;
        if (ImGui.Checkbox("Show Selected Object Bounds", ref showSelectedObjectBounds))
            _worldScene.ShowSelectedObjectBounds = showSelectedObjectBounds;

        DrawObjectPathFilterControls();

        ImGui.TextDisabled("UniqueId ranges and playback are in Tools > Archeology.");

        if (!_worldScene.WlLoadAttempted)
        {
            if (ImGui.Button("Load WL Liquids"))
            {
                _worldScene.ShowWlLiquids = true;
            }
        }
        else if (_worldScene.WlLoadAttempted && (_worldScene.WlLoader == null || !_worldScene.WlLoader.HasData))
        {
            ImGui.TextDisabled("WL Liquids: none found");
        }

        // AreaTriggers — lazy-loaded on first request
        if (_worldScene.AreaTriggerLoader != null && _worldScene.AreaTriggerLoader.Count > 0)
        {
            bool showTriggers = _worldScene.ShowAreaTriggers;
            if (ImGui.Checkbox($"AreaTriggers ({_worldScene.AreaTriggerLoader.Count})", ref showTriggers))
                _worldScene.ShowAreaTriggers = showTriggers;
            if (_worldScene.ShowAreaTriggers && ImGui.IsItemHovered())
                ImGui.SetTooltip("Instance portals, event markers, and script triggers.\nGreen spheres/boxes from AreaTrigger.dbc");
        }
        else if (!_worldScene.AreaTriggerLoadAttempted)
        {
            if (ImGui.Button("Load AreaTriggers"))
            {
                _worldScene.ShowAreaTriggers = true;
            }
        }
        else if (_worldScene.AreaTriggerLoadAttempted && (_worldScene.AreaTriggerLoader == null || _worldScene.AreaTriggerLoader.Count == 0))
        {
            ImGui.TextDisabled("AreaTriggers: none found");
        }

        // WMO placements
        if (_worldScene.ModfPlacements.Count > 0 && ImGui.TreeNode($"WMO Placements ({_worldScene.ModfPlacements.Count})"))
        {
            if (ImGui.BeginChild("##WmoPlacements", new Vector2(0, 220f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(_worldScene.ModfPlacements.Count, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var p = _worldScene.ModfPlacements[i];
                    string name = p.NameIndex < _worldScene.WmoModelNames.Count
                        ? Path.GetFileName(_worldScene.WmoModelNames[p.NameIndex]) : "?";
                    string label = $"[{i}] {name}";
                    if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        {
                            _camera.Position = p.Position + new System.Numerics.Vector3(0, 0, 50);
                            _camera.Pitch = -30f;
                        }
                    }
                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})");
                        ImGui.Text($"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})");
                        ImGui.Text($"Flags: 0x{p.Flags:X4}");
                        ImGui.Text($"Bounds: ({p.BoundsMin.X:F0},{p.BoundsMin.Y:F0},{p.BoundsMin.Z:F0}) - ({p.BoundsMax.X:F0},{p.BoundsMax.Y:F0},{p.BoundsMax.Z:F0})");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < _worldScene.ModfPlacements.Count)
                    ImGui.Dummy(new Vector2(0, (_worldScene.ModfPlacements.Count - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        // MDX placements (show first 200 to avoid UI lag)
        int mddfCount = _worldScene.MddfPlacements.Count;
        int mddfShow = Math.Min(mddfCount, 200);
        if (mddfCount > 0 && ImGui.TreeNode($"MDX Placements ({mddfCount}{(mddfCount > mddfShow ? $", showing {mddfShow}" : "")})"))
        {
            if (ImGui.BeginChild("##MdxPlacements", new Vector2(0, 220f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(mddfShow, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var p = _worldScene.MddfPlacements[i];
                    string name = p.NameIndex < _worldScene.MdxModelNames.Count
                        ? Path.GetFileName(_worldScene.MdxModelNames[p.NameIndex]) : "?";
                    string label = $"[{i}] {name} s={p.Scale:F2}";
                    if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        {
                            _camera.Position = p.Position + new System.Numerics.Vector3(0, 0, 20);
                            _camera.Pitch = -30f;
                        }
                    }
                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1})");
                        ImGui.Text($"Rotation: ({p.Rotation.X:F1}, {p.Rotation.Y:F1}, {p.Rotation.Z:F1})");
                        ImGui.Text($"Scale: {p.Scale:F3}");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < mddfShow)
                    ImGui.Dummy(new Vector2(0, (mddfShow - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        // Area POI list
        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0 &&
            ImGui.TreeNode($"Area POIs ({_worldScene.PoiLoader.Entries.Count})"))
        {
            if (ImGui.BeginChild("##AreaPoiList", new Vector2(0, 200f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                int poiCount = _worldScene.PoiLoader.Entries.Count;
                GetVisibleListRange(poiCount, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var poi = _worldScene.PoiLoader.Entries[i];
                    string label = $"[{poi.Id}] {poi.Name}";
                    bool isSelected = _selectedAreaPoiId == poi.Id;
                    if (ImGui.Selectable(label, isSelected, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        _taxiAndAreaPoi.SelectAreaPoi(poi.Id, toggle: false);
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        {
                            _camera.Position = poi.Position + new System.Numerics.Vector3(0, 0, 50);
                            _camera.Pitch = -30f;
                        }
                    }
                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Position: ({poi.Position.X:F1}, {poi.Position.Y:F1}, {poi.Position.Z:F1})");
                        ImGui.Text($"WoW Pos: ({poi.WoWPosition.X:F1}, {poi.WoWPosition.Y:F1}, {poi.WoWPosition.Z:F1})");
                        ImGui.Text($"Icon: {poi.Icon}  Importance: {poi.Importance}  Flags: 0x{poi.Flags:X}");
                        ImGui.EndTooltip();
                    }
                }

                if (endIndex < poiCount)
                    ImGui.Dummy(new Vector2(0, (poiCount - endIndex) * rowHeight));

                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

    }

    private void PersistObjectPathFiltersForCurrentMap()
    {
        if (_worldScene == null)
            return;

        string? currentMapName = _dataSourceSession.GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return;

        List<SavedObjectPathFilterEntry> savedEntries = _worldScene.ObjectPathFilters
            .Where(entry => !string.IsNullOrWhiteSpace(entry.PathPrefix) && (entry.AppliesToWmo || entry.AppliesToMdx))
            .OrderBy(entry => entry.PathPrefix, StringComparer.OrdinalIgnoreCase)
            .Select(entry => new SavedObjectPathFilterEntry
            {
                PathPrefix = entry.PathPrefix,
                AppliesToWmo = entry.AppliesToWmo,
                AppliesToMdx = entry.AppliesToMdx,
            })
            .ToList();

        if (savedEntries.Count == 0 && _worldScene.ObjectPathFiltersEnabled)
        {
            _savedObjectPathFiltersByMap.Remove(currentMapName);
            SaveViewerSettings();
            return;
        }

        _savedObjectPathFiltersByMap[currentMapName] = new SavedObjectPathFilterMap
        {
            MapName = currentMapName,
            Enabled = _worldScene.ObjectPathFiltersEnabled,
            Filters = savedEntries,
        };

        SaveViewerSettings();
    }

    internal bool TryGetSelectedWorldObjectModelPath(out string modelPath, out bool isWmo)
    {
        modelPath = string.Empty;
        isWmo = false;

        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return false;

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        if (string.IsNullOrWhiteSpace(selected.ModelPath))
            return false;

        modelPath = selected.ModelPath.Trim().Replace('/', '\\').Trim('\\');
        if (string.IsNullOrWhiteSpace(modelPath))
            return false;

        isWmo = modelPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);
        return true;
    }

    private static List<string> BuildObjectPathFilterPrefixCandidates(string modelPath)
    {
        var prefixes = new List<string>();
        if (string.IsNullOrWhiteSpace(modelPath))
            return prefixes;

        string normalizedPath = modelPath.Trim().Replace('/', '\\').Trim('\\');
        if (string.IsNullOrWhiteSpace(normalizedPath))
            return prefixes;

        string[] segments = normalizedPath.Split('\\', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length == 0)
            return prefixes;

        string currentPrefix = string.Empty;
        for (int i = 0; i < segments.Length; i++)
        {
            currentPrefix = string.IsNullOrEmpty(currentPrefix)
                ? segments[i]
                : $"{currentPrefix}\\{segments[i]}";

            if (i < segments.Length - 1 || !Path.HasExtension(segments[i]) || segments.Length == 1)
                prefixes.Add(currentPrefix);
        }

        if (!prefixes.Contains(normalizedPath, StringComparer.OrdinalIgnoreCase))
            prefixes.Add(normalizedPath);

        return prefixes;
    }
}
