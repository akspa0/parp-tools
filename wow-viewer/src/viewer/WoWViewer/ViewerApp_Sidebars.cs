using System.Numerics;
using System.Diagnostics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Workbench;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;
using WoWViewer.Population;
using WoWViewer.UI;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

/// <summary>
/// Partial class containing the large sidebar and inspector UI blocks.
/// </summary>
public partial class ViewerApp
{
    void IViewerAppHost.DrawWorkspaceBarsPanelContent() => _viewerChrome.DrawWorkspaceBarsPanelContent();

    private void DrawLegacyRightSidebar()
    {
        if (!_shellLayout.HasAnyShellPanelsInLane(ShellPanelLane.Right))
            return;

        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
        float sidebarHeight = io.DisplaySize.Y - topOffset - BottomBarHeight - StatusBarHeight;
        if (_useDockspaceUi)
        {
            DrawDockedShellPanelsForLane(ShellPanelLane.Right, sidebarHeight);
            return;
        }

        _rightSidebarWidth = _viewerChrome.ClampFixedSidebarWidth(_rightSidebarWidth, isLeftSidebar: false, io.DisplaySize.X);
        ImGui.SetNextWindowPos(new Vector2(io.DisplaySize.X - _rightSidebarWidth, topOffset), ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(_rightSidebarWidth, sidebarHeight), ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        if (ImGui.Begin("##LegacyRightSidebar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
        {
            _viewerChrome.DrawFixedSidebarWidthControl(
                "Inspector Width",
                ref _rightSidebarWidth,
                isLeftSidebar: false,
                io.DisplaySize.X,
                "Resize the fixed inspector without relying on the edge splitter.");

            DrawUnifiedToolSidebar();
        }
        ImGui.End();
        ImGui.PopStyleVar();
    }

    private int _activeInspectorTab;

    // T604: utility pages have one dispatcher and one active page index. The
    // Quick destination hosts the page bodies in dedicated collapsible
    // sections; the legacy UI uses the nullable route below as an in-sidebar
    // compatibility adapter.
    private bool _quickUtilitiesExpanded;
    private UtilitiesBottomTab? _pendingQuickUtilityPage;
    private UtilitiesBottomTab? _legacyUtilityPage;
    ref UtilitiesBottomTab? IViewerAppHost.LegacyUtilityPage => ref _legacyUtilityPage;
    private bool _legacyUtilityScrollPending;

    private enum InspectorContextSection
    {
        None,
        SceneInvestigation,
        Mcnk,
        WorldContext,
        Archeology,
        Animations,
        Actions,
    }

    private InspectorContextSection _pendingInspectorContextSection;

    private void DrawUnifiedToolSidebar()
    {
        if (_worldScene != null)
        {
            string[] tabs = ["Inspector", "World", "Model", "Settings", "PM4"];
            int current = _activeInspectorTab;
            if (current < 0 || current >= tabs.Length)
                current = 0;

            if (ImGui.BeginTabBar("##InspectorTabs"))
            {
                for (int i = 0; i < tabs.Length; i++)
                {
                    if (ImGui.BeginTabItem(tabs[i]))
                    {
                        _activeInspectorTab = i;
                        switch (i)
                        {
                            case 0: DrawUnifiedInspectorContent(); break;
                            case 1: _worldObjectsPanel.DrawWorldObjectsPanelContent(); break;
                            case 2: _modelInspector.DrawModelInfoPanelContent(); break;
                            case 3: DrawUnifiedViewerSettingsSidebarContent(); break;
                            case 4: _pm4Workbench.DrawPm4WorkbenchInspector(); break;
                        }
                        ImGui.EndTabItem();
                    }
                }
                ImGui.EndTabBar();
            }
        }
        else if (_renderer is IModelRenderer || _renderer is WmoRenderer)
        {
            string[] tabs = ["Model", "Inspector", "Settings"];
            int current = _activeInspectorTab;
            if (current < 0 || current >= tabs.Length)
                current = 0;

            if (ImGui.BeginTabBar("##StandaloneModelTabs"))
            {
                for (int i = 0; i < tabs.Length; i++)
                {
                    if (ImGui.BeginTabItem(tabs[i]))
                    {
                        _activeInspectorTab = i;
                        switch (tabs[i])
                        {
                            case "Model": _modelInspector.DrawModelInfoPanelContent(); break;
                            case "Inspector": DrawUnifiedInspectorContent(); break;
                            case "Settings": DrawUnifiedViewerSettingsSidebarContent(); break;
                        }
                        ImGui.EndTabItem();
                    }
                }
                ImGui.EndTabBar();
            }
        }
        else
        {
            // No scene or model loaded — just show settings
            DrawUnifiedViewerSettingsSidebarContent();
        }
    }

    private void DrawUnifiedViewerSettingsSidebarContent()
    {
        ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
        ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
        ImGui.Separator();
        _themes.DrawUiThemeSettingsContent();
        ImGui.Separator();
        DrawCameraControlsContent();

        if (_terrainManager != null || _vlmTerrainManager != null)
        {
            ImGui.Separator();
            _terrainControlsPanel.DrawTerrainControlsAdjustmentContent();
        }
    }


    /// <summary>
    /// Single inline owner for the current world/model context. Detail surfaces
    /// use the same compact dropdown pattern as the other canonical routes.
    /// </summary>
    private void DrawUnifiedInspectorContent()
    {
        var content = BuildInspectorContent();
        if (content.HasContent)
        {
            InspectorContentHost.Draw(content, HandleInspectorAction);
        }
        else
        {
            ImGui.TextDisabled("Current context");
            ImGui.TextDisabled("Move the camera over a loaded ADT/MCNK or select a model, world object, or PM4 surface.");
        }

        if (_worldScene == null && (_renderer is IModelRenderer || _renderer is WmoRenderer))
        {
            ImGui.Separator();
            _modelInspector.DrawModelInfoContent();
        }
    }
    void IViewerAppHost.DrawUnifiedInspectorContent() => DrawUnifiedInspectorContent();

    private void DrawInspectorWorldContextContent()
    {
        if (_worldScene == null)
            return;

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

        _worldObjectsPanel.DrawObjectPathFilterControls();

        ImGui.Separator();
        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0)
        {
            bool showPoi = _worldScene.ShowPoi;
            if (ImGui.Checkbox($"Area POIs ({_worldScene.PoiLoader.Entries.Count})", ref showPoi))
                _worldScene.ShowPoi = showPoi;
        }
        else if (!_worldScene.PoiLoadAttempted)
        {
            if (ImGui.Button("Load Area POIs"))
                _worldScene.ShowPoi = true;
        }
        else
        {
            ImGui.TextDisabled("Area POIs: none found");
        }

        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0
            && ImGui.TreeNode($"Area POI Details ({_worldScene.PoiLoader.Entries.Count})"))
        {
            int poiCount = _worldScene.PoiLoader.Entries.Count;
            if (ImGui.BeginChild("##InspectAreaPoiList", new Vector2(0, 200f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(poiCount, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var poi = _worldScene.PoiLoader.Entries[i];
                    bool selected = _selectedAreaPoiId == poi.Id;
                    if (ImGui.Selectable($"[{poi.Id}] {poi.Name}", selected, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        _taxiAndAreaPoi.SelectAreaPoi(poi.Id, toggle: false);
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        {
                            _camera.Position = poi.Position + new Vector3(0, 0, 50);
                            _camera.Pitch = -30f;
                        }
                    }
                }

                if (endIndex < poiCount)
                    ImGui.Dummy(new Vector2(0, (poiCount - endIndex) * rowHeight));
                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        ImGui.Separator();
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
                _worldScene.ShowAreaTriggers = true;
        }
        else
        {
            ImGui.TextDisabled("AreaTriggers: none found");
        }
    }








    private void DrawDockedShellPanelsForLane(ShellPanelLane lane, float sidebarHeight)
    {
        foreach (var panel in ShellPanelDefinitions)
        {
            if (panel.Lane != lane || !_shellLayout.IsShellPanelActive(panel.Id))
                continue;

            float defaultHeight = lane == ShellPanelLane.Left
                ? sidebarHeight
                : Math.Clamp(sidebarHeight * 0.65f, 260f, sidebarHeight);

            if (_pendingFocusedShellPanel == panel.Id)
                ImGui.SetNextWindowFocus();

            _shellLayout.PrepareDockableShellPanelWindow(
                panel.Id,
                new Vector2(panel.DefaultWidth, defaultHeight),
                new Vector2(panel.CompactMinWidth, 220f),
                new Vector2(panel.MaxWidth, sidebarHeight));

            if (ImGui.Begin(panel.WindowName))
            {
                _shellLayout.CaptureDockPanelState(panel.Id);
                DrawShellPanelContent(panel.Id);
            }

            ImGui.End();

            if (_pendingFocusedShellPanel == panel.Id)
                _pendingFocusedShellPanel = null;
        }
    }
    void IViewerAppHost.DrawDockedShellPanelsForLane(ShellPanelLane lane, float sidebarHeight) => DrawDockedShellPanelsForLane(lane, sidebarHeight);
    void IViewerAppHost.DrawFixedSidebarWidthControl(string label, ref float width, bool isLeftSidebar, float displayWidth, string tooltip) => _viewerChrome.DrawFixedSidebarWidthControl(label, ref width, isLeftSidebar, displayWidth, tooltip);

    private void DrawShellPanelContent(ShellPanelId panelId)
    {
        switch (panelId)
        {
            case ShellPanelId.WorkspaceBars:
                _viewerChrome.DrawWorkspaceBarsPanelContent();
                break;
            case ShellPanelId.Navigator:
                _navigatorPanel.DrawNavigatorPanelContent();
                break;
            case ShellPanelId.Inspector:
                DrawSelectionPanelContent();
                break;
            case ShellPanelId.Pm4Workbench:
                _pm4Workbench.DrawPm4WorkbenchInspector();
                break;
            case ShellPanelId.TerrainControls:
                _terrainControlsPanel.DrawTerrainControlsPanelContent();
                break;
            case ShellPanelId.RuntimeStats:
                _viewerChrome.DrawRuntimeStatsPanelContent();
                break;
            case ShellPanelId.WorldObjects:
                _worldObjectsPanel.DrawWorldObjectsPanelContent();
                break;
            case ShellPanelId.ModelInfo:
                _modelInspector.DrawModelInfoPanelContent();
                break;
            case ShellPanelId.Pm4Info:
                _pm4Workbench.DrawPm4InfoPanelContent();
                break;
            case ShellPanelId.Pm4SceneGraph:
                _pm4Workbench.DrawPm4SceneGraphPanelContent();
                break;
        }
    }


    private void DrawSelectionPanelContent()
    {
        ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
        ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
        ImGui.Separator();

        bool hasSelectedPm4 = _worldScene?.Pm4Overlay.HasSelectedPm4Object == true;
        bool hasSelectedObject = DrawSelectedObjectSummaryContent();
        if (!hasSelectedObject)
        {
            if (hasSelectedPm4)
            {
                ImGui.TextDisabled("A PM4 object is selected. Use the PM4 Workbench panel for evidence and correlation.");
                if (ImGui.Button("Focus PM4 Workbench"))
                    _pm4Workbench.OpenPm4Workbench(Pm4WorkbenchTab.Selection);
            }
            else
            {
                ImGui.TextDisabled("Select a world object to inspect its identity and controls here.");
            }
        }

        ImGui.Separator();
        DrawCameraControlsContent();
    }

    private void DrawCameraControlsContent()
    {
        ImGui.SliderFloat("Camera Speed", ref _cameraSpeed, 1f, 500f, "%.0f");
        ImGui.Text("Hold Shift for 5x boost");
        ImGui.SliderFloat("FOV", ref _fovDegrees, 20f, 90f, "%.0f°");

        if (_terrainManager != null && !_terrainManager.Adapter.IsWmoBased)
        {
            ImGui.Separator();

            bool autoAdtBudget = _terrainManager.DetailedTileCountOverride <= 0;
            int adtDetailTiles = autoAdtBudget
                ? _terrainManager.EffectiveDetailedTileCount
                : _terrainManager.DetailedTileCountOverride;

            if (ImGui.SliderInt("ADT Detail Tiles", ref adtDetailTiles, 1, TerrainManager.MaxManualDetailedTileCount))
            {
                _terrainManager.DetailedTileCountOverride = adtDetailTiles;
                _savedDetailedAdtTileCountOverride = _terrainManager.DetailedTileCountOverride;
            }

            if (ImGui.IsItemDeactivatedAfterEdit())
                _settings.SaveViewerSettings();

            ImGui.SameLine();
            if (ImGui.SmallButton("Auto"))
            {
                _terrainManager.DetailedTileCountOverride = 0;
                _savedDetailedAdtTileCountOverride = 0;
                _settings.SaveViewerSettings();
            }

            int retainedTileRadius = _terrainManager.RetainedTileRadius;
            if (ImGui.SliderInt("ADT Retain Radius", ref retainedTileRadius, TerrainManager.MinRetainedTileRadius, TerrainManager.MaxRetainedTileRadius))
                _terrainManager.RetainedTileRadius = retainedTileRadius;

            ImGui.TextDisabled(autoAdtBudget
                ? $"Active submission: {_terrainManager.EffectiveDetailedTileCount} / retained window: {_terrainManager.EffectiveRetainedTileCount} (radius {_terrainManager.EffectiveRetainedTileRadius})"
                : $"Active submission: {_terrainManager.DetailedTileCountOverride} / retained window: {_terrainManager.EffectiveRetainedTileCount} (radius {_terrainManager.EffectiveRetainedTileRadius})");
        }
    }

    private bool DrawSelectedObjectSummaryContent()
    {
        bool hasSelectedPm4 = _worldScene?.Pm4Overlay.HasSelectedPm4Object == true;
        if (string.IsNullOrEmpty(_selectedObjectInfo)
            || hasSelectedPm4
            || _selectedObjectType.StartsWith("Taxi", StringComparison.OrdinalIgnoreCase))
            return false;

        ImGui.TextWrapped(_selectedObjectInfo);
        if (_worldObjectsPanel.TryGetSelectedWorldObjectModelPath(out string selectedModelPath, out _))
        {
            ImGui.Separator();
            _navigatorPanel.DrawAssetPathActions("Selected Asset", selectedModelPath, "SelectedWorldObject");
        }

        _modelInspector.DrawSelectedWmoControls();
        _sqlSpawnStreaming.DrawSelectedSqlGameObjectAnimationControls();
        return true;
    }

    private bool DrawSelectedObjectInspectorSection(bool defaultOpen = true)
    {
        bool hasSelectedPm4 = _worldScene?.Pm4Overlay.HasSelectedPm4Object == true;
        if (string.IsNullOrEmpty(_selectedObjectInfo) || hasSelectedPm4)
            return false;

        ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None;
        if (!ImGui.CollapsingHeader("Selected Object", flags))
            return true;

        DrawSelectedObjectSummaryContent();
        return true;
    }


    internal static float GetUniformListRowHeight()
    {
        return MathF.Max(ImGui.GetTextLineHeightWithSpacing(), ImGui.GetFrameHeightWithSpacing());
    }

    internal static void GetVisibleListRange(int itemCount, float rowHeight, out int startIndex, out int endIndex)
    {
        if (itemCount <= 0)
        {
            startIndex = 0;
            endIndex = 0;
            return;
        }

        float safeRowHeight = MathF.Max(1f, rowHeight);
        float scrollY = ImGui.GetScrollY();
        float windowHeight = ImGui.GetWindowHeight();
        const int overscan = 4;

        startIndex = Math.Max((int)MathF.Floor(scrollY / safeRowHeight) - overscan, 0);
        endIndex = Math.Min((int)MathF.Ceiling((scrollY + windowHeight) / safeRowHeight) + overscan, itemCount);
        if (endIndex < startIndex)
            endIndex = startIndex;
    }
    void IViewerAppHost.DrawTerrainControlsAdjustmentWeakSignalContent() => _terrainControlsPanel.DrawTerrainControlsAdjustmentWeakSignalContent();
    void IViewerAppHost.DrawTemporalStratigraphySubTab() => _terrainControlsPanel.DrawTemporalStratigraphySubTab();

    private void DrawRightSidebar()
    {
        // 071: right sidebar = workbench. Fixed position, full height.
        if (!_useTabUi || !_showRightSidebar)
            return;

        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
        float sidebarHeight = io.DisplaySize.Y - topOffset - BottomBarHeight - StatusBarHeight;

        _rightSidebarWidth = _viewerChrome.ClampFixedSidebarWidth(_rightSidebarWidth, isLeftSidebar: false, io.DisplaySize.X);
        ImGui.SetNextWindowPos(new Vector2(io.DisplaySize.X - _rightSidebarWidth, topOffset), ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(_rightSidebarWidth, sidebarHeight), ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0.08f, 0.08f, 0.10f, 0.85f));

        if (!ImGui.Begin("##RightSidebar", ref _workbenchOpen,
            ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
        {
            ImGui.End();
            ImGui.PopStyleColor();
            ImGui.PopStyleVar();
            return;
        }

        DrawWorkbenchContent();

        ImGui.End();
        ImGui.PopStyleColor();
        ImGui.PopStyleVar();
    }

    private void DrawWorkbenchContent()
    {
        // Settings written by the pre-223 shell can still contain Scene,
        // Utilities, or Experimental. Migrate those values at the boundary
        // so the running shell has exactly four visible destinations.
        NormalizeWorkbenchStateAfterLoad();

        string modeLabel = _workspaceMode switch
        {
            WorkspaceMode.Editor => "Editor workspace",
            WorkspaceMode.Archaeology => "Archaeology workspace",
            _ => "Viewer workspace",
        };
        ImGui.TextDisabled(modeLabel);

        if (!ImGui.GetIO().WantTextInput)
        {
            if (ImGui.IsKeyPressed(ImGuiKey.F1))
                OpenWorkbenchTab(WorkbenchTab.Quick);
            else if (ImGui.IsKeyPressed(ImGuiKey.F2))
                OpenWorkbenchTab(WorkbenchTab.Inspect);
            else if (ImGui.IsKeyPressed(ImGuiKey.F3))
                OpenWorkbenchTab(WorkbenchTab.Editor);
            else if (ImGui.IsKeyPressed(ImGuiKey.F4))
                OpenWorkbenchTab(WorkbenchTab.Archaeology);
        }

        // Canonical four-tab IA. Keep the strip compact enough for a narrow
        // sidebar; Archaeology is intentionally the only long label.
        DrawTopTabButton(WorkbenchTab.Quick, "Quick");
        ImGui.SameLine(0f, ImGui.GetStyle().ItemSpacing.X);
        DrawTopTabButton(WorkbenchTab.Inspect, "Inspector");
        ImGui.SameLine(0f, ImGui.GetStyle().ItemSpacing.X);
        DrawTopTabButton(WorkbenchTab.Editor, "Editor");
        ImGui.SameLine(0f, ImGui.GetStyle().ItemSpacing.X);
        DrawTopTabButton(WorkbenchTab.Archaeology, "Archaeology");
        ImGui.Separator();

        string[] labels = WorkbenchNavigator.GetBottomTabLabels(_activeTopTab);
        if (labels.Length > 0)
        {
            int activePageIndex = _activeBottomTabIndex;
            if (activePageIndex < 0 || activePageIndex >= labels.Length)
                activePageIndex = 0;

            activePageIndex = DrawPageCombo(
                "##WorkbenchPage", labels, activePageIndex);
            _activeBottomTabIndex = activePageIndex;
            ImGui.Separator();
        }

        if (ImGui.BeginChild("##WorkbenchSubTabContent", new Vector2(0, 0), false,
            ImGuiWindowFlags.None))
        {
            switch (_activeTopTab)
            {
                case WorkbenchTab.Quick:
                    DrawQuickControlsContent();
                    break;
                case WorkbenchTab.Inspect:
                    DrawInspectorWorkbenchSubTabContent();
                    break;
                case WorkbenchTab.Archaeology:
                    _archaeologyPanel.DrawArchaeologyWorkbenchSubTabContent();
                    break;
                case WorkbenchTab.Editor:
                    DrawEditorWorkbenchSubTabContent();
                    break;
            }
        }
        ImGui.EndChild();
    }

    /// <summary>
    /// Compatibility migration for pre-223 settings. This runs on the draw
    /// boundary as well as being safe to call from the settings loader in a
    /// future shell pass.
    /// </summary>
    private void NormalizeWorkbenchStateAfterLoad()
    {
        switch (_activeTopTab)
        {
            case WorkbenchTab.Scene:
                _activeTopTab = WorkbenchTab.Inspect;
                _activeBottomTabIndex = _activeBottomTabIndex == 1
                    ? (int)InspectBottomTab.LodBudget
                    : (int)InspectBottomTab.Placements;
                break;

            case WorkbenchTab.Utilities:
                // Utility page indices remain authoritative in the single
                // dispatcher; Quick simply hosts that dispatcher now.
                _activeTopTab = WorkbenchTab.Quick;
                _quickUtilitiesExpanded = true;
                _activeBottomTabIndex = 0;
                break;

            case WorkbenchTab.Experimental:
                // Preserve every old Experimental page by routing it to its
                // canonical visible owner.
                int experimentalPage = _activeBottomTabIndex;
                switch (experimentalPage)
                {
                    case 1: // PM4
                        _activeTopTab = WorkbenchTab.Archaeology;
                        _activeBottomTabIndex = 4;
                        break;
                    case 2: // Converters
                        _activeTopTab = WorkbenchTab.Editor;
                        _activeBottomTabIndex = 3;
                        break;
                    case 3: // Population
                        _activeTopTab = WorkbenchTab.Editor;
                        _activeBottomTabIndex = 0;
                        break;
                    default: // Terrain Lab
                        _activeTopTab = WorkbenchTab.Editor;
                        _activeBottomTabIndex = 1;
                        break;
                }
                break;
        }

        if (_activeTopTab is not (WorkbenchTab.Quick or WorkbenchTab.Inspect or WorkbenchTab.Editor or WorkbenchTab.Archaeology))
            _activeTopTab = WorkbenchTab.Quick;

        NormalizeInspectorPageState();
    }

    private void NormalizeInspectorPageState()
    {
        if (_activeTopTab != WorkbenchTab.Inspect)
            return;

        if (_activeBottomTabIndex < 0)
            _activeBottomTabIndex = (int)InspectBottomTab.Context;

        if (_activeBottomTabIndex <= (int)InspectBottomTab.LodBudget)
            return;

        _pendingInspectorContextSection = _activeBottomTabIndex switch
        {
            (int)InspectBottomTab.SceneInvestigation => InspectorContextSection.SceneInvestigation,
            (int)InspectBottomTab.Mcnk => InspectorContextSection.Mcnk,
            (int)InspectBottomTab.WorldContext => InspectorContextSection.WorldContext,
            (int)InspectBottomTab.Archeology => InspectorContextSection.Archeology,
            (int)InspectBottomTab.Animations => InspectorContextSection.Animations,
            (int)InspectBottomTab.Actions => InspectorContextSection.Actions,
            _ => InspectorContextSection.None,
        };
        _activeBottomTabIndex = (int)InspectBottomTab.Context;
    }

    private void DrawInspectorWorkbenchSubTabContent()
    {
        InspectBottomTab page = (InspectBottomTab)Math.Clamp(
            _activeBottomTabIndex,
            0,
            WorkbenchNavigator.GetInspectBottomTabLabels().Length - 1);

        switch (page)
        {
            case InspectBottomTab.Context:
                DrawInspectorContextPage();
                break;

            case InspectBottomTab.Placements:
                _worldObjectsPanel.DrawWorldPlacementsSubTab();
                break;

            case InspectBottomTab.LodBudget:
                DrawWorldLodSubTab();
                break;

            default:
                // Hidden compatibility page identifiers are normalized into
                // Context before the combo is drawn. Keep this fail-closed
                // fallback for settings written by an older build.
                DrawInspectorContextPage();
                break;
        }
    }

    private void DrawInspectorContextPage()
    {
        DrawUnifiedInspectorContent();

        if ((_terrainManager != null || _vlmTerrainManager != null)
            && SharedUiWidgets.SectionHeader(
                "MCNK Flag Overlay",
                "Filter and highlight raw MCNK flags in the loaded terrain. Diagonal weak-corner markers share this control.",
                defaultOpen: false,
                id: "InspectorMcnkFlags"))
        {
            _investigation.DrawMcnkFlagOverlayControls();
        }

        InspectorContextSection requested = _pendingInspectorContextSection;
        if (requested == InspectorContextSection.None)
            return;

        _pendingInspectorContextSection = InspectorContextSection.None;
        switch (requested)
        {
            case InspectorContextSection.SceneInvestigation:
                if (SharedUiWidgets.SectionHeader("Scene Investigation", defaultOpen: true, id: "InspectorSceneInvestigation"))
                    _investigation.DrawVisualInvestigationToolbox(showWorldObjectRangeControls: _worldScene != null);
                break;
            case InspectorContextSection.Mcnk:
                if ((_terrainManager != null || _vlmTerrainManager != null)
                    && SharedUiWidgets.SectionHeader("MCNK Flag Overlay", defaultOpen: true, id: "InspectorMcnkFlagsLegacy"))
                    _investigation.DrawMcnkFlagOverlayControls();
                break;
            case InspectorContextSection.WorldContext:
                if (SharedUiWidgets.SectionHeader("World Context", defaultOpen: true, id: "InspectorWorldContext"))
                {
                    if (_worldScene == null)
                        SharedUiWidgets.CompactStatus("Load a world scene to inspect world context.");
                    else
                        DrawInspectorWorldContextContent();
                }
                break;
            case InspectorContextSection.Archeology:
                if (SharedUiWidgets.SectionHeader("Archeology", defaultOpen: true, id: "InspectorArcheology"))
                    _archaeologyPanel.DrawArcheologySubTabContent();
                break;
            case InspectorContextSection.Animations:
                if (SharedUiWidgets.SectionHeader("Animations", defaultOpen: true, id: "InspectorAnimations"))
                    _modelInspector.DrawModelAnimationsSubTab();
                break;
            case InspectorContextSection.Actions:
                if (SharedUiWidgets.SectionHeader("Actions", defaultOpen: true, id: "InspectorActions"))
                    _modelInspector.DrawModelActionsSubTab();
                break;
        }
    }

    private void DrawEditorWorkbenchSubTabContent()
    {
        EnsureEditorHost();
        // Spec 231: the Editor destination delegates to the four owned page
        // classes; legacy content remains reachable through their sections.
        EnsureEditorPages().Draw(_activeBottomTabIndex);
    }

    // Spec 232 (operator): per-top-tab page memory — switching workbench tabs returns the
    // operator to the page (right-sidebar dropdown selection) they were on, instead of
    // snapping back to the first page.
    private readonly Dictionary<WorkbenchTab, int> _lastPageByTopTab = new();

    private void DrawTopTabButton(WorkbenchTab tab, string label)
    {
        bool selected = _activeTopTab == tab;
        if (selected)
            ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.38f, 0.25f, 0.08f, 1f));
        bool clicked = ImGui.Button($"{label}##WorkbenchTop_{tab}");
        if (selected)
            ImGui.PopStyleColor();

        if (clicked && _activeTopTab != tab)
        {
            _lastPageByTopTab[_activeTopTab] = _activeBottomTabIndex;
            _activeTopTab = tab;
            _activeBottomTabIndex = _lastPageByTopTab.TryGetValue(tab, out int remembered)
                ? Math.Clamp(remembered, 0, Math.Max(0, WorkbenchNavigator.GetBottomTabLabels(tab).Length - 1))
                : tab == WorkbenchTab.Archaeology
                    ? 5 // Cartography opens on its Layers sub-tab.
                    : 0;
        }
    }

    private void OpenWorkbenchTab(WorkbenchTab topTab, int bottomIndex = -1)
    {
        // Spec 232 FR-4: the default Archaeology destination is Cartography's Map Layers
        // page. An explicit caller page (including Range/UniqueId) and remembered page remain
        // authoritative; only no-page navigation takes this default.
        if (bottomIndex < 0)
            bottomIndex = topTab == WorkbenchTab.Archaeology ? 5 : 0;

        // Adapt legacy destinations at the call boundary. This keeps menu,
        // keyboard, and saved-layout callers functional while exposing only
        // Quick, Inspector, Editor, and Archaeology in the shell.
        switch (topTab)
        {
            case WorkbenchTab.Scene:
                OpenWorkbenchTab(
                    WorkbenchTab.Inspect,
                    bottomIndex == 1
                        ? (int)InspectBottomTab.LodBudget
                        : (int)InspectBottomTab.Placements);
                return;

            case WorkbenchTab.Utilities:
                _activeUtilitiesTabIndex = Math.Clamp(
                    bottomIndex,
                    0,
                    (int)UtilitiesBottomTab.Audio);
                _quickUtilitiesExpanded = true;
                OpenWorkbenchTab(WorkbenchTab.Quick);
                return;

            case WorkbenchTab.Experimental:
                _activeTopTab = WorkbenchTab.Experimental;
                _activeBottomTabIndex = bottomIndex;
                NormalizeWorkbenchStateAfterLoad();
                return;
        }

        if (!_useTabUi)
            return;

        _activeTopTab = topTab;
        if (topTab == WorkbenchTab.Inspect && bottomIndex > (int)InspectBottomTab.LodBudget)
        {
            _activeBottomTabIndex = bottomIndex;
            NormalizeInspectorPageState();
            bottomIndex = _activeBottomTabIndex;
        }

        string[] labels = WorkbenchNavigator.GetBottomTabLabels(topTab);
        int pageIndex = labels.Length > 0
            ? Math.Clamp(bottomIndex, 0, labels.Length - 1)
            : 0;
        _activeBottomTabIndex = pageIndex;
        _showRightSidebar = true;
        _workbenchOpen = true;
    }
    void IViewerAppHost.OpenWorkbenchTab(UtilitiesBottomTab tab) => OpenWorkbenchTab(tab);
    void IViewerAppHost.OpenWorkbenchTab(ToolsBottomTab tab) => OpenWorkbenchTab(tab);
    void IViewerAppHost.OpenWorkbenchTab(WorldBottomTab tab) => OpenWorkbenchTab(tab);
    void IViewerAppHost.OpenWorkbenchTab(ModelBottomTab tab) => OpenWorkbenchTab(tab);
    void IViewerAppHost.OpenWorkbenchTab(WorkbenchTab topTab, int bottomIndex) => OpenWorkbenchTab(topTab, bottomIndex);

    private void OpenWorkbenchTab(ModelBottomTab tab)
    {
        _pendingInspectorContextSection = tab switch
        {
            ModelBottomTab.Animations => InspectorContextSection.Animations,
            ModelBottomTab.Actions => InspectorContextSection.Actions,
            _ => InspectorContextSection.None,
        };
        OpenWorkbenchTab(WorkbenchTab.Inspect, (int)InspectBottomTab.Context);
    }

    private void OpenWorkbenchTab(WorldBottomTab tab)
    {
        if (tab == WorldBottomTab.SelectionTools)
        {
            OpenWorkbenchTab(WorkbenchTab.Inspect, (int)InspectBottomTab.Context);
            return;
        }

        if (tab == WorldBottomTab.Tiles)
        {
            OpenWorkbenchTab(WorkbenchTab.Editor, 1); // Terrain Lab → Terrain Tools (Spec 231)
            return;
        }

        int page = tab switch
        {
            WorldBottomTab.Placements => (int)InspectBottomTab.Placements,
            WorldBottomTab.Lod => (int)InspectBottomTab.LodBudget,
            _ => (int)InspectBottomTab.Context,
        };
        OpenWorkbenchTab(WorkbenchTab.Inspect, page);
    }

    private void OpenWorkbenchTab(ToolsBottomTab tab)
    {
        switch (tab)
        {
            case ToolsBottomTab.Quick:
                OpenWorkbenchTab(WorkbenchTab.Quick);
                break;
            case ToolsBottomTab.Pm4:
                OpenWorkbenchTab(WorkbenchTab.Archaeology, 4);
                break;
            case ToolsBottomTab.Archeology:
                OpenWorkbenchTab(WorkbenchTab.Archaeology, 1);
                break;
            case ToolsBottomTab.Utilities:
                OpenWorkbenchTab((UtilitiesBottomTab)Math.Clamp(
                    _activeUtilitiesTabIndex,
                    0,
                    (int)UtilitiesBottomTab.Audio));
                break;
            case ToolsBottomTab.Converters:
                OpenWorkbenchTab(WorkbenchTab.Editor, 3); // Converters (Spec 231)
                break;
            case ToolsBottomTab.Terrain:
            default:
                OpenWorkbenchTab(WorkbenchTab.Editor, 1); // Terrain Lab → Terrain Tools (Spec 231)
                break;
        }
    }

    private void OpenWorkbenchTab(UtilitiesBottomTab tab)
    {
        _activeUtilitiesTabIndex = (int)tab;
        _quickUtilitiesExpanded = true;
        OpenWorkbenchTab(WorkbenchTab.Quick);
    }

    /// <summary>Used by keyboard/capture routing to identify the visible utility page.</summary>
    private bool IsWorkbenchUtilityVisible(UtilitiesBottomTab tab)
    {
        return _legacyUtilityPage == tab
            || (_useTabUi
                && _activeTopTab == WorkbenchTab.Quick
                && _quickUtilitiesExpanded
                && _activeUtilitiesTabIndex == (int)tab);
    }

    internal static int DrawPageCombo(
        string id,
        string[] labels,
        int activeIndex)
    {
        if (labels.Length == 0)
            return 0;

        int selected = Math.Clamp(activeIndex, 0, labels.Length - 1);
        ImGui.SetNextItemWidth(-1f);
        if (ImGui.BeginCombo(id, labels[selected]))
        {
            for (int i = 0; i < labels.Length; i++)
            {
                bool isSelected = selected == i;
                if (ImGui.Selectable(labels[i], isSelected))
                    selected = i;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }
        return selected;
    }




    private void DrawWorldLodSubTab()
    {
        ImGui.TextDisabled("WDL visibility, detailed ADT budget, and distance LOD state.");
        ImGui.Separator();

        if (_worldScene == null && _terrainManager == null && _vlmTerrainManager == null)
        {
            ImGui.TextDisabled("Load a world map to inspect World LOD state.");
            return;
        }

        if (_worldScene != null)
        {
            bool showWdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("Show WDL Terrain", ref showWdl))
                _worldScene.ShowWdlTerrain = showWdl;

            bool showBoundingBoxes = _worldScene.ShowBoundingBoxes;
            if (ImGui.Checkbox("World Bounding Boxes", ref showBoundingBoxes))
                _worldScene.ShowBoundingBoxes = showBoundingBoxes;

            bool showPm4Overlay = _worldScene.Pm4Overlay.ShowPm4Overlay;
            if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
                _worldScene.Pm4Overlay.ShowPm4Overlay = showPm4Overlay;
            if (_worldScene.Pm4Overlay.ShowPm4Overlay && ImGui.IsItemHovered())
                ImGui.SetTooltip(_worldScene.Pm4Overlay.Pm4Status);
        }

        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer != null)
        {
            int loadedTiles = _terrainManager?.LoadedTileCount ?? _vlmTerrainManager?.LoadedTileCount ?? 0;
            ImGui.Text($"Loaded tiles: {loadedTiles}");
            ImGui.Text($"Terrain chunks: {renderer.ChunksRendered} rendered / {renderer.ChunksCulled} culled");
        }

        if (_terrainManager != null)
        {
            int adtDetailTiles = _terrainManager.DetailedTileCountOverride <= 0
                ? _terrainManager.EffectiveDetailedTileCount
                : _terrainManager.DetailedTileCountOverride;
            if (ImGui.SliderInt("ADT Detail Tiles", ref adtDetailTiles, 1, TerrainManager.MaxManualDetailedTileCount))
            {
                _terrainManager.DetailedTileCountOverride = adtDetailTiles;
                _savedDetailedAdtTileCountOverride = _terrainManager.DetailedTileCountOverride;
            }
            if (ImGui.IsItemDeactivatedAfterEdit())
                _settings.SaveViewerSettings();

            ImGui.SameLine();
            if (ImGui.SmallButton("Auto##WorldLodAdtDetail"))
            {
                _terrainManager.DetailedTileCountOverride = 0;
                _savedDetailedAdtTileCountOverride = 0;
                _settings.SaveViewerSettings();
            }

            int retainedTileRadius = _terrainManager.RetainedTileRadius;
            if (ImGui.SliderInt("ADT Retain Radius##WorldLod", ref retainedTileRadius, TerrainManager.MinRetainedTileRadius, TerrainManager.MaxRetainedTileRadius))
                _terrainManager.RetainedTileRadius = retainedTileRadius;

            ImGui.TextDisabled(_terrainManager.DetailedTileCountOverride <= 0
                ? $"Active submission: {_terrainManager.EffectiveDetailedTileCount} / retained window: {_terrainManager.EffectiveRetainedTileCount} (radius {_terrainManager.EffectiveRetainedTileRadius})"
                : $"Active submission: {_terrainManager.DetailedTileCountOverride} / retained window: {_terrainManager.EffectiveRetainedTileCount} (radius {_terrainManager.EffectiveRetainedTileRadius})");
        }

        ImGui.Separator();
        ImGui.TextDisabled("More World LOD facts belong here after the right-sidebar audit identifies the WDL data owner.");
    }

    private void DrawQuickControlsContent()
    {
        // 1. Atmosphere & Fog (Authoritative Fog Controls, Fog End rendered first per US5/FR-6)
        ImGui.Text("Atmosphere & Fog");
        ImGui.Separator();
        DrawAuthoritativeFogControls(showDescription: false);

        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        if (lighting != null)
        {
            ImGui.Spacing();
            ViewerChromeService.DrawTimeOfDayControl(lighting);
            float gameTime = lighting.GameTime;
            string timeLabel = gameTime switch
            {
                < 0.15f => "Night",
                < 0.25f => "Dawn",
                < 0.35f => "Morning",
                < 0.65f => "Day",
                < 0.75f => "Evening",
                < 0.85f => "Dusk",
                _ => "Night"
            };
            ImGui.SameLine();
            ImGui.Text(timeLabel);
        }

        // 2. Camera & Viewport
        ImGui.Spacing();
        ImGui.Text("Camera & Viewport");
        ImGui.Separator();
        ImGui.SliderFloat("Camera Speed", ref _cameraSpeed, 1f, 500f, "%.0f");
        ImGui.TextDisabled("Hold Shift for 5x boost");
        ImGui.SliderFloat("FOV", ref _fovDegrees, 20f, 90f, "%.0f°");

        if (_terrainManager != null && !_terrainManager.Adapter.IsWmoBased)
        {
            ImGui.Spacing();
            bool autoAdtBudget = _terrainManager.DetailedTileCountOverride <= 0;
            int adtDetailTiles = autoAdtBudget
                ? _terrainManager.EffectiveDetailedTileCount
                : _terrainManager.DetailedTileCountOverride;
            if (ImGui.SliderInt("ADT Detail Tiles", ref adtDetailTiles, 1, TerrainManager.MaxManualDetailedTileCount))
            {
                _terrainManager.DetailedTileCountOverride = adtDetailTiles;
                _savedDetailedAdtTileCountOverride = _terrainManager.DetailedTileCountOverride;
            }
            if (ImGui.IsItemDeactivatedAfterEdit())
                _settings.SaveViewerSettings();
            ImGui.SameLine();
            if (ImGui.SmallButton("Auto"))
            {
                _terrainManager.DetailedTileCountOverride = 0;
                _savedDetailedAdtTileCountOverride = 0;
                _settings.SaveViewerSettings();
            }
        }

        // Reset view & wireframe
        ImGui.Spacing();
        if (ImGui.Button("Reset Camera"))
            ResetCamera();
        ImGui.SameLine();
        if (ImGui.Button("Toggle Wireframe"))
            _renderer?.ToggleWireframe();

        if (_worldScene == null && (_renderer is IModelRenderer || _renderer is WmoRenderer))
        {
            ImGui.Spacing();
            ImGui.Text("Model & Animations");
            ImGui.Separator();
            _modelInspector.DrawModelInfoContent();
        }

        // 3. Profile-tailored Quick controls (US5, 223-T501)
        switch (_workspaceMode)
        {
            case WorkspaceMode.Editor:
                DrawEditorQuickSection();
                break;
            case WorkspaceMode.Archaeology:
                _archaeologyPanel.DrawArchaeologyQuickSection();
                break;
            default:
                DrawViewerQuickSection();
                break;
        }

        DrawQuickUtilitiesSection();
    }

    /// <summary>
    /// T604 utility home. Every former Utilities page is still rendered by
    /// <see cref="DrawUtilitiesSubTabContent"/>; this expandable group only
    /// supplies its canonical Quick destination and never forks page logic.
    /// </summary>
    private void DrawQuickUtilitiesSection()
    {
        bool open = _quickUtilitiesExpanded;
        if (SharedUiWidgets.SectionHeader(
                "Utilities",
                "Minimap, log, performance, render quality, taxi, capture, asset catalog, runtime stats, lighting, and audio.",
                defaultOpen: open,
                id: "QuickUtilities"))
        {
            _quickUtilitiesExpanded = true;
            string[] labels = WorkbenchNavigator.GetUtilitiesBottomTabLabels();
            int selected = Math.Clamp(_activeUtilitiesTabIndex, 0, labels.Length - 1);
            selected = DrawPageCombo("##QuickUtilityPage", labels, selected);
            _activeUtilitiesTabIndex = selected;
            SharedUiWidgets.Divider();
            DrawUtilitiesSubTabContent();
        }
        else if (open)
        {
            // ImGui remembers the open state, but retaining this bit lets a
            // legacy caller explicitly reopen the group after profile changes.
            _quickUtilitiesExpanded = false;
        }
    }

    private void DrawViewerQuickSection()
    {
        ImGui.Spacing();
        ImGui.Text("Scene & UI");
        ImGui.Separator();
        ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
        ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
        ImGui.Spacing();

        bool hideUi = _hideUiChrome;
        if (ImGui.Checkbox("Hide UI Chrome (Tab key)", ref hideUi))
            _hideUiChrome = hideUi;

        ImGui.Spacing();
        ImGui.Text("Quick Navigation");
        ImGui.Separator();
        if (ImGui.Button("Open Inspector##ViewerQuick"))
            OpenWorkbenchTab(WorkbenchTab.Inspect, 0);
        ImGui.SameLine();
        if (ImGui.Button("Open Settings...##ViewerQuick"))
            _showSettingsWindow = true;

        ImGui.Spacing();
        ImGui.Text("UI Theme");
        ImGui.Separator();
        _themes.DrawUiThemeSettingsContent();
    }

    private void DrawEditorQuickSection()
    {
        ImGui.Spacing();
        ImGui.Text("Editor Tasks");
        ImGui.Separator();
        ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
        ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
        ImGui.Spacing();

        foreach (EditorWorkspaceTask task in Enum.GetValues<EditorWorkspaceTask>())
        {
            bool isAvailable = IsEditorTaskAvailable(task);
            if (!isAvailable)
                ImGui.BeginDisabled();

            bool isSelected = task == _editorWorkspaceTask;
            if (isSelected)
                ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.38f, 0.25f, 0.08f, 1f));

            if (ImGui.Button($"{GetEditorWorkspaceTaskLabel(task)}##QuickEditorTask_{task}"))
            {
                SetEditorWorkspaceTask(task);
                OpenWorkbenchTab(WorkbenchTab.Editor, 0);
            }

            if (isSelected)
                ImGui.PopStyleColor();

            if (ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled))
                ImGui.SetTooltip(GetEditorWorkspaceTooltip(task));

            if (!isAvailable)
                ImGui.EndDisabled();

            ImGui.SameLine();
        }
        ImGui.NewLine();

        ImGui.Spacing();
        ImGui.Text("Staged Placement Actions");
        ImGui.Separator();
        _placementEditing.DrawPlacementSaveQueueActions(includeCurrentSourceSave: true);

        ImGui.Spacing();
        ImGui.Text("Quick Export & Conversion");
        ImGui.Separator();
        if (ImGui.Button("Map Converter...##Quick"))
        {
            _mainMenuBar.PrepareMapConverterDialogInputs();
            _showMapConverterDialog = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Export GLB Scene##Quick"))
        {
            _wantExportGlb = true;
        }

        if (_chunkClipboard != null || _chunkClipboardSet != null || _selectedChunks.Count > 0)
        {
            ImGui.Spacing();
            if (ImGui.Button("Clear Chunk Clipboard##Quick"))
            {
                _chunkClipboard = null;
                _chunkClipboardSet = null;
                _chunkClipboardLockedTargetKey = null;
                _selectedChunks.Clear();
                _chunkClipboardStatus = "Clipboard cleared.";
            }
        }
    }

    // ── Converters sub-tab content ──────────────────────────────────────────
    private void DrawConvertersSubTabContent()
    {
        ImGui.TextDisabled("Converter commands launch external tools. Each card runs the existing CLI and captures output.");
        ImGui.Separator();

        if (ImGui.CollapsingHeader("Map Converter", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextDisabled("Converts modern ADT/WDT to Alpha-era formats.");
            if (ImGui.Button("Launch Map Converter"))
            {
                _mainMenuBar.PrepareMapConverterDialogInputs();
                _showMapConverterDialog = true;
            }
            ImGui.SameLine();
            ImGui.TextDisabled("Tools > Offline Data / Conversion > Map Converter...");
        }

        if (ImGui.CollapsingHeader("WMO Converter", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextDisabled("Converts WMO v17 to v14 (Alpha) and vice versa.");
            if (ImGui.Button("Launch WMO Converter"))
            {
                _mainMenuBar.PrepareWmoConverterDialogInputs();
                _showWmoConverterDialog = true;
            }
            ImGui.SameLine();
            ImGui.TextDisabled("Tools > Offline Data / Conversion > WMO Converter...");
        }

        if (ImGui.CollapsingHeader("M2 / MDX Converter", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextDisabled("Converts between M2 (Wrath+) and MDX (Alpha/Vanilla) model formats.");
            ImGui.TextDisabled("Not yet implemented — CLI tool exists in gillijimproject_refactor.");
        }

        if (ImGui.CollapsingHeader("ADT Utilities", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextDisabled("Split/merge ADT, texture transfer, alpha mask tools.");
            ImGui.TextDisabled("Not yet implemented — CLI tools exist in gillijimproject_refactor.");
        }

        if (ImGui.CollapsingHeader("Round-trip Validation", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextDisabled("Validate converter output against source data.");
            ImGui.TextDisabled("Not yet implemented.");
        }
    }

    // ── Utilities page content ─────────────────────────────────────────────
    private void DrawUtilitiesSubTabContent()
    {
        switch ((UtilitiesBottomTab)_activeUtilitiesTabIndex)
        {
            case UtilitiesBottomTab.Minimap:
                DrawUtilitiesMinimap();
                break;
            case UtilitiesBottomTab.Log:
                DrawLogViewerContent();
                break;
            case UtilitiesBottomTab.Perf:
                _pm4Workbench.DrawPerfContent();
                break;
            case UtilitiesBottomTab.RenderQuality:
                DrawRenderQualityContent();
                break;
            case UtilitiesBottomTab.Taxi:
                if (_worldScene != null) _taxiPanel.DrawTaxiContent();
                else ImGui.TextDisabled("Load a world to enable taxi tools.");
                break;
            case UtilitiesBottomTab.Capture:
                DrawCapturePanelContent();
                break;
            case UtilitiesBottomTab.AssetCatalog:
                if (_catalogView == null)
                {
                    _catalogView = new Catalog.AssetCatalogView(_gl);
                    _catalogView.SetDataSource(_dataSource);
                    _catalogView.OnLoadModelRequested = _modelLoader.OnCatalogLoadModel;
                }
                _catalogView.DrawContent();
                break;
            case UtilitiesBottomTab.RuntimeStats:
                _viewerChrome.DrawRuntimeStatsPanelContent();
                break;
            case UtilitiesBottomTab.Lighting:
                _lightingPanel.DrawLightingContent();
                break;
            case UtilitiesBottomTab.Audio:
                _audioPanel.DrawAudioContent();
                break;
        }
    }

    private void DrawUtilitiesMinimap()
    {
        if (TryGetActiveMinimapState(out var existingTiles, out var isTileLoaded, out int loadedTileCount, out string? mapName))
        {
            DrawMinimapContent(loadedTileCount, mapName, existingTiles, isTileLoaded);
        }
        else
        {
            DrawMinimapContent(0, null, null!, null!);
        }
    }
}
