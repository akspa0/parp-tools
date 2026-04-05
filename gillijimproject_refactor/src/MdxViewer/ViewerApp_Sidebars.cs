using System.Numerics;
using ImGuiNET;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using MdxViewer.Terrain;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace MdxViewer;

/// <summary>
/// Partial class containing the large sidebar and inspector UI blocks.
/// </summary>
public partial class ViewerApp
{
    private bool HasLoadedContent()
    {
        return _terrainManager != null
            || _vlmTerrainManager != null
            || _worldScene != null
            || _loadedWmo != null
            || _loadedMdx != null
            || !string.IsNullOrWhiteSpace(_loadedFilePath);
    }

    private void DrawToolbar()
    {
        var io = ImGui.GetIO();
        ImGui.SetNextWindowPos(new Vector2(0, MenuBarHeight));
        ImGui.SetNextWindowSize(new Vector2(io.DisplaySize.X, ToolbarHeight));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(8, 6));
        ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, new Vector2(6, 0));
        if (ImGui.Begin("##Toolbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings))
        {
            DrawWorkspaceToolbarControls();
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
            ImGui.SameLine();

            TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

            if (renderer != null)
            {
                bool l0 = renderer.ShowLayer0;
                if (ImGui.Checkbox("Base", ref l0)) renderer.ShowLayer0 = l0;
                ImGui.SameLine();
                bool l1 = renderer.ShowLayer1;
                if (ImGui.Checkbox("L1", ref l1)) renderer.ShowLayer1 = l1;
                ImGui.SameLine();
                bool l2 = renderer.ShowLayer2;
                if (ImGui.Checkbox("L2", ref l2)) renderer.ShowLayer2 = l2;
                ImGui.SameLine();
                bool l3 = renderer.ShowLayer3;
                if (ImGui.Checkbox("L3", ref l3)) renderer.ShowLayer3 = l3;

                ImGui.SameLine();
                bool terrainHolesEnabled = !(_terrainManager?.IgnoreTerrainHolesGlobally
                    ?? _vlmTerrainManager?.IgnoreTerrainHolesGlobally
                    ?? false);
                if (ImGui.Checkbox("Holes", ref terrainHolesEnabled))
                {
                    if (SetIgnoreTerrainHolesGlobally(!terrainHolesEnabled))
                    {
                        _statusMessage = terrainHolesEnabled
                            ? "Terrain hole masking enabled."
                            : "Terrain hole masking disabled.";
                    }
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Toggle terrain hole masking on or off.");

                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                ImGui.SameLine();

                bool chunkGrid = renderer.ShowChunkGrid;
                if (ImGui.Checkbox("Chunks", ref chunkGrid)) renderer.ShowChunkGrid = chunkGrid;
                ImGui.SameLine();
                bool tileGrid = renderer.ShowTileGrid;
                if (ImGui.Checkbox("Tiles", ref tileGrid)) renderer.ShowTileGrid = tileGrid;

                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                ImGui.SameLine();

                bool alphaMask = renderer.ShowAlphaMask;
                if (ImGui.Checkbox("Alpha", ref alphaMask)) renderer.ShowAlphaMask = alphaMask;
                ImGui.SameLine();
                bool shadowMap = renderer.ShowShadowMap;
                if (ImGui.Checkbox("Shadows", ref shadowMap)) renderer.ShowShadowMap = shadowMap;
                ImGui.SameLine();
                bool useMccv = renderer.UseMccv;
                if (ImGui.Checkbox("MCCV", ref useMccv)) renderer.UseMccv = useMccv;
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Toggle MCCV terrain vertex-color tinting.");
                ImGui.SameLine();
                bool contours = renderer.ShowContours;
                if (ImGui.Checkbox("Contours", ref contours)) renderer.ShowContours = contours;

                if (liquidRenderer != null)
                {
                    ImGui.SameLine();
                    bool showLiquid = liquidRenderer.ShowLiquid;
                    if (ImGui.Checkbox($"Liquid Terrain ({liquidRenderer.MeshCount})", ref showLiquid))
                        liquidRenderer.ShowLiquid = showLiquid;
                }

                if (_worldScene != null)
                {
                    ImGui.SameLine();
                    int wlCount = liquidRenderer?.WlMeshCount ?? 0;
                    bool showWlTop = _worldScene.ShowWlLiquids;
                    if (ImGui.Checkbox($"WL* ({wlCount})", ref showWlTop))
                        _worldScene.ShowWlLiquids = showWlTop;
                }

                if (_worldScene != null)
                {
                    ImGui.SameLine();
                    bool showWdl = _worldScene.ShowWdlTerrain;
                    if (ImGui.Checkbox("WDL", ref showWdl))
                        _worldScene.ShowWdlTerrain = showWdl;
                }

                if (_worldScene != null)
                {
                    ImGui.SameLine();
                    ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                    ImGui.SameLine();
                    bool showBB = _worldScene.ShowBoundingBoxes;
                    if (ImGui.Checkbox("BBs", ref showBB))
                        _worldScene.ShowBoundingBoxes = showBB;

                    ImGui.SameLine();
                    bool showPm4 = _worldScene.ShowPm4Overlay;
                    if (ImGui.Checkbox("PM4", ref showPm4))
                        _worldScene.ShowPm4Overlay = showPm4;
                    if (_worldScene.IsPm4Loading)
                    {
                        ImGui.SameLine();
                        ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), "loading");
                    }
                    if (_worldScene.ShowPm4Overlay && ImGui.IsItemHovered())
                        ImGui.SetTooltip(_worldScene.Pm4Status);
                }
            }
            else
            {
                bool hasLoadedContent = HasLoadedContent();
                if (!hasLoadedContent)
                {
                    ImGui.TextDisabled("Welcome");
                    ImGui.SameLine();
                    ImGui.Text("Start with File > Open Game Folder (MPQ). Standalone Open File is mainly for direct model/WMO inspection.");
                    ImGui.SameLine();
                }
                else
                {
                    ImGui.TextDisabled("Scene");
                    ImGui.SameLine();
                    string sceneLabel = !string.IsNullOrWhiteSpace(_loadedFileName)
                        ? _loadedFileName!
                        : !string.IsNullOrWhiteSpace(_loadedFilePath)
                            ? Path.GetFileName(_loadedFilePath)
                            : _loadedWmo != null
                                ? "Standalone WMO"
                                : _loadedMdx != null
                                    ? "Standalone model"
                                    : _worldScene != null
                                        ? "World scene"
                                        : "Loaded";
                    ImGui.Text(sceneLabel);
                    ImGui.SameLine();
                }

                if (ImGui.Button("Open Game Folder..."))
                {
                    _showFolderInput = true;
                    _folderInputBuf = string.IsNullOrWhiteSpace(_lastGameFolderPath) ? "" : _lastGameFolderPath;
                }

                ImGui.SameLine();
                if (ImGui.Button("Open File..."))
                    _wantOpenFile = true;

                if (_dataSource != null)
                {
                    ImGui.SameLine();
                    ImGui.TextColored(new Vector4(0.7f, 0.78f, 0.88f, 1f), $"Source ready: {_dataSource.Name}");

                    if (_discoveredMaps.Count > 0)
                    {
                        ImGui.SameLine();
                        ImGui.TextColored(new Vector4(0.65f, 0.82f, 0.68f, 1f), $"Maps: {_discoveredMaps.Count}");
                    }
                }
            }
        }
        ImGui.End();
        ImGui.PopStyleVar(2);
    }

    private void DrawLeftSidebar()
    {
        if (!HasAnyShellPanelsInLane(ShellPanelLane.Left))
            return;

        var io = ImGui.GetIO();
        float topOffset = MenuBarHeight + ToolbarHeight;
        float sidebarHeight = io.DisplaySize.Y - topOffset - StatusBarHeight;
        if (_useDockspaceUi)
        {
            DrawDockedShellPanelsForLane(ShellPanelLane.Left, sidebarHeight);
            return;
        }

        _leftSidebarWidth = ClampFixedSidebarWidth(_leftSidebarWidth, isLeftSidebar: true, io.DisplaySize.X);
        ImGui.SetNextWindowPos(new Vector2(0, topOffset), ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(_leftSidebarWidth, sidebarHeight), ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        if (ImGui.Begin("##LeftSidebar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
            DrawNavigatorPanelContent();
        ImGui.End();
        ImGui.PopStyleVar();
    }

    private void DrawNavigatorPanelContent()
    {
        bool hasWorldLoaded = _worldScene != null || _terrainManager != null || _vlmTerrainManager != null;

        if (_workspaceMode == WorkspaceMode.Editor)
        {
            DrawEditorWorkspaceNavigator(hasWorldLoaded);
            return;
        }

        if (hasWorldLoaded)
        {
            ImGui.SetNextItemOpen(true, ImGuiCond.Once);
            if (ImGui.CollapsingHeader("World Overview", ImGuiTreeNodeFlags.DefaultOpen))
                DrawWorldOverviewContent();
        }

        ImGui.SetNextItemOpen(!hasWorldLoaded, ImGuiCond.Once);
        if (_showFileBrowser && ImGui.CollapsingHeader("File Browser"))
            DrawFileBrowserContent();

        ImGui.SetNextItemOpen(!hasWorldLoaded, ImGuiCond.Once);
        if (_discoveredMaps.Count > 0 && ImGui.CollapsingHeader("World Maps"))
            DrawMapDiscoveryContent();
    }

    private void DrawWorldOverviewContent()
    {
        string sceneLabel = _terrainManager?.MapName
            ?? _vlmTerrainManager?.MapName
            ?? _loadedFileName
            ?? (!string.IsNullOrWhiteSpace(_loadedFilePath)
                ? Path.GetFileName(_loadedFilePath)
                : "World");

        ImGui.Text(sceneLabel);

        if (_worldScene != null || _terrainManager != null || _vlmTerrainManager != null)
        {
            int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
            int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
            ImGui.TextDisabled($"Camera tile: ({tileX}, {tileY})");
        }

        if (!string.IsNullOrWhiteSpace(_currentAreaName))
            ImGui.TextDisabled($"Area: {_currentAreaName}");

        if (_worldScene != null && (_worldScene.ShowPm4Overlay || _worldScene.Pm4LoadAttempted))
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4VisibleObjectCount}/{_worldScene.Pm4ObjectCount} visible objects");

        if (ImGui.Button(_showMinimapWindow ? "Hide Minimap" : "Show Minimap"))
            _showMinimapWindow = !_showMinimapWindow;

        ImGui.SameLine();
        if (ImGui.Button(_fullscreenMinimap ? "Exit Full Minimap" : "Full Minimap"))
            ToggleFullscreenMinimap();

        if (_pendingMinimapTeleportTile.HasValue)
            ImGui.TextDisabled($"Teleport armed: ({_pendingMinimapTeleportTile.Value.tileX}, {_pendingMinimapTeleportTile.Value.tileY}) {_pendingMinimapTeleportClickCount}/{MinimapTeleportConfirmClicks}");
    }

    private void DrawMapDiscoveryContent()
    {
        if (_discoveredMaps.Count == 0) return;

        ImGui.Text($"{_discoveredMaps.Count} maps discovered");
        var previewWarmup = GetWdlPreviewWarmupStats();
        if (previewWarmup.total > 0)
            ImGui.TextDisabled($"WDL previews: {previewWarmup.ready}/{previewWarmup.total} cached, {previewWarmup.loading} warming, {previewWarmup.failed} failed");
        ImGui.Separator();

        float listHeight = 300f;
        if (ImGui.BeginChild("MapList", new Vector2(0, listHeight), true))
        {
            var style = ImGui.GetStyle();
            float rowHeight = GetUniformListRowHeight();
            GetVisibleListRange(_discoveredMaps.Count, rowHeight, out int startIndex, out int endIndex);
            if (startIndex > 0)
                ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

            for (int i = startIndex; i < endIndex; i++)
            {
                var map = _discoveredMaps[i];
                bool hasWdt = map.HasWdt;
                bool hasWdl = map.HasWdl;
                string label = map.HasDbcEntry
                    ? $"[{map.Id:D3}] {map.Name}"
                    : $"[custom] {map.Name}";
                float loadButtonWidth = ImGui.CalcTextSize("Load").X + style.FramePadding.X * 2f;
                float spawnButtonWidth = ImGui.CalcTextSize("Spawn").X + style.FramePadding.X * 2f;
                float reservedActionWidth = spawnButtonWidth + style.ItemSpacing.X;
                if (hasWdt)
                    reservedActionWidth += loadButtonWidth + style.ItemSpacing.X;
                float labelWidth = MathF.Max(1f, ImGui.GetContentRegionAvail().X - reservedActionWidth);
                if (!hasWdt) ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(0.5f, 0.5f, 0.5f, 1f));

                if (ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick, new Vector2(labelWidth, 0f)))
                {
                    if (hasWdt && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        LoadMapAtDefaultSpawn(map);
                }

                if (!hasWdt) ImGui.PopStyleColor();

                if (hasWdt)
                {
                    ImGui.SameLine();
                    if (ImGui.SmallButton($"Load##{map.Directory}"))
                        LoadMapAtDefaultSpawn(map);
                }

                bool canPreview = hasWdl && CanUseWdlPreviewFeature();
                WdlPreviewWarmState previewState = canPreview && _wdlPreviewCacheService != null
                    ? _wdlPreviewCacheService.GetState(map.Directory)
                    : (canPreview ? WdlPreviewWarmState.Ready : WdlPreviewWarmState.NotQueued);
                bool canSelectSpawn = hasWdt && canPreview && previewState != WdlPreviewWarmState.Failed;

                ImGui.SameLine();
                if (!canSelectSpawn) ImGui.BeginDisabled();
                if (ImGui.SmallButton($"Spawn##{map.Directory}") && canSelectSpawn)
                    OpenWdlPreview(map);
                if (!canSelectSpawn) ImGui.EndDisabled();

                if (ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled))
                {
                    ImGui.BeginTooltip();
                    ImGui.Text($"Directory: {map.Directory}");
                    ImGui.Text($"Source: {(map.HasDbcEntry ? "Map.dbc + data source" : "Loose data source only")}");
                    ImGui.Text($"WDT: {(hasWdt ? "Found" : "Missing")}");
                    ImGui.Text($"WDL: {(hasWdl ? "Found" : "Missing")}");
                    if (previewState == WdlPreviewWarmState.Ready)
                        ImGui.TextColored(new Vector4(0f, 1f, 0f, 1f), "WDL preview ready. Click 'Spawn' to choose a start tile.");
                    else if (!hasWdl)
                        ImGui.TextDisabled("No WDL preview is available. 'Load' will use the default map spawn.");
                    else if (previewState is WdlPreviewWarmState.Loading or WdlPreviewWarmState.NotQueued)
                        ImGui.TextDisabled("WDL preview will continue preparing when you open the spawn chooser.");
                    else if (previewState == WdlPreviewWarmState.Failed)
                        ImGui.TextDisabled("WDL preview failed. 'Load' will fall back to the default map spawn.");
                    ImGui.EndTooltip();
                }
            }

            if (endIndex < _discoveredMaps.Count)
                ImGui.Dummy(new Vector2(0, (_discoveredMaps.Count - endIndex) * rowHeight));
            ImGui.EndChild();
        }
    }

    private void DrawFileBrowserContent()
    {
        if (_dataSource == null || !_dataSource.IsLoaded)
        {
            ImGui.TextWrapped("No data source loaded.\nUse File > Open Game Folder to load MPQ archives.");
            return;
        }

        ImGui.Text($"Source: {_dataSource.Name}");
        ImGui.Separator();

        if (ImGui.BeginCombo("Type", GetExtensionFilterLabel(_extensionFilter)))
        {
            (string value, string label)[] filters =
            {
                (".mdx", ".mdx/.mdl"),
                (".wmo", ".wmo"),
                (".m2", ".m2"),
                (".blp", ".blp"),
                (".wdt", ".wdt")
            };
            foreach (var filter in filters)
            {
                if (ImGui.Selectable(filter.label, _extensionFilter == filter.value))
                {
                    _extensionFilter = filter.value;
                    RefreshFileList();
                }
            }
            ImGui.EndCombo();
        }

        var search = _searchFilter;
        if (ImGui.InputText("Search", ref search, 256))
        {
            _searchFilter = search;
            RefreshFileList();
        }

        if (TryGetSelectedBrowserAssetPath(out string selectedAssetPath))
        {
            if (ImGui.Button("Open Selected"))
                LoadFileFromDataSource(selectedAssetPath);

            ImGui.SameLine();
            if (ImGui.Button("Copy Path"))
                CopyTextToClipboard(selectedAssetPath, "asset path");

            if (TryGetTaxiActorOverrideRouteId(out _)
                && IsTaxiActorModelPath(selectedAssetPath))
            {
                ImGui.SameLine();
                if (ImGui.Button("Use For Taxi Override"))
                    TryApplySelectedBrowserAssetToTaxiOverride();
            }

            ImGui.TextDisabled(selectedAssetPath);
        }

        if (HasWorldReturnTarget() && _worldScene == null)
        {
            if (ImGui.Button("Return To Last World"))
                ReturnToLastWorldScene();
        }

        ImGui.Text($"{_filteredFiles.Count} files");
        ImGui.Separator();

        float remainingH = ImGui.GetContentRegionAvail().Y;
        if (_discoveredMaps.Count > 0)
            remainingH = MathF.Max(remainingH - 360f, 100f);
        if (ImGui.BeginChild("FileList", new Vector2(0, remainingH), true))
        {
            float rowHeight = GetUniformListRowHeight();
            GetVisibleListRange(_filteredFiles.Count, rowHeight, out int startIndex, out int endIndex);
            if (startIndex > 0)
                ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

            for (int i = startIndex; i < endIndex; i++)
            {
                var file = _filteredFiles[i];
                var displayName = Path.GetFileName(file);
                bool selected = i == _selectedFileIndex;

                if (ImGui.Selectable(displayName, selected, ImGuiSelectableFlags.AllowDoubleClick))
                {
                    _selectedFileIndex = i;
                    if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        LoadFileFromDataSource(file);
                }

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip(file);
            }

            if (endIndex < _filteredFiles.Count)
                ImGui.Dummy(new Vector2(0, (_filteredFiles.Count - endIndex) * rowHeight));
            ImGui.EndChild();
        }
    }

    private static string GetExtensionFilterLabel(string extensionFilter)
    {
        return extensionFilter.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            ? ".mdx/.mdl"
            : extensionFilter;
    }

    private void DrawRightSidebar()
    {
        if (!HasAnyShellPanelsInLane(ShellPanelLane.Right))
            return;

        var io = ImGui.GetIO();
        float topOffset = MenuBarHeight + ToolbarHeight;
        float sidebarHeight = io.DisplaySize.Y - topOffset - StatusBarHeight;
        if (_useDockspaceUi)
        {
            DrawDockedShellPanelsForLane(ShellPanelLane.Right, sidebarHeight);
            return;
        }

        _rightSidebarWidth = ClampFixedSidebarWidth(_rightSidebarWidth, isLeftSidebar: false, io.DisplaySize.X);
        ImGui.SetNextWindowPos(new Vector2(io.DisplaySize.X - _rightSidebarWidth, topOffset), ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(_rightSidebarWidth, sidebarHeight), ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        if (ImGui.Begin("##RightSidebar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
        {
            GetFixedSidebarWidthRange(isLeftSidebar: false, io.DisplaySize.X, out float minInspectorWidth, out float maxInspectorWidth);
            if (maxInspectorWidth > minInspectorWidth)
            {
                float inspectorWidth = _rightSidebarWidth;
                ImGui.SetNextItemWidth(-1f);
                if (ImGui.SliderFloat("Inspector Width", ref inspectorWidth, minInspectorWidth, maxInspectorWidth, "%.0f px"))
                    _rightSidebarWidth = ClampFixedSidebarWidth(inspectorWidth, isLeftSidebar: false, io.DisplaySize.X);

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Resize the fixed inspector without relying on the edge splitter.");

                ImGui.Separator();
            }

            if (_workspaceMode == WorkspaceMode.Editor)
            {
                DrawEditorWorkspaceInspector();
            }
            else
            {
                bool hasSelectedPm4 = _worldScene?.HasSelectedPm4Object == true;
                bool hasSelectedObject = DrawSelectedObjectSummaryContent();
                if (hasSelectedObject)
                    ImGui.Spacing();

                if (_worldScene != null)
                {
                    bool defaultOpenPm4Workbench = _worldScene.ShowPm4Overlay || hasSelectedPm4 || _pm4ObjectCollection.Count > 0;
                    ImGui.SetNextItemOpen(defaultOpenPm4Workbench, ImGuiCond.Once);
                    if (ImGui.CollapsingHeader("PM4 Workbench"))
                        DrawPm4WorkbenchInspector();
                }

                ImGui.SetNextItemOpen(!string.IsNullOrEmpty(_modelInfo), ImGuiCond.Once);
                if (_showModelInfo && ImGui.CollapsingHeader("Model Info"))
                    DrawModelInfoContent();

                ImGui.SetNextItemOpen(false, ImGuiCond.Once);
                if (ImGui.CollapsingHeader("Camera"))
                    DrawCameraControlsContent();

                if (_showTerrainControls && (_terrainManager != null || _vlmTerrainManager != null))
                {
                    ImGui.SetNextItemOpen(true, ImGuiCond.Once);
                    if (ImGui.CollapsingHeader("Terrain Controls"))
                        DrawTerrainControlsPanelContent();
                }

                if ((_terrainManager != null || _vlmTerrainManager != null || _worldScene != null) && ImGui.CollapsingHeader("Runtime Stats"))
                    DrawRuntimeStatsPanelContent();

                if (_worldScene != null)
                {
                    ImGui.SetNextItemOpen(true, ImGuiCond.Once);
                    if (ImGui.CollapsingHeader("World Objects"))
                        DrawWorldObjectsPanelContent();
                }
            }
        }
        ImGui.End();
        ImGui.PopStyleVar();
    }

    private void DrawDockedShellPanelsForLane(ShellPanelLane lane, float sidebarHeight)
    {
        foreach (var panel in ShellPanelDefinitions)
        {
            if (panel.Lane != lane || !IsShellPanelActive(panel.Id))
                continue;

            float defaultHeight = lane == ShellPanelLane.Left
                ? sidebarHeight
                : Math.Clamp(sidebarHeight * 0.65f, 260f, sidebarHeight);

            if (_pendingFocusedShellPanel == panel.Id)
                ImGui.SetNextWindowFocus();

            ImGui.SetNextWindowSize(new Vector2(panel.DefaultWidth, defaultHeight), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowSizeConstraints(
                new Vector2(panel.CompactMinWidth, 220f),
                new Vector2(panel.MaxWidth, sidebarHeight));

            if (ImGui.Begin(panel.WindowName))
            {
                CaptureDockPanelState(panel.Id);
                DrawShellPanelContent(panel.Id);
            }

            ImGui.End();

            if (_pendingFocusedShellPanel == panel.Id)
                _pendingFocusedShellPanel = null;
        }
    }

    private void DrawShellPanelContent(ShellPanelId panelId)
    {
        switch (panelId)
        {
            case ShellPanelId.Navigator:
                DrawNavigatorPanelContent();
                break;
            case ShellPanelId.Inspector:
                DrawSelectionPanelContent();
                break;
            case ShellPanelId.Pm4Workbench:
                DrawPm4WorkbenchInspector();
                break;
            case ShellPanelId.TerrainControls:
                DrawTerrainControlsPanelContent();
                break;
            case ShellPanelId.RuntimeStats:
                DrawRuntimeStatsPanelContent();
                break;
            case ShellPanelId.WorldObjects:
                DrawWorldObjectsPanelContent();
                break;
            case ShellPanelId.ModelInfo:
                DrawModelInfoPanelContent();
                break;
        }
    }

    private void DrawSelectionPanelContent()
    {
        if (_workspaceMode == WorkspaceMode.Editor)
        {
            ImGui.TextColored(new Vector4(0.75f, 0.88f, 1f, 1f), $"{GetWorkspaceModeLabel(_workspaceMode)} Workspace");
            ImGui.SameLine();
            ImGui.TextDisabled(GetEditorWorkspaceTaskLabel(_editorWorkspaceTask));
            ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
            ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
            ImGui.Separator();
        }

        bool hasSelectedPm4 = _worldScene?.HasSelectedPm4Object == true;
        bool hasSelectedObject = DrawSelectedObjectSummaryContent();
        if (!hasSelectedObject)
        {
            if (hasSelectedPm4)
            {
                ImGui.TextDisabled("A PM4 object is selected. Use the PM4 Workbench panel for evidence and correlation.");
                if (ImGui.Button("Focus PM4 Workbench"))
                    OpenPm4Workbench(Pm4WorkbenchTab.Selection);
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
    }

    private bool DrawSelectedObjectSummaryContent()
    {
        bool hasSelectedPm4 = _worldScene?.HasSelectedPm4Object == true;
        if (string.IsNullOrEmpty(_selectedObjectInfo) || hasSelectedPm4)
            return false;

        ImGui.TextWrapped(_selectedObjectInfo);
        DrawSelectedTaxiControls();
        DrawSelectedWmoControls();
        DrawSelectedSqlGameObjectAnimationControls();
        return true;
    }

    private void DrawTerrainControlsPanelContent()
    {
        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (lighting == null || renderer == null)
        {
            ImGui.TextWrapped("Load a terrain-backed world to use terrain controls.");
            return;
        }

        DrawTerrainControlsAdjustmentContent();
        ImGui.Separator();
        if (ImGui.Button("Open Chunk Clipboard"))
            _showChunkClipboardWindow = true;
        ImGui.SameLine();
        ImGui.TextDisabled("Chunk copy/paste stays available as its own panel window.");
    }

    private void DrawWorldObjectsPanelContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextWrapped("Load a world scene to inspect object, SQL population, POI, taxi, and PM4 overlay workflows.");
            return;
        }

        DrawWorldObjectsContentCore();
    }

    private void DrawModelInfoPanelContent()
    {
        if (string.IsNullOrWhiteSpace(_modelInfo))
        {
            ImGui.TextDisabled("No model info is available for the current selection or loaded asset.");
            return;
        }

        DrawModelInfoContent();
    }

    private void DrawFixedSidebarSplitters()
    {
        if (!IsShellPanelActive(ShellPanelId.Navigator) && !IsShellPanelActive(ShellPanelId.Inspector))
            return;

        var io = ImGui.GetIO();
        float topOffset = MenuBarHeight + ToolbarHeight;
        float panelHeight = io.DisplaySize.Y - topOffset - StatusBarHeight;
        if (panelHeight <= 0f)
            return;

        if (IsShellPanelActive(ShellPanelId.Navigator))
        {
            float splitterX = _leftSidebarWidth - SidebarSplitterWidth * 0.5f;
            DrawFixedSidebarSplitterWindow(
                "##LeftSidebarSplitter",
                splitterX,
                topOffset,
                panelHeight,
                io.MouseDelta.X,
                isLeftSidebar: true,
                io.DisplaySize.X);
        }

        if (IsShellPanelActive(ShellPanelId.Inspector))
        {
            float splitterX = io.DisplaySize.X - _rightSidebarWidth - SidebarSplitterWidth * 0.5f;
            DrawFixedSidebarSplitterWindow(
                "##RightSidebarSplitter",
                splitterX,
                topOffset,
                panelHeight,
                -io.MouseDelta.X,
                isLeftSidebar: false,
                io.DisplaySize.X);
        }
    }

    private void DrawFixedSidebarSplitterWindow(string id, float splitterX, float topOffset, float panelHeight, float deltaWidth, bool isLeftSidebar, float displayWidth)
    {
        ImGui.SetNextWindowPos(new Vector2(splitterX, topOffset), ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(SidebarSplitterWidth, panelHeight), ImGuiCond.Always);
        ImGui.SetNextWindowBgAlpha(0f);

        ImGuiWindowFlags flags = ImGuiWindowFlags.NoTitleBar
            | ImGuiWindowFlags.NoResize
            | ImGuiWindowFlags.NoMove
            | ImGuiWindowFlags.NoCollapse
            | ImGuiWindowFlags.NoSavedSettings
            | ImGuiWindowFlags.NoScrollbar
            | ImGuiWindowFlags.NoScrollWithMouse
            | ImGuiWindowFlags.NoBackground
            | ImGuiWindowFlags.NoBringToFrontOnFocus
            | ImGuiWindowFlags.NoNavFocus;

        if (!ImGui.Begin(id, flags))
        {
            ImGui.End();
            return;
        }

        ImGui.InvisibleButton("##drag", new Vector2(SidebarSplitterWidth, panelHeight));
        bool hovered = ImGui.IsItemHovered();
        bool active = ImGui.IsItemActive();
        if (hovered || active)
            ImGui.SetMouseCursor(ImGuiMouseCursor.ResizeEW);

        if (active)
        {
            if (isLeftSidebar)
                _leftSidebarWidth = ClampFixedSidebarWidth(_leftSidebarWidth + deltaWidth, isLeftSidebar: true, displayWidth);
            else
                _rightSidebarWidth = ClampFixedSidebarWidth(_rightSidebarWidth + deltaWidth, isLeftSidebar: false, displayWidth);
        }

        uint color = ImGui.GetColorU32(hovered || active
            ? new Vector4(0.52f, 0.68f, 0.86f, 0.95f)
            : new Vector4(0.24f, 0.28f, 0.34f, 0.75f));
        var drawList = ImGui.GetWindowDrawList();
        Vector2 windowPos = ImGui.GetWindowPos();
        drawList.AddRectFilled(
            windowPos,
            windowPos + new Vector2(SidebarSplitterWidth, panelHeight),
            color,
            2f);

        ImGui.End();
    }

    private float ClampFixedSidebarWidth(float width, bool isLeftSidebar, float displayWidth)
    {
        GetFixedSidebarWidthRange(isLeftSidebar, displayWidth, out float minWidth, out float maxWidth);
        return Math.Clamp(width, minWidth, maxWidth);
    }

    private void GetFixedSidebarWidthRange(bool isLeftSidebar, float displayWidth, out float minWidth, out float maxWidth)
    {
        float otherSidebarWidth = 0f;
        if (isLeftSidebar)
        {
            if (IsShellPanelActive(ShellPanelId.Inspector))
                otherSidebarWidth = _rightSidebarWidth;
        }
        else if (IsShellPanelActive(ShellPanelId.Navigator))
        {
            otherSidebarWidth = _leftSidebarWidth;
        }

        float preferredMaxWidth = displayWidth - otherSidebarWidth - SceneViewportPreferredMinWidth;
        float hardMaxWidth = displayWidth - otherSidebarWidth - SceneViewportHardMinWidth;
        maxWidth = MathF.Min(SidebarMaxWidth, MathF.Max(SidebarCompactMinWidth, MathF.Max(preferredMaxWidth, hardMaxWidth)));
        minWidth = MathF.Min(SidebarMinWidth, maxWidth);
    }

    private bool DrawSelectedObjectInspectorSection(bool defaultOpen = true)
    {
        bool hasSelectedPm4 = _worldScene?.HasSelectedPm4Object == true;
        if (string.IsNullOrEmpty(_selectedObjectInfo) || hasSelectedPm4)
            return false;

        ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None;
        if (!ImGui.CollapsingHeader("Selected Object", flags))
            return true;

        DrawSelectedObjectSummaryContent();
        return true;
    }

    private void DrawModelInfoContent()
    {
        if (string.IsNullOrEmpty(_modelInfo))
        {
            ImGui.TextWrapped("No model loaded.");
            return;
        }

        ImGui.TextWrapped(_modelInfo);

        if (_renderer is MdxRenderer || _renderer is WmoRenderer)
        {
            ImGui.Separator();
            ImGui.Checkbox("Auto-frame on load", ref _autoFrameModelOnLoad);
            if (ImGui.Button("Frame Model"))
                FrameCurrentModel();
        }

        if (_renderer is WmoRenderer wmoR && wmoR.DoodadSetCount > 0)
        {
            ImGui.Separator();
            ImGui.Text("Doodad Set:");
            int activeSet = wmoR.ActiveDoodadSet;
            string currentSetName = wmoR.GetDoodadSetName(activeSet);
            if (ImGui.BeginCombo("##DoodadSet", currentSetName))
            {
                for (int s = 0; s < wmoR.DoodadSetCount; s++)
                {
                    bool selected = s == activeSet;
                    if (ImGui.Selectable(wmoR.GetDoodadSetName(s), selected))
                        wmoR.SetActiveDoodadSet(s);
                    if (selected) ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
        }

        if (_renderer is WmoRenderer)
        {
            ImGui.Separator();
            DrawWmoLiquidRotationControls("standalone");
        }

        if (_renderer is IModelRenderer modelRenderer && modelRenderer.Animator != null && modelRenderer.Animator.Sequences.Count > 0)
        {
            ImGui.Separator();
            ImGui.Text("Animation:");

            var animator = modelRenderer.Animator;
            int currentSeq = animator.CurrentSequence;
            string currentSeqName = currentSeq >= 0 && currentSeq < animator.Sequences.Count
                ? animator.Sequences[currentSeq].Name
                : "None";

            if (ImGui.BeginCombo("##AnimSequence", currentSeqName))
            {
                for (int s = 0; s < animator.Sequences.Count; s++)
                {
                    bool selected = s == currentSeq;
                    string seqName = animator.Sequences[s].Name;
                    if (string.IsNullOrEmpty(seqName))
                        seqName = $"Sequence {s}";

                    if (ImGui.Selectable(seqName, selected))
                        animator.SetSequence(s);
                    if (selected) ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }

            if (currentSeq >= 0 && currentSeq < animator.Sequences.Count)
            {
                var seq = animator.Sequences[currentSeq];
                float seqStart = seq.Time.Start;
                float seqEnd = seq.Time.End;
                float duration = seqEnd - seqStart;
                float currentAbs = animator.CurrentFrame;
                float currentRel = currentAbs - seqStart;

                bool isPlaying = animator.IsPlaying;
                if (ImGui.Button(isPlaying ? "⏸ Pause" : "▶ Play"))
                    animator.IsPlaying = !isPlaying;

                ImGui.SameLine();
                if (ImGui.Button("◀"))
                {
                    animator.IsPlaying = false;
                    animator.StepToPrevKeyframe();
                }

                ImGui.SameLine();
                if (ImGui.Button("▶"))
                {
                    animator.IsPlaying = false;
                    animator.StepToNextKeyframe();
                }

                ImGui.SetNextItemWidth(-1);
                if (ImGui.SliderFloat("##Timeline", ref currentRel, 0, duration, $"Frame: {currentAbs:F0} / {seqEnd:F0}"))
                {
                    animator.IsPlaying = false;
                    animator.CurrentFrame = seqStart + currentRel;
                }

                ImGui.Text($"Duration: {duration:F0}ms ({duration / 1000.0f:F2}s)");

                if (ImGui.TreeNode("Animation Debug"))
                {
                    ImGui.Text($"Current Seq: {currentSeq}");
                    ImGui.Text($"Current Abs Frame: {currentAbs:F2}");
                    ImGui.Text($"Seq Range: [{seqStart}, {seqEnd}]");

                    var stats = animator.GetTrackDebugStatsForCurrentSequence();
                    ImGui.Text($"T keys total/in-range: {stats.TranslationKeysTotal}/{stats.TranslationKeysInSequence}");
                    ImGui.Text($"R keys total/in-range: {stats.RotationKeysTotal}/{stats.RotationKeysInSequence}");
                    ImGui.Text($"S keys total/in-range: {stats.ScalingKeysTotal}/{stats.ScalingKeysInSequence}");

                    string minKey = stats.MinKeyTime?.ToString() ?? "n/a";
                    string maxKey = stats.MaxKeyTime?.ToString() ?? "n/a";
                    ImGui.Text($"All key range: [{minKey}, {maxKey}]");

                    ImGui.Separator();
                    ImGui.Text("Sequences (first 12):");
                    int previewCount = Math.Min(12, animator.Sequences.Count);
                    for (int i = 0; i < previewCount; i++)
                    {
                        var s = animator.Sequences[i];
                        string name = string.IsNullOrWhiteSpace(s.Name) ? "<empty>" : s.Name;
                        ImGui.Text($"{i}: {name} [{s.Time.Start}-{s.Time.End}]");
                    }

                    ImGui.TreePop();
                }
            }
        }

        if (_renderer != null && _renderer.SubObjectCount > 0)
        {
            ImGui.Separator();
            ImGui.Text("Visibility:");

            DrawRendererVisibilityControls(_renderer, "standalone");
        }
    }

    private void DrawSelectedTaxiControls()
    {
        if (_worldScene == null)
            return;

        bool hasTaxiSelection = _worldScene.SelectedTaxiNodeId >= 0 || _worldScene.SelectedTaxiRouteId >= 0;
        if (!hasTaxiSelection)
            return;

        ImGui.Separator();
        ImGui.Text("Taxi Route Controls");

        if (ImGui.Button("Focus Selected Taxi"))
            FocusSelectedTaxi();

        bool showTaxiActors = _worldScene.ShowTaxiActors;
        if (ImGui.Checkbox("Show Animated Taxi Actor", ref showTaxiActors))
            _worldScene.ShowTaxiActors = showTaxiActors;

        float speedMultiplier = _worldScene.TaxiActorSpeedMultiplier;
        if (ImGui.SliderFloat("Taxi Speed", ref speedMultiplier, 0.1f, 8f, "%.2fx"))
            _worldScene.TaxiActorSpeedMultiplier = speedMultiplier;

        if (TryGetTaxiActorOverrideRouteId(out int routeId))
        {
            IReadOnlyList<TaxiPathLoader.TaxiRoute> candidateRoutes = GetTaxiActorOverrideCandidateRoutes();

            if (_worldScene.SelectedTaxiNodeId >= 0)
            {
                ImGui.TextDisabled($"Selected taxi node: {_worldScene.SelectedTaxiNodeId}");

                string previewLabel = GetTaxiRouteDisplayLabel(routeId);
                if (ImGui.BeginCombo("Override Target Route", previewLabel))
                {
                    foreach (TaxiPathLoader.TaxiRoute candidateRoute in candidateRoutes)
                    {
                        bool isSelected = candidateRoute.PathId == routeId;
                        if (ImGui.Selectable(GetTaxiRouteDisplayLabel(candidateRoute.PathId), isSelected))
                        {
                            _taxiActorModelOverrideTargetRouteId = candidateRoute.PathId;
                            SyncTaxiActorModelOverrideInput(candidateRoute.PathId);
                        }

                        if (isSelected)
                            ImGui.SetItemDefaultFocus();
                    }

                    ImGui.EndCombo();
                }
            }
            else if (_worldScene.SelectedTaxiRouteId >= 0)
            {
                ImGui.TextDisabled($"Selected taxi route: {_worldScene.SelectedTaxiRouteId}");
            }

            SyncTaxiActorModelOverrideInput(routeId);

            string resolvedActorModelPath = _worldScene.GetResolvedTaxiActorModelPath(routeId) ?? "not found";
            string? actorOverridePath = _worldScene.GetTaxiActorModelOverride(routeId);
            ImGui.TextWrapped($"Override Route: {GetTaxiRouteDisplayLabel(routeId)}");
            ImGui.TextWrapped($"Resolved Actor Model: {resolvedActorModelPath}");
            ImGui.TextDisabled($"Override: {actorOverridePath ?? "auto"}");

            string actorModelPath = _taxiActorModelOverrideInput;
            if (ImGui.InputText("Actor Model Path", ref actorModelPath, 512))
                _taxiActorModelOverrideInput = actorModelPath;

            if (ImGui.Button("Apply Model Override"))
            {
                ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
                SyncTaxiActorModelOverrideInput(routeId);
                RefreshSelectedTaxiInfo();
            }

            ImGui.SameLine();
            if (ImGui.Button("Clear Override"))
            {
                ApplyTaxiActorModelOverride(routeId, null);
                _taxiActorModelOverrideInput = string.Empty;
                _taxiActorModelOverrideInputRouteId = routeId;
                RefreshSelectedTaxiInfo();
            }

            if (TryGetSelectedBrowserModelPath(out string selectedBrowserModelPath))
            {
                if (ImGui.Button("Use Selected Browser Asset"))
                {
                    _taxiActorModelOverrideInput = selectedBrowserModelPath.Replace('/', '\\');
                    _taxiActorModelOverrideInputRouteId = routeId;
                    ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
                    RefreshSelectedTaxiInfo();
                }

                ImGui.SameLine();
                ImGui.TextDisabled(Path.GetFileName(selectedBrowserModelPath));
            }

            if (TryGetLoadedTaxiActorModelPath(out string loadedModelPath))
            {
                if (ImGui.Button("Use Loaded Model"))
                {
                    _taxiActorModelOverrideInput = loadedModelPath;
                    _taxiActorModelOverrideInputRouteId = routeId;
                    ApplyTaxiActorModelOverride(routeId, loadedModelPath);
                    RefreshSelectedTaxiInfo();
                }

                ImGui.SameLine();
                ImGui.TextDisabled(Path.GetFileName(loadedModelPath));
            }

            if (!string.IsNullOrWhiteSpace(actorOverridePath))
            {
                if (ImGui.Button("Copy Override Path"))
                    CopyTextToClipboard(actorOverridePath, "override path");

                ImGui.SameLine();
                if (ImGui.Button("Open Override Asset"))
                    LoadFileFromDataSource(actorOverridePath);

                if (HasWorldReturnTarget() && _worldScene == null)
                {
                    ImGui.SameLine();
                    if (ImGui.Button("Return To Last World"))
                        ReturnToLastWorldScene();
                }
            }
        }
        else if (_worldScene.SelectedTaxiNodeId >= 0)
        {
            ImGui.TextDisabled($"Selected taxi node: {_worldScene.SelectedTaxiNodeId}");
            ImGui.TextDisabled("No connected routes were found for this taxi node.");
        }
        else if (_worldScene.SelectedTaxiRouteId >= 0)
        {
            ImGui.TextDisabled($"Selected taxi route: {_worldScene.SelectedTaxiRouteId}");
        }
    }

    private void DrawSelectedWmoControls()
    {
        if (_worldScene == null || _worldScene.SelectedObjectType != Terrain.ObjectType.Wmo || !_worldScene.SelectedInstance.HasValue)
            return;

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        string normalizedKey = WorldAssetManager.NormalizeKey(selected.ModelPath);
        WmoRenderer? wmoRenderer = _worldScene.Assets.GetWmo(normalizedKey);
        if (wmoRenderer == null)
        {
            ImGui.Separator();
            ImGui.TextDisabled("Selected WMO controls unavailable: renderer not loaded.");
            return;
        }

        ImGui.Separator();
        ImGui.Text("Selected WMO Controls");
        ImGui.TextDisabled("Changes apply to all loaded instances of this WMO model.");

        if (wmoRenderer.DoodadSetCount > 0)
        {
            ImGui.Text("Doodad Set:");
            int activeSet = wmoRenderer.ActiveDoodadSet;
            string currentSetName = wmoRenderer.GetDoodadSetName(activeSet);
            if (ImGui.BeginCombo("##SelectedWmoDoodadSet", currentSetName))
            {
                for (int setIndex = 0; setIndex < wmoRenderer.DoodadSetCount; setIndex++)
                {
                    bool selectedSet = setIndex == activeSet;
                    if (ImGui.Selectable(wmoRenderer.GetDoodadSetName(setIndex), selectedSet))
                        wmoRenderer.SetActiveDoodadSet(setIndex);
                    if (selectedSet)
                        ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
        }

        ImGui.Text("Groups / Doodads:");
        DrawRendererVisibilityControls(wmoRenderer, "selected_wmo");
    }

    private void DrawRendererVisibilityControls(ISceneRenderer renderer, string idSuffix)
    {
        if (ImGui.SmallButton($"All On##{idSuffix}"))
        {
            for (int i = 0; i < renderer.SubObjectCount; i++)
                renderer.SetSubObjectVisible(i, true);
        }

        ImGui.SameLine();
        if (ImGui.SmallButton($"All Off##{idSuffix}"))
        {
            for (int i = 0; i < renderer.SubObjectCount; i++)
                renderer.SetSubObjectVisible(i, false);
        }

        ImGui.TextDisabled($"Entries: {renderer.SubObjectCount}");
        float listHeight = MathF.Min(220f, MathF.Max(110f, GetUniformListRowHeight() * Math.Min(renderer.SubObjectCount, 8)));
        if (!ImGui.BeginChild($"##SubObjectVisibility_{idSuffix}", new Vector2(0, listHeight), true))
        {
            ImGui.EndChild();
            return;
        }

        float rowHeight = GetUniformListRowHeight();
        GetVisibleListRange(renderer.SubObjectCount, rowHeight, out int startIndex, out int endIndex);
        if (startIndex > 0)
            ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

        for (int i = startIndex; i < endIndex; i++)
        {
            bool visible = renderer.GetSubObjectVisible(i);
            string label = $"{renderer.GetSubObjectName(i)}##subobj_{idSuffix}_{i}";
            if (ImGui.Checkbox(label, ref visible))
                renderer.SetSubObjectVisible(i, visible);
        }

        if (endIndex < renderer.SubObjectCount)
            ImGui.Dummy(new Vector2(0, (renderer.SubObjectCount - endIndex) * rowHeight));

        ImGui.EndChild();
    }

    private void FrameCurrentModel()
    {
        if (_renderer is IModelRenderer modelRenderer)
        {
            var bmin = modelRenderer.BoundsMin;
            var bmax = modelRenderer.BoundsMax;
            FrameBounds(bmin, bmax, mdxMirrorX: true);
        }
        else if (_renderer is WmoRenderer wmoR)
        {
            FrameBounds(wmoR.BoundsMin, wmoR.BoundsMax, mdxMirrorX: false);
        }
    }

    private void FrameBounds(Vector3 boundsMin, Vector3 boundsMax, bool mdxMirrorX)
    {
        var center = (boundsMin + boundsMax) * 0.5f;
        var extent = boundsMax - boundsMin;
        float radius = MathF.Max(extent.Length() * 0.5f, 1f);

        if (mdxMirrorX)
            center.X = -center.X;

        float dist = MathF.Max(radius * 3.0f, 10f);
        _camera.Position = center + new Vector3(-dist, 0, radius * 0.6f);
        _camera.Yaw = 0f;
        _camera.Pitch = -15f;
    }

    private void DrawTerrainControlsAdjustmentContent()
    {
        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (lighting == null || renderer == null) return;

        float gameTime = lighting.GameTime;
        if (ImGui.SliderFloat("Time of Day", ref gameTime, 0f, 1f, "%.2f"))
            lighting.GameTime = gameTime;
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

        float fogStart = lighting.FogStart;
        float fogEnd = lighting.FogEnd;
        if (ImGui.SliderFloat("Fog Start", ref fogStart, 0f, 2000f))
            lighting.FogStart = fogStart;
        if (ImGui.SliderFloat("Fog End", ref fogEnd, 100f, 5000f))
            lighting.FogEnd = fogEnd;

        if (_worldScene != null)
        {
            bool showWdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("Show WDL Far Terrain", ref showWdl))
                _worldScene.ShowWdlTerrain = showWdl;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Toggle low-detail WDL background terrain for testing terrain overlap issues.");

            bool showObjects = _worldScene.ObjectsVisible;
            if (ImGui.Checkbox("Show Scene Objects", ref showObjects))
                _worldScene.ObjectsVisible = showObjects;

            bool showWmos = _worldScene.WmosVisible;
            if (ImGui.Checkbox("Show WMOs", ref showWmos))
                _worldScene.WmosVisible = showWmos;
            ImGui.SameLine();
            bool showDoodads = _worldScene.DoodadsVisible;
            if (ImGui.Checkbox("Show Doodads", ref showDoodads))
                _worldScene.DoodadsVisible = showDoodads;

            int visibilityProfileIndex = (int)_worldScene.ObjectVisibilityProfile;
            if (ImGui.Combo("Object Detail", ref visibilityProfileIndex, WorldObjectVisibilityProfileLabels, WorldObjectVisibilityProfileLabels.Length))
                _worldScene.ObjectVisibilityProfile = (WorldObjectVisibilityProfile)visibilityProfileIndex;

            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Quality keeps more far objects alive. Performance culls tiny projected objects and skips low-value off-view loads.");
        }

        if (renderer.ShowContours)
        {
            ImGui.Separator();
            float interval = renderer.ContourInterval;
            if (ImGui.SliderFloat("Contour Interval", ref interval, 0.5f, 20.0f, "%.1f"))
                renderer.ContourInterval = interval;
        }

        ImGui.Separator();
        if (ImGui.Button("Toggle Wireframe"))
            _renderer?.ToggleWireframe();
    }

    private void DrawRuntimeStatsPanelContent()
    {
        int tiles = _terrainManager?.LoadedTileCount ?? _vlmTerrainManager?.LoadedTileCount ?? 0;
        int chunks = _terrainManager?.LoadedChunkCount ?? _vlmTerrainManager?.LoadedChunkCount ?? 0;
        var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (terrainRenderer != null)
            ImGui.Text($"Tiles: {tiles}  Chunks: {terrainRenderer.ChunksRendered}/{chunks}");
        else if (_terrainManager != null || _vlmTerrainManager != null)
            ImGui.Text($"Tiles: {tiles}  Chunks: {chunks}");

        if (_worldScene == null)
        {
            if (_terrainManager == null && _vlmTerrainManager == null)
                ImGui.TextDisabled("Load a world or terrain scene to view runtime stats.");
            return;
        }

        ImGui.Text($"WMO: {_worldScene.WmoRenderedCount}/{_worldScene.WmoInstanceCount}  MDX: {_worldScene.MdxRenderedCount}/{_worldScene.MdxInstanceCount}");
        ImGui.Text($"Asset queue: {_worldScene.Assets.PendingAssetLoadCount}  WMO ok/fail: {_worldScene.Assets.WmoModelsLoaded}/{_worldScene.Assets.WmoModelsFailed}  MDX ok/fail: {_worldScene.Assets.MdxModelsLoaded}/{_worldScene.Assets.MdxModelsFailed}");

        var renderStats = _worldScene.LastRenderFrameStats;
        LiquidRenderer? renderStatsLiquidRenderer = _terrainManager?.LiquidRenderer;
        ImGui.TextDisabled("World render CPU only. UI/layout/input/swap are not included.");
        ImGui.Text($"World CPU: {renderStats.TotalCpuMs:0.00} ms  Pending asset loads: {renderStats.PendingAssetLoadCount}");
        ImGui.Text($"Visible WMO: {renderStats.VisibleWmoCount}  Visible MDX: {renderStats.VisibleMdxCount}  Taxi actors: {renderStats.VisibleTaxiMdxCount}");
        ImGui.Text($"Object stream range: {_worldScene.ObjectStreamingRangeMultiplier:0.00}x");
        ImGui.Text($"Object detail: {_worldScene.ObjectVisibilityProfile}");
        ImGui.Text($"Terrain chunks: {renderStats.TerrainChunksRendered}  WDL tiles: {renderStats.WdlVisibleTileCount}");
        if (terrainRenderer != null)
            ImGui.Text($"Terrain draw/uniform/tex-bind: {terrainRenderer.LastFrameDrawCalls}/{terrainRenderer.LastFrameUniform1Calls}/{terrainRenderer.LastFrameBindTextureCalls}");
        ImGui.Text($"Deferred/taxi/light: {renderStats.DeferredAssetLoads.DurationMs:0.00} / {renderStats.TaxiActorUpdate.DurationMs:0.00} / {renderStats.Lighting.DurationMs:0.00} ms");
        ImGui.Text($"WDL/terrain/liquid: {renderStats.Wdl.DurationMs:0.00} / {renderStats.Terrain.DurationMs:0.00} / {renderStats.Liquid.DurationMs:0.00} ms");
        if (renderStatsLiquidRenderer != null)
            ImGui.Text($"Liquid visible: {renderStatsLiquidRenderer.LastVisibleTerrainMeshCount}/{renderStatsLiquidRenderer.MeshCount}  WL: {renderStatsLiquidRenderer.LastVisibleWlMeshCount}/{renderStatsLiquidRenderer.WlMeshCount}");
        ImGui.Text($"WMO vis/draw: {renderStats.WmoVisibility.DurationMs:0.00} / {renderStats.WmoSubmission.DurationMs:0.00} ms");
        ImGui.Text($"MDX anim/vis/opaque: {renderStats.MdxAnimation.DurationMs:0.00} / {renderStats.MdxVisibility.DurationMs:0.00} / {renderStats.MdxOpaqueSubmission.DurationMs:0.00} ms");
        ImGui.Text($"MDX sort/trans: {renderStats.MdxTransparentSort.DurationMs:0.00} / {renderStats.MdxTransparentSubmission.DurationMs:0.00} ms");
        ImGui.Text($"MDX opaque shared/unbatched: {renderStats.OpaqueBatchedMdxCount}/{renderStats.OpaqueUnbatchedMdxCount}  transparent shared/unbatched: {renderStats.TransparentBatchedMdxCount}/{renderStats.TransparentUnbatchedMdxCount}");
        ImGui.Text($"Sky/backdrop/overlay: {renderStats.Sky.DurationMs:0.00} / {renderStats.SkyboxBackdrop.DurationMs:0.00} / {renderStats.Overlay.DurationMs:0.00} ms");
        ImGui.TextWrapped(_worldScene.RendererOptimizationHint);

        var assetReadStats = _worldScene.Assets.GetReadStats();
        ImGui.Separator();
        ImGui.Text($"Asset I/O req/cache: {assetReadStats.ReadRequests}/{assetReadStats.FileCacheHits}  resolved-cache: {assetReadStats.ResolvedPathCacheHits}  probes hit/miss: {assetReadStats.PathProbeResolutions}/{assetReadStats.PathProbeMisses}");
        ImGui.Text($"Asset misses: failed retry suppress={_worldScene.Assets.SuppressedFailedMdxRetryCount}  known missing M2 skins={_worldScene.Assets.KnownMissingM2SkinCount}  duplicate skin logs={_worldScene.Assets.SuppressedMissingM2SkinLogCount}");

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            var mpqStats = mpqDataSource.GetStatsSnapshot();
            ImGui.Text($"MPQ I/O read cache/miss: {mpqStats.ReadCacheHits}/{mpqStats.ReadCacheMisses}  loose/alpha/mpq/miss: {mpqStats.ReadLooseHits}/{mpqStats.ReadAlphaHits}/{mpqStats.ReadMpqHits}/{mpqStats.ReadMisses}  uncached avg: {mpqStats.AverageUncachedReadMs:0.00} ms");
            ImGui.Text($"MPQ prefetch enq/done/dup/cache: {mpqStats.PrefetchEnqueued}/{mpqStats.PrefetchCompleted}/{mpqStats.PrefetchDuplicateSkips}/{mpqStats.PrefetchCacheSkips}  queue avg: {mpqStats.AveragePrefetchQueueMs:0.00} ms  read avg: {mpqStats.AveragePrefetchReadMs:0.00} ms");
        }
    }

    private void DrawTerrainControlsContent()
    {
        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (lighting == null || renderer == null) return;

        DrawTerrainControlsAdjustmentContent();

        ImGui.Separator();
        DrawRuntimeStatsPanelContent();

        ImGui.Separator();
        if (ImGui.Button("Open Chunk Clipboard"))
            _showChunkClipboardWindow = true;
        ImGui.SameLine();
        ImGui.TextDisabled("Chunk copy/paste now lives in its own dockable panel.");
    }

    private bool SetIgnoreTerrainHolesGlobally(bool enabled)
    {
        bool changed = false;

        if (_terrainManager != null && _terrainManager.IgnoreTerrainHolesGlobally != enabled)
        {
            _terrainManager.IgnoreTerrainHolesGlobally = enabled;
            changed = true;
        }

        if (_vlmTerrainManager != null && _vlmTerrainManager.IgnoreTerrainHolesGlobally != enabled)
        {
            _vlmTerrainManager.IgnoreTerrainHolesGlobally = enabled;
            changed = true;
        }

        return changed;
    }

    private void DrawChunkClipboardWindow()
    {
        var renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
        {
            _showChunkClipboardWindow = false;
            return;
        }

        ImGui.SetNextWindowSize(new Vector2(420f, 0f), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Chunk Clipboard", ref _showChunkClipboardWindow))
        {
            ImGui.End();
            return;
        }

        DrawChunkClipboardContent(renderer);
        ImGui.End();
    }

    private void DrawChunkClipboardContent(TerrainRenderer renderer)
    {
        ImGui.Checkbox("Enable Chunk Tool", ref _chunkToolEnabled);
        ImGui.SameLine();
        ImGui.Checkbox("Show Overlay", ref _chunkClipboardShowOverlay);

        ImGui.TextDisabled("Shift+LMB: toggle selection | Ctrl+LMB: lock paste target | Ctrl+C/Ctrl+V: copy/paste");

        ImGui.Checkbox("Copy Target: Use Mouse", ref _chunkClipboardUseMouse);
        ImGui.Checkbox("Paste Relative Heights", ref _chunkClipboardPasteRelativeHeights);
        ImGui.Checkbox("Include Alpha/Shadow", ref _chunkClipboardIncludeAlphaShadow);
        ImGui.Checkbox("Include Textures", ref _chunkClipboardIncludeTextures);

        ImGui.SetNextItemWidth(160f);
        string[] rotLabels = { "0°", "90°", "180°", "270°" };
        ImGui.Combo("Paste Rotation", ref _chunkClipboardSelectionRotation, rotLabels, rotLabels.Length);

        ImGui.SameLine();
        if (ImGui.SmallButton("Clear Locked Target##chunkTargetClear"))
        {
            _chunkClipboardLockedTargetKey = null;
            _chunkClipboardStatus = "Cleared locked paste target.";
        }

        ImGui.TextDisabled($"Selected: {_selectedChunks.Count}");
        if (_selectedChunks.Count > 0)
        {
            ImGui.SameLine();
            if (ImGui.SmallButton("Clear##chunkSelClear"))
                _selectedChunks.Clear();
        }

        if (_chunkClipboardLockedTargetKey is { } locked)
            ImGui.Text($"Locked Paste Target: tile({locked.tileX},{locked.tileY}) chunk({locked.chunkX},{locked.chunkY})");
        else
            ImGui.TextDisabled("Locked Paste Target: (none)  (Ctrl+LMB to set)");

        var targetChunk = GetChunkClipboardTarget(renderer);
        bool hasChunk = targetChunk.HasValue;
        string targetLabel = _chunkClipboardUseMouse ? "Mouse" : "Camera";
        if (targetChunk is { } c)
        {
            ImGui.TextDisabled($"Copy Target ({targetLabel}): tile({c.TileX},{c.TileY}) chunk({c.ChunkX},{c.ChunkY})");
        }
        else
        {
            ImGui.TextDisabled($"Copy Target ({targetLabel}): (none loaded)");
        }

        if (!hasChunk) ImGui.BeginDisabled();
        if (ImGui.Button(_selectedChunks.Count > 0 ? "Copy Selection" : "Copy Chunk"))
        {
            if (_selectedChunks.Count > 0)
                CopySelectedChunks(renderer);
            else
                CopyChunkAtTarget(renderer);
        }
        if (!hasChunk) ImGui.EndDisabled();

        ImGui.SameLine();
        bool canPaste = (_chunkClipboardSet != null || _chunkClipboard != null);
        if (!canPaste) ImGui.BeginDisabled();
        if (ImGui.Button(_chunkClipboardSet != null ? "Paste Selection" : "Paste Chunk"))
        {
            if (_chunkClipboardSet != null)
                PasteClipboardSetAtTarget(renderer);
            else
                PasteChunkAtTarget(renderer);
        }
        if (!canPaste) ImGui.EndDisabled();

        if (!string.IsNullOrWhiteSpace(_chunkClipboardStatus))
            ImGui.TextWrapped(_chunkClipboardStatus);
    }

    private void DrawWorldObjectsContent()
    {
        // Intentionally moved as-is into a partial file to keep ViewerApp.cs manageable.
        // The implementation remains unchanged and still lives in this partial class.
        DrawWorldObjectsContentCore();
    }

    private static float GetUniformListRowHeight()
    {
        return MathF.Max(ImGui.GetTextLineHeightWithSpacing(), ImGui.GetFrameHeightWithSpacing());
    }

    private static void GetVisibleListRange(int itemCount, float rowHeight, out int startIndex, out int endIndex)
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

    private static void DrawWmoLiquidRotationControls(string idSuffix)
    {
        int quarterTurns = WmoRenderer.MliqRotationQuarterTurns;
        string currentLabel = WmoLiquidRotationLabels[Math.Clamp(quarterTurns, 0, WmoLiquidRotationLabels.Length - 1)];

        if (ImGui.BeginCombo($"WMO MLIQ Rotation##{idSuffix}", currentLabel))
        {
            for (int i = 0; i < WmoLiquidRotationLabels.Length; i++)
            {
                bool selected = i == quarterTurns;
                if (ImGui.Selectable(WmoLiquidRotationLabels[i], selected))
                    WmoRenderer.MliqRotationQuarterTurns = i;
                if (selected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        ImGui.TextDisabled("Applies to all WMO MLIQ surfaces. Changes are live.");
    }
}