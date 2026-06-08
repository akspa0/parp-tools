using System.Numerics;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

/// <summary>
/// Partial class containing the large sidebar and inspector UI blocks.
/// </summary>
public partial class ViewerApp
{
    private static int CountEnabled(params bool[] values)
    {
        int count = 0;
        foreach (bool value in values)
        {
            if (value)
                count++;
        }

        return count;
    }

    private static float MeasureToolbarCheckboxWidth(string label)
    {
        var style = ImGui.GetStyle();
        return ImGui.GetFrameHeight() + style.ItemInnerSpacing.X + ImGui.CalcTextSize(label).X;
    }

    private static float MeasureToolbarSeparatorWidth()
    {
        var style = ImGui.GetStyle();
        return ImGui.CalcTextSize("|").X + style.ItemSpacing.X * 2f;
    }

    private bool HasLoadedContent()
    {
        return _terrainManager != null
            || _vlmTerrainManager != null
            || _worldScene != null
            || _loadedWmo != null
            || _loadedMdx != null
            || !string.IsNullOrWhiteSpace(_loadedFilePath);
    }

    private void DrawToolbarPopupButton(string label, string summary, string popupId, Action drawContent)
    {
        string buttonLabel = string.IsNullOrWhiteSpace(summary)
            ? label
            : $"{label} {summary}";

        if (ImGui.Button(buttonLabel))
            ImGui.OpenPopup(popupId);

        if (ImGui.BeginPopup(popupId))
        {
            drawContent();
            ImGui.EndPopup();
        }
    }

    private float GetDirectTerrainToolbarWidth(TerrainRenderer renderer, LiquidRenderer? liquidRenderer)
    {
        float width = 0f;
        width += MeasureToolbarCheckboxWidth("Base");
        width += MeasureToolbarCheckboxWidth("L1");
        width += MeasureToolbarCheckboxWidth("L2");
        width += MeasureToolbarCheckboxWidth("L3");
        width += MeasureToolbarCheckboxWidth("Holes");
        width += MeasureToolbarSeparatorWidth();
        width += MeasureToolbarCheckboxWidth("Chunks");
        width += MeasureToolbarCheckboxWidth("Tiles");
        width += MeasureToolbarSeparatorWidth();
        width += MeasureToolbarCheckboxWidth("Alpha");
        width += MeasureToolbarCheckboxWidth("Shadows");
        width += MeasureToolbarCheckboxWidth("MCCV");
        width += MeasureToolbarCheckboxWidth("Contours");

        if (liquidRenderer != null || _worldScene != null)
        {
            width += MeasureToolbarSeparatorWidth();
            if (liquidRenderer != null)
                width += MeasureToolbarCheckboxWidth("Liquid");
            if (_worldScene != null)
            {
                width += MeasureToolbarCheckboxWidth("WL*");
                width += MeasureToolbarCheckboxWidth("WDL");
                width += MeasureToolbarCheckboxWidth("BBs");
                width += MeasureToolbarCheckboxWidth("PM4");
            }
        }

        return width;
    }

    private void DrawDirectTerrainToolbarControls(TerrainRenderer renderer, LiquidRenderer? liquidRenderer)
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
        ImGui.SameLine();
        bool contours = renderer.ShowContours;
        if (ImGui.Checkbox("Contours", ref contours)) renderer.ShowContours = contours;

        if (liquidRenderer != null || _worldScene != null)
        {
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
            ImGui.SameLine();
        }

        if (liquidRenderer != null)
        {
            bool showLiquid = liquidRenderer.ShowLiquid;
            if (ImGui.Checkbox("Liquid", ref showLiquid))
                liquidRenderer.ShowLiquid = showLiquid;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip($"Liquid terrain meshes: {liquidRenderer.MeshCount}");
        }

        if (_worldScene != null)
        {
            if (liquidRenderer != null)
                ImGui.SameLine();

            int wlCount = liquidRenderer?.WlMeshCount ?? 0;
            bool showWlTop = _worldScene.ShowWlLiquids;
            if (ImGui.Checkbox("WL*", ref showWlTop))
                _worldScene.ShowWlLiquids = showWlTop;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip($"World liquid overlays: {wlCount}");

            ImGui.SameLine();
            bool showWdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("WDL", ref showWdl))
                _worldScene.ShowWdlTerrain = showWdl;

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
            else if (_worldScene.ShowPm4Overlay && ImGui.IsItemHovered())
            {
                ImGui.SetTooltip(_worldScene.Pm4Status);
            }
        }
    }

    private void DrawCenteredTerrainToolbarWindow(TerrainRenderer renderer, LiquidRenderer? liquidRenderer)
    {
        float laneX = 0f;
        float laneWidth = ImGui.GetIO().DisplaySize.X;
        if (TryGetSceneViewportRect(out float viewportX, out _, out float viewportWidth, out _))
        {
            laneX = viewportX;
            laneWidth = viewportWidth;
        }

        if (laneWidth <= 10f)
            return;

        ImGui.SetNextWindowPos(new Vector2(laneX, MenuBarHeight));
        ImGui.SetNextWindowSize(new Vector2(laneWidth, ToolbarHeight));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(8, 6));
        ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, new Vector2(6, 0));
        if (ImGui.Begin("##CenteredTerrainToolbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar |
            ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoBackground))
        {
            float contentWidth = GetDirectTerrainToolbarWidth(renderer, liquidRenderer);
            float startX = MathF.Max(8f, (laneWidth - contentWidth) * 0.5f);
            ImGui.SetCursorPosX(startX);
            DrawDirectTerrainToolbarControls(renderer, liquidRenderer);
        }
        ImGui.End();
        ImGui.PopStyleVar(2);
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

            TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

            if (renderer != null)
            {
                DrawCenteredTerrainToolbarWindow(renderer, liquidRenderer);
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

    private void DrawWorkspaceBarsPanelContent()
    {
        ImGui.TextDisabled("P toggles this panel | I toggles the inspector set | M fullscreen minimap | Tab hides UI chrome");
        ImGui.Separator();

        ImGui.TextDisabled("Workspace");
        DrawWorkspaceToolbarControls();
        ImGui.Spacing();

        if (ImGui.Button("Open Game Folder..."))
        {
            _showFolderInput = true;
            _folderInputBuf = string.IsNullOrWhiteSpace(_lastGameFolderPath) ? "" : _lastGameFolderPath;
        }

        ImGui.SameLine();
        if (ImGui.Button("Open File..."))
            _wantOpenFile = true;

        if (_dataSource != null)
            ImGui.TextColored(new Vector4(0.70f, 0.78f, 0.88f, 1f), $"Source: {_dataSource.Name}");

        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;
        if (renderer == null)
        {
            ImGui.Spacing();
            ImGui.TextWrapped("Load a terrain-backed world to populate the display bars. Standalone model and WMO inspection still works through the navigator and selection panels.");
            return;
        }

        ImGui.Spacing();
        ImGui.TextDisabled("Terrain Layers");
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

        ImGui.SameLine();
        bool chunkGrid = renderer.ShowChunkGrid;
        if (ImGui.Checkbox("Chunks", ref chunkGrid)) renderer.ShowChunkGrid = chunkGrid;
        ImGui.SameLine();
        bool tileGrid = renderer.ShowTileGrid;
        if (ImGui.Checkbox("Tiles", ref tileGrid)) renderer.ShowTileGrid = tileGrid;

        ImGui.Spacing();
        ImGui.TextDisabled("Overlays");
        bool alphaMask = renderer.ShowAlphaMask;
        if (ImGui.Checkbox("Alpha", ref alphaMask)) renderer.ShowAlphaMask = alphaMask;
        ImGui.SameLine();
        bool shadowMap = renderer.ShowShadowMap;
        if (ImGui.Checkbox("Shadows", ref shadowMap)) renderer.ShowShadowMap = shadowMap;
        ImGui.SameLine();
        bool useMccv = renderer.UseMccv;
        if (ImGui.Checkbox("MCCV", ref useMccv)) renderer.UseMccv = useMccv;
        ImGui.SameLine();
        bool contours = renderer.ShowContours;
        if (ImGui.Checkbox("Contours", ref contours)) renderer.ShowContours = contours;

        ImGui.Spacing();
        ImGui.TextDisabled("World");
        if (liquidRenderer != null)
        {
            bool showLiquid = liquidRenderer.ShowLiquid;
            if (ImGui.Checkbox($"Liquid Terrain ({liquidRenderer.MeshCount})", ref showLiquid))
                liquidRenderer.ShowLiquid = showLiquid;
        }

        if (_worldScene != null)
        {
            int wlCount = liquidRenderer?.WlMeshCount ?? 0;
            bool showWlTop = _worldScene.ShowWlLiquids;
            if (ImGui.Checkbox($"WL* ({wlCount})", ref showWlTop))
                _worldScene.ShowWlLiquids = showWlTop;

            ImGui.SameLine();
            bool showWdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("WDL", ref showWdl))
                _worldScene.ShowWdlTerrain = showWdl;

            bool showBB = _worldScene.ShowBoundingBoxes;
            if (ImGui.Checkbox("Bounding Boxes", ref showBB))
                _worldScene.ShowBoundingBoxes = showBB;

            ImGui.SameLine();
            bool showPm4 = _worldScene.ShowPm4Overlay;
            if (ImGui.Checkbox("PM4 Overlay", ref showPm4))
                _worldScene.ShowPm4Overlay = showPm4;

            if (_worldScene.IsPm4Loading)
                ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), "PM4 overlay is loading...");
        }
    }

    private void DrawLeftSidebar()
    {
        if (!HasAnyShellPanelsInLane(ShellPanelLane.Left))
            return;

        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
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
        {
            DrawFixedSidebarWidthControl(
                "Navigator Width",
                ref _leftSidebarWidth,
                isLeftSidebar: true,
                io.DisplaySize.X,
                "Resize the fixed navigator without relying on the edge splitter.");
            DrawNavigatorPanelContent();
        }
        ImGui.End();
        ImGui.PopStyleVar();
    }

    private void DrawNavigatorPanelContent()
    {
        bool hasWorldLoaded = _worldScene != null || _terrainManager != null || _vlmTerrainManager != null;

        if (hasWorldLoaded)
        {
            ImGui.SetNextItemOpen(true, ImGuiCond.Once);
            if (ImGui.CollapsingHeader("World Overview", ImGuiTreeNodeFlags.DefaultOpen))
                DrawWorldOverviewContent();
        }

        ImGui.SetNextItemOpen(!hasWorldLoaded, ImGuiCond.Once);
        if (_showFileBrowser && ImGui.CollapsingHeader("File Browser", hasWorldLoaded ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None))
            DrawFileBrowserContent(hasWorldLoaded ? 260f : 0f);

        if (_autoOpenWorldMapsPanel)
            ImGui.SetNextItemOpen(true, ImGuiCond.Always);

        if (_discoveredMaps.Count > 0 && ImGui.CollapsingHeader("World Maps"))
            DrawMapDiscoveryContent();

        if (hasWorldLoaded)
        {
            ImGui.SetNextItemOpen(true, ImGuiCond.Once);
            if (ImGui.CollapsingHeader("Runtime Stats", ImGuiTreeNodeFlags.DefaultOpen))
                DrawRuntimeStatsPanelContent();
        }
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

        if (TryGetActiveMinimapState(out var existingTiles, out var isTileLoaded, out int loadedTileCount, out string? mapName))
        {
            float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / MinimapWorldTileSize;
            float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / MinimapWorldTileSize;
            ClampMinimapPanOffset();
            int ctX = (int)MathF.Floor(camTileX);
            int ctY = (int)MathF.Floor(camTileY);

            ImGui.TextDisabled($"Tile: ({ctX}, {ctY})  Loaded: {loadedTileCount}");
            if (_minimapRenderer != null && (_minimapRenderer.IsBusy || _minimapRenderer.UploadedTileCount > 0 || _minimapRenderer.FailedTileCount > 0))
            {
                float progress = _minimapRenderer.LoadingProgress;
                string overlay = _minimapRenderer.IsBusy
                    ? $"Minimap {progress * 100f:F0}%  {_minimapRenderer.PendingTileCount} pending"
                    : $"Minimap ready  {_minimapRenderer.UploadedTileCount} tiles";
                ImGui.ProgressBar(progress, new Vector2(MathF.Min(220f, ImGui.GetContentRegionAvail().X), 0f), overlay);
                if (_minimapRenderer.FailedTileCount > 0)
                    ImGui.TextDisabled($"Missing or failed tiles: {_minimapRenderer.FailedTileCount}");
            }

            float mapSize = ComputeMinimapSquareSize(ImGui.GetContentRegionAvail().X, 220f, 140f);
            var cursorPos = ImGui.GetCursorScreenPos();
            DrawInteractiveMinimapSurface(
                "##sidebarMinimapInteraction",
                cursorPos,
                mapSize,
                existingTiles,
                isTileLoaded,
                mapName,
                MinimapTeleportMode.Armed,
                out _,
                out _,
                out _);
            ImGui.SetCursorPosY(ImGui.GetCursorPosY() + mapSize + 4f);
        }

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

        if (ImGui.Button(_fullscreenMinimap ? "Exit Full Minimap" : "Full Minimap"))
            ToggleFullscreenMinimap();

        ImGui.SameLine();
        if (ImGui.Button(_showMinimapWindow ? "Hide Pop-out" : "Pop Out"))
            _showMinimapWindow = !_showMinimapWindow;

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

    private void DrawFileBrowserContent(float reservedFooterHeight = 0f)
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

        float remainingH = ImGui.GetContentRegionAvail().Y - reservedFooterHeight;
        if (_discoveredMaps.Count > 0)
            remainingH = MathF.Max(remainingH - 360f, 100f);
        else
            remainingH = MathF.Max(remainingH, 100f);
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
        float topOffset = GetTopChromeHeight();
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
            DrawFixedSidebarWidthControl(
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

    private void DrawUnifiedToolSidebar()
    {
        DrawRightSidebarSection(FixedBottomDrawerTab.Workspace, "Viewer Settings", DrawUnifiedViewerSettingsSidebarContent, defaultOpen: true);
        DrawRightSidebarSection(FixedBottomDrawerTab.World, "Selection", DrawUnifiedSelectionSidebarContent, defaultOpen: true);
        DrawRightSidebarSection(FixedBottomDrawerTab.Pm4, "World Tools", DrawUnifiedWorldToolsSidebarContent, _worldScene != null, defaultOpen: false);
        DrawRightSidebarSection(FixedBottomDrawerTab.Diagnostics, "Utilities", DrawViewerDiagnosticsSidebarContent, defaultOpen: false);
    }

    private void DrawUnifiedViewerSettingsSidebarContent()
    {
        ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
        ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
        ImGui.Separator();
        DrawUiThemeSettingsContent();
        ImGui.Separator();
        DrawCameraControlsContent();

        if (_terrainManager != null || _vlmTerrainManager != null)
        {
            ImGui.Separator();
            DrawTerrainControlsAdjustmentContent();
        }
    }

    private void DrawUnifiedSelectionSidebarContent()
    {
        DrawViewerSelectionSummary();

        if (!string.IsNullOrWhiteSpace(_modelInfo))
        {
            ImGui.Separator();
            DrawModelInfoPanelContent();
        }
    }

    private void DrawUnifiedWorldToolsSidebarContent()
    {
        DrawWorldObjectsPanelContent();

        if (_worldScene != null)
        {
            ImGui.Separator();
            DrawPm4WorkbenchInspector();
        }
    }

    private void DrawViewerSelectionSummary()
    {
        bool hasSelectedPm4 = _worldScene?.HasSelectedPm4Object == true;
        bool hasSelectedObject = DrawSelectedObjectSummaryContent();
        if (!hasSelectedObject)
        {
            if (hasSelectedPm4)
            {
                ImGui.TextDisabled("A PM4 object is selected. Use the PM4 section for evidence and correlation.");
                if (ImGui.Button("Open PM4 Tools"))
                    OpenPm4Workbench(Pm4WorkbenchTab.Selection);
            }
            else
            {
                ImGui.TextDisabled("Select a world object to inspect its identity here.");
            }
        }
    }

    private void DrawRightSidebarSection(FixedBottomDrawerTab section, string label, Action drawContent, bool enabled = true, bool defaultOpen = false)
    {
        if (!enabled)
            return;

        bool shouldForceOpen = _pendingRightSidebarSection == section;
        ImGui.SetNextItemOpen(shouldForceOpen || defaultOpen, shouldForceOpen ? ImGuiCond.Always : ImGuiCond.Once);
        if (ImGui.CollapsingHeader(label, defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None))
        {
            _activeBottomDrawerTab = section;
            if (shouldForceOpen)
                _pendingRightSidebarSection = null;
            drawContent();
        }
    }

    private void DrawViewerInspectSidebarContent()
    {
        DrawCameraControlsContent();

        if (!string.IsNullOrWhiteSpace(_modelInfo))
        {
            ImGui.Separator();
            DrawModelInfoPanelContent();
        }
    }

    private void DrawViewerDiagnosticsSidebarContent()
    {
        if (_worldScene != null && !string.IsNullOrWhiteSpace(_worldScene.RendererOptimizationHint))
            ImGui.TextWrapped(_worldScene.RendererOptimizationHint);

        ImGui.Text("Utility Panels");
        DrawToolbarPopupButton("Utility Windows", $"{CountEnabled(_showMinimapWindow, _showLogViewer, _showPerfWindow, _showRenderQualityWindow)} open", "##UtilityWindowsPopup", () =>
        {
            if (ImGui.Button(_showMinimapWindow ? "Hide Minimap" : "Show Minimap"))
            {
                _showMinimapWindow = !_showMinimapWindow;
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button(_showLogViewer ? "Hide Log Viewer" : "Show Log Viewer"))
            {
                _showLogViewer = !_showLogViewer;
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button(_showPerfWindow ? "Hide Perf" : "Show Perf"))
            {
                _showPerfWindow = !_showPerfWindow;
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button(_showRenderQualityWindow ? "Hide Render Quality" : "Show Render Quality"))
            {
                _showRenderQualityWindow = !_showRenderQualityWindow;
                ImGui.CloseCurrentPopup();
            }
        });
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

            PrepareDockableShellPanelWindow(
                panel.Id,
                new Vector2(panel.DefaultWidth, defaultHeight),
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

    private void DrawFixedSidebarWidthControl(string label, ref float width, bool isLeftSidebar, float displayWidth, string tooltip)
    {
        GetFixedSidebarWidthRange(isLeftSidebar, displayWidth, out float minWidth, out float maxWidth);
        if (maxWidth <= minWidth)
            return;

        float updatedWidth = width;
        ImGui.SetNextItemWidth(-1f);
        if (ImGui.SliderFloat(label, ref updatedWidth, minWidth, maxWidth, "%.0f px"))
            width = ClampFixedSidebarWidth(updatedWidth, isLeftSidebar, displayWidth);

        if (ImGui.IsItemHovered())
            ImGui.SetTooltip(tooltip);

        if (ImGui.IsItemDeactivatedAfterEdit())
            SaveViewerSettings();

        ImGui.Separator();
    }

    private void DrawShellPanelContent(ShellPanelId panelId)
    {
        switch (panelId)
        {
            case ShellPanelId.WorkspaceBars:
                DrawWorkspaceBarsPanelContent();
                break;
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
        ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
        ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
        ImGui.Separator();

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
                SaveViewerSettings();

            ImGui.SameLine();
            if (ImGui.SmallButton("Auto"))
            {
                _terrainManager.DetailedTileCountOverride = 0;
                _savedDetailedAdtTileCountOverride = 0;
                SaveViewerSettings();
            }

            ImGui.TextDisabled(autoAdtBudget
                ? $"Auto from fog: {_terrainManager.EffectiveDetailedTileCount} detailed / {_terrainManager.EffectiveRetainedTileCount} retained"
                : $"Manual override: {_terrainManager.DetailedTileCountOverride} detailed / {_terrainManager.EffectiveRetainedTileCount} retained");
        }
    }

    private bool DrawSelectedObjectSummaryContent()
    {
        bool hasSelectedPm4 = _worldScene?.HasSelectedPm4Object == true;
        if (string.IsNullOrEmpty(_selectedObjectInfo)
            || hasSelectedPm4
            || _selectedObjectType.StartsWith("Taxi", StringComparison.OrdinalIgnoreCase))
            return false;

        ImGui.TextWrapped(_selectedObjectInfo);
        if (TryGetSelectedWorldObjectModelPath(out string selectedModelPath, out _))
        {
            ImGui.Separator();
            DrawAssetPathActions("Selected Asset", selectedModelPath, "SelectedWorldObject");
        }

        DrawSelectedWmoControls();
        DrawSelectedSqlGameObjectAnimationControls();
        return true;
    }

    private void DrawWmoDoodadInspector(WmoRenderer wmoRenderer, ref int selectedDoodadIndex, string idSuffix, Func<WmoDoodadInfo, bool>? frameDoodad)
    {
        ImGui.Separator();
        ImGui.Text("WMO Doodad Inspector");

        int doodadCount = wmoRenderer.DoodadInstanceCount;
        if (doodadCount <= 0)
        {
            selectedDoodadIndex = -1;
            ImGui.TextDisabled("The active doodad set has no resolved doodads.");
            return;
        }

        if (selectedDoodadIndex >= doodadCount)
            selectedDoodadIndex = -1;

        ImGui.TextDisabled($"Active set: {wmoRenderer.GetDoodadSetName(wmoRenderer.ActiveDoodadSet)}");
        ImGui.TextDisabled($"Doodads: {doodadCount}");

        float listHeight = MathF.Min(220f, MathF.Max(110f, GetUniformListRowHeight() * Math.Min(doodadCount, 7)));
        if (ImGui.BeginChild($"##WmoDoodadInspector_{idSuffix}", new Vector2(0, listHeight), true))
        {
            for (int doodadIndex = 0; doodadIndex < doodadCount; doodadIndex++)
            {
                if (!wmoRenderer.TryGetDoodadInfo(doodadIndex, out WmoDoodadInfo doodad))
                    continue;

                string label = $"[{doodadIndex}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}";
                if (!doodad.IsLoaded)
                    label += " [deferred]";
                if (!doodad.Visible)
                    label += " [hidden]";

                bool isSelected = doodadIndex == selectedDoodadIndex;
                if (ImGui.Selectable($"{label}##{idSuffix}_{doodadIndex}", isSelected))
                {
                    selectedDoodadIndex = doodadIndex;
                    frameDoodad?.Invoke(doodad);
                }

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip(doodad.ModelPath);
            }
        }
        ImGui.EndChild();

        if (selectedDoodadIndex < 0 || !wmoRenderer.TryGetDoodadInfo(selectedDoodadIndex, out WmoDoodadInfo selectedDoodad))
            return;

        if (frameDoodad != null && ImGui.SmallButton($"Frame Doodad##{idSuffix}_FrameDoodad"))
            frameDoodad(selectedDoodad);

        DrawAssetPathActions("Doodad Asset", selectedDoodad.ModelPath, $"{idSuffix}_DoodadAsset");
        ImGui.TextDisabled($"Def Index: {selectedDoodad.DoodadDefIndex}");
        ImGui.TextDisabled($"Visible: {(selectedDoodad.Visible ? "yes" : "no")}  Loaded: {(selectedDoodad.IsLoaded ? "yes" : "no")}");
        ImGui.TextDisabled($"Local: ({selectedDoodad.LocalPosition.X:F1}, {selectedDoodad.LocalPosition.Y:F1}, {selectedDoodad.LocalPosition.Z:F1})");
    }

    private void DrawObjectPathFilterControls()
    {
        if (_worldScene == null)
            return;

        ImGui.Separator();
        ImGui.Text("Object Path Filters");

        bool filtersEnabled = _worldScene.ObjectPathFiltersEnabled;
        if (ImGui.Checkbox("Enable Path Filters", ref filtersEnabled))
        {
            _worldScene.ObjectPathFiltersEnabled = filtersEnabled;
            PersistObjectPathFiltersForCurrentMap();
        }

        ImGui.SameLine();
        if (_worldScene.ObjectPathFilters.Count == 0)
            ImGui.BeginDisabled();
        if (ImGui.SmallButton("Clear All"))
        {
            _worldScene.ClearObjectPathFilters();
            PersistObjectPathFiltersForCurrentMap();
            _statusMessage = "Cleared object path filters for the current map.";
        }
        if (_worldScene.ObjectPathFilters.Count == 0)
            ImGui.EndDisabled();

        string filterInput = _objectPathFilterInput;
        if (ImGui.InputTextWithHint("Path Prefix", "World\\...", ref filterInput, 512))
            _objectPathFilterInput = filterInput;

        bool appliesToWmo = _objectPathFilterInputAppliesToWmo;
        if (ImGui.Checkbox("WMO##ObjectPathFilterWmo", ref appliesToWmo))
            _objectPathFilterInputAppliesToWmo = appliesToWmo;

        ImGui.SameLine();
        bool appliesToMdx = _objectPathFilterInputAppliesToMdx;
        if (ImGui.Checkbox("MDX##ObjectPathFilterMdx", ref appliesToMdx))
            _objectPathFilterInputAppliesToMdx = appliesToMdx;

        bool canAddFilter = !string.IsNullOrWhiteSpace(_objectPathFilterInput)
            && (_objectPathFilterInputAppliesToWmo || _objectPathFilterInputAppliesToMdx);
        if (!canAddFilter)
            ImGui.BeginDisabled();
        if (ImGui.Button("Add Filter"))
        {
            if (_worldScene.AddObjectPathFilter(_objectPathFilterInput, _objectPathFilterInputAppliesToWmo, _objectPathFilterInputAppliesToMdx))
            {
                PersistObjectPathFiltersForCurrentMap();
                _statusMessage = $"Added object path filter: {_objectPathFilterInput.Trim()}";
            }
            else
            {
                _statusMessage = "Object path filter was empty, duplicated, or had no enabled asset family.";
            }
        }
        if (!canAddFilter)
            ImGui.EndDisabled();

        if (TryGetSelectedWorldObjectModelPath(out string selectedModelPath, out bool selectedIsWmo))
        {
            ImGui.TextDisabled($"Selected: {selectedModelPath}");

            List<string> prefixCandidates = BuildObjectPathFilterPrefixCandidates(selectedModelPath);
            if (prefixCandidates.Count > 0 && ImGui.TreeNode("Quick Add From Selected Object"))
            {
                for (int i = 0; i < prefixCandidates.Count; i++)
                {
                    string prefix = prefixCandidates[i];
                    bool alreadyExists = _worldScene.ObjectPathFilters.Any(entry =>
                        string.Equals(entry.PathPrefix, prefix, StringComparison.OrdinalIgnoreCase)
                        && entry.AppliesToWmo == selectedIsWmo
                        && entry.AppliesToMdx == !selectedIsWmo);

                    if (alreadyExists)
                        ImGui.BeginDisabled();

                    if (ImGui.SmallButton($"{prefix}##QuickObjectPathFilter{i}")
                        && _worldScene.AddObjectPathFilter(prefix, selectedIsWmo, !selectedIsWmo))
                    {
                        PersistObjectPathFiltersForCurrentMap();
                        _statusMessage = $"Added {(selectedIsWmo ? "WMO" : "MDX")} family filter: {prefix}";
                    }

                    if (alreadyExists)
                        ImGui.EndDisabled();
                }

                ImGui.TreePop();
            }
        }

        if (_worldScene.ObjectPathFilters.Count == 0)
        {
            ImGui.TextDisabled("No path filters are saved for the current map.");
            return;
        }

        ImGui.TextDisabled($"Current map filters: {_worldScene.ObjectPathFilters.Count}");
        if (!ImGui.BeginTable("ObjectPathFiltersTable", 3, ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp))
            return;

        ImGui.TableSetupColumn("Family", ImGuiTableColumnFlags.WidthFixed, 84f);
        ImGui.TableSetupColumn("Prefix", ImGuiTableColumnFlags.WidthStretch);
        ImGui.TableSetupColumn("Action", ImGuiTableColumnFlags.WidthFixed, 72f);
        ImGui.TableHeadersRow();

        for (int i = 0; i < _worldScene.ObjectPathFilters.Count; i++)
        {
            ObjectPathFilterEntry entry = _worldScene.ObjectPathFilters[i];
            string familyLabel = entry.AppliesToWmo && entry.AppliesToMdx
                ? "WMO+MDX"
                : entry.AppliesToWmo
                    ? "WMO"
                    : "MDX";

            ImGui.TableNextRow();

            ImGui.TableSetColumnIndex(0);
            ImGui.TextUnformatted(familyLabel);

            ImGui.TableSetColumnIndex(1);
            ImGui.TextUnformatted(entry.PathPrefix);

            ImGui.TableSetColumnIndex(2);
            if (ImGui.SmallButton($"Remove##ObjectPathFilter{i}"))
            {
                _worldScene.RemoveObjectPathFilter(entry.PathPrefix, entry.AppliesToWmo, entry.AppliesToMdx);
                PersistObjectPathFiltersForCurrentMap();
                _statusMessage = $"Removed object path filter: {entry.PathPrefix}";
            }
        }

        ImGui.EndTable();
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
        ImGui.TextDisabled("Open terrain editor windows from the Tools menu.");
    }

    private void DrawTerrainToolsWindow()
    {
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
        {
            _showTerrainToolsWindow = false;
            return;
        }

        ImGui.SetNextWindowSize(new Vector2(560f, 0f), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Terrain Workbench", ref _showTerrainToolsWindow, ImGuiWindowFlags.NoCollapse))
        {
            ImGui.End();
            return;
        }

        ImGui.TextDisabled("Terrain workbench: tile targeting, chunk targeting, live restore tuning, and reusable heightmap saves in one place.");
        ImGui.Separator();
        DrawTerrainWorkbenchSelectionContent(renderer);
        ImGui.Separator();
        DrawTerrainControlsAdjustmentContent();

        ImGui.Separator();
        ImGui.Text("Terrain Export Scope");
        DrawTerrainTileScopeSelector("TerrainToolsExport", includeCurrentTile: true);
        var scopedTiles = GetTileScopeList(_terrainTileScope);
        ImGui.TextDisabled($"Resolved export scope: {scopedTiles.Count} tile(s).");

        ImGui.Separator();
        ImGui.Text("Scoped Export");
        ImGui.TextDisabled("Use Current tile, Loaded tiles, Whole map, Custom list, or a row/column rectangle before exporting partial ADT data.");
        if (ImGui.Button("Export Alpha"))
        {
            if (_terrainTileScope == TerrainTileScope.CurrentTile)
                ExportAlphaCurrentTileChunksFolder();
            else
                ExportAlphaTilesFolder(_terrainTileScope);
        }

        ImGui.SameLine();
        if (ImGui.Button("Export Heightmap"))
        {
            if (_terrainTileScope == TerrainTileScope.CurrentTile)
                ExportHeightmap257CurrentTilePerTile();
            else
                ExportHeightmap257TilesFolderPerTile(_terrainTileScope);
        }

        ImGui.SameLine();
        if (ImGui.Button("Export MCCV"))
        {
            if (_terrainTileScope == TerrainTileScope.CurrentTile)
                ExportMccvCurrentTilePng();
            else
                ExportMccvTilesFolder(_terrainTileScope);
        }

        ImGui.Separator();
        if (ImGui.CollapsingHeader("Clipboard + Save", ImGuiTreeNodeFlags.DefaultOpen))
            DrawChunkClipboardContent(renderer);
        ImGui.End();
    }

    private void DrawTerrainWorkbenchSelectionContent(TerrainRenderer renderer)
    {
        if (!TryGetActiveMinimapState(out var existingTiles, out var isTileLoaded, out int loadedTileCount, out string? mapName)
            || existingTiles == null
            || isTileLoaded == null)
        {
            ImGui.TextDisabled("Load a terrain-backed world to target tiles and chunks in the terrain workbench.");
            return;
        }

        ImGui.Text("Selection Map");
        ImGui.TextDisabled("LMB drag selects ADT tiles. RMB drag pans. Mouse wheel zooms. Click one tile to focus it for chunk-level work.");
        ImGui.TextDisabled($"Loaded tiles: {loadedTileCount}");

        float mapSize = MathF.Max(220f, MathF.Min(ImGui.GetContentRegionAvail().X, 360f));
        Vector2 cursorPos = ImGui.GetCursorScreenPos();
        float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / MinimapWorldTileSize;
        float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / MinimapWorldTileSize;

        MinimapHelpers.RenderMinimapContent(
            cursorPos,
            mapSize,
            existingTiles,
            isTileLoaded,
            _minimapRenderer,
            mapName,
            camTileX,
            camTileY,
            _minimapZoom,
            _minimapPanOffset,
            _camera,
            _worldScene,
            out float viewMinTx,
            out float viewMinTy,
            out float cellSize);

        DrawTerrainWorkbenchSelectionOverlay(cursorPos, mapSize, viewMinTx, viewMinTy, cellSize);
        HandleTerrainWorkbenchSelectionInteraction(cursorPos, mapSize, viewMinTx, viewMinTy, cellSize);

        ImGui.Dummy(new Vector2(mapSize, mapSize));

        if (_terrainWorkbenchFocusedTile == null)
            _terrainWorkbenchFocusedTile = GetCameraTile();

        DrawTerrainWorkbenchFocusedTileSummary();
        DrawTerrainWorkbenchChunkGrid(renderer);
    }

    private void DrawTerrainWorkbenchSelectionOverlay(Vector2 cursorPos, float mapSize, float viewMinTx, float viewMinTy, float cellSize)
    {
        var drawList = ImGui.GetWindowDrawList();
        drawList.PushClipRect(cursorPos, cursorPos + new Vector2(mapSize, mapSize), true);

        if (_terrainTileScope == TerrainTileScope.RectRange)
        {
            GetTerrainTileRange(out int startX, out int startY, out int endX, out int endY);
            Vector2 min = new(
                cursorPos.X + (startY - viewMinTy) * cellSize,
                cursorPos.Y + (startX - viewMinTx) * cellSize);
            Vector2 max = new(
                cursorPos.X + ((endY + 1) - viewMinTy) * cellSize,
                cursorPos.Y + ((endX + 1) - viewMinTx) * cellSize);
            drawList.AddRectFilled(min, max, 0x3FA8FF40);
            drawList.AddRect(min, max, 0xFF7CFF40, 0f, ImDrawFlags.None, 2f);
        }

        if (_terrainWorkbenchFocusedTile is { } focusedTile)
        {
            Vector2 min = new(
                cursorPos.X + (focusedTile.tileY - viewMinTy) * cellSize,
                cursorPos.Y + (focusedTile.tileX - viewMinTx) * cellSize);
            Vector2 max = new(
                cursorPos.X + ((focusedTile.tileY + 1) - viewMinTy) * cellSize,
                cursorPos.Y + ((focusedTile.tileX + 1) - viewMinTx) * cellSize);
            drawList.AddRect(min, max, 0xFFFFFF00, 0f, ImDrawFlags.None, 2f);
        }

        drawList.PopClipRect();
    }

    private void HandleTerrainWorkbenchSelectionInteraction(Vector2 cursorPos, float mapSize, float viewMinTx, float viewMinTy, float cellSize)
    {
        ImGui.SetCursorScreenPos(cursorPos);
        ImGui.InvisibleButton("##terrainWorkbenchSelectionMap", new Vector2(mapSize, mapSize));
        bool hovered = ImGui.IsItemHovered();
        Vector2 mousePos = ImGui.GetMousePos();
        var io = ImGui.GetIO();

        if (hovered && io.MouseWheel != 0f)
            _minimapZoom = Math.Clamp(_minimapZoom - io.MouseWheel * 0.5f, 1f, 32f);

        if (hovered && ImGui.IsMouseClicked(ImGuiMouseButton.Right))
        {
            _terrainWorkbenchMapPanActive = true;
            _terrainWorkbenchMapDragStart = mousePos;
            _terrainWorkbenchMapPanOrigin = _minimapPanOffset;
        }
        else if (_terrainWorkbenchMapPanActive && ImGui.IsMouseDown(ImGuiMouseButton.Right))
        {
            Vector2 delta = mousePos - _terrainWorkbenchMapDragStart;
            _minimapPanOffset = _terrainWorkbenchMapPanOrigin - new Vector2(delta.Y / cellSize, delta.X / cellSize);
            ClampMinimapPanOffset();
        }
        else if (_terrainWorkbenchMapPanActive && ImGui.IsMouseReleased(ImGuiMouseButton.Right))
        {
            _terrainWorkbenchMapPanActive = false;
        }

        if (hovered && ImGui.IsMouseClicked(ImGuiMouseButton.Left)
            && TryGetMinimapClickTarget(mousePos, cursorPos, cellSize, viewMinTx, viewMinTy, out float clickTileX, out float clickTileY))
        {
            int tileX = (int)MathF.Floor(clickTileX);
            int tileY = (int)MathF.Floor(clickTileY);
            _terrainWorkbenchTileSelectionActive = true;
            _terrainWorkbenchTileSelectionAnchor = (tileX, tileY);
            _terrainWorkbenchFocusedTile = (tileX, tileY);
            _terrainTileRangeStartX = tileX;
            _terrainTileRangeEndX = tileX;
            _terrainTileRangeStartY = tileY;
            _terrainTileRangeEndY = tileY;
            _terrainTileScope = TerrainTileScope.RectRange;
            MarkTerrainWeakSignalRestoreDirty();
        }
        else if (_terrainWorkbenchTileSelectionActive && ImGui.IsMouseDown(ImGuiMouseButton.Left)
            && TryGetMinimapClickTarget(mousePos, cursorPos, cellSize, viewMinTx, viewMinTy, out float dragTileX, out float dragTileY)
            && _terrainWorkbenchTileSelectionAnchor is { } anchor)
        {
            _terrainTileRangeStartX = anchor.tileX;
            _terrainTileRangeStartY = anchor.tileY;
            _terrainTileRangeEndX = (int)MathF.Floor(dragTileX);
            _terrainTileRangeEndY = (int)MathF.Floor(dragTileY);
            _terrainTileScope = TerrainTileScope.RectRange;
            MarkTerrainWeakSignalRestoreDirty();
        }
        else if (_terrainWorkbenchTileSelectionActive && ImGui.IsMouseReleased(ImGuiMouseButton.Left))
        {
            _terrainWorkbenchTileSelectionActive = false;
            _terrainWorkbenchTileSelectionAnchor = null;
        }
    }

    private void DrawTerrainWorkbenchFocusedTileSummary()
    {
        if (_terrainWorkbenchFocusedTile is not { } focusedTile)
            return;

        ImGui.Text($"Focused ADT: ({focusedTile.tileX}, {focusedTile.tileY})");
        ImGui.SameLine();
        if (ImGui.SmallButton("Use Camera Tile"))
        {
            _terrainWorkbenchFocusedTile = GetCameraTile();
            MarkTerrainWeakSignalRestoreDirty();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Clear Tile Range"))
        {
            _terrainTileScope = TerrainTileScope.CurrentTile;
            _terrainTileRangeStartX = focusedTile.tileX;
            _terrainTileRangeEndX = focusedTile.tileX;
            _terrainTileRangeStartY = focusedTile.tileY;
            _terrainTileRangeEndY = focusedTile.tileY;
            MarkTerrainWeakSignalRestoreDirty();
        }
    }

    private void DrawTerrainWorkbenchChunkGrid(TerrainRenderer renderer)
    {
        if (_terrainWorkbenchFocusedTile is not { } focusedTile)
            return;

        ImGui.Text("Focused ADT Chunk Grid");
        ImGui.TextDisabled("LMB drag selects chunks in the focused ADT. Ctrl keeps existing selection. Use Clipboard + Save below for copy, paste, invert, and save.");

        float gridSize = MathF.Max(220f, MathF.Min(ImGui.GetContentRegionAvail().X, 320f));
        float cellSize = gridSize / 16f;
        Vector2 origin = ImGui.GetCursorScreenPos();
        var drawList = ImGui.GetWindowDrawList();
        drawList.AddRectFilled(origin, origin + new Vector2(gridSize, gridSize), 0xFF1C1C1C);

        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                Vector2 min = new(origin.X + chunkY * cellSize, origin.Y + chunkX * cellSize);
                Vector2 max = new(min.X + cellSize, min.Y + cellSize);
                bool selected = _selectedChunks.Contains((focusedTile.tileX, focusedTile.tileY, chunkX, chunkY));
                uint fill = selected ? 0x6FA8FF40u : 0x20202020u;
                drawList.AddRectFilled(min, max, fill);
                drawList.AddRect(min, max, 0x50505050);
            }
        }

        ImGui.SetCursorScreenPos(origin);
        ImGui.InvisibleButton("##terrainWorkbenchChunkGrid", new Vector2(gridSize, gridSize));
        bool hovered = ImGui.IsItemHovered();
        Vector2 mousePos = ImGui.GetMousePos();
        if (hovered && ImGui.IsMouseClicked(ImGuiMouseButton.Left))
        {
            Vector2 local = mousePos - origin;
            int chunkY = Math.Clamp((int)(local.X / cellSize), 0, 15);
            int chunkX = Math.Clamp((int)(local.Y / cellSize), 0, 15);
            _terrainWorkbenchChunkSelectionActive = true;
            _terrainWorkbenchChunkSelectionAnchor = (chunkX, chunkY);

            if (!ImGui.GetIO().KeyCtrl)
                ClearSelectedChunksForTile(focusedTile.tileX, focusedTile.tileY);
            MarkTerrainWeakSignalRestoreDirty();
        }
        else if (_terrainWorkbenchChunkSelectionActive && ImGui.IsMouseDown(ImGuiMouseButton.Left) && _terrainWorkbenchChunkSelectionAnchor is { } anchor)
        {
            Vector2 local = Vector2.Clamp(mousePos - origin, Vector2.Zero, new Vector2(gridSize - 1f, gridSize - 1f));
            int chunkY = Math.Clamp((int)(local.X / cellSize), 0, 15);
            int chunkX = Math.Clamp((int)(local.Y / cellSize), 0, 15);
            ClearSelectedChunksForTile(focusedTile.tileX, focusedTile.tileY);
            int minChunkX = Math.Min(anchor.chunkX, chunkX);
            int maxChunkX = Math.Max(anchor.chunkX, chunkX);
            int minChunkY = Math.Min(anchor.chunkY, chunkY);
            int maxChunkY = Math.Max(anchor.chunkY, chunkY);
            for (int selectedChunkY = minChunkY; selectedChunkY <= maxChunkY; selectedChunkY++)
            {
                for (int selectedChunkX = minChunkX; selectedChunkX <= maxChunkX; selectedChunkX++)
                    _selectedChunks.Add((focusedTile.tileX, focusedTile.tileY, selectedChunkX, selectedChunkY));
            }
            _chunkClipboardStatus = $"Selected {_selectedChunks.Count} chunk(s) via terrain workbench.";
            MarkTerrainWeakSignalRestoreDirty();
        }
        else if (_terrainWorkbenchChunkSelectionActive && ImGui.IsMouseReleased(ImGuiMouseButton.Left))
        {
            _terrainWorkbenchChunkSelectionActive = false;
            _terrainWorkbenchChunkSelectionAnchor = null;
        }

        ImGui.Dummy(new Vector2(gridSize, gridSize));
    }

    private void ClearSelectedChunksForTile(int tileX, int tileY)
    {
        _selectedChunks.RemoveWhere(chunk => chunk.tileX == tileX && chunk.tileY == tileY);
        MarkTerrainWeakSignalRestoreDirty();
    }

    private void ApplyTerrainWeakSignalRestoreQuickRange(float minHeight, float maxHeight)
    {
        _terrainWeakSignalRestoreCandidateMinHeight = ClampTerrainWeakSignalRestoreZ(minHeight);
        _terrainWeakSignalRestoreCandidateMaxHeight = ClampTerrainWeakSignalRestoreZ(maxHeight);
        GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
        MarkTerrainWeakSignalRestoreDirty();
        SaveViewerSettings();
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
        if (_useDockspaceUi)
            return;

        if (!IsShellPanelActive(ShellPanelId.Navigator) && !IsShellPanelActive(ShellPanelId.Inspector))
            return;

        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
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

        if (_renderer is IModelRenderer || _renderer is WmoRenderer)
        {
            ImGui.Separator();
            ImGui.Checkbox("Auto-frame on load", ref _autoFrameModelOnLoad);
            DrawToolbarPopupButton("Model Actions", string.Empty, "##ModelActionsPopup", () =>
            {
                if (ImGui.Button("Frame Model"))
                {
                    FrameCurrentModel();
                    ImGui.CloseCurrentPopup();
                }
            });
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

        if (_renderer is WmoRenderer standaloneWmoRenderer)
        {
            if (TryGetStandaloneWmoAssetPath(out string standaloneWmoAssetPath))
            {
                ImGui.Separator();
                DrawAssetPathActions("WMO Asset", standaloneWmoAssetPath, "StandaloneWmoAsset");
            }

            ImGui.Separator();
            DrawStandaloneWmoGroupControls(standaloneWmoRenderer);
            DrawWmoDoodadInspector(
                standaloneWmoRenderer,
                ref _selectedStandaloneWmoDoodadIndex,
                "StandaloneWmo",
                doodad => TryFrameStandaloneWmoDoodad(standaloneWmoRenderer, doodad));
        }

        if (_renderer is MdxRenderer standaloneMdxRenderer)
        {
            DrawStandaloneCharacterVariationControls(standaloneMdxRenderer);
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
                DrawToolbarPopupButton("Animation Actions", isPlaying ? "playing" : "paused", "##AnimationActionsPopup", () =>
                {
                    if (ImGui.Button(isPlaying ? "Pause" : "Play"))
                    {
                        animator.IsPlaying = !isPlaying;
                        ImGui.CloseCurrentPopup();
                    }

                    if (ImGui.Button("Previous Key"))
                    {
                        animator.IsPlaying = false;
                        animator.StepToPrevKeyframe();
                        ImGui.CloseCurrentPopup();
                    }

                    if (ImGui.Button("Next Key"))
                    {
                        animator.IsPlaying = false;
                        animator.StepToNextKeyframe();
                        ImGui.CloseCurrentPopup();
                    }
                });

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

    private void DrawStandaloneCharacterVariationControls(MdxRenderer renderer)
    {
        string? modelPath = renderer.ModelVirtualPath ?? _standaloneCharacterCustomizationModelPath;
        if (string.IsNullOrWhiteSpace(modelPath) || _texResolver == null)
            return;

        string normalizedPath = modelPath.Replace('/', '\\');
        if (!string.Equals(_standaloneCharacterCustomizationModelPath, normalizedPath, StringComparison.OrdinalIgnoreCase))
            RefreshStandaloneCharacterCustomizationState(normalizedPath, isM2AdapterModel: false);

        if (string.IsNullOrWhiteSpace(_standaloneCharacterCustomizationModelPath))
            return;

        bool hasHairOptions = _standaloneCharacterHairVariationIds.Count > 0;
        bool hasFacialOptions = _standaloneCharacterFacialHairVariationIds.Count > 0;
        if (!hasHairOptions && !hasFacialOptions)
            return;

        ImGui.Separator();
        ImGui.Text("Character Variants:");
        ImGui.TextDisabled("Raw DBC variation ids for standalone classic character MDX inspection.");

        bool changed = false;
        if (hasHairOptions)
            changed |= DrawStandaloneCharacterVariationCombo("Hair VariationId", "##StandaloneCharacterHairVariation", _standaloneCharacterHairVariationIds, ref _standaloneCharacterHairVariationOverride);

        if (hasFacialOptions)
            changed |= DrawStandaloneCharacterVariationCombo("Facial VariationId", "##StandaloneCharacterFacialVariation", _standaloneCharacterFacialHairVariationIds, ref _standaloneCharacterFacialHairVariationOverride);

        if ((_standaloneCharacterHairVariationOverride >= 0 || _standaloneCharacterFacialHairVariationOverride >= 0)
            && ImGui.Button("Reset Character Variants"))
        {
            _standaloneCharacterHairVariationOverride = -1;
            _standaloneCharacterFacialHairVariationOverride = -1;
            changed = true;
        }

        if (changed)
            ApplyStandaloneCharacterCustomizationOverrides();
    }

    private static bool DrawStandaloneCharacterVariationCombo(string label, string comboId, IReadOnlyList<int> variationIds, ref int selectedVariationId)
    {
        ImGui.Text(label);
        ImGui.SetNextItemWidth(-1);

        string preview = selectedVariationId >= 0
            ? $"VariationId {selectedVariationId}"
            : "Default (VariationId 0)";
        bool changed = false;

        if (ImGui.BeginCombo(comboId, preview))
        {
            bool defaultSelected = selectedVariationId < 0;
            if (ImGui.Selectable("Default (VariationId 0)", defaultSelected))
            {
                selectedVariationId = -1;
                changed = true;
            }

            if (defaultSelected)
                ImGui.SetItemDefaultFocus();

            foreach (int variationId in variationIds)
            {
                bool selected = selectedVariationId == variationId;
                if (ImGui.Selectable($"VariationId {variationId}", selected))
                {
                    selectedVariationId = variationId;
                    changed = true;
                }

                if (selected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        return changed;
    }

    private void DrawSelectedTaxiControls()
    {
        if (_worldScene == null)
            return;

        if (_worldScene.TaxiLoader != null && _worldScene.TaxiLoader.Routes.Count > 0)
        {
            bool showTaxi = _worldScene.ShowTaxi;
            if (ImGui.Checkbox($"Show Taxi Paths ({_worldScene.TaxiLoader.Routes.Count})", ref showTaxi))
                _worldScene.ShowTaxi = showTaxi;

            if (_worldScene.ShowTaxi && (_worldScene.SelectedTaxiNodeId >= 0 || _worldScene.SelectedTaxiRouteId >= 0))
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Show All"))
                {
                    _worldScene.ClearTaxiSelection();
                    ClearSelectedTaxiInfo();
                }
            }
        }
        else if (!_worldScene.TaxiLoadAttempted)
        {
            if (ImGui.Button("Load Taxi Paths"))
                _worldScene.ShowTaxi = true;

            ImGui.TextDisabled("Load taxi paths to enable viewport picking, route browsing, and actor overrides.");
            return;
        }
        else
        {
            ImGui.TextDisabled("Taxi Paths: none found");
            return;
        }

        bool hasTaxiSelection = _worldScene.SelectedTaxiNodeId >= 0 || _worldScene.SelectedTaxiRouteId >= 0;
        if (hasTaxiSelection && !string.IsNullOrWhiteSpace(_selectedObjectInfo))
        {
            ImGui.Separator();
            ImGui.TextWrapped(_selectedObjectInfo);
        }

        ImGui.Separator();
        ImGui.Text("Taxi Controls");

        if (!hasTaxiSelection)
            ImGui.BeginDisabled();
        if (ImGui.Button("Focus Selected Taxi"))
            FocusSelectedTaxi();
        if (!hasTaxiSelection)
            ImGui.EndDisabled();

        bool hasSelectedTaxiRoute = _worldScene.SelectedTaxiRouteId >= 0;
        bool rideCameraAttachedToSelection = _taxiRideCameraEnabled
            && hasSelectedTaxiRoute
            && _taxiRideCameraRouteId == _worldScene.SelectedTaxiRouteId;
        bool rideCameraActive = _taxiRideCameraEnabled && _taxiRideCameraRouteId >= 0;

        bool canToggleRideCamera = hasSelectedTaxiRoute || _taxiRideCameraEnabled;
        if (!canToggleRideCamera)
            ImGui.BeginDisabled();
        if (ImGui.Button(rideCameraActive ? "Detach Ride Camera" : "Ride Selected Route"))
        {
            if (rideCameraActive)
                StopTaxiRideCamera("Ride camera detached.");
            else
                TryAttachTaxiRideCameraToSelectedRoute();
        }
        if (!canToggleRideCamera)
            ImGui.EndDisabled();

        if (_taxiRideCameraEnabled)
            ImGui.TextDisabled($"Ride Camera: {GetTaxiRouteDisplayLabel(_taxiRideCameraRouteId)}");

        int taxiRideCameraMode = (int)_taxiRideCameraMode;
        string[] taxiRideCameraLabels = { "Cockpit", "Chase" };
        if (ImGui.Combo("Ride Camera Mode", ref taxiRideCameraMode, taxiRideCameraLabels, taxiRideCameraLabels.Length))
            _taxiRideCameraMode = (TaxiRideCameraMode)taxiRideCameraMode;

        if (_taxiRideCameraMode == TaxiRideCameraMode.Cockpit)
        {
            float cockpitHeight = _taxiRideCockpitHeight;
            if (ImGui.SliderFloat("Ride Camera Height", ref cockpitHeight, 2f, 30f, "%.1f"))
                _taxiRideCockpitHeight = cockpitHeight;
        }
        else
        {
            float chaseDistance = _taxiRideChaseDistance;
            if (ImGui.SliderFloat("Ride Chase Distance", ref chaseDistance, 8f, 120f, "%.1f"))
                _taxiRideChaseDistance = chaseDistance;

            float chaseHeight = _taxiRideChaseHeight;
            if (ImGui.SliderFloat("Ride Chase Height", ref chaseHeight, 2f, 40f, "%.1f"))
                _taxiRideChaseHeight = chaseHeight;
        }

        float rideLookAhead = _taxiRideLookAhead;
        if (ImGui.SliderFloat("Ride Look Ahead", ref rideLookAhead, 8f, 80f, "%.1f"))
            _taxiRideLookAhead = rideLookAhead;

        int videoFps = _videoCaptureFps;
        if (ImGui.SliderInt("Ride Video FPS", ref videoFps, 12, 60))
            _videoCaptureFps = videoFps;

        bool videoIncludeUi = _videoCaptureIncludeUi;
        if (ImGui.Checkbox("Ride Video Includes UI", ref videoIncludeUi))
            _videoCaptureIncludeUi = videoIncludeUi;

        if (_activeVideoRecording == null)
        {
            if (!hasSelectedTaxiRoute)
                ImGui.BeginDisabled();
            if (ImGui.Button("Record Selected Route Video"))
                TryStartTaxiRideVideoCapture();
            if (!hasSelectedTaxiRoute)
                ImGui.EndDisabled();
        }
        else
        {
            if (ImGui.Button("Stop Route Video"))
                StopVideoRecording();

            ImGui.SameLine();
            ImGui.TextDisabled(Path.GetFileName(_activeVideoRecording.OutputPath));
        }

        bool showTaxiActors = _worldScene.ShowTaxiActors;
        if (ImGui.Checkbox("Show Animated Taxi Actor", ref showTaxiActors))
            _worldScene.ShowTaxiActors = showTaxiActors;

        float speedMultiplier = _worldScene.TaxiActorSpeedMultiplier;
        if (ImGui.SliderFloat("Taxi Speed", ref speedMultiplier, WorldScene.TaxiActorMinSpeedSetting, WorldScene.TaxiActorMaxSpeedSetting, "%.2f"))
            _worldScene.TaxiActorSpeedMultiplier = speedMultiplier;
        ImGui.TextDisabled("0.10 = 100% speed, 0.01 = 10%, 0.50 = 500%.");

        float scaleMultiplier = _worldScene.TaxiActorScaleMultiplier;
        if (ImGui.SliderFloat("Taxi Actor Scale", ref scaleMultiplier, 0.05f, 5f, "%.2fx"))
            _worldScene.TaxiActorScaleMultiplier = scaleMultiplier;

        ImGui.Separator();
        string[] taxiGroupingLabels = { "None", "From Node", "To Node" };
        ImGui.Text($"Routes ({_worldScene.TaxiLoader.Routes.Count})");
        ImGui.SetNextItemWidth(140f);
        ImGui.Combo("Group By", ref _taxiRouteListGroupingMode, taxiGroupingLabels, taxiGroupingLabels.Length);

        string taxiRouteFilter = _taxiRouteFilter;
        if (ImGui.InputText("Search Routes", ref taxiRouteFilter, 256))
            _taxiRouteFilter = taxiRouteFilter;

        var routeEntries = new List<(TaxiPathLoader.TaxiRoute Route, string FromName, string ToName, string Label, string GroupKey)>();
        foreach (TaxiPathLoader.TaxiRoute route in _worldScene.TaxiLoader.Routes)
        {
            string fromName = _worldScene.GetTaxiNode(route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
            string toName = _worldScene.GetTaxiNode(route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";
            string label = $"{GetTaxiRouteDisplayLabel(route.PathId)} ({route.Waypoints.Count} pts)";
            string searchText = $"{route.PathId} {fromName} {toName} {label}";
            if (!string.IsNullOrWhiteSpace(_taxiRouteFilter)
                && !searchText.Contains(_taxiRouteFilter, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string groupKey = _taxiRouteListGroupingMode switch
            {
                1 => fromName,
                2 => toName,
                _ => string.Empty,
            };

            routeEntries.Add((route, fromName, toName, label, groupKey));
        }

        routeEntries.Sort((left, right) =>
        {
            if (_taxiRouteListGroupingMode != 0)
            {
                int groupCompare = StringComparer.OrdinalIgnoreCase.Compare(left.GroupKey, right.GroupKey);
                if (groupCompare != 0)
                    return groupCompare;
            }

            int primaryCompare = _taxiRouteListGroupingMode switch
            {
                1 => StringComparer.OrdinalIgnoreCase.Compare(left.ToName, right.ToName),
                2 => StringComparer.OrdinalIgnoreCase.Compare(left.FromName, right.FromName),
                _ => 0,
            };
            if (primaryCompare != 0)
                return primaryCompare;

            return left.Route.PathId.CompareTo(right.Route.PathId);
        });

        if (routeEntries.Count != _worldScene.TaxiLoader.Routes.Count)
            ImGui.TextDisabled($"Showing {routeEntries.Count} of {_worldScene.TaxiLoader.Routes.Count} routes");

        if (ImGui.BeginChild("##TaxiRouteSidebarList", new Vector2(0, 220f), true))
        {
            if (routeEntries.Count == 0)
            {
                ImGui.TextDisabled(string.IsNullOrWhiteSpace(_taxiRouteFilter)
                    ? "No taxi routes are available."
                    : "No taxi routes match the current search.");
            }
            else
            {
                Dictionary<string, int>? groupCounts = null;
                string currentGroupKey = string.Empty;
                if (_taxiRouteListGroupingMode != 0)
                {
                    groupCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                    foreach (var entry in routeEntries)
                        groupCounts[entry.GroupKey] = groupCounts.TryGetValue(entry.GroupKey, out int count) ? count + 1 : 1;
                }

                for (int i = 0; i < routeEntries.Count; i++)
                {
                    var entry = routeEntries[i];
                    if (_taxiRouteListGroupingMode != 0 && !string.Equals(currentGroupKey, entry.GroupKey, StringComparison.OrdinalIgnoreCase))
                    {
                        currentGroupKey = entry.GroupKey;
                        if (i > 0)
                            ImGui.Separator();
                        ImGui.TextDisabled($"{currentGroupKey} ({groupCounts![currentGroupKey]})");
                    }

                    bool isSelected = _worldScene.SelectedTaxiRouteId == entry.Route.PathId;
                    if (ImGui.Selectable(entry.Label, isSelected, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        SelectTaxiRoute(entry.Route.PathId, toggle: true);
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                            FocusSelectedTaxi();
                    }

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Cost: {entry.Route.Cost}");
                        ImGui.Text($"From: {entry.FromName}");
                        ImGui.Text($"To: {entry.ToName}");
                        ImGui.Text($"Waypoints: {entry.Route.Waypoints.Count}");
                        ImGui.Text("Single-click selects the route. Double-click focuses the camera.");
                        ImGui.EndTooltip();
                    }
                }
            }

            ImGui.EndChild();
        }

        if (_worldScene.SelectedTaxiNodeId >= 0)
            ImGui.TextDisabled($"Selected taxi node: {_worldScene.SelectedTaxiNodeId}");
        else if (_worldScene.SelectedTaxiRouteId >= 0)
            ImGui.TextDisabled($"Selected taxi route: {_worldScene.SelectedTaxiRouteId}");

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
            IReadOnlyList<string> defaultTaxiActorModels = WorldScene.DefaultTaxiActorModelPaths;
            ImGui.TextWrapped($"Override Route: {GetTaxiRouteDisplayLabel(routeId)}");
            ImGui.TextWrapped($"Resolved Actor Model: {resolvedActorModelPath}");
            ImGui.TextDisabled($"Override: {actorOverridePath ?? "auto"}");

            string actorModelPath = _taxiActorModelOverrideInput;
            if (ImGui.InputText("Actor Model Path", ref actorModelPath, 512))
                _taxiActorModelOverrideInput = actorModelPath;

            if (defaultTaxiActorModels.Count > 0)
            {
                if (ImGui.Button("Use Gryphon Default"))
                {
                    _taxiActorModelOverrideInput = defaultTaxiActorModels[0];
                    _taxiActorModelOverrideInputRouteId = routeId;
                    ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
                    RefreshSelectedTaxiInfo();
                }
            }

            if (defaultTaxiActorModels.Count > 1)
            {
                ImGui.SameLine();
                if (ImGui.Button("Use FelBat Default"))
                {
                    _taxiActorModelOverrideInput = defaultTaxiActorModels[1];
                    _taxiActorModelOverrideInputRouteId = routeId;
                    ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
                    RefreshSelectedTaxiInfo();
                }
            }

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
                SyncTaxiActorModelOverrideInput(routeId);
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
            ImGui.TextDisabled("No connected routes were found for this taxi node.");
        else
            ImGui.TextDisabled("Select a taxi route from the list or click one in the viewport to configure the animated actor.");
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
        DrawWmoDoodadInspector(
            wmoRenderer,
            ref _selectedWorldWmoDoodadIndex,
            "SelectedWmo",
            doodad => TryFrameSelectedWorldWmoDoodad(wmoRenderer, doodad));
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

        float fogStart = Math.Clamp(lighting.FogStart, 0f, MaxTerrainFogDistance - 1f);
        float fogEnd = Math.Clamp(lighting.FogEnd, 100f, MaxTerrainFogDistance);
        bool fogStartChanged = ImGui.SliderFloat("Fog Start", ref fogStart, 0f, MaxTerrainFogDistance - 1f);
        bool fogEndChanged = ImGui.SliderFloat("Fog End", ref fogEnd, 100f, MaxTerrainFogDistance);
        if (fogStartChanged || fogEndChanged)
        {
            if (fogEnd <= fogStart)
            {
                if (fogEndChanged && !fogStartChanged)
                    fogStart = Math.Max(0f, fogEnd - 1f);
                else
                    fogEnd = Math.Min(MaxTerrainFogDistance, fogStart + 1f);
            }

            lighting.FogStart = fogStart;
            lighting.FogEnd = fogEnd;
        }

        if (_worldScene != null)
        {
            bool showWdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("Show WDL Far Terrain", ref showWdl))
                _worldScene.ShowWdlTerrain = showWdl;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Toggle low-detail WDL background terrain for testing terrain overlap issues.");

            bool weakSignalRestore = _terrainWeakSignalRestoreEnabled;
            if (ImGui.Checkbox("Restore Weak-Signal Terrain", ref weakSignalRestore))
            {
                if (SetTerrainWeakSignalRestoreEnabled(weakSignalRestore))
                    SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Amplify weak, era-compressed terrain on the camera tile and its four direct neighbors, then clamp the actual motion to weak per-cell signal regions across the ADT instead of picking one whole chunk or one whole texture bucket.");

            ImGui.TextDisabled("Mode: whole-tile factor, per-cell weak-signal clamp.");

            float restoreRangeMin = _terrainWeakSignalRestoreCandidateMinHeight;
            if (ImGui.InputFloat("Restore Range Min Z", ref restoreRangeMin, 10f, 100f, "%.1f"))
            {
                _terrainWeakSignalRestoreCandidateMinHeight = ClampTerrainWeakSignalRestoreZ(restoreRangeMin);
                GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
                MarkTerrainWeakSignalRestoreDirty();
                SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Early-era buried terrain tends to sit around -10..10. Later-era ocean-floor-compressed data can need something closer to -5000..10.");

            float restoreRangeMax = _terrainWeakSignalRestoreCandidateMaxHeight;
            if (ImGui.InputFloat("Restore Range Max Z", ref restoreRangeMax, 10f, 100f, "%.1f"))
            {
                _terrainWeakSignalRestoreCandidateMaxHeight = ClampTerrainWeakSignalRestoreZ(restoreRangeMax);
                GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
                MarkTerrainWeakSignalRestoreDirty();
                SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Use this with the minimum bound to switch between early 0-floor data and later ocean-floor-compressed tiles.");

            ImGui.TextDisabled("Quick ranges:");
            if (ImGui.SmallButton("Packed +/-2.778"))
                ApplyTerrainWeakSignalRestoreQuickRange(-2.778f, 2.778f);
            ImGui.SameLine();
            if (ImGui.SmallButton("Packed +/-3"))
                ApplyTerrainWeakSignalRestoreQuickRange(-3f, 3f);
            ImGui.SameLine();
            if (ImGui.SmallButton("Early +/-5"))
                ApplyTerrainWeakSignalRestoreQuickRange(-5f, 5f);
            ImGui.SameLine();
            if (ImGui.SmallButton("Early +/-10"))
                ApplyTerrainWeakSignalRestoreQuickRange(-10f, 10f);
            ImGui.SameLine();
            if (ImGui.SmallButton("Late -5000..10"))
                ApplyTerrainWeakSignalRestoreQuickRange(-5000f, 10f);

            ImGui.TextDisabled("Examples: early era -10..10, later era -5000..10.");

            bool weakSignalAuto = _terrainWeakSignalRestoreUseAutoFactor;
            if (ImGui.Checkbox("Auto Restore Scale", ref weakSignalAuto))
            {
                _terrainWeakSignalRestoreUseAutoFactor = weakSignalAuto;
                MarkTerrainWeakSignalRestoreDirty();
                SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Use the WDL-backed whole-tile auto estimate, then clamp the resulting deformation to weak per-cell signal regions across the ADT. Turn this off to A/B the manual restore control instead.");

            var wdlGuideTile = GetCameraTile();
            if (TryGetTerrainWeakSignalWdlTile(wdlGuideTile.tileX, wdlGuideTile.tileY, out var wdlGuide) && wdlGuide != null)
            {
                ImGui.TextDisabled($"WDL guide ({wdlGuideTile.tileX}, {wdlGuideTile.tileY}): {wdlGuide.MinZ:F1}..{wdlGuide.MaxZ:F1}, center {wdlGuide.Height17[8, 8]:F1}, 17x17 + 16x16 samples");
            }
            else
            {
                ImGui.TextDisabled($"WDL guide ({wdlGuideTile.tileX}, {wdlGuideTile.tileY}): no tile data available");
            }

            if (!_terrainWeakSignalRestoreUseAutoFactor)
            {
                float manualRestoreScale = _terrainWeakSignalRestoreManualFactor;
                if (ImGui.InputFloat("Restore Scale", ref manualRestoreScale, 0.25f, 1f, "%.2fx"))
                {
                    _terrainWeakSignalRestoreManualFactor = Math.Clamp(manualRestoreScale, 1f, TerrainWeakSignalRestoreMaxFactor);
                    MarkTerrainWeakSignalRestoreDirty();
                    SaveViewerSettings();
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Manual viewer-only terrain relief multiplier. Type the exact factor you want; the value is clamped to the supported restore range and reapplied from the original tile data so you can A/B without compounding.");
            }

            string restoreScopeSummary = GetTerrainWeakSignalRestoreScopeSummary();
            ImGui.TextDisabled($"Candidates: {restoreScopeSummary}, whole-tile factor with per-cell weak-signal clamp, source Z in {_terrainWeakSignalRestoreCandidateMinHeight:0.#}..{_terrainWeakSignalRestoreCandidateMaxHeight:0.#}.");

            if (!string.IsNullOrWhiteSpace(_terrainWeakSignalRestoreStatus))
                ImGui.TextWrapped(_terrainWeakSignalRestoreStatus);

            bool layoutObjectPreviewMode = _layoutObjectPreviewMode;
            if (ImGui.Checkbox("Pretextured Layout Mode", ref layoutObjectPreviewMode))
                SetLayoutObjectPreviewMode(layoutObjectPreviewMode);
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Keep large textured WMOs visible, suppress doodads, and force Performance object detail for fast zone layout passes.");

            bool showObjects = _worldScene.ObjectsVisible;
            if (_layoutObjectPreviewMode)
                ImGui.BeginDisabled();
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

            if (_layoutObjectPreviewMode)
                ImGui.EndDisabled();

            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Quality keeps more far objects alive. Performance culls tiny projected objects and skips low-value off-view loads.");

            if (_layoutObjectPreviewMode)
                ImGui.TextDisabled("Layout mode keeps WMOs only and turns off doodads until you disable the preset.");
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
        ImGui.Text($"Terrain chunks rendered/culled: {renderStats.TerrainChunksRendered}/{renderStats.TerrainChunksCulled}  WDL visible/hidden: {renderStats.WdlVisibleTileCount}/{renderStats.WdlHiddenTileCount}");
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
        ImGui.TextDisabled("Open Terrain Tools, Chunk Clipboard, Terrain Analysis, and MCNK Explorer from the Tools menu.");
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

        ImGui.SameLine();
        bool canInvert = _selectedChunks.Count > 0 || hasChunk;
        if (!canInvert) ImGui.BeginDisabled();
        if (ImGui.Button(_selectedChunks.Count > 0 ? "Invert Z Selection" : "Invert Z Chunk"))
            InvertSelectedChunkHeights(renderer);
        if (!canInvert) ImGui.EndDisabled();

        ImGui.TextDisabled($"Edited tiles: {GetChunkToolDirtyTileCount()}  Edited chunks: {GetChunkToolDirtyChunkCount()}");
        ImGui.TextDisabled("Saves reusable 257x257 L16 heightmaps plus a manifest under the editor project output folder. Source terrain files stay untouched.");

        bool canSaveEdited = GetChunkToolDirtyTileCount() > 0;
        if (!canSaveEdited) ImGui.BeginDisabled();
        if (ImGui.Button("Save Edited Heightmaps"))
            SaveChunkToolHeightmapOutputs();
        if (!canSaveEdited) ImGui.EndDisabled();

        ImGui.SameLine();
        if (!canSaveEdited) ImGui.BeginDisabled();
        if (ImGui.SmallButton("Clear Dirty##chunkToolDirtyClear"))
            ClearChunkToolDirtyTracking();
        if (!canSaveEdited) ImGui.EndDisabled();

        if (!string.IsNullOrWhiteSpace(_chunkClipboardLastSaveFolder))
            ImGui.TextWrapped($"Last heightmap output: {_chunkClipboardLastSaveFolder}");

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

    private void DrawWmoLiquidRotationControls(string idSuffix)
    {
        int quarterTurns = WmoRenderer.MliqRotationQuarterTurns;
        string currentLabel = WmoLiquidRotationLabels[Math.Clamp(quarterTurns, 0, WmoLiquidRotationLabels.Length - 1)];

        if (ImGui.BeginCombo($"WMO MLIQ Additional Rotation##{idSuffix}", currentLabel))
        {
            for (int i = 0; i < WmoLiquidRotationLabels.Length; i++)
            {
                bool selected = i == quarterTurns;
                if (ImGui.Selectable(WmoLiquidRotationLabels[i], selected))
                {
                    _hasExplicitWmoMliqRotationOverride = i != 0;
                    WmoRenderer.MliqRotationQuarterTurns = i;
                }
                if (selected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        ImGui.TextDisabled("Adds on top of the version-aware WMO MLIQ baseline. Changes are live.");
    }

    // ── Tool windows extracted from right sidebar ──────────────────────

    private void DrawUniqueIdArchaeologyWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(420f, 360f), ImGuiCond.FirstUseEver);
        if (ImGui.Begin("UniqueId Archaeology", ref _showUniqueIdArchaeologyWindow))
        {
            DrawUniqueIdArchaeologyContent();
        }
        ImGui.End();
    }

    private void DrawTaxiWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(460f, 500f), ImGuiCond.FirstUseEver);
        if (ImGui.Begin("Taxi Panel", ref _showTaxiWindow))
        {
            DrawSelectedTaxiControls();
        }
        ImGui.End();
    }

    private void DrawWeakSignalWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(400f, 480f), ImGuiCond.FirstUseEver);
        if (ImGui.Begin("Weak Signal Amplifier", ref _showWeakSignalWindow))
        {
            DrawTerrainControlsAdjustmentWeakSignalContent();
        }
        ImGui.End();
    }

    private void DrawUniqueIdArchaeologyContent()
    {
        // Extracted from DrawUnifiedViewerSettingsSidebarContent (ViewerApp.cs ~line 8368)
        if (_worldScene == null)
            return;

        ImGui.TextDisabled("Scopes UniqueId placement data into layers by consecutive gap analysis.");
        ImGui.Spacing();

        int cameraTileX = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
        int cameraTileY = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
        _worldScene.SetUniqueIdFilterTile(cameraTileX, cameraTileY);

        bool uniqueIdFilterEnabled = _worldScene.UniqueIdFilterEnabled;
        bool uniqueIdFilterChanged = false;
        if (ImGui.Checkbox("Filter UniqueId Range", ref uniqueIdFilterEnabled))
        {
            _worldScene.UniqueIdFilterEnabled = uniqueIdFilterEnabled;
            uniqueIdFilterChanged = true;
        }
        ImGui.SameLine();
        UniqueIdVisibilityScope currentScope = _worldScene.UniqueIdVisibilityScope;
        if (ImGui.BeginCombo("##UniqueIdScope", currentScope == UniqueIdVisibilityScope.PerMap ? "Per-Map" : "Camera Tile"))
        {
            if (ImGui.Selectable("Per-Map", currentScope == UniqueIdVisibilityScope.PerMap))
                _worldScene.UniqueIdVisibilityScope = UniqueIdVisibilityScope.PerMap;
            if (ImGui.Selectable("Camera Tile", currentScope == UniqueIdVisibilityScope.CameraTile))
                _worldScene.UniqueIdVisibilityScope = UniqueIdVisibilityScope.CameraTile;
            ImGui.EndCombo();
        }

        ImGui.Spacing();
        IReadOnlyList<UniqueIdArchaeologyLayer> detectedLayers = _worldScene.GetUniqueIdArchaeologyLayers();
        if (detectedLayers.Count == 0)
        {
            ImGui.TextDisabled("No UniqueId data available for the current scope.");
        }
        else
        {
            if (ImGui.BeginTable("##UniqueIdLayers", 4, ImGuiTableFlags.BordersV | ImGuiTableFlags.RowBg))
            {
                ImGui.TableSetupColumn("Layer");
                ImGui.TableSetupColumn("Range");
                ImGui.TableSetupColumn("Summary");
                ImGui.TableSetupColumn("");
                ImGui.TableHeadersRow();

                for (int i = 0; i < detectedLayers.Count; i++)
                {
                    UniqueIdArchaeologyLayer layer = detectedLayers[i];
                    ImGui.TableNextRow();
                    ImGui.TableNextColumn();
                    ImGui.TextUnformatted($"Layer {layer.LayerNumber}");
                    ImGui.TableNextColumn();
                    ImGui.TextUnformatted($"{layer.MinUniqueId}  —  {layer.MaxUniqueId}");
                    ImGui.TableNextColumn();
                    ImGui.TextUnformatted($"{layer.PlacementCount} placements ({layer.WmoCount} WMO, {layer.MdxCount} M2)");
                    ImGui.TableNextColumn();
                    if (ImGui.SmallButton($"Show##uid_layer_{i}"))
                    {
                        _worldScene.SetUniqueIdFilterRange(layer.MinUniqueId, layer.MaxUniqueId);
                        _worldScene.UniqueIdFilterEnabled = true;
                        uniqueIdFilterChanged = true;
                    }
                }
                ImGui.EndTable();
            }
        }

        // Clear filter handled by enable/disable toggle above
    }

    private void DrawTerrainControlsAdjustmentWeakSignalContent()
    {
        // Extracted from DrawTerrainControlsAdjustmentContent (this file ~line 2576)
        if (_terrainManager == null && _vlmTerrainManager == null)
            return;

        ImGui.Text("Weak Signal Amplifier");
        ImGui.Spacing();

        bool weakSignalEnabled = _terrainWeakSignalRestoreEnabled;
        if (ImGui.Checkbox("Restore Weak-Signal Terrain", ref weakSignalEnabled))
            SetTerrainWeakSignalRestoreEnabled(weakSignalEnabled);

        if (_terrainWeakSignalRestoreEnabled)
        {
            ImGui.InputFloat("Restore Range Min Z", ref _terrainWeakSignalRestoreCandidateMinHeight, 1f, 10f);
            ImGui.InputFloat("Restore Range Max Z", ref _terrainWeakSignalRestoreCandidateMaxHeight, 1f, 10f);
            ImGui.Spacing();

            if (ImGui.Button("Packed +/-2.778")) { _terrainWeakSignalRestoreCandidateMinHeight = -2.778f; _terrainWeakSignalRestoreCandidateMaxHeight = 2.778f; RefreshTerrainWeakSignalRestoreForLoadedTiles(); }
            ImGui.SameLine();
            if (ImGui.Button("Packed +/-3")) { _terrainWeakSignalRestoreCandidateMinHeight = -3f; _terrainWeakSignalRestoreCandidateMaxHeight = 3f; RefreshTerrainWeakSignalRestoreForLoadedTiles(); }
            ImGui.SameLine();
            if (ImGui.Button("Early +/-5")) { _terrainWeakSignalRestoreCandidateMinHeight = -5f; _terrainWeakSignalRestoreCandidateMaxHeight = 5f; RefreshTerrainWeakSignalRestoreForLoadedTiles(); }
            ImGui.SameLine();
            if (ImGui.Button("Early +/-10")) { _terrainWeakSignalRestoreCandidateMinHeight = -10f; _terrainWeakSignalRestoreCandidateMaxHeight = 10f; RefreshTerrainWeakSignalRestoreForLoadedTiles(); }
            ImGui.SameLine();
            if (ImGui.Button("Late -5000..10")) { _terrainWeakSignalRestoreCandidateMinHeight = -5000f; _terrainWeakSignalRestoreCandidateMaxHeight = 10f; RefreshTerrainWeakSignalRestoreForLoadedTiles(); }
            ImGui.Spacing();

            bool autoFactor = _terrainWeakSignalRestoreUseAutoFactor;
            if (ImGui.Checkbox("Auto Restore Scale", ref autoFactor))
                _terrainWeakSignalRestoreUseAutoFactor = autoFactor;

            if (!autoFactor)
            {
                ImGui.InputFloat("Restore Scale", ref _terrainWeakSignalRestoreManualFactor, 0.5f, 8f);
            }

            if (!string.IsNullOrWhiteSpace(_terrainWeakSignalRestoreStatus))
                ImGui.TextDisabled(_terrainWeakSignalRestoreStatus);
        }
    }
}
