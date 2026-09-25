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
using static WoWViewer.ViewerApp;
using static WoWViewer.MinimapAndStatusService;

namespace WoWViewer;

// NavigatorPanelService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class NavigatorPanelService
{

    internal void DrawLeftSidebar()
    {
        if (!_useTabUi || !_showLeftSidebar)
            return;

        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
        float sidebarHeight = io.DisplaySize.Y - topOffset - BottomBarHeight - StatusBarHeight;

        _leftSidebarWidth = ClampFixedSidebarWidth(_leftSidebarWidth, isLeftSidebar: true, io.DisplaySize.X);
        ImGui.SetNextWindowPos(new Vector2(0, topOffset), ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(_leftSidebarWidth, sidebarHeight), ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0.08f, 0.08f, 0.10f, 0.85f));
        if (ImGui.Begin("##LeftSidebar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
        {
            bool hasWorldLoaded = _worldScene != null || _terrainManager != null || _vlmTerrainManager != null;

            DrawWorkspaceBarsPanelContent();

            ImGui.Separator();

            ImGui.Separator();

            DrawSharedWorldOverviewSection(inScrollableChild: true);
            if (hasWorldLoaded)
                ImGui.Separator();

            DrawFileBrowserContent(hasWorldLoaded ? 260f : 0f);

            ImGui.Separator();
            if (_discoveredMaps.Count > 0)
                DrawMapDiscoveryContent();
        }
        ImGui.End();
        ImGui.PopStyleColor();
        ImGui.PopStyleVar();
    }

    internal void DrawLegacyLeftSidebar()
    {
        if (!_shellLayout.HasAnyShellPanelsInLane(ShellPanelLane.Left))
            return;

        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
        float sidebarHeight = io.DisplaySize.Y - topOffset - BottomBarHeight - StatusBarHeight;
        if (_useDockspaceUi)
        {
            DrawDockedShellPanelsForLane(ShellPanelLane.Left, sidebarHeight);
            return;
        }

        _leftSidebarWidth = ClampFixedSidebarWidth(_leftSidebarWidth, isLeftSidebar: true, io.DisplaySize.X);
        ImGui.SetNextWindowPos(new Vector2(0, topOffset), ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(_leftSidebarWidth, sidebarHeight), ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        if (ImGui.Begin("##LegacyLeftSidebar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
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

    internal void DrawNavigatorPanelContent()
    {
        bool hasWorldLoaded = _worldScene != null || _terrainManager != null || _vlmTerrainManager != null;

        DrawSharedWorldOverviewSection(inScrollableChild: false);

        ImGui.SetNextItemOpen(!hasWorldLoaded, ImGuiCond.Once);
        if (_showFileBrowser && ImGui.CollapsingHeader("File Browser", hasWorldLoaded ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None))
            DrawFileBrowserContent(hasWorldLoaded ? 260f : 0f);

        DrawSharedWorldMapsSection(defaultOpenWhenNoWorld: false);
    }

    private void DrawSharedWorldOverviewSection(bool inScrollableChild)
    {
        bool hasWorldLoaded = _worldScene != null || _terrainManager != null || _vlmTerrainManager != null;
        if (!hasWorldLoaded)
            return;

        ImGui.SetNextItemOpen(true, ImGuiCond.Once);
        if (ImGui.CollapsingHeader("World Overview", ImGuiTreeNodeFlags.DefaultOpen))
        {
            if (inScrollableChild)
            {
                float overviewHeight = MathF.Min(340f, MathF.Max(210f, ImGui.GetContentRegionAvail().Y * 0.42f));
                if (ImGui.BeginChild("##LeftWorldOverview", new Vector2(0f, overviewHeight), true,
                    ImGuiWindowFlags.None))
                    DrawWorldOverviewContent();
                ImGui.EndChild();
            }
            else
            {
                DrawWorldOverviewContent();
            }

            DrawMapExportControls();
        }
    }

    /// <summary>
    /// Spec 247: one export button for the loaded map, with the output format picked by checkbox rather
    /// than a separate button per target. Only shown for DAT terrain, which is the only source the
    /// exporter currently reads.
    /// </summary>
    private void DrawMapExportControls()
    {
        if (_terrainManager?.Adapter is not AhdrTerrainAdapter)
            return;

        ImGui.Spacing();
        ImGui.SeparatorText("Export Map");
        Terrain.MapExportFormats.DrawCheckboxes("sidebar");

        ImGui.BeginDisabled(!Terrain.MapExportFormats.Any);
        if (ImGui.Button($"Export as {Terrain.MapExportFormats.Summary}...", new Vector2(-1f, 0f)))
            _cascAhdrSource.ExportLoadedDatMap();
        ImGui.EndDisabled();

        if (ImGui.IsItemHovered() && Terrain.MapExportFormats.Any)
            ImGui.SetTooltip("Writes the loaded DAT folder in every ticked format, plus a manifest naming every field carried and dropped.");
    }

    private void DrawSharedWorldMapsSection(bool defaultOpenWhenNoWorld = false)
    {
        if (_discoveredMaps.Count == 0)
            return;

        bool hasWorldLoaded = _worldScene != null || _terrainManager != null || _vlmTerrainManager != null;
        if (_autoOpenWorldMapsPanel)
            ImGui.SetNextItemOpen(true, ImGuiCond.Always);
        else if (defaultOpenWhenNoWorld)
            ImGui.SetNextItemOpen(!hasWorldLoaded, ImGuiCond.Once);

        if (ImGui.CollapsingHeader("World Maps"))
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
            ImGui.TextDisabled($"Camera tile: ({tileY}, {tileX})");
        }

        if (!string.IsNullOrWhiteSpace(_currentAreaName))
            ImGui.TextDisabled($"Area: {_currentAreaName}");

        if (_worldScene != null && (_worldScene.Pm4Overlay.ShowPm4Overlay || _worldScene.Pm4Overlay.Pm4LoadAttempted))
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4Overlay.Pm4VisibleObjectCount}/{_worldScene.Pm4Overlay.Pm4ObjectCount} visible objects");

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
        DrawMapSortModeSelector("##mapListSort");
        var previewWarmup = _wdlPreview.GetWdlPreviewWarmupStats();
        if (previewWarmup.total > 0)
            ImGui.TextDisabled($"WDL previews: {previewWarmup.ready}/{previewWarmup.total} cached, {previewWarmup.loading} warming, {previewWarmup.failed} failed");
        ImGui.Separator();

        var sortedMaps = MapListSorting.Sort(_discoveredMaps, _mapListSortMode).ToList();

        float listHeight = MathF.Min(300f, MathF.Max(120f, ImGui.GetContentRegionAvail().Y - 34f));
        if (ImGui.BeginChild("MapList", new Vector2(0, listHeight), true))
        {
            var style = ImGui.GetStyle();
            float rowHeight = GetUniformListRowHeight();
            GetVisibleListRange(sortedMaps.Count, rowHeight, out int startIndex, out int endIndex);
            if (startIndex > 0)
                ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

            for (int i = startIndex; i < endIndex; i++)
            {
                var map = sortedMaps[i];
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
                        _worldLoader.LoadMapAtDefaultSpawn(map);
                }

                if (!hasWdt) ImGui.PopStyleColor();

                if (hasWdt)
                {
                    ImGui.SameLine();
                    if (ImGui.SmallButton($"Load##{map.Directory}"))
                        _worldLoader.LoadMapAtDefaultSpawn(map);
                }

                bool canPreview = hasWdl && _wdlPreview.CanUseWdlPreviewFeature();
                WdlPreviewWarmState previewState = canPreview && _wdlPreviewCacheService != null
                    ? _wdlPreviewCacheService.GetState(map.Directory)
                    : (canPreview ? WdlPreviewWarmState.Ready : WdlPreviewWarmState.NotQueued);
                bool canSelectSpawn = hasWdt && canPreview && previewState != WdlPreviewWarmState.Failed;

                ImGui.SameLine();
                if (!canSelectSpawn) ImGui.BeginDisabled();
                if (ImGui.SmallButton($"Spawn##{map.Directory}") && canSelectSpawn)
                    _wdlPreview.OpenWdlPreview(map);
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
                    _dataSourceSession.RefreshFileList();
                }
            }
            ImGui.EndCombo();
        }

        var search = _searchFilter;
        if (ImGui.InputText("Search", ref search, 256))
        {
            _searchFilter = search;
            _dataSourceSession.RefreshFileList();
        }

        if (TryGetSelectedBrowserAssetPath(out string selectedAssetPath))
        {
            if (ImGui.Button("Open Selected"))
                _modelLoader.LoadFileFromDataSource(selectedAssetPath);

            ImGui.SameLine();
            if (ImGui.Button("Copy Path"))
                CopyTextToClipboard(selectedAssetPath, "asset path");

            if (_taxiAndAreaPoi.TryGetTaxiActorOverrideRouteId(out _)
                && TaxiAndAreaPoiSelectionService.IsTaxiActorModelPath(selectedAssetPath))
            {
                ImGui.SameLine();
                if (ImGui.Button("Use For Taxi Override"))
                    _taxiAndAreaPoi.TryApplySelectedBrowserAssetToTaxiOverride();
            }

            ImGui.TextDisabled(selectedAssetPath);
        }

        if (_dataSourceSession.HasWorldReturnTarget() && _worldScene == null)
        {
            if (ImGui.Button("Return To Last World"))
                _dataSourceSession.ReturnToLastWorldScene();
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
                        _modelLoader.LoadFileFromDataSource(file);
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
}
