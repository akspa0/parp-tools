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
using static WoWViewer.ImGuiListLayout;
using static WoWViewer.MinimapAndStatusService;

namespace WoWViewer;

// NavigatorPanelService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class NavigatorPanelService
{

    internal void DrawWorkspaceContent()
    {
        if (SharedUiWidgets.ActionButton("Open Game Folder..."))
        {
            _host.ShowFolderInput = true;
            _host.FolderInputBuf = string.IsNullOrWhiteSpace(_host.LastGameFolderPath) ? "" : _host.LastGameFolderPath;
        }
        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Open File...")) _host.WantOpenFile = true;
        _host.WorkbenchPanels.DrawCameraContent();
        if (SharedUiWidgets.SectionHeader("Fog", defaultOpen: true, id: "WorkspaceFog"))
            _host.DrawAuthoritativeFogControls(showDescription: false);
    }

    internal void DrawMapsContent()
    {
        DrawWorldOverviewContent();
        DrawMapExportControls();
        DrawMapDiscoveryContent();
    }

    internal void DrawFilesContent() => DrawFileBrowserContent();

    internal void DrawDiagnosticsContent()
    {
        if (SharedUiWidgets.ActionButton("Log")) _host.ShowLogViewer = !_host.ShowLogViewer;
        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Performance")) _host.ShowPerfWindow = !_host.ShowPerfWindow;
        _host.ViewerChrome.DrawRuntimeStatsPanelContent();
    }

    internal void DrawProjectOutputContent()
    {
        if (SharedUiWidgets.SectionHeader("Project Output", id: "ProjectOutput"))
        {
            if (ImGui.InputText("Output root", ref _host.ProjectOutputRootDir, 1024))
                _host.HandleProjectOutputRootChanged();
            ImGui.TextWrapped(_host.GetProjectOutputRootDirectory());
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
        SharedUiWidgets.SectionTitle("Export Map");
        Terrain.MapExportFormats.DrawCheckboxes("sidebar");

        ImGui.BeginDisabled(!Terrain.MapExportFormats.Any);
        if (ImGui.Button($"Export as {Terrain.MapExportFormats.Summary}...", new Vector2(-1f, 0f)))
            _cascAhdrSource.ExportLoadedDatMap();
        ImGui.EndDisabled();

        if (ImGui.IsItemHovered() && Terrain.MapExportFormats.Any)
            ImGui.SetTooltip("Writes the loaded DAT folder in every ticked format, plus a manifest naming every field carried and dropped.");
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
            ImGui.TextDisabled($"Camera tile: ({tileY}, {tileX})");
        }

        if (!string.IsNullOrWhiteSpace(_currentAreaName))
            ImGui.TextDisabled($"Area: {_currentAreaName}");

        if (_worldScene != null && (_worldScene.Pm4Overlay.ShowPm4Overlay || _worldScene.Pm4Overlay.Pm4LoadAttempted))
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4Overlay.Pm4VisibleObjectCount}/{_worldScene.Pm4Overlay.Pm4ObjectCount} visible objects");

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
                bool clicked = hasWdt
                    ? ImGui.Selectable(label, false, ImGuiSelectableFlags.AllowDoubleClick, new Vector2(labelWidth, 0f))
                    : SharedUiWidgets.MutedSelectable(label, false, ImGuiSelectableFlags.AllowDoubleClick, new Vector2(labelWidth, 0f));

                if (clicked && hasWdt && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    _worldLoader.LoadMapAtDefaultSpawn(map);

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
                        SharedUiWidgets.SuccessText("WDL preview ready. Click 'Spawn' to choose a start tile.");
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
        _fileBrowserExtras.DrawOpenWc3(
            (root, listfile) => _dataSourceSession.LoadWc3DataSource(root, listfile),
            _dataSource is Wc3DataSource wc3Source ? wc3Source.Summary : null);

        if (_dataSource == null || !_dataSource.IsLoaded)
        {
            SharedUiWidgets.DrawEmptyState(
                "No data source loaded",
                "Use File > Open Game Folder to load MPQ archives, or 'Open WC3 Beta data' above.",
                "Open Game Folder...",
                () =>
                {
                    _host.ShowFolderInput = true;
                    _host.FolderInputBuf = string.IsNullOrWhiteSpace(_host.LastGameFolderPath) ? "" : _host.LastGameFolderPath;
                });
            return;
        }

        ImGui.Text($"Source: {_dataSource.Name}");
        ImGui.Separator();

        if (ImGui.BeginCombo("Type", GetExtensionFilterLabel(_extensionFilter)))
        {
            (string value, string label)[] filters =
            {
                (DataSourceSessionService.AllFilesFilter, "All files"),
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
            if (SharedUiWidgets.ActionButton("Open Selected"))
                _modelLoader.LoadFileFromDataSource(selectedAssetPath);

            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Copy Path"))
                CopyTextToClipboard(selectedAssetPath, "asset path");

            if (_taxiAndAreaPoi.TryGetTaxiActorOverrideRouteId(out _)
                && TaxiAndAreaPoiSelectionService.IsTaxiActorModelPath(selectedAssetPath))
            {
                ImGui.SameLine();
                if (SharedUiWidgets.ActionButton("Use For Taxi Override"))
                    _taxiAndAreaPoi.TryApplySelectedBrowserAssetToTaxiOverride();
            }

            ImGui.TextDisabled(selectedAssetPath);
        }

        if (_dataSourceSession.HasWorldReturnTarget() && _worldScene == null)
        {
            if (SharedUiWidgets.ActionButton("Return To Last World"))
                _dataSourceSession.ReturnToLastWorldScene();
        }

        _fileBrowserExtras.DrawExport(_dataSource,
            TryGetSelectedBrowserAssetPath(out string exportSelected) ? exportSelected : null,
            _filteredFiles, message => _statusMessage = message);

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

            var mipCounts = _dataSourceSession.MipCounts;
            for (int i = startIndex; i < endIndex; i++)
            {
                var file = _filteredFiles[i];
                var displayName = Path.GetFileName(file);
                if (mipCounts.TryGetValue(file, out int mips))
                    displayName += $"  [{mips} mip{(mips == 1 ? "" : "s")}]";
                bool selected = i == _selectedFileIndex;

                ImGui.PushID(i);
                if (ImGui.Selectable(displayName, selected, ImGuiSelectableFlags.AllowDoubleClick))
                {
                    _selectedFileIndex = i;
                    if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        _modelLoader.LoadFileFromDataSource(file);
                    else if (_dataSource is Wc3DataSource && DataFileViewerService.IsImagePath(file))
                        _modelLoader.LoadFileFromDataSource(file); // WC3 folders have no world to lose: preview on click.
                }

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip(file);
                ImGui.PopID();
            }

            if (endIndex < _filteredFiles.Count)
                ImGui.Dummy(new Vector2(0, (_filteredFiles.Count - endIndex) * rowHeight));
            ImGui.EndChild();
        }
    }

    private static string GetExtensionFilterLabel(string extensionFilter)
    {
        if (extensionFilter == DataSourceSessionService.AllFilesFilter)
            return "All files";

        return extensionFilter.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            ? ".mdx/.mdl"
            : extensionFilter;
    }
}
