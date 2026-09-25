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

    private static string FormatBytes(long bytes)
    {
        const double kib = 1024.0;
        const double mib = kib * 1024.0;
        const double gib = mib * 1024.0;

        if (bytes >= gib)
            return $"{bytes / gib:0.00} GiB";
        if (bytes >= mib)
            return $"{bytes / mib:0.0} MiB";
        if (bytes >= kib)
            return $"{bytes / kib:0.0} KiB";
        return $"{bytes} B";
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
        if (_worldScene != null)
            width += MeasureToolbarCheckboxWidth("WDL");
        return width;
    }

    private void DrawDirectTerrainToolbarControls(TerrainRenderer renderer, LiquidRenderer? liquidRenderer)
    {
        // Layer toggles moved to bottom bar. Only keep WDL in toolbar.
        if (_worldScene != null)
        {
            bool showWdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("WDL", ref showWdl))
                _worldScene.ShowWdlTerrain = showWdl;
        }
    }

    private void DrawBottomBar()
    {
        var io = ImGui.GetIO();
        float bottomBarY = io.DisplaySize.Y - BottomBarHeight - StatusBarHeight;

        ImGui.SetNextWindowPos(new Vector2(0, bottomBarY));
        ImGui.SetNextWindowSize(new Vector2(io.DisplaySize.X, BottomBarHeight));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(8, 6));
        ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, new Vector2(6, 0));
        if (ImGui.Begin("##BottomBar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings))
        {
            TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

            if (renderer != null)
            {
                // Grid toggles (operator-specified order 2026-09-06: Tiles, Chunks, Cells)
                bool tileGrid = renderer.ShowTileGrid;
                if (ImGui.Checkbox("Tiles", ref tileGrid)) renderer.ShowTileGrid = tileGrid;
                ImGui.SameLine();
                bool chunkGrid = renderer.ShowChunkGrid;
                if (ImGui.Checkbox("Chunks", ref chunkGrid)) renderer.ShowChunkGrid = chunkGrid;
                ImGui.SameLine();
                bool cellGrid = renderer.ShowCellGrid;
                if (ImGui.Checkbox("Cells", ref cellGrid)) renderer.ShowCellGrid = cellGrid;

                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                ImGui.SameLine();

                // Layer visibility (single source of truth)
                DrawTerrainLayerToggles(renderer);
                ImGui.SameLine();
                bool terrainHolesEnabled = !(_terrainManager?.IgnoreTerrainHolesGlobally
                    ?? _vlmTerrainManager?.IgnoreTerrainHolesGlobally
                    ?? false);
                if (ImGui.Checkbox("Holes", ref terrainHolesEnabled))
                {
                    if (_terrainControlsPanel.SetIgnoreTerrainHolesGlobally(!terrainHolesEnabled))
                    {
                        _statusMessage = terrainHolesEnabled
                            ? "Terrain hole masking enabled."
                            : "Terrain hole masking disabled.";
                    }
                }

                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                ImGui.SameLine();

                // Surface overlays
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
                }

                if (_worldScene != null)
                {
                    if (liquidRenderer != null)
                        ImGui.SameLine();

                    bool showWlTop = _worldScene.ShowWlLiquids;
                    if (ImGui.Checkbox("WL*", ref showWlTop))
                        _worldScene.ShowWlLiquids = showWlTop;

                    ImGui.SameLine();
                    bool showBB = _worldScene.ShowBoundingBoxes;
                    if (ImGui.Checkbox("BBs", ref showBB))
                        _worldScene.ShowBoundingBoxes = showBB;

                    ImGui.SameLine();
                    bool showPm4 = _worldScene.Pm4Overlay.ShowPm4Overlay;
                    if (ImGui.Checkbox("PM4", ref showPm4))
                        _worldScene.Pm4Overlay.ShowPm4Overlay = showPm4;
                    ImGui.SameLine();
                    if (ImGui.SmallButton("Inspect##inspect_pm4_btn"))
                    {
                        _worldScene.Pm4Overlay.ShowPm4Overlay = true;
                        _pm4Workbench.OpenPm4Workbench(Pm4WorkbenchTab.Selection);
                    }
                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("Enable PM4 overlay and focus PM4 selection & collision inspector");

                    if (_worldScene.Pm4Overlay.IsPm4Loading)
                    {
                        ImGui.SameLine();
                        ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), "loading");
                    }
                    else if (_worldScene.Pm4Overlay.ShowPm4Overlay && ImGui.IsItemHovered())
                    {
                        ImGui.SetTooltip(_worldScene.Pm4Overlay.Pm4Status);
                    }
                }

                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                ImGui.SameLine();

                if (_worldScene != null)
                {
                    bool terrainWireframe = _worldScene.TerrainWireframeEnabled;
                    if (ImGui.Checkbox("Terrain WF", ref terrainWireframe))
                        _worldScene.SetTerrainWireframeEnabled(terrainWireframe);

                    ImGui.SameLine();
                    bool objectWireframe = _worldScene.ObjectWireframeEnabled;
                    if (ImGui.Checkbox("M2/WMO WF", ref objectWireframe))
                        _worldScene.SetObjectWireframeEnabled(objectWireframe);

                    ImGui.SameLine();
                    bool worldAnimations = _worldScene.WorldDoodadAnimationEnabled;
                    if (ImGui.Checkbox("Anim", ref worldAnimations))
                        _worldScene.WorldDoodadAnimationEnabled = worldAnimations;
                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("Advance animations on world doodad MDX/M2 models.");

                    // Fast doodad-set switching for the WMO under the cursor, always visible
                    // in the editor toolbar (the Inspector's live-switch section is buried).
                    // While the combo popup is open, the hovered WMO is frozen: moving the
                    // mouse off the model to reach the dropdown would otherwise retarget or
                    // collapse the combo mid-interaction (operator report 2026-09-07).
                    bool doodadSetComboOpen = ImGui.IsPopupOpen("##HoveredWmoDoodadSet");
                    if (doodadSetComboOpen && _hoveredWmoDoodadSetComboWmo is { DoodadSetCount: > 0 } frozenWmo)
                    {
                        ImGui.SameLine();
                        _modelInspector.DrawHoveredWmoDoodadSetCombo(frozenWmo, _hoveredWmoDoodadSetComboSourcePath);
                    }
                    else if (_worldScene.HoveredAssetInfo is { } hoveredAsset
                        && hoveredAsset.SceneObjectType == Terrain.ObjectType.Wmo
                        && hoveredAsset.SceneObjectIndex >= 0)
                    {
                        WmoRenderer? hoveredWmo = _worldScene.Assets.GetWmo(
                            WorldAssetManager.NormalizeKey(hoveredAsset.SourcePath));
                        if (hoveredWmo is { DoodadSetCount: > 0 })
                        {
                            _hoveredWmoDoodadSetComboWmo = hoveredWmo;
                            _hoveredWmoDoodadSetComboSourcePath = hoveredAsset.SourcePath;
                            ImGui.SameLine();
                            _modelInspector.DrawHoveredWmoDoodadSetCombo(hoveredWmo, hoveredAsset.SourcePath);
                        }
                        else
                        {
                            _hoveredWmoDoodadSetComboWmo = null;
                        }
                    }
                    else
                    {
                        _hoveredWmoDoodadSetComboWmo = null;
                    }
                }
                else
                {
                    bool wireframe = _renderer?.IsWireframe ?? false;
                    if (ImGui.Checkbox("Wireframe", ref wireframe))
                        _renderer?.ToggleWireframe();
                }
            }

            if (renderer == null && _renderer != null)
            {
                bool wireframe = _renderer.IsWireframe;
                if (ImGui.Checkbox(_renderer is WmoRenderer ? "WMO WF" : "Model WF", ref wireframe))
                    _renderer.ToggleWireframe();
            }

            if (_renderer is WmoRenderer)
            {
                if (renderer != null || _renderer != null)
                {
                    ImGui.SameLine();
                    ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
                    ImGui.SameLine();
                }

                ImGui.Checkbox("WMO Group BBs", ref _standaloneWmoGroupOverlayEnabled);
                ImGui.SameLine();
                ImGui.Checkbox("Group Names", ref _standaloneWmoGroupLabelsAllEnabled);
            }

            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
            ImGui.SameLine();
            if (ImGui.Button("Settings"))
                _showSettingsWindow = true;

        }
        ImGui.End();
        ImGui.PopStyleVar(2);
    }


    private void DrawToolbar()
    {
        var io = ImGui.GetIO();
        float toolbarX = 0f;
        float toolbarWidth = io.DisplaySize.X;

        ImGui.SetNextWindowPos(new Vector2(toolbarX, MenuBarHeight));
        ImGui.SetNextWindowSize(new Vector2(toolbarWidth, ToolbarHeight));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(8, 6));
        ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, new Vector2(6, 0));
        if (ImGui.Begin("##Toolbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings))
        {
            DrawVisualInvestigationModeButton(VisualInvestigationMode.Auto, "\u25CE Auto", "Follow the current hovered visual target.");
            ImGui.SameLine();
            DrawVisualInvestigationModeButton(VisualInvestigationMode.Adt, "\u25A6 ADT", "Inspect terrain chunks, layers, alpha, and assigned MTEX textures.");
            ImGui.SameLine();
            DrawVisualInvestigationModeButton(VisualInvestigationMode.Wmo, "\u25A3 WMO", "Limit hover inspection to WMO placements.");
            ImGui.SameLine();
            DrawVisualInvestigationModeButton(VisualInvestigationMode.M2, "\u25C7 M2", "Limit hover inspection to MDX/M2 doodad placements.");
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
            ImGui.SameLine();

            TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

            if (renderer != null)
            {
                DrawDirectTerrainToolbarControls(renderer, liquidRenderer);
            }
            else
            {
                bool hasLoadedContent = HasLoadedContent();
                if (!hasLoadedContent)
                {
                    ImGui.TextDisabled("Welcome");
                    ImGui.SameLine();
                    ImGui.Text("Open a game folder or file from the left sidebar or File menu.");
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

        // These toggles are duplicated by the bottom display bar; keep the sidebar
        // section collapsed by default so the navigator leads with scene context.
        if (ImGui.CollapsingHeader("Layers, Grids & Overlays"))
            DrawLayersGridsOverlaysContent(renderer, liquidRenderer);
    }

    /// <summary>
    /// Base + L1..L3 always; L4..L7 appear once resident terrain has chunks with that many layers
    /// (modern maps carry up to 8 MCLY layers per chunk).
    /// </summary>
    private static void DrawTerrainLayerToggles(TerrainRenderer renderer)
    {
        int layerCount = Math.Clamp(renderer.MaxResidentLayerCount, 4, TerrainTileMeshBuilder.MaxLayers);
        for (int layer = 0; layer < layerCount; layer++)
        {
            if (layer > 0)
                ImGui.SameLine();

            bool visible = renderer.GetShowLayer(layer);
            if (ImGui.Checkbox(layer == 0 ? "Base" : $"L{layer}", ref visible))
                renderer.SetShowLayer(layer, visible);
        }
    }

    /// <summary>
    /// Terrain layer, grid, hole, and overlay toggles. Drawn collapsed inside the
    /// workspace sidebar (the bottom bar exposes the same controls) and reusable
    /// wherever the full toggle set is needed.
    /// </summary>
    private void DrawLayersGridsOverlaysContent(TerrainRenderer renderer, LiquidRenderer? liquidRenderer)
    {
        ImGui.TextDisabled("Terrain Layers");
        DrawTerrainLayerToggles(renderer);

        bool terrainHolesEnabled = !(_terrainManager?.IgnoreTerrainHolesGlobally
            ?? _vlmTerrainManager?.IgnoreTerrainHolesGlobally
            ?? false);
        if (ImGui.Checkbox("Holes", ref terrainHolesEnabled))
        {
            if (_terrainControlsPanel.SetIgnoreTerrainHolesGlobally(!terrainHolesEnabled))
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
        ImGui.SameLine();
        bool cellGrid = renderer.ShowCellGrid;
        if (ImGui.Checkbox("Cells", ref cellGrid)) renderer.ShowCellGrid = cellGrid;

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
            bool showPm4 = _worldScene.Pm4Overlay.ShowPm4Overlay;
            if (ImGui.Checkbox("PM4 Overlay", ref showPm4))
                _worldScene.Pm4Overlay.ShowPm4Overlay = showPm4;

            if (_worldScene.Pm4Overlay.IsPm4Loading)
                ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), "PM4 overlay is loading...");
        }
    }

    private void DrawLeftSidebar()
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

    private void DrawLegacyLeftSidebar()
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

    private void DrawNavigatorPanelContent()
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

        _rightSidebarWidth = ClampFixedSidebarWidth(_rightSidebarWidth, isLeftSidebar: false, io.DisplaySize.X);
        ImGui.SetNextWindowPos(new Vector2(io.DisplaySize.X - _rightSidebarWidth, topOffset), ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(_rightSidebarWidth, sidebarHeight), ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(6, 6));
        if (ImGui.Begin("##LegacyRightSidebar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings))
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
        DrawUiThemeSettingsContent();
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
            _settings.SaveViewerSettings();

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
                _pm4Workbench.DrawPm4WorkbenchInspector();
                break;
            case ShellPanelId.TerrainControls:
                _terrainControlsPanel.DrawTerrainControlsPanelContent();
                break;
            case ShellPanelId.RuntimeStats:
                DrawRuntimeStatsPanelContent();
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
            DrawAssetPathActions("Selected Asset", selectedModelPath, "SelectedWorldObject");
        }

        _modelInspector.DrawSelectedWmoControls();
        _sqlSpawnStreaming.DrawSelectedSqlGameObjectAnimationControls();
        return true;
    }

    private void DrawFixedSidebarSplitters()
    {
        if (_useDockspaceUi)
            return;

        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
        float panelHeight = io.DisplaySize.Y - topOffset - StatusBarHeight;
        if (panelHeight <= 0f)
            return;

        bool hasLeft = _useTabUi || _shellLayout.IsShellPanelActive(ShellPanelId.Navigator);
        bool hasRight = _useTabUi || _shellLayout.IsShellPanelActive(ShellPanelId.Inspector);

        if (hasLeft)
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

        if (hasRight)
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
            if (_shellLayout.IsShellPanelActive(ShellPanelId.Inspector))
                otherSidebarWidth = _rightSidebarWidth;
        }
        else if (_shellLayout.IsShellPanelActive(ShellPanelId.Navigator))
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
        bool hasSelectedPm4 = _worldScene?.Pm4Overlay.HasSelectedPm4Object == true;
        if (string.IsNullOrEmpty(_selectedObjectInfo) || hasSelectedPm4)
            return false;

        ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None;
        if (!ImGui.CollapsingHeader("Selected Object", flags))
            return true;

        DrawSelectedObjectSummaryContent();
        return true;
    }

    private void DrawRuntimeStatsPanelContent()
    {
        using Process process = Process.GetCurrentProcess();
        long managedHeap = GC.GetTotalMemory(forceFullCollection: false);
        long totalAllocated = GC.GetTotalAllocatedBytes(precise: false);
        ImGui.Text($"Process memory: working={FormatBytes(process.WorkingSet64)}  private={FormatBytes(process.PrivateMemorySize64)}");
        ImGui.Text($"Managed heap: live={FormatBytes(managedHeap)}  allocated={FormatBytes(totalAllocated)}  GC={GC.CollectionCount(0)}/{GC.CollectionCount(1)}/{GC.CollectionCount(2)}");

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
        ImGui.TextDisabled("Frame timing over time and hitch detection live on Utilities > Perf.");

        ImGui.Text($"Visible WMO: {renderStats.VisibleWmoCount}  Visible MDX: {renderStats.VisibleMdxCount}  Taxi actors: {renderStats.VisibleTaxiMdxCount}");
        ImGui.Text($"Object stream range: {_worldScene.ObjectStreamingRangeMultiplier:0.00}x");
        ImGui.Text($"Object detail: {_worldScene.ObjectVisibilityProfile}");
        var graphDiagnostics = _worldScene.SceneGraphTraversalDiagnostics;
        ImGui.Text($"ADT graph: {(_worldScene.IsHierarchicalSceneTraversalActive ? "active" : "inactive")}  roots={_worldScene.SceneGraphResidentAdtCount}  external={(_worldScene.SceneGraphHasExternalRoot ? "yes" : "no")}");
        // "skipped" requires per-kind attribution, which recursively walks every rejected subtree.
        // That is off on production frames, so it is opt-in here rather than a permanent frame cost.
        bool detailedGraphDiagnostics = _worldScene.SceneGraphDetailedDiagnosticsEnabled;
        if (ImGui.Checkbox("Detailed graph attribution (costs frame time)", ref detailedGraphDiagnostics))
            _worldScene.SceneGraphDetailedDiagnosticsEnabled = detailedGraphDiagnostics;
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Collecting skipped-descendant counts walks every culled subtree each frame.\nLeave off unless you are reading the breakdown.");

        string skippedText = graphDiagnostics.DetailedCollectionEnabled
            ? graphDiagnostics.SkippedDescendantCount.ToString()
            : "off";
        ImGui.Text($"Graph visited/tested/rejected/skipped: {graphDiagnostics.VisitedNodeCount}/{graphDiagnostics.IndividuallyTestedNodeCount}/{graphDiagnostics.RejectedNodeCount}/{skippedText}");
        ImGui.Text($"AOI camera tile: ({_worldScene.Terrain.CameraTileX},{_worldScene.Terrain.CameraTileY})  loaded={_worldScene.Terrain.LoadedTileCount}  detailed/retained={_worldScene.Terrain.EffectiveDetailedTileCount}/{_worldScene.Terrain.EffectiveRetainedTileCount}");
        if (_worldScene.Terrain.TileUnloadEventCount > 0)
            ImGui.Text($"Last ADT unload: ({_worldScene.Terrain.LastUnloadedTileX},{_worldScene.Terrain.LastUnloadedTileY})  WMO placements={_worldScene.LastUnloadedWmoInstanceCount}");
        ImGui.Text($"Terrain chunks rendered/culled: {renderStats.TerrainChunksRendered}/{renderStats.TerrainChunksCulled}  WDL visible/hidden: {renderStats.WdlVisibleTileCount}/{renderStats.WdlHiddenTileCount}");
        if (terrainRenderer != null)
            ImGui.Text($"Terrain draw/uniform/tex-bind: {terrainRenderer.LastFrameDrawCalls}/{terrainRenderer.LastFrameUniform1Calls}/{terrainRenderer.LastFrameBindTextureCalls}");
        ImGui.Text($"Deferred/taxi/light: {renderStats.DeferredAssetLoads.DurationMs:0.00} / {renderStats.TaxiActorUpdate.DurationMs:0.00} / {renderStats.Lighting.DurationMs:0.00} ms");
        ImGui.Text($"WDL/terrain/liquid: {renderStats.Wdl.DurationMs:0.00} / {renderStats.Terrain.DurationMs:0.00} / {renderStats.Liquid.DurationMs:0.00} ms");
        if (renderStatsLiquidRenderer != null)
            ImGui.Text($"Liquid visible: {renderStatsLiquidRenderer.LastVisibleTerrainMeshCount}/{renderStatsLiquidRenderer.MeshCount}  WL: {renderStatsLiquidRenderer.LastVisibleWlMeshCount}/{renderStatsLiquidRenderer.WlMeshCount}");
        ImGui.Text($"WMO vis/opaque/trans: {renderStats.WmoVisibility.DurationMs:0.00} / {renderStats.WmoSubmission.DurationMs:0.00} / {renderStats.WmoTransparentSubmission.DurationMs:0.00} ms");
        ImGui.Text($"WMO draws batch/fallback/liquid/doodad: {renderStats.WmoBatchDrawCallCount}/{renderStats.WmoGroupFallbackDrawCallCount}/{renderStats.WmoLiquidDrawCallCount}/{renderStats.WmoDoodadSubmissionCount}  instances={renderStats.WmoOpaqueBatchInstanceCount} groups={renderStats.WmoVisibleGroupSubmissionCount}");
        ImGui.Text($"MDX anim/vis/opaque: {renderStats.MdxAnimation.DurationMs:0.00} / {renderStats.MdxVisibility.DurationMs:0.00} / {renderStats.MdxOpaqueSubmission.DurationMs:0.00} ms");
        ImGui.Text($"MDX sort/trans: {renderStats.MdxTransparentSort.DurationMs:0.00} / {renderStats.MdxTransparentSubmission.DurationMs:0.00} ms");
        ImGui.Text($"Models opaque inst/hoisted/unbatched: {renderStats.OpaqueModelSubmission.Instanced}/{renderStats.OpaqueModelSubmission.StateHoisted}/{renderStats.OpaqueModelSubmission.Unbatched}"
            + $"  transparent: {renderStats.TransparentModelSubmission.Unbatched}  draw calls: {renderStats.OpaqueModelSubmission.DrawCalls}+{renderStats.TransparentModelSubmission.DrawCalls}");
        ImGui.Text($"Sky/backdrop/overlay: {renderStats.Sky.DurationMs:0.00} / {renderStats.SkyboxBackdrop.DurationMs:0.00} / {renderStats.Overlay.DurationMs:0.00} ms");
        ImGui.TextWrapped(_worldScene.RendererOptimizationHint);

        var assetReadStats = _worldScene.Assets.GetReadStats();
        ImGui.Separator();
        ImGui.Text($"Asset I/O req/cache: {assetReadStats.ReadRequests}/{assetReadStats.FileCacheHits}  resolved-cache: {assetReadStats.ResolvedPathCacheHits}  probes hit/miss: {assetReadStats.PathProbeResolutions}/{assetReadStats.PathProbeMisses}");
        ImGui.Text($"Asset raw cache: {assetReadStats.FileCacheCount} files / {FormatBytes(assetReadStats.FileCacheBytes)}");

        // Spec 153 Phase 5. DeferredLoadBudget records these and, until now, nothing read them.
        // OversizedAdmissionCount is documented in that class as "the honest measure of the residual
        // the off-thread decode still owes": every one of these is a frame that paid a full
        // synchronous load because the policy guarantees progress even when the load cannot fit.
        DeferredLoadBudget loadBudget = _worldScene.Assets.LoadBudget;
        ImGui.Text($"Deferred loads: oversized admissions {loadBudget.OversizedAdmissionCount}"
            + $"  budget deferrals {loadBudget.BudgetDeferralCount}"
            + $"  worst single load {loadBudget.WorstObservedLoadMs:0.0} ms");
        ImGui.Text($"  predicted cost: MDX {loadBudget.PredictedCostMs(DeferredLoadKind.Mdx):0.0} ms"
            + $"  WMO {loadBudget.PredictedCostMs(DeferredLoadKind.Wmo):0.0} ms");
        if (loadBudget.OversizedAdmissionCount > 0)
        {
            ImGui.TextWrapped(
                "Oversized admissions are synchronous loads larger than the whole frame budget, "
                + "admitted anyway so the asset eventually appears. They are the DeferredAssetLoads "
                + "hitches. The budget cannot subdivide one load - only moving decode off the render "
                + "thread can.");
        }
        ImGui.Text($"Asset misses: failed retry suppress={_worldScene.Assets.SuppressedFailedMdxRetryCount}  known missing M2 skins={_worldScene.Assets.KnownMissingM2SkinCount}  duplicate skin logs={_worldScene.Assets.SuppressedMissingM2SkinLogCount}");

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            var mpqStats = mpqDataSource.GetStatsSnapshot();
            ImGui.Text($"MPQ I/O read cache/miss: {mpqStats.ReadCacheHits}/{mpqStats.ReadCacheMisses}  loose/alpha/mpq/miss: {mpqStats.ReadLooseHits}/{mpqStats.ReadAlphaHits}/{mpqStats.ReadMpqHits}/{mpqStats.ReadMisses}  uncached avg: {mpqStats.AverageUncachedReadMs:0.00} ms");
            ImGui.Text($"MPQ raw cache: {mpqStats.ReadCacheEntryCount} files / {FormatBytes(mpqStats.ReadCacheBytes)}  prefetch queue: {mpqStats.PrefetchQueueDepth}");
            ImGui.Text($"MPQ prefetch enq/done/dup/cache: {mpqStats.PrefetchEnqueued}/{mpqStats.PrefetchCompleted}/{mpqStats.PrefetchDuplicateSkips}/{mpqStats.PrefetchCacheSkips}  queue avg: {mpqStats.AveragePrefetchQueueMs:0.00} ms  read avg: {mpqStats.AveragePrefetchReadMs:0.00} ms");
        }
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

        _rightSidebarWidth = ClampFixedSidebarWidth(_rightSidebarWidth, isLeftSidebar: false, io.DisplaySize.X);
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
            DrawMcnkFlagOverlayControls();
        }

        InspectorContextSection requested = _pendingInspectorContextSection;
        if (requested == InspectorContextSection.None)
            return;

        _pendingInspectorContextSection = InspectorContextSection.None;
        switch (requested)
        {
            case InspectorContextSection.SceneInvestigation:
                if (SharedUiWidgets.SectionHeader("Scene Investigation", defaultOpen: true, id: "InspectorSceneInvestigation"))
                    DrawVisualInvestigationToolbox(showWorldObjectRangeControls: _worldScene != null);
                break;
            case InspectorContextSection.Mcnk:
                if ((_terrainManager != null || _vlmTerrainManager != null)
                    && SharedUiWidgets.SectionHeader("MCNK Flag Overlay", defaultOpen: true, id: "InspectorMcnkFlagsLegacy"))
                    DrawMcnkFlagOverlayControls();
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

    internal static void DrawTimeOfDayControl(TerrainLighting lighting)
    {
        bool automatic = lighting.IsAutomaticTimeOfDay;
        if (ImGui.Checkbox("Automatic 24-minute cycle", ref automatic))
            lighting.SetAutomaticTimeOfDay(automatic);

        float gameTime = lighting.GameTime;
        if (ImGui.SliderFloat("Time of Day", ref gameTime, 0f, 1f, "%.2f"))
        {
            lighting.GameTime = gameTime;
            lighting.HasManualGameTimeOverride = true;
        }

        ImGui.TextDisabled(lighting.IsAutomaticTimeOfDay
            ? "Live Alpha 0.5.3 clock: one game day per 24 real minutes."
            : "Time frozen at the selected value; enable the cycle to resume.");
    }

    // Frozen hovered-WMO source for the toolbar doodad-set quick combo: while the
    // combo popup is open the reference must not follow the mouse (operator bug
    // report 2026-09-07 — the dropdown collapsed as soon as the cursor left the WMO).
    private WmoRenderer? _hoveredWmoDoodadSetComboWmo;
    private string _hoveredWmoDoodadSetComboSourcePath = string.Empty;

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
            DrawTimeOfDayControl(lighting);
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
        DrawUiThemeSettingsContent();
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
                DrawRuntimeStatsPanelContent();
                break;
            case UtilitiesBottomTab.Lighting:
                DrawLightingContent();
                break;
            case UtilitiesBottomTab.Audio:
                DrawAudioContent();
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
