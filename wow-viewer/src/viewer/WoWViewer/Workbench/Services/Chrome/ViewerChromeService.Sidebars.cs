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

namespace WoWViewer;

// ViewerChromeService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class ViewerChromeService
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

    internal void DrawToolbarPopupButton(string label, string summary, string popupId, Action drawContent)
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

    internal void DrawBottomBar()
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


    internal void DrawToolbar()
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

    internal void DrawWorkspaceBarsPanelContent()
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

    internal void DrawFixedSidebarWidthControl(string label, ref float width, bool isLeftSidebar, float displayWidth, string tooltip)
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

    internal void DrawFixedSidebarSplitters()
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

    internal float ClampFixedSidebarWidth(float width, bool isLeftSidebar, float displayWidth)
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

    internal void DrawRuntimeStatsPanelContent()
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
}
