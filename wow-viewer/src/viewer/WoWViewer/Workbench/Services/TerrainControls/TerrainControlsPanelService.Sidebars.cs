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
using static WoWViewer.ViewerChromeService;

namespace WoWViewer;

// TerrainControlsPanelService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class TerrainControlsPanelService
{

    internal void DrawTerrainControlsPanelContent()
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
            _terrainTileIo.GetTerrainTileRange(out int startX, out int startY, out int endX, out int endY);
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
            _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
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
            _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
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

        ImGui.Text($"Focused ADT: ({focusedTile.tileY}, {focusedTile.tileX})");
        ImGui.SameLine();
        if (ImGui.SmallButton("Use Camera Tile"))
        {
            _terrainWorkbenchFocusedTile = GetCameraTile();
            _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Clear Tile Range"))
        {
            _terrainTileScope = TerrainTileScope.CurrentTile;
            _terrainTileRangeStartX = focusedTile.tileX;
            _terrainTileRangeEndX = focusedTile.tileX;
            _terrainTileRangeStartY = focusedTile.tileY;
            _terrainTileRangeEndY = focusedTile.tileY;
            _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
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
            _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
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
            _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
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
        _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
    }

    private void ApplyTerrainWeakSignalRestoreQuickRange(float minHeight, float maxHeight)
    {
        _terrainWeakSignalRestoreCandidateMinHeight = TerrainWeakSignalRestoreService.ClampTerrainWeakSignalRestoreZ(minHeight);
        _terrainWeakSignalRestoreCandidateMaxHeight = TerrainWeakSignalRestoreService.ClampTerrainWeakSignalRestoreZ(maxHeight);
        _terrainWeakSignalRestore.GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
        _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
        _settings.SaveViewerSettings();
    }

    internal void DrawTerrainControlsAdjustmentContent()
    {
        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (lighting == null || renderer == null) return;

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

        // Keep every terrain route on the same fog state owner. Reading the
        // renderer-facing TerrainLighting fields here would overwrite a drag on
        // the following frame when WorldScene recomposes DBC/LIT lighting.
        DrawAuthoritativeFogControls(showDescription: false);

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
                if (_terrainWeakSignalRestore.SetTerrainWeakSignalRestoreEnabled(weakSignalRestore))
                    _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Amplify weak, era-compressed terrain on the camera tile and its four direct neighbors, then clamp the actual motion to weak per-cell signal regions across the ADT instead of picking one whole chunk or one whole texture bucket.");

            ImGui.TextDisabled("Mode: whole-tile factor, per-cell weak-signal clamp.");

            float restoreRangeMin = _terrainWeakSignalRestoreCandidateMinHeight;
            if (ImGui.InputFloat("Restore Range Min Z", ref restoreRangeMin, 10f, 100f, "%.1f"))
            {
                _terrainWeakSignalRestoreCandidateMinHeight = TerrainWeakSignalRestoreService.ClampTerrainWeakSignalRestoreZ(restoreRangeMin);
                _terrainWeakSignalRestore.GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Early-era buried terrain tends to sit around -10..10. Later-era ocean-floor-compressed data can need something closer to -5000..10.");

            float restoreRangeMax = _terrainWeakSignalRestoreCandidateMaxHeight;
            if (ImGui.InputFloat("Restore Range Max Z", ref restoreRangeMax, 10f, 100f, "%.1f"))
            {
                _terrainWeakSignalRestoreCandidateMaxHeight = TerrainWeakSignalRestoreService.ClampTerrainWeakSignalRestoreZ(restoreRangeMax);
                _terrainWeakSignalRestore.GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
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
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Use the WDL-backed whole-tile auto estimate, then clamp the resulting deformation to weak per-cell signal regions across the ADT. Turn this off to A/B the manual restore control instead.");

            var wdlGuideTile = GetCameraTile();
            if (_terrainWeakSignalRestore.TryGetTerrainWeakSignalWdlTile(wdlGuideTile.tileX, wdlGuideTile.tileY, out var wdlGuide) && wdlGuide != null)
            {
                ImGui.TextDisabled($"WDL guide ({wdlGuideTile.tileX}, {wdlGuideTile.tileY}): {wdlGuide.MinZ:F1}..{wdlGuide.MaxZ:F1}, center {wdlGuide.Height17[8, 8]:F1}, 17x17 + 16x16 samples");
            }
            else
            {
                ImGui.TextDisabled($"WDL guide ({wdlGuideTile.tileX}, {wdlGuideTile.tileY}): no tile data available");
            }

            if (!_terrainWeakSignalRestoreUseAutoFactor)
            {
                DrawStratigraphyFactorControl();
            }

            string restoreScopeSummary = _terrainWeakSignalRestore.GetTerrainWeakSignalRestoreScopeSummary();
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

    internal void DrawTerrainControlsContent()
    {
        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (lighting == null || renderer == null) return;

        DrawTerrainControlsAdjustmentContent();

        ImGui.Separator();
        ImGui.TextDisabled("Open Terrain Tools, Chunk Clipboard, Terrain Analysis, and MCNK Explorer from the Tools menu.");
    }

    internal bool SetIgnoreTerrainHolesGlobally(bool enabled)
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

    internal void DrawSharedChunkClipboardSection(TerrainRenderer? renderer, bool withHeader = true, string headerTitle = "Chunk Clipboard")
    {
        if (renderer == null)
        {
            ImGui.TextDisabled("Terrain renderer not available for clipboard.");
            return;
        }

        if (withHeader)
        {
            if (ImGui.CollapsingHeader(headerTitle, ImGuiTreeNodeFlags.DefaultOpen))
                _chunkEdit.DrawChunkClipboardContent(renderer);
        }
        else
        {
            _chunkEdit.DrawChunkClipboardContent(renderer);
        }
    }

    internal void DrawWeakSignalWindow()
    {
        // 069 Phase 16: wrapper keeps legacy floating-window behavior.
        // Workbench sub-tab uses DrawWeakSignalContent directly.
        ImGui.SetNextWindowSize(new Vector2(400f, 480f), ImGuiCond.FirstUseEver);
        if (ImGui.Begin("Weak Signal Amplifier", ref _showWeakSignalWindow))
        {
            DrawWeakSignalContent();
        }
        ImGui.End();
    }

    private void DrawWeakSignalContent()
    {
        DrawTerrainControlsAdjustmentWeakSignalContent();
    }

    internal void DrawTerrainControlsAdjustmentWeakSignalContent()
    {
        DrawTemporalStratigraphySubTab();
    }

    internal void DrawTemporalStratigraphySubTab()
    {
        if (_terrainManager == null && _vlmTerrainManager == null && _worldScene == null)
        {
            ImGui.TextDisabled("Load a terrain-backed world or map to inspect temporal stratigraphy.");
            return;
        }

        ImGui.Text("Temporal Stratigraphy & Dev Mesh Restoration");
        ImGui.TextDisabled("Recover compressed historical development terrain (1/0.03 = 33.334x) & hidden HoleMask dev geometry.");
        ImGui.Spacing();

        // 1. Live Camera Tile & Analysis Summary
        var cameraTile = GetCameraTile();
        if (_stratigraphyTileAnalyses.TryGetValue((cameraTile.tileX, cameraTile.tileY), out var analysis))
        {
            ImGui.TextColored(new Vector4(0.3f, 0.8f, 1f, 1f), $"Tile ({cameraTile.tileY}, {cameraTile.tileX}) Dominant Stratum: {analysis.DominantStratum}");
            ImGui.TextDisabled($"Surviving Levels: {analysis.TotalSurvivingLevels:N0} | Z: {analysis.MinHeight:F2}m .. {analysis.MaxHeight:F2}m ({analysis.HeightRange:F3}m range)");
            ImGui.TextDisabled($"Chunks: {analysis.ActiveChunkCount} Active, {analysis.SqueezedChunkCount} Squeezed, {analysis.HoledChunkCount} Holed, {analysis.FlatChunkCount} Flat");
            if (analysis.SeamProfile != null)
            {
                ImGui.TextDisabled($"Seam Profile: {analysis.SeamProfile.InferredMergeOrigin}");
            }
        }
        else
        {
            ImGui.TextDisabled($"Tile ({cameraTile.tileY}, {cameraTile.tileX}) — Click 'Analyze Active Tile' for full stratigraphic breakdown.");
        }

        ImGui.Separator();

        // 2. Master Restoration Toggle
        bool restoreEnabled = _terrainWeakSignalRestoreEnabled;
        if (ImGui.Checkbox("Enable Temporal Stratigraphy Restoration", ref restoreEnabled))
        {
            _terrainWeakSignalRestore.SetTerrainWeakSignalRestoreEnabled(restoreEnabled);
            _settings.SaveViewerSettings();
        }

        if (_terrainWeakSignalRestoreEnabled)
        {
            ImGui.Spacing();
            DrawStratigraphyFactorControl();

            ImGui.Spacing();
            bool unhideHoles = _stratigraphyUnhideDevMeshes;
            if (ImGui.Checkbox("Unhide Dev Meshes (Bypass HoleMask)", ref unhideHoles))
            {
                _stratigraphyUnhideDevMeshes = unhideHoles;
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Renders intact full-scale geometry hidden behind MCNK HoleMask flags (dev caves, subterranean paths, Outland blockouts).");

            ImGui.SameLine();
            bool stitch = _stratigraphyStitchBoundaries;
            if (ImGui.Checkbox("Stitch Active Boundaries", ref stitch))
            {
                _stratigraphyStitchBoundaries = stitch;
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Smoothly feathers height deltas at borders adjoining full-scale active terrain to prevent cliff edge artifacts.");

            bool preserveFloor = _stratigraphyPreserveNegativeFloor;
            if (ImGui.Checkbox("Preserve Negative Elevation Floor", ref preserveFloor))
            {
                _stratigraphyPreserveNegativeFloor = preserveFloor;
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("When minimum Z is negative, anchors scaling to the negative floor so sunken basins and deep valleys scale downward naturally.");

            ImGui.Spacing();
            bool invertPolarity = _stratigraphyPolarityInverted;
            if (ImGui.Checkbox("Invert Polarity (Negative Scaling / Dragon Isles Fix)", ref invertPolarity))
            {
                _stratigraphyPolarityInverted = invertPolarity;
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Inverts scale factor (-1x) so developmental terrain compressed against a ceiling reconstructs downward without massive vertical wall spikes.");

            int anchorModeIndex = (int)_stratigraphyAnchorMode;
            string[] anchorModeLabels = ["Lowest Z (Floor)", "Highest Z (Ceiling / Inverted)", "Mean Z", "Neighbor Edge", "WDL Lattice", "Custom Datum"];
            if (anchorModeIndex >= anchorModeLabels.Length) anchorModeIndex = 0;
            ImGui.SetNextItemWidth(220f);
            if (ImGui.Combo("Anchor Datum", ref anchorModeIndex, anchorModeLabels, anchorModeLabels.Length))
            {
                _stratigraphyAnchorMode = (WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode)anchorModeIndex;
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }

            bool useAutoFit = _stratigraphyUseNeighborAutoFit;
            if (ImGui.Checkbox("Auto-Fit to Neighbor Mesh Heights (1-3 Chunk Radius)", ref useAutoFit))
            {
                _stratigraphyUseNeighborAutoFit = useAutoFit;
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Samples boundary vertices from adjacent active terrain within 1-3 chunks and calculates the optimal scale factor, polarity, and Z offset minimizing seam error.");

            bool useWdl = _stratigraphyUseWdlMagnetization;
            if (ImGui.Checkbox("Magnetize to WDL Macro-Lattice", ref useWdl))
            {
                _stratigraphyUseWdlMagnetization = useWdl;
                _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Uses low-frequency 17x17 WDL heights as macro topographical guides, adding high-frequency ADT weak signals as micro-relief.");

            if (_stratigraphyUseWdlMagnetization)
            {
                float strength = _stratigraphyWdlMagnetizationStrength;
                ImGui.SetNextItemWidth(180f);
                if (ImGui.SliderFloat("WDL Magnet Strength", ref strength, 0f, 1f, "%.2f"))
                {
                    _stratigraphyWdlMagnetizationStrength = strength;
                    _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
                    _settings.SaveViewerSettings();
                }
            }
        }

        ImGui.Separator();

        // 3. Actions: Analyze, Revert, Save
        if (ImGui.Button("Analyze Active Tile"))
        {
            _stratigraphy.AnalyzeActiveCameraTileStratigraphy();
        }
        ImGui.SameLine();
        if (ImGui.Button("Analyze All Loaded Tiles"))
        {
            _stratigraphy.AnalyzeAllLoadedTilesStratigraphy();
        }

        ImGui.Spacing();
        if (ImGui.Button("Save Restored ADT / WDT Tiles..."))
        {
            _stratigraphy.OpenStratigraphySaveDialog();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Exports pre-computed loose LK ADT files and monolithic Alpha WDT maps to disk with current stratigraphy restorations applied.");

        if (!string.IsNullOrWhiteSpace(_terrainWeakSignalRestoreStatus))
        {
            ImGui.Spacing();
            ImGui.TextWrapped(_terrainWeakSignalRestoreStatus);
        }
    }

    private void DrawStratigraphyFactorControl()
    {
        ImGui.Text("Restoration Gradient Factor:");

        // 1. Direct High-Precision Numeric Input
        float factor = _terrainWeakSignalRestoreManualFactor;
        ImGui.SetNextItemWidth(140f);
        if (ImGui.InputFloat("##StratigraphyFactorInput", ref factor, 0.1f, 1.0f, "%.4fx"))
        {
            SetStratigraphyFactor(factor);
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Type an exact amplification multiplier (e.g. 33.334, 80, 10, 16, 64). Press Enter to apply.");

        // Quick Steppers
        ImGui.SameLine();
        if (ImGui.SmallButton("-10x")) SetStratigraphyFactor(_terrainWeakSignalRestoreManualFactor - 10f);
        ImGui.SameLine();
        if (ImGui.SmallButton("-1x")) SetStratigraphyFactor(_terrainWeakSignalRestoreManualFactor - 1f);
        ImGui.SameLine();
        if (ImGui.SmallButton("-0.1x")) SetStratigraphyFactor(_terrainWeakSignalRestoreManualFactor - 0.1f);
        ImGui.SameLine();
        if (ImGui.SmallButton("+0.1x")) SetStratigraphyFactor(_terrainWeakSignalRestoreManualFactor + 0.1f);
        ImGui.SameLine();
        if (ImGui.SmallButton("+1x")) SetStratigraphyFactor(_terrainWeakSignalRestoreManualFactor + 1f);
        ImGui.SameLine();
        if (ImGui.SmallButton("+10x")) SetStratigraphyFactor(_terrainWeakSignalRestoreManualFactor + 10f);

        // 2. Wide Logarithmic Slider (1x to 512x)
        factor = _terrainWeakSignalRestoreManualFactor;
        ImGui.SetNextItemWidth(-1f);
        if (ImGui.SliderFloat("##StratigraphyFactorSlider", ref factor, 1f, 512f, "Slider: %.3fx", ImGuiSliderFlags.Logarithmic))
        {
            SetStratigraphyFactor(factor);
        }

        // 3. Historical Era Preset Buttons
        ImGui.TextDisabled("Historical Era Presets:");
        if (ImGui.SmallButton("1x (Reset)")) SetStratigraphyFactor(1f);
        ImGui.SameLine();
        if (ImGui.SmallButton("3.33x (Late Alpha)")) SetStratigraphyFactor(3.333f);
        ImGui.SameLine();
        if (ImGui.SmallButton("10x (Pre-Release)")) SetStratigraphyFactor(10f);
        ImGui.SameLine();
        if (ImGui.SmallButton("16x (Blockout)")) SetStratigraphyFactor(16f);
        ImGui.SameLine();
        if (ImGui.SmallButton("33.334x (Classic 1/0.03)")) SetStratigraphyFactor(WowViewer.Core.Runtime.World.Terrain.Stratigraphy.TemporalStratigraphyOptions.DefaultClassicFactor);

        if (ImGui.SmallButton("64x (Early Proto)")) SetStratigraphyFactor(64f);
        ImGui.SameLine();
        if (ImGui.SmallButton("80x (Deep Erasure)")) SetStratigraphyFactor(80f);
        ImGui.SameLine();
        if (ImGui.SmallButton("128x (Sub-Grid)")) SetStratigraphyFactor(128f);
        ImGui.SameLine();
        if (ImGui.SmallButton("256x (Deep Lattice)")) SetStratigraphyFactor(256f);
        ImGui.SameLine();
        if (ImGui.SmallButton("512x (Max)")) SetStratigraphyFactor(512f);
    }

    private void SetStratigraphyFactor(float factor)
    {
        _terrainWeakSignalRestoreManualFactor = Math.Clamp(factor, 1f, 512f);
        _terrainWeakSignalRestoreUseAutoFactor = false;
        _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
        _settings.SaveViewerSettings();
    }

    // (DrawQuickControlsPopoutBody removed — Quick is now a direct workbench destination.)
    // (DrawSubTabWindow removed — replaced by single Workbench popout)

    // ── Scene sub-tab content ──────────────────────────────────────────────
    internal void DrawTerrainLabSubTab()
    {
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null && !HasTerrainOrWorldLoaded())
        {
            ImGui.TextDisabled("Load a terrain-backed world to use Terrain Lab.");
            return;
        }

        ImGui.TextDisabled("Tile targeting and MCNK/chunk clipboard operations share this experimental surface.");
        ImGui.Separator();
        if (renderer == null)
        {
            ImGui.TextDisabled("Terrain renderer is not available for the clipboard or selection map.");
            return;
        }

        DrawTerrainWorkbenchSelectionContent(renderer);
        ImGui.Separator();
        DrawSharedChunkClipboardSection(renderer, withHeader: true, headerTitle: "Chunk Clipboard + Save");
        ImGui.Separator();
        DrawTerrainControlsAdjustmentContent();
    }

    private bool HasTerrainOrWorldLoaded() => _terrainManager != null || _vlmTerrainManager != null || _worldScene != null;
}
