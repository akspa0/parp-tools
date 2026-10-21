using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.UI;

namespace WoWViewer;

internal sealed partial class ViewerChromeService
{
    /// <summary>Draws into the manager-owned Options region; never opens or places chrome.</summary>
    internal void DrawOptionsContent()
    {
        ImGui.PushID("ViewerOptions");
        TerrainRenderer? terrain = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        LiquidRenderer? liquid = _terrainManager?.LiquidRenderer ?? _vlmTerrainManager?.LiquidRenderer;

        if (SharedUiWidgets.DrawSectionHeader("Display", id: "OptionsDisplay"))
        {
            if (SharedUiWidgets.DrawSectionHeader("Render Quality", id: "OptionsRenderQuality"))
                _host.DrawRenderQualityContent();
            if (SharedUiWidgets.DrawSectionHeader("Lighting", id: "OptionsLighting"))
                _host.LightingPanel.DrawLightingContent();
            if (SharedUiWidgets.DrawSectionHeader("Theme", id: "OptionsTheme"))
                _host.Themes.DrawUiThemeSettingsContent();
        }

        if (SharedUiWidgets.DrawSectionHeader("Overlays", id: "OptionsOverlays"))
            DrawOptionsOverlaysContent(terrain, liquid);
        if (SharedUiWidgets.DrawSectionHeader("Layers", id: "OptionsLayers"))
            DrawOptionsLayersContent(terrain);
        if (SharedUiWidgets.DrawSectionHeader("Grids", id: "OptionsGrids"))
            DrawOptionsGridsContent(terrain);
        if (SharedUiWidgets.DrawSectionHeader("Objects", id: "OptionsObjects"))
            DrawOptionsObjectsContent();
        if (SharedUiWidgets.DrawSectionHeader("Audio", id: "OptionsAudio"))
            _host.AudioPanel.DrawAudioContent();
        if (SharedUiWidgets.DrawSectionHeader("Simulation", id: "OptionsSimulation"))
            DrawOptionsSimulationContent();

        ImGui.PopID();
    }

    /// <summary>Compact readout for the manager's persistent bottom bar, in any destination.</summary>
    internal void DrawStatsContent()
    {
        float fps = ImGui.GetIO().Framerate;
        SharedUiWidgets.CompactStatus(fps > 0f
            ? $"{fps:0} FPS | {1000f / fps:0.0} ms"
            : "FPS: -- | -- ms");
    }

    private void DrawOptionsOverlaysContent(TerrainRenderer? terrain, LiquidRenderer? liquid)
    {
        if (terrain != null)
        {
            bool alpha = terrain.ShowAlphaMask;
            if (ImGui.Checkbox("Alpha", ref alpha)) terrain.ShowAlphaMask = alpha;
            bool shadows = terrain.ShowShadowMap;
            if (ImGui.Checkbox("Shadows", ref shadows)) terrain.ShowShadowMap = shadows;
            bool mccv = terrain.UseMccv;
            if (ImGui.Checkbox("MCCV", ref mccv)) terrain.UseMccv = mccv;
            bool contours = terrain.ShowContours;
            if (ImGui.Checkbox("Contours", ref contours)) terrain.ShowContours = contours;
        }
        if (liquid != null)
        {
            bool visible = liquid.ShowLiquid;
            if (ImGui.Checkbox($"Liquid Terrain ({liquid.MeshCount})", ref visible))
                liquid.ShowLiquid = visible;
        }
        if (_worldScene != null)
        {
            bool wl = _worldScene.ShowWlLiquids;
            if (ImGui.Checkbox($"WL* ({liquid?.WlMeshCount ?? 0})", ref wl))
                _worldScene.ShowWlLiquids = wl;
            bool boxes = _worldScene.ShowBoundingBoxes;
            if (ImGui.Checkbox("Bounding Boxes", ref boxes))
                _worldScene.ShowBoundingBoxes = boxes;
            bool pm4 = _worldScene.Pm4Overlay.ShowPm4Overlay;
            if (ImGui.Checkbox("PM4 Overlay", ref pm4))
                _worldScene.Pm4Overlay.ShowPm4Overlay = pm4;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip(_worldScene.Pm4Overlay.Pm4Status);
            if (_worldScene.Pm4Overlay.IsPm4Loading)
                SharedUiWidgets.CompactStatus("PM4 overlay is loading...");
            bool colorSpawns = _worldScene.ExternalSpawns.ColorSpawnBoundingBoxes;
            if (ImGui.Checkbox("Color Spawns", ref colorSpawns))
                _worldScene.ExternalSpawns.ColorSpawnBoundingBoxes = colorSpawns;
        }
        if (_renderer is WmoRenderer)
        {
            ImGui.Checkbox("WMO Group BBs", ref _standaloneWmoGroupOverlayEnabled);
            ImGui.Checkbox("Group Names", ref _standaloneWmoGroupLabelsAllEnabled);
        }
        if (terrain == null && liquid == null && _worldScene == null && _renderer is not WmoRenderer)
            SharedUiWidgets.CompactStatus("Load terrain or a WMO to inspect overlays.");
    }

    private void DrawOptionsLayersContent(TerrainRenderer? terrain)
    {
        if (terrain != null)
        {
            int layerCount = Math.Clamp(terrain.MaxResidentLayerCount, 4, TerrainTileMeshBuilder.MaxLayers);
            for (int layer = 0; layer < layerCount; layer++)
            {
                bool visible = terrain.GetShowLayer(layer);
                if (ImGui.Checkbox(layer == 0 ? "Base Layer (L0)" : $"Layer {layer} (L{layer})", ref visible))
                    terrain.SetShowLayer(layer, visible);
            }
            if (SharedUiWidgets.ActionButton("All Layers", small: true))
            {
                for (int layer = 0; layer < layerCount; layer++) terrain.SetShowLayer(layer, true);
            }
            if (SharedUiWidgets.ActionButton("Base Only", small: true))
            {
                for (int layer = 0; layer < layerCount; layer++) terrain.SetShowLayer(layer, layer == 0);
            }
            if (SharedUiWidgets.ActionButton("Hide All", small: true))
            {
                for (int layer = 0; layer < layerCount; layer++) terrain.SetShowLayer(layer, false);
            }

            bool holes = !(_terrainManager?.IgnoreTerrainHolesGlobally
                ?? _vlmTerrainManager?.IgnoreTerrainHolesGlobally ?? false);
            if (ImGui.Checkbox("Holes", ref holes)
                && _terrainControlsPanel.SetIgnoreTerrainHolesGlobally(!holes))
                _statusMessage = holes ? "Terrain hole masking enabled." : "Terrain hole masking disabled.";
        }
        else
        {
            SharedUiWidgets.CompactStatus("Load terrain to control texture layers and holes.");
        }
        if (_worldScene != null)
        {
            bool wdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("WDL", ref wdl)) _worldScene.ShowWdlTerrain = wdl;
        }
    }

    private static void DrawOptionsGridsContent(TerrainRenderer? terrain)
    {
        if (terrain == null)
        {
            SharedUiWidgets.CompactStatus("Load terrain to control grids.");
            return;
        }
        bool tiles = terrain.ShowTileGrid;
        if (ImGui.Checkbox("Tiles", ref tiles)) terrain.ShowTileGrid = tiles;
        bool chunks = terrain.ShowChunkGrid;
        if (ImGui.Checkbox("Chunks", ref chunks)) terrain.ShowChunkGrid = chunks;
        bool cells = terrain.ShowCellGrid;
        if (ImGui.Checkbox("Cells", ref cells)) terrain.ShowCellGrid = cells;
    }

    private void DrawOptionsObjectsContent()
    {
        bool hasWmo = _worldScene != null || _renderer is WmoRenderer || _loadedWmo != null;
        bool hasM2 = _worldScene != null || (_renderer != null && _renderer is not WmoRenderer) || _loadedMdx != null;
        if (hasWmo || hasM2)
            DrawObjectTransparencyContent("Options", hasWmo, hasM2, sliderWidth: -1f);

        if (_worldScene != null)
        {
            bool terrainWireframe = _worldScene.TerrainWireframeEnabled;
            if (ImGui.Checkbox("Terrain WF", ref terrainWireframe))
                _worldScene.SetTerrainWireframeEnabled(terrainWireframe);
            bool objectWireframe = _worldScene.ObjectWireframeEnabled;
            if (ImGui.Checkbox("M2/WMO WF", ref objectWireframe))
                _worldScene.SetObjectWireframeEnabled(objectWireframe);
            bool animation = _worldScene.WorldDoodadAnimationEnabled;
            if (ImGui.Checkbox("Anim", ref animation))
                _worldScene.WorldDoodadAnimationEnabled = animation;
        }
        else if (_renderer != null)
        {
            bool wireframe = _renderer.IsWireframe;
            if (ImGui.Checkbox(_renderer is WmoRenderer ? "WMO WF" : "Model WF", ref wireframe))
                _renderer.ToggleWireframe();
        }
        if (_sceneCursorRenderer != null)
        {
            bool cursor3d = _sceneCursorRenderer.Style != CursorStyle.ClassicOSArrow;
            if (ImGui.Checkbox("3D Cursor", ref cursor3d))
            {
                _sceneCursorRenderer.Style = cursor3d ? CursorStyle.AuthenticWoWGauntlet : CursorStyle.ClassicOSArrow;
                _settings.SaveViewerSettings();
            }

            // Report which client asset is actually backing the cursor. Without this the operator
            // cannot tell an authentic cursor.mdx/cursor.m2 from the procedural fallback, which is
            // exactly how the M2 cursor stayed silently broken.
            if (cursor3d)
            {
                if (_sceneCursorRenderer.HasAuthenticCursorModel)
                {
                    SharedUiWidgets.CompactStatus(
                        $"Cursor model: {_sceneCursorRenderer.ModelKind} {_sceneCursorRenderer.LoadedCursorModelPath}");
                }
                else if (!string.IsNullOrWhiteSpace(_sceneCursorRenderer.LastLoadError))
                {
                    SharedUiWidgets.CompactStatus($"Cursor model unavailable: {_sceneCursorRenderer.LastLoadError}");
                }
                else
                {
                    SharedUiWidgets.CompactStatus("Cursor model: not resolved yet (open a client).");
                }

                if (ImGui.Button("Reload cursor model"))
                    _sceneCursorRenderer.ReloadCursorModel();
            }
        }
    }

    private void DrawOptionsSimulationContent()
    {
        // Auto-Stream is private to SqlSpawnStreamingService. Its chat control remains
        // authoritative until the parent provides an existing-state content/accessor API.
        SharedUiWidgets.CompactStatus("Auto-Stream is available in the chat population controls.");
        if (_worldScene == null)
        {
            SharedUiWidgets.CompactStatus("Load a world to control spawn bounding boxes.");
            return;
        }
        bool spawnsOnly = _worldScene.ExternalSpawns.ShowSpawnBoundingBoxesOnly;
        if (ImGui.Checkbox("Spawn BBs Only", ref spawnsOnly))
            _worldScene.ExternalSpawns.ShowSpawnBoundingBoxesOnly = spawnsOnly;
    }
}
