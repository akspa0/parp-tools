using System;
using System.Collections.Generic;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.UI;
using WoWViewer.Workbench;

namespace WoWViewer;

/// <summary>
/// Persistent bottom toolbar with the scene toggles, per-liquid-family toggles, the texture
/// layers the resident terrain actually carries, one alpha selector per non-base layer, and
/// M2/MDX + WMO opacity controls.
/// </summary>
internal sealed class BottomBarService
{
    private readonly IViewerAppHost _host;
    private WorldScene? _lastAppliedScene;
    private string _lastAppliedMap = string.Empty;
    private int _lastAppliedLayerCount = -1;
    private bool _showLayerPopup;

    internal bool BottomToggleLiquidWater { get; set; } = true;
    internal bool BottomToggleLiquidOcean { get; set; } = true;
    internal bool BottomToggleLiquidMagma { get; set; } = true;
    internal bool BottomToggleLiquidSlime { get; set; } = true;
    internal bool BottomToggleDoodads { get; set; } = true;
    internal bool BottomToggleWmos { get; set; } = true;
    internal bool BottomToggleTerrain { get; set; } = true;
    internal bool BottomTogglePm4 { get; set; }
    internal bool BottomToggleTileGrid { get; set; }
    internal bool BottomToggleChunkGrid { get; set; }
    internal bool BottomToggleCellGrid { get; set; }
    internal bool BottomToggleAudioMute { get; set; } = true;
    internal bool BottomToggleAlphaMask { get; set; }
    internal int BottomAlphaMaskChannel { get; set; } = 1;

    internal BottomBarService(IViewerAppHost host)
    {
        _host = host;
    }

    internal void ApplyToScene(WorldScene? scene, TerrainManager? terrainManager, VlmTerrainManager? vlmTerrain)
    {
        if (scene is not null)
        {
            scene.Pm4Overlay.ShowPm4Overlay = BottomTogglePm4;
            scene.DoodadsVisible = BottomToggleDoodads;
            scene.WmosVisible = BottomToggleWmos;
            scene.ShowWdlTerrain = BottomToggleTerrain;
            scene.SetAudioMuted(BottomToggleAudioMute);
        }

        if (terrainManager is not null)
        {
            terrainManager.TerrainVisible = BottomToggleTerrain;
            ApplyGridToggles(terrainManager.Renderer);
        }

        if (vlmTerrain is not null)
        {
            vlmTerrain.TerrainVisible = BottomToggleTerrain;
            ApplyGridToggles(vlmTerrain.Renderer);
        }

        ApplyLiquidState(scene, terrainManager?.LiquidRenderer, vlmTerrain?.LiquidRenderer);

        TerrainRenderer? renderer = terrainManager?.Renderer ?? vlmTerrain?.Renderer;
        int layerCount = BottomBarControlPolicy.ResolveLayerCount(renderer?.MaxResidentLayerCount ?? 0);
        int alphaChannelCount = BottomBarControlPolicy.ResolveAlphaChannelCount(layerCount);
        if (alphaChannelCount > 0)
            BottomAlphaMaskChannel = Math.Clamp(BottomAlphaMaskChannel, 1, alphaChannelCount);
        bool showAlpha = BottomToggleAlphaMask && alphaChannelCount > 0;
        ApplyAlphaState(terrainManager?.Renderer, showAlpha, BottomAlphaMaskChannel);
        if (vlmTerrain?.Renderer != terrainManager?.Renderer)
            ApplyAlphaState(vlmTerrain?.Renderer, showAlpha, BottomAlphaMaskChannel);
        _lastAppliedLayerCount = layerCount;
    }

    /// <summary>Refreshes the persisted quick-toggle snapshot from the live scene before settings save.</summary>
    internal void CaptureSceneState(WorldScene? scene, TerrainManager? terrainManager, VlmTerrainManager? vlmTerrain)
    {
        if (scene is not null)
        {
            BottomToggleDoodads = scene.DoodadsVisible;
            BottomToggleWmos = scene.WmosVisible;
            BottomTogglePm4 = scene.Pm4Overlay.ShowPm4Overlay;
            BottomToggleAudioMute = scene.AudioMuted;
        }

        if (terrainManager is not null)
        {
            BottomToggleTerrain = terrainManager.TerrainVisible;
            CaptureGridToggles(terrainManager.Renderer);
            CaptureLiquidState(terrainManager.LiquidRenderer);
        }

        if (vlmTerrain is not null)
        {
            BottomToggleTerrain &= vlmTerrain.TerrainVisible;
            CaptureGridToggles(vlmTerrain.Renderer);
            CaptureLiquidState(vlmTerrain.LiquidRenderer);
        }

        TerrainRenderer? renderer = terrainManager?.Renderer ?? vlmTerrain?.Renderer;
        if (renderer is not null)
        {
            BottomToggleAlphaMask = renderer.ShowAlphaMask;
            BottomAlphaMaskChannel = Math.Clamp(renderer.AlphaMaskChannel, 1, TerrainTileMeshBuilder.MaxLayers - 1);
        }
    }

    internal void DrawBottom(UiRect region, float scale)
    {
        WorldScene? scene = _host.WorldScene;
        TerrainManager? terrainManager = _host.TerrainManager;
        VlmTerrainManager? vlmTerrain = _host.VlmTerrainManager;
        TerrainRenderer? terrainRenderer = terrainManager?.Renderer ?? vlmTerrain?.Renderer;
        LiquidRenderer? liquidRenderer = terrainManager?.LiquidRenderer ?? vlmTerrain?.LiquidRenderer;
        int layerCount = BottomBarControlPolicy.ResolveLayerCount(terrainRenderer?.MaxResidentLayerCount ?? 0);
        int alphaLayerCount = BottomBarControlPolicy.ResolveAlphaChannelCount(layerCount);
        string mapName = terrainManager?.MapName ?? vlmTerrain?.MapName ?? string.Empty;
        if (scene is not null &&
            (!ReferenceEquals(scene, _lastAppliedScene) || !string.Equals(mapName, _lastAppliedMap, StringComparison.OrdinalIgnoreCase)))
        {
            ApplyToScene(scene, terrainManager, vlmTerrain);
            _lastAppliedScene = scene;
            _lastAppliedMap = mapName;
        }

        if (layerCount != _lastAppliedLayerCount)
        {
            if (alphaLayerCount > 0)
                BottomAlphaMaskChannel = Math.Clamp(BottomAlphaMaskChannel, 1, alphaLayerCount);
            bool showAlpha = BottomToggleAlphaMask && alphaLayerCount > 0;
            ApplyAlphaState(terrainManager?.Renderer, showAlpha, BottomAlphaMaskChannel);
            if (vlmTerrain?.Renderer != terrainManager?.Renderer)
                ApplyAlphaState(vlmTerrain?.Renderer, showAlpha, BottomAlphaMaskChannel);
            _lastAppliedLayerCount = layerCount;
        }

        string[] toggleLabels = ["Water", "Ocean", "Magma", "Slime", "Doodads", "WMOs", "Terrain", "PM4", "Tile", "Chunk", "Cell", "Mute"];
        string[] toggleIds =
        [
            "toggle-liquid-water", "toggle-liquid-ocean", "toggle-liquid-magma", "toggle-liquid-slime",
            "toggle-doodads", "toggle-wmos", "toggle-terrain", "toggle-pm4",
            "toggle-tile-grid", "toggle-chunk-grid", "toggle-cell-grid", "toggle-audio-mute",
        ];
        var items = new List<FooterItemMeasure>(toggleIds.Length + 3);
        for (int index = 0; index < toggleIds.Length; index++)
        {
            float width = ImGui.CalcTextSize(toggleLabels[index]).X + ImGui.GetFrameHeight() + 5f;
            items.Add(new FooterItemMeasure(toggleIds[index], width, width, FooterItemPriority.Essential));
        }
        float layersWidth = ImGui.CalcTextSize("Layers").X + ImGui.GetFrameHeight() + 5f;
        items.Add(new FooterItemMeasure("layers", layersWidth, layersWidth, FooterItemPriority.Essential));
        float m2LabelWidth = ImGui.CalcTextSize("M2/MDX Opacity").X;
        float wmoLabelWidth = ImGui.CalcTextSize("WMO Opacity").X;
        items.Add(new FooterItemMeasure("opacity-m2", m2LabelWidth + 70f * scale, m2LabelWidth + 54f * scale, FooterItemPriority.Essential));
        items.Add(new FooterItemMeasure("opacity-wmo", wmoLabelWidth + 70f * scale, wmoLabelWidth + 54f * scale, FooterItemPriority.Essential));

        FooterFitResult fit = ViewerFooterLayout.Fit(region, scale, items, gap: 2f);
        float originX = ImGui.GetCursorPosX();
        float y = ImGui.GetCursorPosY();

        void DrawToggle(string id, bool current, Action<bool> onChange, string tooltip)
        {
            FooterItemPlacement placement = fit.Get(id);
            if (!placement.Visible) return;
            ImGui.SetCursorPos(new Vector2(originX + placement.X, y));
            bool val = current;
            bool changed = id switch
            {
                "toggle-liquid-water" => ImGui.Checkbox("Water##toggle-liquid-water", ref val),
                "toggle-liquid-ocean" => ImGui.Checkbox("Ocean##toggle-liquid-ocean", ref val),
                "toggle-liquid-magma" => ImGui.Checkbox("Magma##toggle-liquid-magma", ref val),
                "toggle-liquid-slime" => ImGui.Checkbox("Slime##toggle-liquid-slime", ref val),
                "toggle-doodads" => ImGui.Checkbox("Doodads##toggle-doodads", ref val),
                "toggle-wmos" => ImGui.Checkbox("WMOs##toggle-wmos", ref val),
                "toggle-terrain" => ImGui.Checkbox("Terrain##toggle-terrain", ref val),
                "toggle-pm4" => ImGui.Checkbox("PM4##toggle-pm4", ref val),
                "toggle-tile-grid" => ImGui.Checkbox("Tile##toggle-tile-grid", ref val),
                "toggle-chunk-grid" => ImGui.Checkbox("Chunk##toggle-chunk-grid", ref val),
                "toggle-cell-grid" => ImGui.Checkbox("Cell##toggle-cell-grid", ref val),
                _ => ImGui.Checkbox("Mute##toggle-audio-mute", ref val),
            };
            if (changed)
            {
                onChange(val);
                _host.Settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered()) ImGui.SetTooltip(tooltip);
        }

        DrawToggle("toggle-liquid-water", liquidRenderer?.ShowWaterLiquid ?? BottomToggleLiquidWater, v =>
        {
            BottomToggleLiquidWater = v;
            ApplyLiquidState(scene, terrainManager?.LiquidRenderer, vlmTerrain?.LiquidRenderer);
        }, "Show MCLQ/MH2O and WL* water and river liquid.");

        DrawToggle("toggle-liquid-ocean", liquidRenderer?.ShowOceanLiquid ?? BottomToggleLiquidOcean, v =>
        {
            BottomToggleLiquidOcean = v;
            ApplyLiquidState(scene, terrainManager?.LiquidRenderer, vlmTerrain?.LiquidRenderer);
        }, "Show MCLQ/MH2O and WL* ocean liquid.");

        DrawToggle("toggle-liquid-magma", liquidRenderer?.ShowMagmaLiquid ?? BottomToggleLiquidMagma, v =>
        {
            BottomToggleLiquidMagma = v;
            ApplyLiquidState(scene, terrainManager?.LiquidRenderer, vlmTerrain?.LiquidRenderer);
        }, "Show MCLQ/MH2O and WL* magma and lava liquid.");

        DrawToggle("toggle-liquid-slime", liquidRenderer?.ShowSlimeLiquid ?? BottomToggleLiquidSlime, v =>
        {
            BottomToggleLiquidSlime = v;
            ApplyLiquidState(scene, terrainManager?.LiquidRenderer, vlmTerrain?.LiquidRenderer);
        }, "Show MCLQ/MH2O and WL* slime liquid.");

        DrawToggle("toggle-doodads", scene?.DoodadsVisible ?? BottomToggleDoodads, v =>
        {
            BottomToggleDoodads = v;
            if (scene is not null) scene.DoodadsVisible = v;
        }, "Toggle M2/MDX doodad visibility.");

        DrawToggle("toggle-wmos", scene?.WmosVisible ?? BottomToggleWmos, v =>
        {
            BottomToggleWmos = v;
            if (scene is not null) scene.WmosVisible = v;
        }, "Toggle WMO visibility.");

        bool terrainVisible = terrainManager?.TerrainVisible ?? vlmTerrain?.TerrainVisible ?? BottomToggleTerrain;
        DrawToggle("toggle-terrain", terrainVisible, v =>
        {
            BottomToggleTerrain = v;
            if (terrainManager is not null) terrainManager.TerrainVisible = v;
            if (vlmTerrain is not null) vlmTerrain.TerrainVisible = v;
            if (scene is not null) scene.ShowWdlTerrain = v;
        }, "Toggle terrain tiles and WDL far terrain.");

        DrawToggle("toggle-pm4", scene?.Pm4Overlay.ShowPm4Overlay ?? BottomTogglePm4, v =>
        {
            BottomTogglePm4 = v;
            if (scene is not null) scene.Pm4Overlay.ShowPm4Overlay = v;
        }, "Toggle the PM4 navigation geometry overlay.");

        DrawToggle("toggle-tile-grid", terrainRenderer?.ShowTileGrid ?? BottomToggleTileGrid, v =>
        {
            BottomToggleTileGrid = v;
            if (terrainManager?.Renderer is { } renderer) renderer.ShowTileGrid = v;
            if (vlmTerrain?.Renderer is { } vlmRenderer && vlmRenderer != terrainManager?.Renderer)
                vlmRenderer.ShowTileGrid = v;
        }, "Toggle the tile-boundary grid overlay.");

        DrawToggle("toggle-chunk-grid", terrainRenderer?.ShowChunkGrid ?? BottomToggleChunkGrid, v =>
        {
            BottomToggleChunkGrid = v;
            if (terrainManager?.Renderer is { } renderer) renderer.ShowChunkGrid = v;
            if (vlmTerrain?.Renderer is { } vlmRenderer && vlmRenderer != terrainManager?.Renderer)
                vlmRenderer.ShowChunkGrid = v;
        }, "Toggle the chunk-boundary grid overlay.");

        DrawToggle("toggle-cell-grid", terrainRenderer?.ShowCellGrid ?? BottomToggleCellGrid, v =>
        {
            BottomToggleCellGrid = v;
            if (terrainManager?.Renderer is { } renderer) renderer.ShowCellGrid = v;
            if (vlmTerrain?.Renderer is { } vlmRenderer && vlmRenderer != terrainManager?.Renderer)
                vlmRenderer.ShowCellGrid = v;
        }, "Toggle the cell-boundary grid overlay.");

        DrawToggle("toggle-audio-mute", scene?.AudioMuted ?? BottomToggleAudioMute, v =>
        {
            BottomToggleAudioMute = v;
            scene?.SetAudioMuted(v);
        }, "Mute or unmute world audio output.");

        FooterItemPlacement layersPlacement = fit.Get("layers");
        if (layersPlacement.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + layersPlacement.X, y));
            if (ImGui.Button("Layers##bottom-layers", new Vector2(layersPlacement.Width, 0f)))
                _showLayerPopup = true;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Open per-layer visibility and alpha-channel controls.");
        }

        if (_showLayerPopup)
        {
            ImGui.SetNextWindowSize(new Vector2(380f * scale, 460f * scale), ImGuiCond.FirstUseEver);
            if (ImGui.Begin("Layer Controls###BottomLayerPopup", ref _showLayerPopup))
            {
                ImGui.TextDisabled($"{layerCount} resident layer(s), {alphaLayerCount} alpha channel(s)");
                for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
                {
                    int layer = layerIndex;
                    bool visible = terrainRenderer?.GetShowLayer(layer) ?? true;
                    if (ImGui.Checkbox($"L{layer}##popup-layer-{layer}", ref visible))
                    {
                        terrainRenderer?.SetShowLayer(layer, visible);
                        _host.Settings.SaveViewerSettings();
                    }
                    ImGui.SameLine();
                    if (ImGui.SmallButton($"Details##popup-layer-details-{layer}"))
                        ImGui.OpenPopup($"layer-detail-{layer}");
                    if (ImGui.BeginPopup($"layer-detail-{layer}"))
                    {
                        ImGui.TextUnformatted($"Layer {layer}");
                        bool detailVisible = terrainRenderer?.GetShowLayer(layer) ?? true;
                        if (ImGui.Checkbox("Visible##popup-layer-detail-visible", ref detailVisible))
                        {
                            terrainRenderer?.SetShowLayer(layer, detailVisible);
                            _host.Settings.SaveViewerSettings();
                        }
                        if (layer > 0 && ImGui.Button("Show alpha map##popup-layer-detail-alpha"))
                        {
                            int channel = Math.Clamp(layer, 1, Math.Max(1, alphaLayerCount));
                            BottomToggleAlphaMask = true;
                            BottomAlphaMaskChannel = channel;
                            ApplyAlphaState(terrainManager?.Renderer, true, channel);
                            if (vlmTerrain?.Renderer != terrainManager?.Renderer)
                                ApplyAlphaState(vlmTerrain?.Renderer, true, channel);
                            _host.Settings.SaveViewerSettings();
                        }
                        ImGui.EndPopup();
                    }
                }

                ImGui.Separator();
                for (int channel = 1; channel <= alphaLayerCount; channel++)
                {
                    int alphaChannel = channel;
                    bool selected = terrainRenderer is not null
                        ? terrainRenderer.ShowAlphaMask && terrainRenderer.AlphaMaskChannel == channel
                        : BottomToggleAlphaMask && BottomAlphaMaskChannel == channel;
                    if (ImGui.Selectable($"A{channel}##popup-alpha-channel-{channel}", selected))
                    {
                        bool showAlpha = !selected;
                        BottomToggleAlphaMask = showAlpha;
                        BottomAlphaMaskChannel = alphaChannel;
                        ApplyAlphaState(terrainManager?.Renderer, showAlpha, alphaChannel);
                        if (vlmTerrain?.Renderer != terrainManager?.Renderer)
                            ApplyAlphaState(vlmTerrain?.Renderer, showAlpha, alphaChannel);
                        _host.Settings.SaveViewerSettings();
                    }
                }
            }
            ImGui.End();
        }

        DrawOpacity("opacity-m2", M2Renderer.GlobalOpacity, value =>
        {
            M2Renderer.GlobalOpacity = value;
            MdxRenderer.GlobalOpacity = value;
        });
        DrawOpacity("opacity-wmo", WmoRenderer.GlobalOpacity, value => WmoRenderer.GlobalOpacity = value);

        void DrawOpacity(string id, float current, Action<float> onChange)
        {
            FooterItemPlacement placement = fit.Get(id);
            if (!placement.Visible) return;
            ImGui.SetCursorPos(new Vector2(originX + placement.X, y));
            float opacityPercent = Math.Clamp(current, 0f, 1f) * 100f;
            string label = id == "opacity-m2" ? "M2/MDX Opacity" : "WMO Opacity";
            float labelWidth = ImGui.CalcTextSize(label).X;
            ImGui.SetNextItemWidth(MathF.Max(48f, placement.Width - labelWidth - 4f));
            bool changed = id == "opacity-m2"
                ? ImGui.SliderFloat("M2/MDX Opacity##opacity-m2", ref opacityPercent, 0f, 100f, "%.0f%%")
                : ImGui.SliderFloat("WMO Opacity##opacity-wmo", ref opacityPercent, 0f, 100f, "%.0f%%");
            if (changed)
                onChange(Math.Clamp(opacityPercent / 100f, 0f, 1f));
            if (ImGui.IsItemDeactivatedAfterEdit()) _host.Settings.SaveViewerSettings();
        }
    }

    private void ApplyLiquidState(WorldScene? scene, LiquidRenderer? terrainLiquids, LiquidRenderer? vlmLiquids)
    {
        bool anyLiquid = BottomToggleLiquidWater || BottomToggleLiquidOcean || BottomToggleLiquidMagma || BottomToggleLiquidSlime;
        if (scene is not null)
        {
            scene.ShowWlLiquids = anyLiquid;
            scene.EnableRuntimeWmoGroupLiquids = anyLiquid;
        }

        ApplyLiquidState(terrainLiquids);
        if (!ReferenceEquals(vlmLiquids, terrainLiquids))
            ApplyLiquidState(vlmLiquids);

        WmoRenderer.ShowWaterLiquids = BottomToggleLiquidWater;
        WmoRenderer.ShowOceanLiquids = BottomToggleLiquidOcean;
        WmoRenderer.ShowMagmaLiquids = BottomToggleLiquidMagma;
        WmoRenderer.ShowSlimeLiquids = BottomToggleLiquidSlime;
    }

    private void ApplyLiquidState(LiquidRenderer? renderer)
    {
        if (renderer is null) return;
        bool anyLiquid = BottomToggleLiquidWater || BottomToggleLiquidOcean || BottomToggleLiquidMagma || BottomToggleLiquidSlime;
        renderer.ShowLiquid = anyLiquid;
        renderer.ShowWlLiquids = anyLiquid;
        renderer.ShowWaterLiquid = BottomToggleLiquidWater;
        renderer.ShowOceanLiquid = BottomToggleLiquidOcean;
        renderer.ShowMagmaLiquid = BottomToggleLiquidMagma;
        renderer.ShowSlimeLiquid = BottomToggleLiquidSlime;
    }

    private void CaptureLiquidState(LiquidRenderer? renderer)
    {
        if (renderer is null) return;
        BottomToggleLiquidWater &= renderer.ShowWaterLiquid;
        BottomToggleLiquidOcean &= renderer.ShowOceanLiquid;
        BottomToggleLiquidMagma &= renderer.ShowMagmaLiquid;
        BottomToggleLiquidSlime &= renderer.ShowSlimeLiquid;
    }

    private void ApplyGridToggles(TerrainRenderer? renderer)
    {
        if (renderer is null) return;
        renderer.ShowTileGrid = BottomToggleTileGrid;
        renderer.ShowChunkGrid = BottomToggleChunkGrid;
        renderer.ShowCellGrid = BottomToggleCellGrid;
    }

    private void CaptureGridToggles(TerrainRenderer renderer)
    {
        BottomToggleTileGrid = renderer.ShowTileGrid;
        BottomToggleChunkGrid = renderer.ShowChunkGrid;
        BottomToggleCellGrid = renderer.ShowCellGrid;
    }

    private static void ApplyAlphaState(TerrainRenderer? renderer, bool show, int channel)
    {
        if (renderer is null) return;
        renderer.AlphaMaskChannel = Math.Clamp(channel, 1, TerrainTileMeshBuilder.MaxLayers - 1);
        renderer.ShowAlphaMask = show;
    }

}
