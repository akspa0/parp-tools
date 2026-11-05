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
/// Persistent bottom toolbar with the nine quick scene toggles and M2/MDX + WMO opacity controls.
/// Existing L0–L7/Alpha quick checks remain; the Texture Masks button is not part of this toolbar.
/// </summary>
internal sealed class BottomBarService
{
    private readonly IViewerAppHost _host;
    private WorldScene? _lastAppliedScene;
    private string _lastAppliedMap = string.Empty;

    internal bool BottomToggleSky { get; set; } = true;
    internal bool BottomToggleFog { get; set; } = true;
    internal bool BottomToggleWater { get; set; } = true;
    internal bool BottomToggleDoodads { get; set; } = true;
    internal bool BottomToggleWmos { get; set; } = true;
    internal bool BottomToggleTerrain { get; set; } = true;
    internal bool BottomTogglePm4 { get; set; }
    internal bool BottomToggleGrid { get; set; }
    internal bool BottomToggleAudioMute { get; set; }

    internal BottomBarService(IViewerAppHost host)
    {
        _host = host;
    }

    internal void ApplyToScene(WorldScene? scene, TerrainManager? terrainManager, VlmTerrainManager? vlmTerrain)
    {
        if (scene is not null)
        {
            scene.Pm4Overlay.ShowPm4Overlay = BottomTogglePm4;
            scene.ShowSky = BottomToggleSky;
            scene.ObjectFogEnabled = BottomToggleFog;
            scene.ShowWlLiquids = BottomToggleWater;
            scene.EnableRuntimeWmoGroupLiquids = BottomToggleWater;
            scene.DoodadsVisible = BottomToggleDoodads;
            scene.WmosVisible = BottomToggleWmos;
            scene.ShowWdlTerrain = BottomToggleTerrain;
            scene.SetAudioMuted(BottomToggleAudioMute);
        }

        if (terrainManager is not null)
        {
            terrainManager.TerrainVisible = BottomToggleTerrain;
            terrainManager.LiquidRenderer.ShowLiquid = BottomToggleWater;
            ApplyGridToggle(terrainManager.Renderer, BottomToggleGrid);
        }

        if (vlmTerrain is not null)
        {
            vlmTerrain.TerrainVisible = BottomToggleTerrain;
            vlmTerrain.LiquidRenderer.ShowLiquid = BottomToggleWater;
            ApplyGridToggle(vlmTerrain.Renderer, BottomToggleGrid);
        }
    }

    /// <summary>Refreshes the persisted quick-toggle snapshot from the live scene before settings save.</summary>
    internal void CaptureSceneState(WorldScene? scene, TerrainManager? terrainManager, VlmTerrainManager? vlmTerrain)
    {
        if (scene is not null)
        {
            BottomToggleSky = scene.ShowSky;
            BottomToggleFog = scene.ObjectFogEnabled;
            BottomToggleDoodads = scene.DoodadsVisible;
            BottomToggleWmos = scene.WmosVisible;
            BottomTogglePm4 = scene.Pm4Overlay.ShowPm4Overlay;
            BottomToggleAudioMute = scene.AudioMuted;
        }

        if (terrainManager is not null)
        {
            BottomToggleTerrain = terrainManager.TerrainVisible;
            BottomToggleWater = terrainManager.LiquidRenderer.ShowLiquid;
            BottomToggleGrid = ReadGridToggle(terrainManager.Renderer);
        }

        if (vlmTerrain is not null)
        {
            BottomToggleTerrain &= vlmTerrain.TerrainVisible;
            BottomToggleWater &= vlmTerrain.LiquidRenderer.ShowLiquid;
            BottomToggleGrid |= ReadGridToggle(vlmTerrain.Renderer);
        }

        if (scene is not null)
            BottomToggleWater &= scene.ShowWlLiquids && scene.EnableRuntimeWmoGroupLiquids;
    }

    internal void DrawBottom(UiRect region, float scale)
    {
        WorldScene? scene = _host.WorldScene;
        TerrainManager? terrainManager = _host.TerrainManager;
        VlmTerrainManager? vlmTerrain = _host.VlmTerrainManager;
        TerrainRenderer? terrainRenderer = terrainManager?.Renderer ?? vlmTerrain?.Renderer;
        int layerCount = terrainRenderer is null
            ? 4
            : Math.Clamp(terrainRenderer.MaxResidentLayerCount, 4, TerrainTileMeshBuilder.MaxLayers);
        string mapName = terrainManager?.MapName ?? vlmTerrain?.MapName ?? string.Empty;
        if (scene is not null &&
            (!ReferenceEquals(scene, _lastAppliedScene) || !string.Equals(mapName, _lastAppliedMap, StringComparison.OrdinalIgnoreCase)))
        {
            ApplyToScene(scene, terrainManager, vlmTerrain);
            _lastAppliedScene = scene;
            _lastAppliedMap = mapName;
        }

        string[] toggleLabels = ["Sky", "Fog", "Water", "Doodads", "WMOs", "Terrain", "PM4", "Grid", "Mute"];
        string[] toggleIds = ["toggle-sky", "toggle-fog", "toggle-water", "toggle-doodads", "toggle-wmos", "toggle-terrain", "toggle-pm4", "toggle-grid", "toggle-audio-mute"];
        var items = new List<FooterItemMeasure>(12 + layerCount);
        for (int index = 0; index < toggleIds.Length; index++)
        {
            float width = ImGui.CalcTextSize(toggleLabels[index]).X + ImGui.GetFrameHeight() + 5f;
            items.Add(new FooterItemMeasure(toggleIds[index], width, width, FooterItemPriority.Essential));
        }
        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
        {
            string layerLabel = layerIndex == 0 ? "L0" : $"L{layerIndex}";
            float layerWidth = ImGui.CalcTextSize(layerLabel).X + ImGui.GetFrameHeight() + 5f;
            items.Add(new FooterItemMeasure($"toggle-layer-{layerIndex}", layerWidth, layerWidth, FooterItemPriority.Secondary));
        }
        float alphaWidth = ImGui.CalcTextSize("Alpha").X + ImGui.GetFrameHeight() + 5f;
        items.Add(new FooterItemMeasure("toggle-alpha", alphaWidth, alphaWidth, FooterItemPriority.Secondary));
        float m2LabelWidth = ImGui.CalcTextSize("M2/MDX Opacity").X;
        float wmoLabelWidth = ImGui.CalcTextSize("WMO Opacity").X;
        items.Add(new FooterItemMeasure("opacity-m2", m2LabelWidth + 100f * scale, m2LabelWidth + 72f * scale, FooterItemPriority.Essential));
        items.Add(new FooterItemMeasure("opacity-wmo", wmoLabelWidth + 100f * scale, wmoLabelWidth + 72f * scale, FooterItemPriority.Essential));

        FooterFitResult fit = ViewerFooterLayout.Fit(region, scale, items, gap: 5f);
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
                "toggle-sky" => ImGui.Checkbox("Sky##toggle-sky", ref val),
                "toggle-fog" => ImGui.Checkbox("Fog##toggle-fog", ref val),
                "toggle-water" => ImGui.Checkbox("Water##toggle-water", ref val),
                "toggle-doodads" => ImGui.Checkbox("Doodads##toggle-doodads", ref val),
                "toggle-wmos" => ImGui.Checkbox("WMOs##toggle-wmos", ref val),
                "toggle-terrain" => ImGui.Checkbox("Terrain##toggle-terrain", ref val),
                "toggle-pm4" => ImGui.Checkbox("PM4##toggle-pm4", ref val),
                "toggle-grid" => ImGui.Checkbox("Grid##toggle-grid", ref val),
                _ => ImGui.Checkbox("Mute##toggle-audio-mute", ref val),
            };
            if (changed)
            {
                onChange(val);
                _host.Settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered()) ImGui.SetTooltip(tooltip);
        }

        DrawToggle("toggle-sky", scene?.ShowSky ?? BottomToggleSky, v =>
        {
            BottomToggleSky = v;
            if (scene is not null) scene.ShowSky = v;
        }, "Show or hide the sky dome and backdrop.");

        DrawToggle("toggle-fog", scene?.ObjectFogEnabled ?? BottomToggleFog, v =>
        {
            BottomToggleFog = v;
            if (scene is not null) scene.ObjectFogEnabled = v;
        }, "Toggle object fog.");

        bool waterVisible = scene is null
            ? BottomToggleWater
            : scene.ShowWlLiquids && scene.EnableRuntimeWmoGroupLiquids &&
              (terrainManager?.LiquidRenderer.ShowLiquid ?? true) &&
              (vlmTerrain?.LiquidRenderer.ShowLiquid ?? true);
        DrawToggle("toggle-water", waterVisible, v =>
        {
            BottomToggleWater = v;
            if (scene is not null)
            {
                scene.ShowWlLiquids = v;
                scene.EnableRuntimeWmoGroupLiquids = v;
            }
            if (terrainManager is not null) terrainManager.LiquidRenderer.ShowLiquid = v;
            if (vlmTerrain is not null) vlmTerrain.LiquidRenderer.ShowLiquid = v;
        }, "Toggle terrain, WMO-group, and loose world liquids.");

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

        bool gridVisible = terrainRenderer is null ? BottomToggleGrid : ReadGridToggle(terrainRenderer);
        DrawToggle("toggle-grid", gridVisible, v =>
        {
            BottomToggleGrid = v;
            ApplyGridToggle(terrainManager?.Renderer, v);
            if (vlmTerrain?.Renderer != terrainManager?.Renderer)
                ApplyGridToggle(vlmTerrain?.Renderer, v);
        }, "Toggle tile, chunk, and cell grid overlays.");

        DrawToggle("toggle-audio-mute", scene?.AudioMuted ?? BottomToggleAudioMute, v =>
        {
            BottomToggleAudioMute = v;
            scene?.SetAudioMuted(v);
        }, "Mute or unmute world audio output.");

        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
        {
            string itemId = $"toggle-layer-{layerIndex}";
            FooterItemPlacement placement = fit.Get(itemId);
            if (!placement.Visible) continue;
            ImGui.SetCursorPos(new Vector2(originX + placement.X, y));
            bool visible = terrainRenderer?.GetShowLayer(layerIndex) ?? true;
            bool changed = layerIndex switch
            {
                0 => ImGui.Checkbox("L0##toggle-layer-0", ref visible),
                1 => ImGui.Checkbox("L1##toggle-layer-1", ref visible),
                2 => ImGui.Checkbox("L2##toggle-layer-2", ref visible),
                3 => ImGui.Checkbox("L3##toggle-layer-3", ref visible),
                4 => ImGui.Checkbox("L4##toggle-layer-4", ref visible),
                5 => ImGui.Checkbox("L5##toggle-layer-5", ref visible),
                6 => ImGui.Checkbox("L6##toggle-layer-6", ref visible),
                _ => ImGui.Checkbox("L7##toggle-layer-7", ref visible),
            };
            if (changed)
            {
                terrainRenderer?.SetShowLayer(layerIndex, visible);
                _host.Settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered()) ImGui.SetTooltip(layerIndex == 0 ? "Toggle the base terrain texture layer." : $"Toggle terrain texture layer {layerIndex}.");
        }

        FooterItemPlacement alphaPlacement = fit.Get("toggle-alpha");
        if (alphaPlacement.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + alphaPlacement.X, y));
            bool alphaVisible = terrainRenderer?.ShowAlphaMask ?? false;
            if (ImGui.Checkbox("Alpha##toggle-alpha", ref alphaVisible))
            {
                if (terrainRenderer is not null) terrainRenderer.ShowAlphaMask = alphaVisible;
                _host.Settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Show or hide the terrain alpha-mask overlay.");
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
            ImGui.SetNextItemWidth(MathF.Max(48f, placement.Width - labelWidth - 6f));
            bool changed = id == "opacity-m2"
                ? ImGui.SliderFloat("M2/MDX Opacity##opacity-m2", ref opacityPercent, 0f, 100f, "%.0f%%")
                : ImGui.SliderFloat("WMO Opacity##opacity-wmo", ref opacityPercent, 0f, 100f, "%.0f%%");
            if (changed)
                onChange(Math.Clamp(opacityPercent / 100f, 0f, 1f));
            if (ImGui.IsItemDeactivatedAfterEdit()) _host.Settings.SaveViewerSettings();
        }

    }

    private static bool ReadGridToggle(TerrainRenderer renderer)
        => renderer.ShowTileGrid || renderer.ShowChunkGrid || renderer.ShowCellGrid;

    private static void ApplyGridToggle(TerrainRenderer? renderer, bool enabled)
    {
        if (renderer is null) return;
        renderer.ShowTileGrid = enabled;
        renderer.ShowChunkGrid = enabled;
        renderer.ShowCellGrid = enabled;
    }
}
