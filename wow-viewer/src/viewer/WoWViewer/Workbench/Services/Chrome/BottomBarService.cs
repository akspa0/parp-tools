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
/// Persistent bottom bar service hosting the quick toggles the operator asked for:
/// PM4 overlay, one visibility toggle per texture layer, the alpha-mask toggle, and a
/// button that opens the Texture Masks panel. Destination switching lives in the
/// sidebars/action rails, not here.
/// </summary>
internal sealed class BottomBarService
{
    private readonly IViewerAppHost _host;

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
        if (scene != null)
        {
            scene.Pm4Overlay.ShowPm4Overlay = BottomTogglePm4;
        }
    }

    internal void DrawBottom(UiRect region, float scale)
    {
        float height = ImGui.GetFrameHeight();
        WorldScene? scene = _host.WorldScene;
        TerrainManager? terrainManager = _host.TerrainManager;
        VlmTerrainManager? vlmTerrain = _host.VlmTerrainManager;

        TerrainRenderer? layerTerrain = terrainManager?.Renderer ?? vlmTerrain?.Renderer;
        int layerCount = layerTerrain != null
            ? Math.Clamp(layerTerrain.MaxResidentLayerCount, 4, TerrainTileMeshBuilder.MaxLayers)
            : 4;

        var items = new List<FooterItemMeasure>
        {
            new("toggle-pm4", 62f * scale, 48f * scale, FooterItemPriority.Essential),
        };
        for (int i = 0; i < layerCount; i++)
            items.Add(new($"toggle-layer-{i}", 46f * scale, 40f * scale, FooterItemPriority.Essential));
        items.Add(new("toggle-alpha", 74f * scale, 58f * scale, FooterItemPriority.Essential));
        items.Add(new("open-texture-masks", 120f * scale, 92f * scale, FooterItemPriority.Essential));

        FooterFitResult fit = ViewerFooterLayout.Fit(region, scale, items);
        float originX = ImGui.GetCursorPosX();
        float y = ImGui.GetCursorPosY();

        void DrawToggle(string id, string label, bool current, Action<bool> onChange, string tooltip)
        {
            FooterItemPlacement placement = fit.Get(id);
            if (!placement.Visible) return;
            ImGui.SetCursorPos(new Vector2(originX + placement.X, y));
            bool val = current;
            if (ImGui.Checkbox($"{label}##{id}", ref val))
            {
                onChange(val);
                _host.Settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered()) ImGui.SetTooltip(tooltip);
        }

        // 1. PM4 quick toggle
        bool pm4Val = scene?.Pm4Overlay?.ShowPm4Overlay ?? BottomTogglePm4;
        DrawToggle("toggle-pm4", "PM4", pm4Val, v =>
        {
            BottomTogglePm4 = v;
            if (scene?.Pm4Overlay != null) scene.Pm4Overlay.ShowPm4Overlay = v;
        }, "Toggle PM4 pre-release navigation geometry overlay");

        // 2. One visibility toggle per texture layer (L0 = base, then overlays).
        for (int i = 0; i < layerCount; i++)
        {
            int layer = i;
            bool visible = layerTerrain?.GetShowLayer(layer) ?? true;
            DrawToggle($"toggle-layer-{layer}", layer == 0 ? "L0" : $"L{layer}", visible, v =>
            {
                layerTerrain?.SetShowLayer(layer, v);
            }, layer == 0
                ? "Toggle the base terrain texture layer"
                : $"Toggle terrain texture layer {layer}");
        }

        // 3. Alpha mask overlay toggle
        bool alphaOn = layerTerrain?.ShowAlphaMask ?? false;
        DrawToggle("toggle-alpha", "Alpha", alphaOn, v =>
        {
            if (layerTerrain != null) layerTerrain.ShowAlphaMask = v;
        }, "Show or hide the terrain alpha-mask overlay");

        // 4. Texture Masks panel button (per-layer opacity, blending, channel detail).
        FooterItemPlacement masksPlacement = fit.Get("open-texture-masks");
        if (masksPlacement.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + masksPlacement.X, y));
            if (ImGui.Button("Texture Masks##open-texture-masks", new Vector2(masksPlacement.Width, height)))
                _host.UiManager.OpenLeftDock(LeftPanel.Layers);
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Open the Texture Masks panel: per-layer opacity, alpha-mask blending, and channel detail");
        }

    }
}
