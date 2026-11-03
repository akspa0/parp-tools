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
/// Persistent bottom bar service hosting quick-toggle PM4 checkbox,
/// drop-up Alpha Masks & Layer Opacities menu, and 4 destination mode switches
/// (Viewer, Editor, Archaeology, Options).
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
        ViewerUiManager ui = _host.UiManager;

        var items = new List<FooterItemMeasure>
        {
            new("toggle-pm4", 62f * scale, 48f * scale, FooterItemPriority.Essential),
            new("dropup-layers", 120f * scale, 92f * scale, FooterItemPriority.Essential),
            new("mode-viewer", 76f * scale, 58f * scale, FooterItemPriority.Essential),
            new("mode-editor", 74f * scale, 56f * scale, FooterItemPriority.Essential),
            new("mode-archaeology", 104f * scale, 80f * scale, FooterItemPriority.Essential),
            new("mode-options", 76f * scale, 58f * scale, FooterItemPriority.Essential),
        };

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

        void DrawModeButton(string id, string label, bool active, Action onClick, string tooltip)
        {
            FooterItemPlacement placement = fit.Get(id);
            if (!placement.Visible) return;
            ImGui.SetCursorPos(new Vector2(originX + placement.X, y));
            if (active)
            {
                ImGui.PushStyleColor(ImGuiCol.Button, UiTokens.BrassDim);
                ImGui.PushStyleColor(ImGuiCol.Text, UiTokens.Text);
            }
            if (ImGui.Button($"{label}##{id}", new Vector2(placement.Width, height)))
            {
                onClick();
            }
            if (active)
            {
                ImGui.PopStyleColor(2);
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

        // 2. Drop-up Layers & Opacity button
        FooterItemPlacement layersPlacement = fit.Get("dropup-layers");
        if (layersPlacement.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + layersPlacement.X, y));
            if (ImGui.Button("Layers & Opacity##dropup-layers", new Vector2(layersPlacement.Width, height)))
            {
                ImGui.OpenPopup("BottomLayersDropupPopup");
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Quick drop-up: Alpha masks blend, layer opacities, and Photoshop-style blending");

            if (ImGui.BeginPopup("BottomLayersDropupPopup"))
            {
                TerrainRenderer? terrain = terrainManager?.Renderer ?? vlmTerrain?.Renderer;
                _host.TerrainLayersPanel.DrawQuickLayersPopup(terrain);
                ImGui.EndPopup();
            }
        }

        // 3. 4 Mode switches: Viewer, Editor, Archaeology, Options
        bool isViewer = ui.Destination == WorkbenchTab.Viewer;
        bool isEditor = ui.Destination == WorkbenchTab.Editor;
        bool isArchaeology = ui.Destination == WorkbenchTab.Archaeology;
        bool isOptions = ui.LeftDockOpen && ui.ActiveLeftPanel == LeftPanel.Options;

        DrawModeButton("mode-viewer", "Viewer", isViewer, () =>
        {
            ui.Navigate(WorkbenchTab.Viewer);
            ui.OpenViewerDock(ui.ViewerPanel);
        }, "Switch to Viewer mode (Inspector / World / PM4 / Exports)");

        DrawModeButton("mode-editor", "Editor", isEditor, () =>
        {
            ui.Navigate(WorkbenchTab.Editor);
        }, "Switch to Editor mode (Placement / Terrain Tools / Data I/O / Converters)");

        DrawModeButton("mode-archaeology", "Archaeology", isArchaeology, () =>
        {
            ui.Navigate(WorkbenchTab.Archaeology);
        }, "Switch to Archaeology mode (Timeline / Layers / Provenance)");

        DrawModeButton("mode-options", "Options", isOptions, () =>
        {
            ui.ToggleOptions();
        }, "Toggle Options panel in left sidebar");
    }
}
