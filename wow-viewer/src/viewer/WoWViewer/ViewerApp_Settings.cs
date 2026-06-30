using System;
using ImGuiNET;
using WoWViewer.Terrain;

namespace WoWViewer;

public partial class ViewerApp
{
    private bool _showSettingsWindow;

    private void DrawSettingsWindow()
    {
        if (!ImGui.Begin("Settings", ref _showSettingsWindow))
        {
            ImGui.End();
            return;
        }
        DrawSettingsContent();
        ImGui.End();
    }

    private void DrawSettingsContent()
    {
        if (ImGui.CollapsingHeader("Render Quality", ImGuiTreeNodeFlags.DefaultOpen))
        {
            DrawRenderQualityContent();
        }

        ImGui.Separator();

        if (ImGui.CollapsingHeader("Fog Defaults", ImGuiTreeNodeFlags.DefaultOpen))
        {
            DrawFogDefaultsContent();
        }

        ImGui.Separator();

        if (ImGui.CollapsingHeader("Interface", ImGuiTreeNodeFlags.DefaultOpen))
        {
            DrawInterfaceSettingsContent();
        }
    }

    private void DrawFogDefaultsContent()
    {
        ImGui.TextDisabled("Global fog defaults applied when terrain loads.");

        float fogStart = Math.Clamp(_defaultFogStart, 0f, MaxTerrainFogDistance - 1f);
        if (ImGui.DragFloat("Fog Start", ref fogStart, 10f, 0f, MaxTerrainFogDistance - 1f))
        {
            _defaultFogStart = fogStart;
            SaveViewerSettings();
        }

        float fogEnd = Math.Clamp(_defaultFogEnd, 100f, MaxTerrainFogDistance);
        if (ImGui.DragFloat("Fog End", ref fogEnd, 10f, 100f, MaxTerrainFogDistance))
        {
            _defaultFogEnd = fogEnd;
            SaveViewerSettings();
        }
    }

    private void DrawInterfaceSettingsContent()
    {
        bool useTabUi = _useTabUi;
        if (ImGui.Checkbox("Use Tabbed UI", ref useTabUi))
        {
            _useTabUi = useTabUi;
            SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Toggle between the modern tabbed workbench and legacy dockspace layout.");

        bool showMinimap = _showMinimapWindow;
        if (ImGui.Checkbox("Show Minimap", ref showMinimap))
        {
            _showMinimapWindow = showMinimap;
            SaveViewerSettings();
        }
    }

    /// <summary>
    /// Apply saved fog defaults to terrain lighting after terrain loads.
    /// Call this after terrain manager creation.
    /// </summary>
    private void ApplyGlobalFogDefaults(TerrainLighting lighting)
    {
        lighting.FogStart = _defaultFogStart;
        lighting.FogEnd = _defaultFogEnd;
    }
}
