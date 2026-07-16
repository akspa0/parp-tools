using System;
using ImGuiNET;
using WoWViewer.Terrain;
using WowViewer.Core.Maps;

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

        ImGui.Separator();

        if (ImGui.CollapsingHeader("Camera", ImGuiTreeNodeFlags.DefaultOpen))
        {
            DrawCameraDefaultsContent();
        }
    }

    private void DrawFogDefaultsContent()
    {
        ImGui.TextDisabled("Global fog defaults apply when terrain loads without an active user override.");

        float fogStart = Math.Clamp(_defaultFogStart, 0f, MaxTerrainFogDistance - 1f);
        float fogEnd = Math.Clamp(_defaultFogEnd, 100f, MaxTerrainFogDistance);
        bool fogStartChanged = ImGui.SliderFloat("Fog Start", ref fogStart, 0f, MaxTerrainFogDistance - 1f);
        bool fogEndChanged = ImGui.SliderFloat("Fog End", ref fogEnd, 100f, MaxTerrainFogDistance);
        if (fogStartChanged || fogEndChanged)
        {
            (_defaultFogStart, _defaultFogEnd) = TerrainLightingMath.NormalizeFogRange(fogStart, fogEnd);
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

    private void DrawCameraDefaultsContent()
    {
        ImGui.TextDisabled("Free-fly camera defaults saved with viewer settings.");

        float cameraSpeed = Math.Clamp(_cameraSpeed, 1f, 500f);
        if (ImGui.DragFloat("Camera Speed", ref cameraSpeed, 1f, 1f, 500f, "%.0f"))
        {
            _cameraSpeed = cameraSpeed;
            SaveViewerSettings();
        }

        float fovDegrees = Math.Clamp(_fovDegrees, 20f, 90f);
        if (ImGui.DragFloat("FOV", ref fovDegrees, 0.5f, 20f, 90f, "%.0f°"))
        {
            _fovDegrees = fovDegrees;
            SaveViewerSettings();
        }

        if (ImGui.Button("Reset Camera Defaults"))
        {
            _cameraSpeed = 50f;
            _fovDegrees = 45f;
            SaveViewerSettings();
        }
    }

    /// <summary>
    /// Apply saved fog defaults to terrain lighting after terrain loads.
    /// Call this after terrain manager creation.
    /// </summary>
    private void ApplyGlobalFogDefaults(TerrainLighting lighting)
    {
        (lighting.FogStart, lighting.FogEnd) = TerrainLightingMath.NormalizeFogRange(_defaultFogStart, _defaultFogEnd);
    }
}
