using System;
using ImGuiNET;
using WoWViewer.Terrain;
using WoWViewer.Rendering;
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

        if (ImGui.CollapsingHeader("Dataset Versions", ImGuiTreeNodeFlags.DefaultOpen))
        {
            DrawDatasetVersionSettingsContent();
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
        ImGui.Text("UI Typography & Font Size:");
        float fontScale = _uiFontScale;
        if (ImGui.SliderFloat("Text Font Size", ref fontScale, 0.85f, 2.20f, "%.2fx"))
        {
            _uiFontScale = fontScale;
            if (HasImGuiContext())
            {
                ImGui.GetIO().FontGlobalScale = _uiFontScale;
            }
            SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Scales all UI text, labels, inspector panels, and menus across the entire application.");

        ImGui.TextDisabled("Presets:");
        ImGui.SameLine();
        if (ImGui.SmallButton("100%"))
        {
            _uiFontScale = 1.0f;
            if (HasImGuiContext()) ImGui.GetIO().FontGlobalScale = _uiFontScale;
            SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("120%"))
        {
            _uiFontScale = 1.20f;
            if (HasImGuiContext()) ImGui.GetIO().FontGlobalScale = _uiFontScale;
            SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("135%"))
        {
            _uiFontScale = 1.35f;
            if (HasImGuiContext()) ImGui.GetIO().FontGlobalScale = _uiFontScale;
            SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("150%"))
        {
            _uiFontScale = 1.50f;
            if (HasImGuiContext()) ImGui.GetIO().FontGlobalScale = _uiFontScale;
            SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("175%"))
        {
            _uiFontScale = 1.75f;
            if (HasImGuiContext()) ImGui.GetIO().FontGlobalScale = _uiFontScale;
            SaveViewerSettings();
        }

        ImGui.Spacing();
        ImGui.Separator();

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

        ImGui.Spacing();
        ImGui.Separator();
        ImGui.Text("3D In-Scene Cursor:");

        int currentStyle = (int)(_sceneCursorRenderer?.Style ?? CursorStyle.ClassicOSArrow);
        string[] styleNames = { "Authentic WoW Gauntlet", "Procedural 3D Pointer", "3D Target Reticle", "Classic OS Arrow" };
        if (ImGui.Combo("Cursor Style", ref currentStyle, styleNames, styleNames.Length))
        {
            if (_sceneCursorRenderer != null)
                _sceneCursorRenderer.Style = (CursorStyle)currentStyle;
            SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Authentic WoW Gauntlet loads Interface\\Cursor\\Cursor.mdx. Procedural meshes render OpenSCAD 3D primitives.");

        float cursorScale = _sceneCursorRenderer?.UserScale ?? 1.0f;
        if (ImGui.SliderFloat("Cursor Scale", ref cursorScale, 0.5f, 3.0f, "%.2fx"))
        {
            if (_sceneCursorRenderer != null)
                _sceneCursorRenderer.UserScale = cursorScale;
        }

        ImGui.Spacing();
        ImGui.Separator();
        ImGui.Text("Camera-Rigged 3D HUD (OpenSCAD):");

        if (_cameraHudRig3D != null)
        {
            bool hudEnabled = _cameraHudRig3D.Enabled;
            if (ImGui.Checkbox("Enable 3D Camera HUD", ref hudEnabled))
            {
                _cameraHudRig3D.Enabled = hudEnabled;
            }

            if (hudEnabled)
            {
                bool showGimbal = _cameraHudRig3D.ShowGimbal;
                if (ImGui.Checkbox("3D Attitude & Heading Gimbal", ref showGimbal))
                {
                    _cameraHudRig3D.ShowGimbal = showGimbal;
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Renders an authored 3D OpenSCAD flight attitude & compass ring mounted to the camera entity.");

                bool showBrackets = _cameraHudRig3D.ShowBrackets;
                if (ImGui.Checkbox("3D Viewport Corner Brackets", ref showBrackets))
                {
                    _cameraHudRig3D.ShowBrackets = showBrackets;
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Renders 3D geometric framing brackets at the camera viewport boundary.");
            }
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
