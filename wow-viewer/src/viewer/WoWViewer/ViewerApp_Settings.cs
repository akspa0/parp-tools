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
        DrawAuthoritativeFogControls(showDescription: true);
    }

    /// <summary>
    /// The sole fog editor for the viewer. It reads and writes WorldScene's user
    /// override when a scene exists; TerrainLighting is a derived render-time value.
    /// </summary>
    private void DrawAuthoritativeFogControls(bool showDescription = true)
    {
        if (showDescription)
            ImGui.TextDisabled("Changes apply to the loaded world immediately and become the saved defaults for the next terrain load.");

        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        (float currentFogStart, float currentFogEnd) = GetAuthoritativeFogRange(lighting);

        float fogEnd = Math.Clamp(currentFogEnd, 100f, MaxTerrainFogDistance);
        float fogStart = Math.Clamp(currentFogStart, 0f, MaxTerrainFogDistance - 1f);

        // Operator correction 2026-09-06: natural order - Fog Start first, Fog End
        // after. This supersedes Spec 223 T501's "Fog End first in every profile"
        // requirement, which caused the operator to adjust the wrong slider.
        bool fogStartChanged = ImGui.SliderFloat("Fog Start", ref fogStart, 0f, MaxTerrainFogDistance - 1f, "%.0f");
        bool fogEndChanged = ImGui.SliderFloat("Fog End", ref fogEnd, 100f, MaxTerrainFogDistance, "%.0f");

        if (fogEndChanged || fogStartChanged)
        {
            // A crossing drag (Start >= End) must adjust Start, never snap both
            // values to defaults; NormalizeFogRange's default fallback would
            // otherwise rubberband both sliders back to 200/1500.
            fogStart = Math.Clamp(fogStart, 0f, MathF.Max(0f, fogEnd - TerrainLightingMath.MinimumFogRangeSpan));
            SetAuthoritativeFogRange(fogStart, fogEnd, persistAsDefault: true);
        }

        if (lighting != null && _worldScene != null)
        {
            bool useLitFog = _worldScene.UseLitFogOverride;
            if (ImGui.Checkbox("Use LIT fog", ref useLitFog))
                _worldScene.UseLitFogOverride = useLitFog;
            ImGui.TextDisabled($"Fog/detail range: {currentFogStart:F0}–{currentFogEnd:F0}; WDL horizon clips at {ComputeSceneFarPlane(currentFogEnd):F0} (+2500).");
        }
    }

    /// <summary>
    /// Returns the range owned by the user-facing fog editor. WorldScene composes
    /// lighting recommendations each frame, so its mutable TerrainLighting values
    /// must not be used to repopulate a slider after writing an override.
    /// </summary>
    private (float FogStart, float FogEnd) GetAuthoritativeFogRange(TerrainLighting? lighting = null)
    {
        if (_worldScene is { HasUserFogRangeOverride: true } scene)
            return (scene.UserFogStart, scene.UserFogEnd);

        if (_worldScene != null)
            return (_worldScene.ActiveFogStart, _worldScene.ActiveFogEnd);

        if (lighting != null)
            return (lighting.FogStart, lighting.FogEnd);

        return (_defaultFogStart, _defaultFogEnd);
    }

    private void SetAuthoritativeFogRange(float start, float end, bool persistAsDefault = true)
    {
        (float normalizedStart, float normalizedEnd) = TerrainLightingMath.NormalizeFogRange(start, end);

        if (_worldScene != null)
        {
            _worldScene.SetUserFogRangeOverride(normalizedStart, normalizedEnd);
        }
        else
        {
            TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
            if (lighting != null)
            {
                lighting.FogStart = normalizedStart;
                lighting.FogEnd = normalizedEnd;
            }
        }

        if (persistAsDefault)
        {
            _defaultFogStart = normalizedStart;
            _defaultFogEnd = normalizedEnd;
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
        ImGui.TextDisabled("Decorative camera HUD only; the interactive spatial workbench remains Spec 212 Phase 3+.");

        if (_cameraHudRig != null)
        {
            bool hudEnabled = _cameraHudRig.Enabled;
            if (ImGui.Checkbox("Enable 3D Camera HUD", ref hudEnabled))
            {
                _cameraHudRig.Enabled = hudEnabled;
            }

            if (hudEnabled)
            {
                bool showGimbal = _cameraHudRig.ShowGimbal;
                if (ImGui.Checkbox("3D Attitude & Heading Gimbal", ref showGimbal))
                {
                    _cameraHudRig.ShowGimbal = showGimbal;
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Renders an authored 3D OpenSCAD flight attitude & compass ring mounted to the camera entity.");

                bool showBrackets = _cameraHudRig.ShowBrackets;
                if (ImGui.Checkbox("3D Viewport Corner Brackets", ref showBrackets))
                {
                    _cameraHudRig.ShowBrackets = showBrackets;
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
