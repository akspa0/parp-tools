using System;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using Silk.NET.OpenGL;

namespace WoWViewer;

public partial class ViewerApp
{
    private TextureFilteringMode _textureFilteringMode = TextureFilteringMode.Trilinear;
    private bool _enableMultisample = true;
    private bool _enableTerrainBackfaceCulling = true;
    private bool _enableWmoBackfaceCulling;
    private int _sampleBufferCount;
    private int _sampleCount;

    private bool SupportsRuntimeMultisampleToggle => _sampleBufferCount > 0 && _sampleCount > 0;

    private void DetectRenderQualityCapabilities()
    {
        _sampleBufferCount = _gl.GetInteger(GetPName.SampleBuffers);
        _sampleCount = _gl.GetInteger(GetPName.Samples);
    }

    private void ApplyRenderQualitySettings(bool refreshTextures)
    {
        RenderQualitySettings.TextureFilteringMode = _textureFilteringMode;
        RenderQualitySettings.EnableTerrainBackfaceCulling = _enableTerrainBackfaceCulling;
        RenderQualitySettings.EnableWmoBackfaceCulling = _enableWmoBackfaceCulling;

        if (SupportsRuntimeMultisampleToggle && _enableMultisample)
            _gl.Enable(EnableCap.Multisample);
        else
            _gl.Disable(EnableCap.Multisample);

        if (!refreshTextures)
            return;

        if (_renderer is WorldScene worldScene)
        {
            worldScene.ApplyTextureSamplingSettings();
            return;
        }

        if (_renderer is IModelRenderer modelRenderer)
            modelRenderer.ApplyTextureSamplingSettings();

        if (_renderer is WmoRenderer wmoRenderer)
            wmoRenderer.ApplyTextureSamplingSettings();

        _terrainManager?.Renderer.ApplyTextureSamplingSettings();
        _vlmTerrainManager?.Renderer.ApplyTextureSamplingSettings();
    }

    private void DrawRenderQualityWindow()
    {
        // 069 Phase 16: wrapper keeps legacy floating-window behavior.
        // Workbench sub-tab uses DrawRenderQualityContent directly.
        if (!ImGui.Begin("Render Quality", ref _showRenderQualityWindow, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }
        DrawRenderQualityContent();
        ImGui.End();
    }

    private void DrawRenderQualityContent()
    {
        if (ImGui.BeginCombo("Texture Filtering", RenderQualitySettings.GetLabel(_textureFilteringMode)))
        {
            foreach (TextureFilteringMode mode in Enum.GetValues(typeof(TextureFilteringMode)))
            {
                bool selected = mode == _textureFilteringMode;
                if (ImGui.Selectable(RenderQualitySettings.GetLabel(mode), selected))
                {
                    _textureFilteringMode = mode;
                    ApplyRenderQualitySettings(refreshTextures: true);
                    SaveViewerSettings();
                }

                if (selected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        if (SupportsRuntimeMultisampleToggle)
        {
            bool enabled = _enableMultisample;
            if (ImGui.Checkbox($"Object MSAA ({_sampleCount}x)", ref enabled))
            {
                _enableMultisample = enabled;
                ApplyRenderQualitySettings(refreshTextures: false);
                SaveViewerSettings();
            }

            ImGui.TextDisabled($"Swapchain sample buffers: {_sampleBufferCount}");
        }
        else
        {
            bool disabled = false;
            ImGui.BeginDisabled();
            ImGui.Checkbox("Object MSAA", ref disabled);
            ImGui.EndDisabled();
            ImGui.TextDisabled("Current GL window did not provide multisample buffers, so object AA cannot be toggled live in this context.");
        }

        bool terrainCull = _enableTerrainBackfaceCulling;
        if (ImGui.Checkbox("Cull Terrain Backfaces", ref terrainCull))
        {
            _enableTerrainBackfaceCulling = terrainCull;
            ApplyRenderQualitySettings(refreshTextures: false);
            SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Terrain chunk winding is generated locally, so this can safely skip underside fragments in the normal terrain pass.");

        bool wmoCull = _enableWmoBackfaceCulling;
        if (ImGui.Checkbox("Cull WMO Backfaces (Experimental)", ref wmoCull))
        {
            _enableWmoBackfaceCulling = wmoCull;
            ApplyRenderQualitySettings(refreshTextures: false);
            SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Applies backface culling to WMO shell passes. Keep this off if a scene has intentionally double-sided WMO materials until we finish a material-flag-aware rollout.");

        if (ImGui.Button("Reapply To Loaded Textures"))
            ApplyRenderQualitySettings(refreshTextures: true);

        ImGui.TextDisabled("Applies live to standalone MDX, standalone WMO, terrain, and world object renderer caches.");
    }
}
