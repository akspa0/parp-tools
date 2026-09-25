using System;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using Silk.NET.OpenGL;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Render quality settings: presets and quality controls.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class RenderQualityService
{
    private readonly IViewerAppHost _host;

    internal RenderQualityService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see RenderQualityService.Host.cs.

    internal TextureFilteringMode _textureFilteringMode = TextureFilteringMode.Trilinear;
    internal bool _enableMultisample = true;
    internal bool _enableTerrainBackfaceCulling = true;
    private int _sampleBufferCount;
    private int _sampleCount;
    internal float _defaultFogStart = 200f;
    internal float _defaultFogEnd = 1500f;

    private bool SupportsRuntimeMultisampleToggle => _sampleBufferCount > 0 && _sampleCount > 0;

    internal void DetectRenderQualityCapabilities()
    {
        _sampleBufferCount = _gl.GetInteger(GetPName.SampleBuffers);
        _sampleCount = _gl.GetInteger(GetPName.Samples);
    }

    internal void ApplyRenderQualitySettings(bool refreshTextures)
    {
        RenderQualitySettings.TextureFilteringMode = _textureFilteringMode;
        RenderQualitySettings.EnableTerrainBackfaceCulling = _enableTerrainBackfaceCulling;

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


    internal void DrawRenderQualityContent()
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
                    _settings.SaveViewerSettings();
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
                _settings.SaveViewerSettings();
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
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Terrain chunk winding is generated locally, so this can safely skip underside fragments in the normal terrain pass.");

        if (ImGui.Button("Reapply To Loaded Textures"))
            ApplyRenderQualitySettings(refreshTextures: true);

        ImGui.TextDisabled("Applies live to standalone MDX, standalone WMO, terrain, and world object renderer caches.");
    }
}
