using System;
using ImGuiNET;
using MdxViewer.Rendering;
using MdxViewer.Terrain;
using Silk.NET.OpenGL;

namespace MdxViewer;

public partial class ViewerApp
{
    private TextureFilteringMode _textureFilteringMode = TextureFilteringMode.Trilinear;
    private bool _enableMultisample = true;
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
        if (!ImGui.Begin("Render Quality", ref _showRenderQualityWindow, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

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

        if (ImGui.Button("Reapply To Loaded Textures"))
            ApplyRenderQualitySettings(refreshTextures: true);

        ImGui.TextDisabled("Applies live to standalone MDX, standalone WMO, terrain, and world object renderer caches.");

        ImGui.End();
    }
}