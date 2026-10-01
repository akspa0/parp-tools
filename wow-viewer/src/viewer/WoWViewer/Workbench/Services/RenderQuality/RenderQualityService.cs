using System;
using System.Numerics;
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

        ImGui.Separator();
        ImGui.Text("World Map Objects (WMO)");

        bool wmoTex = WmoRenderer.TexturesEnabled;
        if (ImGui.Checkbox("Enable WMO Textures", ref wmoTex))
        {
            WmoRenderer.TexturesEnabled = wmoTex;
            _settings.SaveViewerSettings();
        }

        float wmoOpacity = WmoRenderer.GlobalOpacity * 100f;
        if (ImGui.SliderFloat("WMO Opacity", ref wmoOpacity, 0f, 100f, "%.0f%%"))
        {
            WmoRenderer.GlobalOpacity = Math.Clamp(wmoOpacity / 100f, 0f, 1f);
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Global WMO Opacity (0% - 100%, e.g. 75% for see-through). Right-click to type.");

        bool wallTex = WmoRenderer.CollisionWallTexturesEnabled;
        if (ImGui.Checkbox("Enable Collision Wall Textures", ref wallTex))
        {
            WmoRenderer.CollisionWallTexturesEnabled = wallTex;
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Toggle textures for invisible/boundary collision walls in WoW:Forever and classic clients.");

        float wallOpacity = WmoRenderer.CollisionWallOpacity * 100f;
        if (ImGui.SliderFloat("Collision Wall Opacity", ref wallOpacity, 0f, 100f, "%.0f%%"))
        {
            WmoRenderer.CollisionWallOpacity = Math.Clamp(wallOpacity / 100f, 0f, 1f);
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Dedicated opacity for collision wall WMOs (0% = invisible, 50% = translucent, 100% = solid).");

        if (ImGui.SmallButton("Hide Collision Walls (0%)"))
        {
            WmoRenderer.CollisionWallOpacity = 0.0f;
            _settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Translucent Walls (50%)"))
        {
            WmoRenderer.CollisionWallOpacity = 0.5f;
            _settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Reset Walls (100%)"))
        {
            WmoRenderer.CollisionWallOpacity = 1.0f;
            _settings.SaveViewerSettings();
        }

        ImGui.Separator();
        ImGui.Text("M2 & MDX Models");

        float m2Opacity = M2Renderer.GlobalOpacity * 100f;
        if (ImGui.SliderFloat("Model Opacity", ref m2Opacity, 0f, 100f, "%.0f%%"))
        {
            M2Renderer.GlobalOpacity = Math.Clamp(m2Opacity / 100f, 0f, 1f);
            MdxRenderer.GlobalOpacity = M2Renderer.GlobalOpacity;
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Global opacity for M2 and MDX models and particle effects (0% - 100%). Right-click to type.");

        if (ImGui.SmallButton("Invisible (0%)##m2"))
        {
            M2Renderer.GlobalOpacity = 0.0f;
            MdxRenderer.GlobalOpacity = 0.0f;
            _settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Translucent (50%)##m2"))
        {
            M2Renderer.GlobalOpacity = 0.5f;
            MdxRenderer.GlobalOpacity = 0.5f;
            _settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("See-Through (75%)##m2"))
        {
            M2Renderer.GlobalOpacity = 0.75f;
            MdxRenderer.GlobalOpacity = 0.75f;
            _settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Solid (100%)##m2"))
        {
            M2Renderer.GlobalOpacity = 1.0f;
            MdxRenderer.GlobalOpacity = 1.0f;
            _settings.SaveViewerSettings();
        }

        ImGui.Separator();
        ImGui.Text("Wireframe Overlay Styling");

        var wireColor = WireframeOverlaySettings.DefaultColor;
        if (ImGui.ColorEdit3("Wireframe Color", ref wireColor, ImGuiColorEditFlags.NoInputs))
        {
            WireframeOverlaySettings.DefaultColor = wireColor;
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Color used for wireframe overlays across WMO, M2, and MDX geometry.");

        ImGui.SameLine();
        if (ImGui.SmallButton("Gold##wf"))
        {
            WireframeOverlaySettings.DefaultColor = new Vector3(1.0f, 0.85f, 0.3f);
            _settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Cyan##wf"))
        {
            WireframeOverlaySettings.DefaultColor = new Vector3(0.2f, 0.8f, 1.0f);
            _settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Green##wf"))
        {
            WireframeOverlaySettings.DefaultColor = new Vector3(0.2f, 1.0f, 0.4f);
            _settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("White##wf"))
        {
            WireframeOverlaySettings.DefaultColor = new Vector3(1.0f, 1.0f, 1.0f);
            _settings.SaveViewerSettings();
        }

        float lineWidth = WireframeOverlaySettings.LineWidth;
        if (ImGui.SliderFloat("Wireframe Width", ref lineWidth, 0.5f, 4.0f, "%.1f px"))
        {
            WireframeOverlaySettings.LineWidth = Math.Clamp(lineWidth, 0.5f, 4.0f);
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Rasterized line thickness in pixels for wireframe meshes.");

        float baseIntensity = WireframeOverlaySettings.BaseIntensity * 100f;
        if (ImGui.SliderFloat("Wireframe Intensity", ref baseIntensity, 10f, 100f, "%.0f%%"))
        {
            WireframeOverlaySettings.BaseIntensity = Math.Clamp(baseIntensity / 100f, 0.1f, 1.0f);
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Base opacity of the wireframe lines so they do not overpower the underlying textures.");

        bool followOpacity = WireframeOverlaySettings.FollowModelOpacity;
        if (ImGui.Checkbox("Tie Wireframe to Model/WMO Opacity", ref followOpacity))
        {
            WireframeOverlaySettings.FollowModelOpacity = followOpacity;
            _settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("When checked, wireframe line alpha automatically fades down when the model or WMO opacity is reduced.");
    }
}
