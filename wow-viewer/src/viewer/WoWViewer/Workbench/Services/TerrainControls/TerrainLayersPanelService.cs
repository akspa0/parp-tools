using System;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.UI;

namespace WoWViewer.Workbench.Services.TerrainControls;

/// <summary>
/// Photoshop-style Terrain Layers workbench providing per-layer opacity sliders,
/// alpha mask overlay blending, holes control, and Ghidra client binary analysis tooltips
/// decoding MCLY flags and ground effect clutter IDs (Spec 287 Phase 3).
/// </summary>
internal sealed class TerrainLayersPanelService
{
    private readonly IViewerAppHost _host;

    public TerrainLayersPanelService(IViewerAppHost host)
    {
        _host = host ?? throw new ArgumentNullException(nameof(host));
    }

    /// <summary>
    /// Draws the complete Photoshop-style Layers workbench content.
    /// </summary>
    public void DrawLayersPanelContent()
    {
        TerrainRenderer? terrain = _host.TerrainManager?.Renderer ?? _host.VlmTerrainManager?.Renderer;

        if (terrain == null)
        {
            SharedUiWidgets.DrawEmptyState(
                "No terrain loaded",
                "Load a terrain-backed world map to inspect texture layers, adjust per-layer opacities, and blend alpha masks.");
            if (SharedUiWidgets.ActionButton("Browse Maps...", small: true))
            {
                _host.UiManager.OpenLeftDock(LeftPanel.Maps);
            }
            return;
        }

        int maxLayers = Math.Clamp(terrain.MaxResidentLayerCount, 4, TerrainTileMeshBuilder.MaxLayers);

        ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "Photoshop-Style Layer Blending");
        ImGui.TextDisabled("Adjust individual layer opacity multipliers and blend raw alpha masks over diffuse terrain.");
        ImGui.Separator();

        // 1. Alpha Mask Overlay Section
        if (SharedUiWidgets.SectionHeader("Alpha Mask Overlay Blending", defaultOpen: true, id: "LayersAlphaBlendHeader"))
        {
            float alphaBlend = terrain.AlphaMaskBlend;
            ImGui.SetNextItemWidth(180f);
            if (ImGui.SliderFloat("Alpha Overlay##AlphaMaskBlendSlider", ref alphaBlend, 0f, 1f, "%.2f"))
            {
                terrain.AlphaMaskBlend = Math.Clamp(alphaBlend, 0f, 1f);
            }
            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip(
                    "Blends raw grayscale alpha mask directly over painted terrain diffuse.\n" +
                    "0.0 = Pure diffuse tileset\n" +
                    "1.0 = Pure grayscale alpha mask\n" +
                    "0.5 = 50% split showing both simultaneously (Photoshop overlay style)");
            }

            ImGui.SameLine();
            if (ImGui.SmallButton("Reset##ResetAlphaBlend"))
            {
                terrain.AlphaMaskBlend = 0f;
            }

            int alphaChannel = terrain.AlphaMaskChannel;
            string[] channelLabels = { "All Combined (Product)", "Layer 1 Alpha", "Layer 2 Alpha", "Layer 3 Alpha", "Layer 4 Alpha", "Layer 5 Alpha", "Layer 6 Alpha", "Layer 7 Alpha" };
            ImGui.SetNextItemWidth(200f);
            if (ImGui.Combo("Channel##AlphaMaskChannelCombo", ref alphaChannel, channelLabels, Math.Min(maxLayers, channelLabels.Length)))
            {
                terrain.AlphaMaskChannel = alphaChannel;
            }
            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip("Selects which layer's alpha map drives the grayscale overlay.\nChannel 0 combines all active layers via (1 - product(1 - a_i)).");
            }
        }

        ImGui.Separator();

        // 2. Per-Layer Stack
        if (SharedUiWidgets.SectionHeader("Resident Texture Layers Stack", defaultOpen: true, id: "LayersStackHeader"))
        {
            ImGui.TextDisabled($"Max resident layers in current tiles: {maxLayers}");

            for (int layer = 0; layer < maxLayers; layer++)
            {
                ImGui.PushID($"LayerRow_{layer}");

                bool visible = terrain.GetShowLayer(layer);
                if (ImGui.Checkbox(layer == 0 ? "Base (L0)" : $"L{layer}", ref visible))
                {
                    terrain.SetShowLayer(layer, visible);
                }

                ImGui.SameLine();
                float opacity = terrain.LayerOpacities[layer];
                ImGui.SetNextItemWidth(140f);
                if (ImGui.SliderFloat("##Opacity", ref opacity, 0f, 1f, "%.2f"))
                {
                    terrain.LayerOpacities[layer] = Math.Clamp(opacity, 0f, 1f);
                }

                ImGui.SameLine();
                ImGui.TextDisabled(layer == 0 ? "Base Diffuse" : $"Layer {layer}");

                ImGui.SameLine();
                DrawGhidraLayerFlagsHelpMarker(layer);

                ImGui.PopID();
            }

            ImGui.Spacing();
            if (ImGui.Button("All 100%##ResetAllOpacities"))
            {
                for (int i = 0; i < 8; i++)
                {
                    terrain.LayerOpacities[i] = 1f;
                    terrain.SetShowLayer(i, true);
                }
            }
            ImGui.SameLine();
            if (ImGui.Button("Base Only##ShowBaseOnly"))
            {
                for (int i = 0; i < 8; i++)
                {
                    terrain.SetShowLayer(i, i == 0);
                    terrain.LayerOpacities[i] = 1f;
                }
            }
            ImGui.SameLine();
            if (ImGui.Button("Hide All##HideAllLayers"))
            {
                for (int i = 0; i < 8; i++)
                {
                    terrain.SetShowLayer(i, false);
                }
            }
        }

        ImGui.Separator();

        // 3. Terrain Holes & WDL
        if (SharedUiWidgets.SectionHeader("Terrain Holes & WDL", defaultOpen: true, id: "LayersHolesHeader"))
        {
            bool holes = !(_host.TerrainManager?.IgnoreTerrainHolesGlobally
                ?? _host.VlmTerrainManager?.IgnoreTerrainHolesGlobally ?? false);
            if (ImGui.Checkbox("Holes##TerrainHolesCheckbox", ref holes)
                && _host.TerrainControlsPanel.SetIgnoreTerrainHolesGlobally(!holes))
            {
            }

            ImGui.SameLine();
            if (_host.WorldScene != null)
            {
                bool wdl = _host.WorldScene.ShowWdlTerrain;
                if (ImGui.Checkbox("WDL##WdlTerrainCheckbox", ref wdl))
                    _host.WorldScene.ShowWdlTerrain = wdl;
            }

            if (SharedUiWidgets.ActionButton("Derive Holes from Burgundy MCCV", small: true))
            {
                if (_host.TerrainManager != null)
                    _host.MccvHoleDetection.DeriveBurgundyHoles(_host.TerrainManager);
            }
            if (_host.MccvHoleDetection.LastStatusMessage != null)
                ImGui.TextWrapped(_host.MccvHoleDetection.LastStatusMessage);
        }

        ImGui.Separator();

        // 4. Reverse-Engineered Format Reference & Decoding
        if (SharedUiWidgets.SectionHeader("MCLY Flags & Ground Clutter Decoding", defaultOpen: false, id: "LayersGhidraDecodingHeader"))
        {
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "Ghidra Client Binary Analysis: MCLY & EffectId Ground Truth");
            ImGui.TextWrapped(
                "MCLY records in ADT terrain chunks define texture layering, UV transformation, and procedural foliage placement.\n\n" +
                "* 0x0001 (FLAG_ANIM_ENABLED): Translates texture UVs across time using direction and speed.\n" +
                "* 0x0002 (FLAG_OVERBRIGHT): Multiplies layer lighting brightness by 2.0x.\n" +
                "* 0x0004 (FLAG_USE_ALPHA_MAP): Indicates valid MCAL alpha data exists for this layer.\n" +
                "* 0x0007 (FLAG_ANIM_DIRECTION): UV scroll direction (3 bits): 0=0°, 1=45°, 2=90°, 3=135°, 4=180°, 5=225°, 6=270°, 7=315°.\n" +
                "* 0x0038 (FLAG_ANIM_SPEED): UV scroll speed multiplier (3 bits): scales scroll velocity.\n" +
                "* 0x0080 (FLAG_SPECULAR_MAP): Cataclysm+ specular gloss map reflection.\n" +
                "* 0x0100 (FLAG_UNCOMPRESSED_ALPHA): Raw 4096-byte uncompressed 8-bit alpha map (BigAlpha).\n" +
                "* 0x0200 (FLAG_COMPRESSED_ALPHA): Run-length encoded (RLE) compressed alpha chunks.\n" +
                "* 0x0400 (FLAG_CUBEMAP_REFLECTION): Reflects active skybox / cubemap on the surface.\n" +
                "* 0x8000 (FLAG_DYNAMIC_TEXTURE): Client animated dynamic texture surface.\n\n" +
                "* EffectId: GroundEffectTexture.dbc / GroundEffectDoodad.dbc record ID. Drives procedural 3D detail mesh instances (grass, flowers, rocks) spawned on this layer's alpha weights.");
        }
    }

    public void DrawQuickLayersPopup(TerrainRenderer? terrain)
    {
        if (terrain == null)
        {
            ImGui.TextDisabled("No terrain loaded");
            ImGui.TextUnformatted("Load a terrain map to adjust texture layers and alpha masks.");
            ImGui.Spacing();
            if (ImGui.Button("Browse Maps...##QuickBrowseMaps"))
            {
                _host.UiManager.OpenLeftDock(LeftPanel.Maps);
                ImGui.CloseCurrentPopup();
            }
            return;
        }

        ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "Alpha Masks & Layer Opacities");
        ImGui.Separator();

        // Alpha Mask Overlay
        float alphaBlend = terrain.AlphaMaskBlend;
        ImGui.SetNextItemWidth(150f);
        if (ImGui.SliderFloat("Alpha Overlay##QuickAlphaBlend", ref alphaBlend, 0f, 1f, "%.2f"))
        {
            terrain.AlphaMaskBlend = Math.Clamp(alphaBlend, 0f, 1f);
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Reset##ResetQuickAlpha"))
        {
            terrain.AlphaMaskBlend = 0f;
        }

        int alphaChannel = terrain.AlphaMaskChannel;
        string[] channelLabels = { "All Combined", "L1 Alpha", "L2 Alpha", "L3 Alpha", "L4 Alpha", "L5 Alpha", "L6 Alpha", "L7 Alpha" };
        int maxLayers = Math.Clamp(terrain.MaxResidentLayerCount, 4, TerrainTileMeshBuilder.MaxLayers);
        ImGui.SetNextItemWidth(150f);
        if (ImGui.Combo("Channel##QuickAlphaChannel", ref alphaChannel, channelLabels, Math.Min(maxLayers, channelLabels.Length)))
        {
            terrain.AlphaMaskChannel = alphaChannel;
        }

        ImGui.Separator();
        ImGui.TextDisabled("Texture Layers Stack:");
        for (int layer = 0; layer < maxLayers; layer++)
        {
            ImGui.PushID($"QuickLayer_{layer}");
            bool visible = terrain.GetShowLayer(layer);
            if (ImGui.Checkbox(layer == 0 ? "Base (L0)" : $"L{layer}", ref visible))
            {
                terrain.SetShowLayer(layer, visible);
            }
            ImGui.SameLine();
            float opacity = terrain.LayerOpacities[layer];
            ImGui.SetNextItemWidth(120f);
            if (ImGui.SliderFloat("##Opacity", ref opacity, 0f, 1f, "%.2f"))
            {
                terrain.LayerOpacities[layer] = Math.Clamp(opacity, 0f, 1f);
            }
            ImGui.SameLine();
            ImGui.TextDisabled(layer == 0 ? "Base Diffuse" : $"Layer {layer}");
            ImGui.PopID();
        }

        ImGui.Spacing();
        if (ImGui.SmallButton("All 100%##QuickAll100"))
        {
            for (int i = 0; i < 8; i++) { terrain.LayerOpacities[i] = 1f; terrain.SetShowLayer(i, true); }
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Base Only##QuickBaseOnly"))
        {
            for (int i = 0; i < 8; i++) { terrain.SetShowLayer(i, i == 0); terrain.LayerOpacities[i] = 1f; }
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Hide All##QuickHideAll"))
        {
            for (int i = 0; i < 8; i++) { terrain.SetShowLayer(i, false); }
        }

        ImGui.Separator();
        if (ImGui.Button("Open Full Layers Workbench...##QuickOpenFull"))
        {
            _host.UiManager.OpenLeftDock(LeftPanel.Layers);
            ImGui.CloseCurrentPopup();
        }
    }

    private static void DrawGhidraLayerFlagsHelpMarker(int layer)
    {
        ImGui.TextDisabled("[?]");
        if (ImGui.IsItemHovered())
        {
            ImGui.BeginTooltip();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), $"Layer {layer} MCLY & EffectId Spec");
            ImGui.Text(layer == 0
                ? "Base layer (Layer 0): Provides full opaque background coverage; no alpha map needed."
                : $"Overlay layer (Layer {layer}): Blended over underlying layers via MCAL alpha weight.");
            ImGui.Separator();
            ImGui.Text("Ghidra analysis flags ground truth:");
            ImGui.Text("• 0x07: UV Anim Direction (0..7 x 45°)");
            ImGui.Text("• 0x38: UV Anim Speed multiplier");
            ImGui.Text("• 0x80: Specular Gloss Map reflection");
            ImGui.Text("• 0x100: Uncompressed 4096-byte 8-bit alpha");
            ImGui.Text("• 0x200: RLE Compressed alpha");
            ImGui.Text("• 0x400: Cubemap / skybox reflection");
            ImGui.Text("• EffectId: Procedural ground clutter doodad ID");
            ImGui.EndTooltip();
        }
    }
}
