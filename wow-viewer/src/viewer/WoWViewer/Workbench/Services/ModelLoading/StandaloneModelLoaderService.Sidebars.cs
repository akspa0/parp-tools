using System.Numerics;
using System.Diagnostics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Workbench;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;
using WoWViewer.Population;
using WoWViewer.UI;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// StandaloneModelLoaderService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class StandaloneModelLoaderService
{

    internal void DrawStandaloneCharacterVariationControls(IModelRenderer renderer)
    {
        string? modelPath = (renderer as MdxRenderer)?.ModelVirtualPath
            ?? (renderer as M2Renderer)?.SourceModelPath
            ?? _standaloneCharacterCustomizationModelPath;
        if (string.IsNullOrWhiteSpace(modelPath) || _texResolver == null)
            return;

        string normalizedPath = modelPath.Replace('/', '\\');
        if (!string.Equals(_standaloneCharacterCustomizationModelPath, normalizedPath, StringComparison.OrdinalIgnoreCase))
        {
            bool isM2 = renderer is M2Renderer || (renderer as MdxRenderer)?.IsM2AdapterModel == true;
            RefreshStandaloneCharacterCustomizationState(normalizedPath, isM2AdapterModel: isM2);
        }

        if (string.IsNullOrWhiteSpace(_standaloneCharacterCustomizationModelPath))
            return;

        bool hasHairOptions = _standaloneCharacterHairVariationIds.Count > 0;
        bool hasFacialOptions = _standaloneCharacterFacialHairVariationIds.Count > 0;
        if (!hasHairOptions && !hasFacialOptions)
            return;

        ImGui.Separator();
        ImGui.Text("Character Variants:");
        ImGui.TextDisabled("Raw DBC variation ids for standalone classic character MDX inspection.");

        bool changed = false;
        if (hasHairOptions)
            changed |= DrawStandaloneCharacterVariationCombo("Hair VariationId", "##StandaloneCharacterHairVariation", _standaloneCharacterHairVariationIds, ref _standaloneCharacterHairVariationOverride);

        if (hasFacialOptions)
            changed |= DrawStandaloneCharacterVariationCombo("Facial VariationId", "##StandaloneCharacterFacialVariation", _standaloneCharacterFacialHairVariationIds, ref _standaloneCharacterFacialHairVariationOverride);

        if ((_standaloneCharacterHairVariationOverride >= 0 || _standaloneCharacterFacialHairVariationOverride >= 0)
            && ImGui.Button("Reset Character Variants"))
        {
            _standaloneCharacterHairVariationOverride = -1;
            _standaloneCharacterFacialHairVariationOverride = -1;
            changed = true;
        }

        if (changed)
            ApplyStandaloneCharacterCustomizationOverrides();
    }

    private static bool DrawStandaloneCharacterVariationCombo(string label, string comboId, IReadOnlyList<int> variationIds, ref int selectedVariationId)
    {
        ImGui.Text(label);
        ImGui.SetNextItemWidth(-1);

        string preview = selectedVariationId >= 0
            ? $"VariationId {selectedVariationId}"
            : "Default (VariationId 0)";
        bool changed = false;

        if (ImGui.BeginCombo(comboId, preview))
        {
            bool defaultSelected = selectedVariationId < 0;
            if (ImGui.Selectable("Default (VariationId 0)", defaultSelected))
            {
                selectedVariationId = -1;
                changed = true;
            }

            if (defaultSelected)
                ImGui.SetItemDefaultFocus();

            foreach (int variationId in variationIds)
            {
                bool selected = selectedVariationId == variationId;
                if (ImGui.Selectable($"VariationId {variationId}", selected))
                {
                    selectedVariationId = variationId;
                    changed = true;
                }

                if (selected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        return changed;
    }
}
