using System;
using System.Collections.Generic;

namespace WoWViewer;

/// <summary>One bottom-toolbar liquid-family toggle.</summary>
internal readonly record struct BottomLiquidToggle(string Id, string Label, string Tooltip);

/// <summary>
/// Pure bottom-toolbar sizing policy. The toolbar offers only the texture layers that the
/// resident terrain actually carries, and one alpha selector per non-base layer. Keeping the
/// arithmetic here lets the headless footer tests exercise the same rule the viewer draws with.
/// </summary>
internal static class BottomBarControlPolicy
{
    /// <summary>Hard cap shared with <c>TerrainTileMeshBuilder.MaxLayers</c>.</summary>
    public const int MaxLayers = 8;

    /// <summary>Number of L0..Ln visibility toggles for the resident terrain.</summary>
    public static int ResolveLayerCount(int maxResidentLayerCount)
        => Math.Clamp(maxResidentLayerCount, 0, MaxLayers);

    /// <summary>Number of A1..An alpha-channel selectors (base layer has no alpha map).</summary>
    public static int ResolveAlphaChannelCount(int layerCount)
        => Math.Max(0, layerCount - 1);

    /// <summary>
    /// Liquid families carried by MCLQ/MH2O and WL* data. A single "Water" switch hid magma,
    /// slime, and ocean surfaces, so each family gets its own toolbar toggle.
    /// </summary>
    public static IReadOnlyList<BottomLiquidToggle> LiquidToggles { get; } = new[]
    {
        new BottomLiquidToggle("toggle-liquid-water", "Water", "Show MCLQ/MH2O and WL* water and river liquid."),
        new BottomLiquidToggle("toggle-liquid-ocean", "Ocean", "Show MCLQ/MH2O and WL* ocean liquid."),
        new BottomLiquidToggle("toggle-liquid-magma", "Magma", "Show MCLQ/MH2O and WL* magma and lava liquid."),
        new BottomLiquidToggle("toggle-liquid-slime", "Slime", "Show MCLQ/MH2O and WL* slime liquid."),
    };
}
