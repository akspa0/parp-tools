using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>Effective colors and fog range consumed by the interactive terrain renderer.</summary>
public readonly record struct TerrainViewerLightingState(
    Vector3 DirectionalColor,
    Vector3 AmbientColor,
    Vector3 FogColor,
    float FogStart,
    float FogEnd);

/// <summary>
/// Composes an authored local lighting profile over the viewer's always-present global sun.
/// A missing or zero-weight local profile is an identity operation.
/// </summary>
public static class TerrainViewerLightingComposer
{
    public static TerrainViewerLightingState ComposeGlobalWithLocal(
        TerrainViewerLightingState global,
        TerrainViewerLightingState local,
        float localWeight)
    {
        float weight = float.IsFinite(localWeight)
            ? Math.Clamp(localWeight, 0f, 1f)
            : 0f;

        if (weight <= 0f)
            return global;

        float fogStart = global.FogStart + ((local.FogStart - global.FogStart) * weight);
        float fogEnd = global.FogEnd + ((local.FogEnd - global.FogEnd) * weight);
        (fogStart, fogEnd) = TerrainLightingMath.NormalizeFogRange(
            fogStart,
            fogEnd,
            global.FogStart,
            global.FogEnd);

        return new TerrainViewerLightingState(
            Vector3.Lerp(global.DirectionalColor, local.DirectionalColor, weight),
            Vector3.Lerp(global.AmbientColor, local.AmbientColor, weight),
            Vector3.Lerp(global.FogColor, local.FogColor, weight),
            fogStart,
            fogEnd);
    }
}
