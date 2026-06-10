using WoWFormatLib.Structs.WMO;

namespace ParpToolbox.Formats.WMO;

/// <summary>
/// Utility helpers for determining whether a WMO group should be treated as visible/renderable.
/// Logic is heuristic and may be refined as we encounter more files.
/// </summary>
internal static class WmoVisibility
{
    /// <summary>
    /// Determines if the provided group flags correspond to geometry that should be rendered
    /// in normal visual exports (i.e. not collision shells).
    /// </summary>
    public static bool IsRenderable(uint rawFlags)
    {
        var flags = (MOGPFlags)rawFlags;

        // Flags that typically mark invisible / collision-only / LOD proxy groups
        const MOGPFlags InvisibleMask =
            MOGPFlags.mogp_unreachable |  // 0x80 - collision hulls
            MOGPFlags.mogp_lod;          // 0x400 - low-detail proxy

        // Render if none of the invisible bits are set.
        return (flags & InvisibleMask) == 0;
    }
}
