using System.Numerics;

namespace WoWViewer.Rendering;

/// <summary>
/// Configurable styling options for the wireframe overlay across WMO, M2, and MDX renderers.
/// </summary>
public static class WireframeOverlaySettings
{
    /// <summary>Default wireframe line color (RGB).</summary>
    public static Vector3 DefaultColor { get; set; } = new(1.0f, 0.85f, 0.3f); // Warm gold

    /// <summary>Wireframe line raster width in pixels.</summary>
    public static float LineWidth { get; set; } = 1.0f;

    /// <summary>Base opacity multiplier (0.1 to 1.0) so the overlay is not overpowering.</summary>
    public static float BaseIntensity { get; set; } = 0.75f;

    /// <summary>Whether the wireframe opacity dynamically scales with the object's opacity slider.</summary>
    public static bool FollowModelOpacity { get; set; } = true;
}
