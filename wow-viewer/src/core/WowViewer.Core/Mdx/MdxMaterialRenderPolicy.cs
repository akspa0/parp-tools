namespace WowViewer.Core.Mdx;

/// <summary>
/// Shared safety and selection rules for the legacy MDX material path.
/// The OpenGL shader mirrors these bounded values; this contract keeps the
/// CPU-side decisions deterministic and testable without requiring a GL host.
/// </summary>
public static class MdxMaterialRenderPolicy
{
    public const float MaxEmissiveGain = 1.0f;
    public const float MaxLocalLightComponent = 4.0f;

    public static float ClampFinite(float value, float min, float max)
        => float.IsFinite(value) ? Math.Clamp(value, min, max) : 0.0f;

    public static int SelectUvSet(int coordId, int availableUvSets)
        => coordId == 1 && availableUvSets > 1 ? 1 : 0;

    public static bool UsesSphereEnvironmentMap(uint layerFlags)
        => (layerFlags & 0x2u) != 0;
}
