using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>CPU reference math for the terrain Lambert plus MCSH lighting contract.</summary>
public static class TerrainLightingMath
{
    /// <summary>
    /// Classic LightFloatBand/LIT fog distances use the same 1/36 fixed scale
    /// as the outdoor-light spatial records. Renderer coordinates are world units.
    /// </summary>
    public const float ClientFixedUnitsPerWorldUnit = 36f;

    /// <summary>
    /// Authored fallback until the exact client shadow-darkness coefficient is recovered.
    /// MCSH itself is client-authored data; only this visibility coefficient is approximate.
    /// </summary>
    public const float DefaultAuthoredMcshShadowStrength = 0.60f;

    public static Vector3 Evaluate(
        Vector3 normal,
        Vector3 lightDirection,
        Vector3 directionalColor,
        Vector3 ambientColor,
        float shadowMask,
        float shadowStrength = DefaultAuthoredMcshShadowStrength)
    {
        normal = NormalizeOrUp(normal);
        lightDirection = NormalizeOrUp(lightDirection);

        float lambert = MathF.Max(0f, Vector3.Dot(normal, lightDirection));
        float visibility = 1f - (Math.Clamp(shadowMask, 0f, 1f) * Math.Clamp(shadowStrength, 0f, 1f));
        return ambientColor + (directionalColor * lambert * visibility);
    }

    /// <summary>
    /// Convert a renderer-unit FogEnd/FogStartScalar pair to renderer distances.
    /// The scalar describes how much of the range is fogged, so 0.25 starts
    /// fog at 75 percent of FogEnd rather than at 25 percent.
    /// </summary>
    public static (float FogStart, float FogEnd) ComputeFogRange(
        float fogEnd,
        float fogStartScalar)
    {
        float end = float.IsFinite(fogEnd) && fogEnd > 1f ? fogEnd : 1500f;
        float scalar = float.IsFinite(fogStartScalar)
            ? Math.Clamp(fogStartScalar, 0f, 1f)
            : 0.25f;
        float start = Math.Clamp(end * (1f - scalar), 0f, end - 0.001f);
        return (start, end);
    }

    /// <summary>
    /// Convert a classic client fixed-unit FogEnd/FogStartScalar pair to renderer distances.
    /// </summary>
    public static (float FogStart, float FogEnd) ComputeClientFogRange(
        float fogEndFixedUnits,
        float fogStartScalar)
    {
        float rendererFogEnd = float.IsFinite(fogEndFixedUnits)
            ? fogEndFixedUnits / ClientFixedUnitsPerWorldUnit
            : float.NaN;
        return ComputeFogRange(rendererFogEnd, fogStartScalar);
    }

    private static Vector3 NormalizeOrUp(Vector3 value)
    {
        return value.LengthSquared() > 1e-10f ? Vector3.Normalize(value) : Vector3.UnitZ;
    }
}
