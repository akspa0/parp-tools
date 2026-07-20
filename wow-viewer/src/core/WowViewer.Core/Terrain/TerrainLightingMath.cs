using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>CPU reference math for the terrain Lambert plus MCSH lighting contract.</summary>
public static class TerrainLightingMath
{
    public const float DefaultFogStart = 200f;
    public const float DefaultFogEnd = 1500f;
    public const float MinimumFogRangeSpan = 1f;

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
        float shadowStrength = DefaultAuthoredMcshShadowStrength,
        bool toneMapped = false)
    {
        normal = NormalizeOrUp(normal);
        lightDirection = NormalizeOrUp(lightDirection);

        float lambert = MathF.Max(0f, Vector3.Dot(normal, lightDirection));
        return Evaluate(lambert, directionalColor, ambientColor, shadowMask, shadowStrength, toneMapped);
    }

    /// <summary>
    /// Reinhard-with-exposure tone map: <c>x' = x*exposure; mapped = x'/(1+x')</c>. Smoothly
    /// saturates toward 1.0 (never hard-clips) while still lifting dim/ambient-dominated values,
    /// unlike a flat linear multiplier which fixes underexposed shadows only by proportionally
    /// blowing out already-bright highlights (a flat 2.79x multiplier -- the first attempt at this
    /// fix -- closed the same brightness deficit but hard-clipped ~4% of pixels to solid white on
    /// steep, well-lit slopes).
    ///
    /// Calibrated 2026-07-20 by iterative real-render search against the T010b 2.4.3/Expansion01
    /// comparison set (6 tiles: 24,24 / 26,26 / 27,27 / 21,28 / 23,30 / 28,30), measuring mean
    /// synthesized/authored pixel-brightness ratio and clipped-pixel fraction after each attempt:
    /// exposure=4 -> ratio 0.628; exposure=7 -> 0.779; exposure=12 -> 0.902; exposure=20 -> 0.990,
    /// with 0.00% clipped pixels (>=250/255) at every tested exposure. 20.0 is the settled value.
    /// </summary>
    public const float ToneMapExposure = 20.0f;

    /// <summary>
    /// Applies terrain lighting after the caller has evaluated the Lambert term at terrain
    /// vertices and interpolated that scalar across the terrain primitive. This mirrors the
    /// fixed-function terrain path more closely than interpolating or re-normalizing normals
    /// at the output pixel.
    /// </summary>
    /// <param name="toneMapped">
    /// Opt-in only: <see cref="MinimapShadingMatch"/> sweeps this function across every hour of
    /// the day and depends on the raw linear response (headroom against clipping across the whole
    /// sweep, not just at noon), so the default stays unchanged for every existing caller. Only
    /// <see cref="TerrainMinimapLighting.CreateNoonWhiteGlobal"/> opts in.
    /// </param>
    public static Vector3 Evaluate(
        float interpolatedLambert,
        Vector3 directionalColor,
        Vector3 ambientColor,
        float shadowMask,
        float shadowStrength = DefaultAuthoredMcshShadowStrength,
        bool toneMapped = false)
    {
        float lambert = Math.Clamp(float.IsFinite(interpolatedLambert) ? interpolatedLambert : 0f, 0f, 1f);
        float visibility = 1f - (Math.Clamp(shadowMask, 0f, 1f) * Math.Clamp(shadowStrength, 0f, 1f));
        Vector3 raw = ambientColor + (directionalColor * lambert * visibility);
        if (!toneMapped)
            return raw;

        Vector3 exposed = raw * ToneMapExposure;
        return exposed / (Vector3.One + exposed);
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
        float end = float.IsFinite(fogEnd) && fogEnd > MinimumFogRangeSpan ? fogEnd : DefaultFogEnd;
        float scalar = float.IsFinite(fogStartScalar)
            ? Math.Clamp(fogStartScalar, 0f, 1f)
            : 0.25f;
        float start = Math.Clamp(end * (1f - scalar), 0f, end - MinimumFogRangeSpan);
        return (start, end);
    }

    /// <summary>
    /// Produces the only fog-range shape renderers and visibility code may consume.
    /// A malformed or collapsed source range falls back to a visible range instead of
    /// reaching shaders as a zero denominator.
    /// </summary>
    public static (float FogStart, float FogEnd) NormalizeFogRange(
        float fogStart,
        float fogEnd,
        float fallbackStart = DefaultFogStart,
        float fallbackEnd = DefaultFogEnd)
    {
        float safeFallbackEnd = float.IsFinite(fallbackEnd) && fallbackEnd > MinimumFogRangeSpan
            ? fallbackEnd
            : DefaultFogEnd;
        float safeFallbackStart = float.IsFinite(fallbackStart)
            ? Math.Clamp(fallbackStart, 0f, safeFallbackEnd - MinimumFogRangeSpan)
            : Math.Min(DefaultFogStart, safeFallbackEnd - MinimumFogRangeSpan);

        if (!float.IsFinite(fogStart)
            || !float.IsFinite(fogEnd)
            || fogEnd <= MinimumFogRangeSpan
            || fogEnd <= fogStart)
        {
            return (safeFallbackStart, safeFallbackEnd);
        }

        return (Math.Clamp(fogStart, 0f, fogEnd - MinimumFogRangeSpan), fogEnd);
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
