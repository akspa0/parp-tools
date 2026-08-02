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

    /// <summary>
    /// Default darkening for a fully occluded analytic terrain cast shadow (see
    /// <see cref="TerrainCastShadowMap"/>). Deliberately below
    /// <see cref="DefaultAuthoredMcshShadowStrength"/>: a cast shadow still receives full ambient
    /// sky in this model and should read as shading, not as a hole punched in the terrain.
    ///
    /// Unlike the MCSH coefficient this is a rendering choice, not an approximation of a recovered
    /// client constant, and has not yet been calibrated against authored minimaps. Paired with
    /// <see cref="DefaultSyntheticMinimapAmbient"/>: together they set how deep shadows read, while
    /// <see cref="ResolveLinearLightGain"/> holds overall brightness steady as they change.
    /// </summary>
    public const float DefaultCastShadowStrength = 0.70f;

    /// <summary>
    /// Ambient (sky) term for synthetic minimap export, lower than the 0.25 the shared
    /// <c>CreateWhiteTopEdge</c> diagnostic light uses.
    ///
    /// Ambient is the single biggest control on how deep a shadow can get, because it is the floor
    /// every shadowed pixel lands on: at ambient 0.25 a fully cast-shadowed patch of flat ground is
    /// only ~18% darker than lit ground no matter how high the shadow strength goes. At 0.12 with
    /// <see cref="DefaultCastShadowStrength"/> the same patch reads ~37% darker, which is in the
    /// range authored 0.5.3 minimaps show.
    ///
    /// Not calibrated against authored tiles -- exposed as <c>--ambient</c> on the CLI precisely so
    /// it can be dialled in against real comparisons without a rebuild.
    /// </summary>
    public const float DefaultSyntheticMinimapAmbient = 0.12f;

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
    /// Linear-space light gain for the synthetic-minimap path, replacing the exposure-20 Reinhard
    /// curve as the brightness control.
    ///
    /// WHY THIS REPLACED THE TONE MAP: exposure 20 was fitted against *mean* authored brightness
    /// (ratio 0.990) using a saturating curve. Matching the first moment that way destroys the
    /// second. With ambient 0.25 and directional 1.0 the whole Lambert range 0..1 maps through
    /// <c>20x/(1+20x)</c> to just 0.833..0.962 -- a surface pointing completely away from the sun
    /// came out only 17% darker than a fully lit one, so the terrain hillshade was compressed into
    /// 12.8% of albedo and read as "no shadows at all" against authored minimaps.
    ///
    /// A plain linear gain applied in LINEAR light space, with the sRGB encode on output doing the
    /// perceptual curve, holds the same mid-tone brightness while restoring the full shading range.
    /// On a mid-grey (sRGB 0.5) texel: old toned path 0.417..0.481 (range 0.064); this path
    /// 0.274..0.591 (range 0.317), roughly 5x the shading contrast at the same mid-tone.
    ///
    /// DO NOT hardcode this gain -- derive it with <see cref="ResolveLinearLightGain"/>. A fixed
    /// constant has to pick one lighting condition to be correct at, and picking the wrong one is
    /// exactly how the first attempt at this fix went wrong: 1.166 was anchored at lambert 0.5,
    /// but flat terrain under the noon sun sits at lambert 0.894, so ordinary ground rendered 19%
    /// too bright and the result read as washed out. Deriving the gain also decouples brightness
    /// from contrast -- ambient and cast-shadow strength can then be tuned freely without dragging
    /// the whole image lighter or darker.
    /// </summary>
    /// <remarks>
    /// The reference albedo is fixed at mid-grey. The sRGB encode is non-linear, so strictly the
    /// gain that preserves a given output level depends on albedo, but the dependence is tiny over
    /// the useful range (under 0.3% at sRGB 0.25 and under 1.5% at sRGB 0.75) -- far below the
    /// calibration uncertainty this whole path already carries.
    /// </remarks>
    public const float LinearGainReferenceAlbedo = 0.5f;

    /// <summary>
    /// The lambert term of flat ground under the authored solar direction, which is the modal
    /// condition across a real tile and therefore the right anchor for brightness. The sun holds a
    /// fixed north-west bearing and only cycles elevation, so at solar noon this is the Z component
    /// of <c>normalize(0.3536, 0.3536, 1.0)</c>.
    /// </summary>
    public const float FlatGroundNoonLambert = 0.8944272f;

    /// <summary>
    /// The ambient term the legacy exposure-<see cref="ToneMapExposure"/> path ran with. The
    /// brightness target is whatever THAT path produced, so it must be evaluated at its own ambient
    /// -- not at whatever ambient the caller has since dialled in, or the target moves every time
    /// the contrast knobs are touched, which is the whole thing this derivation prevents.
    /// </summary>
    public const float LegacyCalibratedAmbient = 0.25f;

    /// <summary>
    /// Returns the linear-space gain that makes a surface at <paramref name="anchorLambert"/> render
    /// at the same brightness the legacy exposure-<see cref="ToneMapExposure"/> path produced -- the
    /// response that was actually calibrated against authored minimaps -- while leaving the shading
    /// range uncompressed everywhere else.
    ///
    /// Because the target is fixed and only the divisor moves with <paramref name="ambient"/>,
    /// lowering ambient to deepen shadows automatically raises the gain to compensate. Brightness
    /// and contrast become independent controls.
    /// </summary>
    /// <param name="ambient">Ambient (sky) term of the lighting profile, single channel.</param>
    /// <param name="anchorLambert">
    /// Lighting condition to preserve brightness at. Defaults to <see cref="FlatGroundNoonLambert"/>
    /// because most of a terrain tile is near-flat; anchoring anywhere else biases the whole image.
    /// </param>
    public static float ResolveLinearLightGain(float ambient, float anchorLambert = FlatGroundNoonLambert)
    {
        float safeAmbient = float.IsFinite(ambient) ? Math.Clamp(ambient, 0f, 4f) : LegacyCalibratedAmbient;
        float safeLambert = float.IsFinite(anchorLambert) ? Math.Clamp(anchorLambert, 0f, 1f) : FlatGroundNoonLambert;
        float raw = safeAmbient + safeLambert;
        if (raw <= 1e-4f)
            return 1f;

        // Solve enc(dec(albedo) * gain * raw) == legacyOutput for gain, where legacyOutput is the
        // calibrated sRGB-space response at the anchor under the legacy ambient.
        float legacyRaw = LegacyCalibratedAmbient + safeLambert;
        float exposed = legacyRaw * ToneMapExposure;
        float legacyOutput = LinearGainReferenceAlbedo * (exposed / (1f + exposed));
        float gain = SrgbToLinear(legacyOutput) / (SrgbToLinear(LinearGainReferenceAlbedo) * raw);
        return float.IsFinite(gain) && gain > 0f ? gain : 1f;
    }

    /// <summary>
    /// Decodes an sRGB-encoded channel (0..1) to linear light. Terrain BLP texels are authored
    /// sRGB; multiplying them by a linear light factor without decoding first darkens shadowed
    /// areas incorrectly and was part of the brightness deficit the tone map was papering over.
    /// </summary>
    public static float SrgbToLinear(float srgb)
    {
        if (!float.IsFinite(srgb))
            return 0f;

        srgb = Math.Clamp(srgb, 0f, 1f);
        return srgb <= 0.04045f
            ? srgb / 12.92f
            : MathF.Pow((srgb + 0.055f) / 1.055f, 2.4f);
    }

    /// <summary>Encodes a linear light value (0..1) back to sRGB for 8-bit output.</summary>
    public static float LinearToSrgb(float linear)
    {
        if (!float.IsFinite(linear))
            return 0f;

        linear = Math.Clamp(linear, 0f, 1f);
        return linear <= 0.0031308f
            ? linear * 12.92f
            : (1.055f * MathF.Pow(linear, 1f / 2.4f)) - 0.055f;
    }

    /// <summary>Per-channel <see cref="SrgbToLinear"/>.</summary>
    public static Vector3 SrgbToLinear(Vector3 srgb) =>
        new(SrgbToLinear(srgb.X), SrgbToLinear(srgb.Y), SrgbToLinear(srgb.Z));

    /// <summary>Per-channel <see cref="LinearToSrgb"/>.</summary>
    public static Vector3 LinearToSrgb(Vector3 linear) =>
        new(LinearToSrgb(linear.X), LinearToSrgb(linear.Y), LinearToSrgb(linear.Z));

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
    ///
    /// SUPERSEDED for the synthetic-minimap path by <see cref="SyntheticMinimapLinearLightGain"/>:
    /// this calibration only ever checked mean brightness and clipped-pixel fraction, never shading
    /// contrast, and at exposure 20 the curve flattens the hillshade to 12.8% of albedo. Retained
    /// because it is still the documented behaviour of every caller that opts into
    /// <c>toneMapped: true</c>.
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
        bool toneMapped = false,
        float toneMapExposure = ToneMapExposure)
    {
        float lambert = Math.Clamp(float.IsFinite(interpolatedLambert) ? interpolatedLambert : 0f, 0f, 1f);
        float visibility = 1f - (Math.Clamp(shadowMask, 0f, 1f) * Math.Clamp(shadowStrength, 0f, 1f));
        Vector3 raw = ambientColor + (directionalColor * lambert * visibility);
        if (!toneMapped)
            return raw;

        float exposure = float.IsFinite(toneMapExposure) && toneMapExposure > 0f ? toneMapExposure : ToneMapExposure;
        Vector3 exposed = raw * exposure;
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
