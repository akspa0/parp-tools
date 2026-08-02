using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.IO.Maps;

/// <summary>How a build's sun bearing behaves over the day.</summary>
public enum SolarAzimuthModel
{
    /// <summary>Bearing is constant all day; only elevation cycles.</summary>
    Fixed,

    /// <summary>Bearing sweeps east-to-west across the day, passing the noon bearing at solar noon.</summary>
    EastToWestSweep,
}

/// <summary>
/// How confident we are in an era profile's solar model. Recorded in the synthesis manifest so a
/// generated corpus never implies more certainty than the evidence supports.
/// </summary>
public enum SolarModelProvenance
{
    /// <summary>Read out of the running client with a debugger.</summary>
    TracedFromClient,

    /// <summary>Carried over from a different build because this one has not been traced.</summary>
    AssumedFromOtherBuild,

    /// <summary>Inferred by scoring candidates against authored minimaps.</summary>
    MeasuredFromAuthoredMinimaps,
}

/// <summary>
/// Era-scoped minimap generation profile.
///
/// Blizzard changed how minimaps were generated as the client evolved: 0.5.3 is Alpha, 0.6.0 is
/// Beta 1 roughly five months later, and 1.0.0 differs again. Treating any one build's behaviour as
/// universal silently misrepresents the others, so every era-sensitive parameter -- solar model,
/// water palette, shadow contrast -- lives here behind a build lookup rather than as a global
/// default scattered across the renderer.
///
/// The profile name is recorded in the synthesis manifest, so a corpus always states which era's
/// rules produced it.
/// </summary>
public sealed record MinimapEraProfile(
    string Name,
    string EraLabel,
    SolarAzimuthModel AzimuthModel,
    float NoonAzimuthDegrees,
    SolarModelProvenance AzimuthProvenance,
    MinimapLiquidPalette Liquid,
    Vector3 AmbientColor,
    float CastShadowStrength,
    float CastShadowSoftness)
{
    /// <summary>
    /// 0.5.3.3368 -- Alpha. The current restoration target.
    ///
    /// The solar azimuth here is <b>assumed, not traced</b>: it is carried over from the 1.0.0
    /// debugger trace because nobody has traced 0.5.3. The user's expectation is that this era's sun
    /// travels east-to-west, which would make <see cref="SolarAzimuthModel.Fixed"/> wrong for Alpha.
    /// Resolve it by measuring against authored 0.5.3 minimaps, then set
    /// <see cref="SolarModelProvenance.MeasuredFromAuthoredMinimaps"/> here.
    /// </summary>
    public static MinimapEraProfile Alpha053 { get; } = new(
        "alpha_0_5_3",
        "Alpha (0.5.3)",
        AzimuthModel: SolarAzimuthModel.Fixed,
        NoonAzimuthDegrees: TerrainSolarDirection.TracedSourceAzimuthDegrees,
        AzimuthProvenance: SolarModelProvenance.AssumedFromOtherBuild,
        Liquid: MinimapLiquidPalette.PreAlpha053,
        AmbientColor: new Vector3(TerrainLightingMath.DefaultSyntheticMinimapAmbient),
        CastShadowStrength: TerrainLightingMath.DefaultCastShadowStrength,
        CastShadowSoftness: TerrainCastShadowMap.DefaultSoftnessWorldUnits);

    /// <summary>
    /// 0.6.0 -- Beta 1. Roughly five months after Alpha, with known differences in how minimaps were
    /// produced. Nothing era-specific has been measured for this build yet: every value below is
    /// inherited from Alpha and must not be read as a Beta-1 finding.
    /// </summary>
    public static MinimapEraProfile Beta060 { get; } = Alpha053 with
    {
        Name = "beta1_0_6_0",
        EraLabel = "Beta 1 (0.6.0)",
    };

    /// <summary>
    /// 1.0.0 and later -- release-era generation. This is the only era whose solar model is actually
    /// traced: <c>SetDirection</c> at 0x006bca40 holds theta = 225 degrees across all four sampled
    /// <c>thetaTable</c> entries, so the source bearing is a constant north-west.
    /// </summary>
    public static MinimapEraProfile Release100 { get; } = Alpha053 with
    {
        Name = "release_1_0_0",
        EraLabel = "Release (1.0.0+)",
        AzimuthProvenance = SolarModelProvenance.TracedFromClient,
        Liquid = MinimapLiquidPalette.ViewerFlatV1,
    };

    /// <summary>Restoration target era, used when a build cannot be resolved.</summary>
    public static MinimapEraProfile Default => Alpha053;

    /// <summary>True when this profile's solar model rests on evidence from a different build.</summary>
    public bool HasUnverifiedSolarModel => AzimuthProvenance == SolarModelProvenance.AssumedFromOtherBuild;

    /// <summary>
    /// Resolves the era profile for a build string. <paramref name="exactEraMatch"/> reports whether
    /// the build was actually recognised; callers must record it rather than let an unrecognised
    /// build quietly inherit Alpha's rules.
    /// </summary>
    public static MinimapEraProfile ResolveForBuild(string? buildVersion, out bool exactEraMatch)
    {
        exactEraMatch = true;
        string build = (buildVersion ?? string.Empty).Trim().Replace('_', '.');

        if (build.StartsWith("0.5.", StringComparison.Ordinal))
            return Alpha053;
        if (build.StartsWith("0.6.", StringComparison.Ordinal) || build.StartsWith("0.7.", StringComparison.Ordinal))
            return Beta060;
        if (build.StartsWith("0.8.", StringComparison.Ordinal)
            || build.StartsWith("0.9.", StringComparison.Ordinal)
            || build.StartsWith("1.", StringComparison.Ordinal)
            || build.StartsWith("2.", StringComparison.Ordinal)
            || build.StartsWith("3.", StringComparison.Ordinal))
        {
            return Release100;
        }

        exactEraMatch = false;
        return Default;
    }

    /// <summary>Resolves an era by explicit name for CLI selection. Returns null when unknown.</summary>
    public static MinimapEraProfile? TryResolveByName(string? name) => name?.Trim().ToLowerInvariant() switch
    {
        null or "" => null,
        "alpha" or "alpha053" or "0.5.3" or "0_5_3" => Alpha053,
        "beta" or "beta1" or "beta060" or "0.6.0" or "0_6_0" => Beta060,
        "release" or "release100" or "1.0.0" or "1_0_0" => Release100,
        _ => null,
    };

    public static string AvailableNames => "alpha (0.5.3) | beta1 (0.6.0) | release (1.0.0+)";

    /// <summary>Light-source bearing at a given time of day under this era's azimuth model.</summary>
    public float ResolveAzimuthDegrees(float gameTime) => AzimuthModel switch
    {
        SolarAzimuthModel.EastToWestSweep =>
            TerrainSolarDirection.EvaluateSweepAzimuthDegrees(gameTime, NoonAzimuthDegrees),
        _ => NoonAzimuthDegrees,
    };

    /// <summary>Light direction at a given time of day under this era's azimuth model.</summary>
    public Vector3 ResolveLightDirection(float gameTime) =>
        TerrainSolarDirection.Evaluate(gameTime, ResolveAzimuthDegrees(gameTime));

    public string RenderProfile =>
        $"minimap_era_{Name}+azimuth_{AzimuthModel.ToString().ToLowerInvariant()}_{NoonAzimuthDegrees:0}deg_{AzimuthProvenance.ToString().ToLowerInvariant()}";
}
