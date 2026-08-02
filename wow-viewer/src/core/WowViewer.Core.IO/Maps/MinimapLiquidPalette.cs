using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>Flat overlay colour and opacity for one liquid class, in sRGB 0..1.</summary>
public readonly record struct MinimapLiquidStyle(float Red, float Green, float Blue, float Opacity);

/// <summary>
/// Named liquid colour palette for synthesized minimaps.
///
/// This is era-scoped on purpose. Water's rendered tint changed across WoW's history, and this
/// project restores a specific era, so a single global constant would silently misrepresent whichever
/// era it was not tuned for. The palette name is recorded in the synthesis manifest so any generated
/// corpus states which water it was rendered with.
/// </summary>
public sealed record MinimapLiquidPalette(
    string Name,
    MinimapLiquidStyle Water,
    MinimapLiquidStyle Ocean,
    MinimapLiquidStyle Magma,
    MinimapLiquidStyle Slime)
{
    /// <summary>Profile identifier recorded in the synthesis manifest.</summary>
    public string RenderProfile => $"viewer_flat_liquid_overlay_{Name}";

    /// <summary>
    /// 0.5.3 pre-alpha era water: a bright cyan-teal, markedly greener than the later slate-blue.
    ///
    /// CALIBRATION STATUS: read off an authored-vs-synthesized comparison tile BY EYE, not measured.
    /// The authored river/ocean reads near sRGB (0.33, 0.72, 0.80) composited, against
    /// <see cref="ViewerFlatV1"/>'s (0.15, 0.35, 0.65) -- the green channel is the big miss, roughly
    /// double. Replace these with measured values off real authored tiles; see <c>--water-color</c>
    /// for dialling without a rebuild.
    ///
    /// OPACITY IS NOT COSMETIC. Authored 0.5.3 water is genuinely translucent -- seabed relief and
    /// shoreline detail read through it -- so water pixels in a real minimap still carry terrain
    /// signal. Rendering water as an opaque slab destroys that signal in every synthesized training
    /// row, which matters to anything that learns terrain or detects liquid from minimap RGB, not
    /// just to how the picture looks.
    /// </summary>
    public static MinimapLiquidPalette PreAlpha053 { get; } = new(
        "prealpha_0_5_3_teal_v2",
        Water: new MinimapLiquidStyle(0.33f, 0.72f, 0.80f, 0.75f),
        Ocean: new MinimapLiquidStyle(0.20f, 0.58f, 0.72f, 0.78f),
        Magma: new MinimapLiquidStyle(0.85f, 0.30f, 0.05f, 0.75f),
        Slime: new MinimapLiquidStyle(0.20f, 0.70f, 0.15f, 0.65f));

    /// <summary>
    /// The original palette, kept verbatim: it mirrors the live viewer's flat liquid pass, so it
    /// stays available for renders that must match the viewer rather than an authored 0.5.3 minimap,
    /// and for reproducing any corpus generated before the era-scoped palette existed.
    /// </summary>
    public static MinimapLiquidPalette ViewerFlatV1 { get; } = new(
        "v1",
        Water: new MinimapLiquidStyle(0.15f, 0.35f, 0.65f, 0.55f),
        Ocean: new MinimapLiquidStyle(0.10f, 0.25f, 0.55f, 0.60f),
        Magma: new MinimapLiquidStyle(0.85f, 0.30f, 0.05f, 0.75f),
        Slime: new MinimapLiquidStyle(0.20f, 0.70f, 0.15f, 0.65f));

    /// <summary>The restoration target era is 0.5.3, so that palette is the default.</summary>
    public static MinimapLiquidPalette Default => PreAlpha053;

    public MinimapLiquidStyle Resolve(AdtLiquidBasicType type) => type switch
    {
        AdtLiquidBasicType.Ocean => Ocean,
        AdtLiquidBasicType.Magma => Magma,
        AdtLiquidBasicType.Slime => Slime,
        _ => Water,
    };

    /// <summary>
    /// Returns this palette with water (and ocean, scaled to keep its relative depth) overridden.
    /// Ocean tracks water so a single <c>--water-color</c> does not leave coastlines mismatched.
    /// </summary>
    public MinimapLiquidPalette WithWaterColor(float red, float green, float blue, float? opacity = null)
    {
        float waterOpacity = opacity ?? Water.Opacity;
        float oceanScale = Water.Green > 1e-4f ? Ocean.Green / Water.Green : 0.8f;
        return this with
        {
            Name = $"{Name}+custom_water",
            Water = new MinimapLiquidStyle(red, green, blue, waterOpacity),
            Ocean = new MinimapLiquidStyle(
                red * oceanScale,
                green * oceanScale,
                blue * oceanScale,
                opacity ?? Ocean.Opacity),
        };
    }

    /// <summary>Resolves a palette by name for CLI selection. Returns null when unknown.</summary>
    public static MinimapLiquidPalette? TryResolve(string? name) => name?.Trim().ToLowerInvariant() switch
    {
        null or "" or "default" or "prealpha" or "prealpha053" or "0.5.3" or "teal" => PreAlpha053,
        "viewer" or "viewerflatv1" or "v1" or "legacy" => ViewerFlatV1,
        _ => null,
    };

    public static string AvailableNames => "prealpha053 (default, 0.5.3 teal) | viewer (legacy viewer-flat v1)";
}
