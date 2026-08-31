namespace WowViewer.Core.Maps;

/// <summary>
/// The input family used by the map conversion workflow. This is deliberately separate from the
/// output format: a split ADT family may be normalized to an older monolithic format, but it must
/// never be mistaken for an already-compatible LK document.
/// </summary>
public enum MapConversionSourceFormat
{
    AlphaWdt053,
    SplitAdtFamily,
}

/// <summary>
/// Explicit serialization targets exposed by the map conversion workflow.
/// </summary>
public enum MapConversionTargetFormat
{
    AlphaWdt053,
    LkAdtV18,
    MopSplitAdt,
}

/// <summary>
/// Result of validating a conversion route before any writer is invoked.
/// </summary>
public readonly record struct MapConversionValidationResult(
    bool IsSupported,
    bool IsLossy,
    string? Error,
    IReadOnlyList<string> Warnings)
{
    public bool HasWarnings => Warnings.Count != 0;
}

/// <summary>
/// Names, writer availability, and route validation for cross-era map serialization.
/// Keep this contract in Core so UI and CLI cannot silently disagree about the selected target.
/// </summary>
public static class MapConversionFormats
{
    public static IReadOnlyList<MapConversionTargetFormat> AllTargets { get; } =
    [
        MapConversionTargetFormat.AlphaWdt053,
        MapConversionTargetFormat.LkAdtV18,
        MapConversionTargetFormat.MopSplitAdt,
    ];

    public static string GetDisplayName(MapConversionTargetFormat format) => format switch
    {
        MapConversionTargetFormat.AlphaWdt053 => "Alpha 0.5.3 monolithic WDT",
        MapConversionTargetFormat.LkAdtV18 => "WotLK/LK v18 monolithic ADT",
        MapConversionTargetFormat.MopSplitAdt => "Cataclysm/MoP split ADT family",
        _ => format.ToString(),
    };

    public static string GetCommandValue(MapConversionTargetFormat format) => format switch
    {
        MapConversionTargetFormat.AlphaWdt053 => "alpha-wdt-0.5.3",
        MapConversionTargetFormat.LkAdtV18 => "lk-adt-v18",
        MapConversionTargetFormat.MopSplitAdt => "mop-split-adt",
        _ => throw new ArgumentOutOfRangeException(nameof(format), format, "Unknown map conversion target format."),
    };

    public static bool TryParseTarget(string? value, out MapConversionTargetFormat format)
    {
        string normalized = (value ?? string.Empty).Trim().ToLowerInvariant();
        switch (normalized)
        {
            case "alpha":
            case "alpha-wdt":
            case "alpha-wdt-053":
            case "alpha-wdt-0.5.3":
            case "alpha053":
                format = MapConversionTargetFormat.AlphaWdt053;
                return true;

            case "lk":
            case "lk-adt":
            case "lk-v18":
            case "lk-adt-v18":
            case "wotlk":
                format = MapConversionTargetFormat.LkAdtV18;
                return true;

            case "mop":
            case "mop-split":
            case "mop-split-adt":
            case "cataclysm-mop-split":
                format = MapConversionTargetFormat.MopSplitAdt;
                return true;

            default:
                format = default;
                return false;
        }
    }

    public static bool HasWriter(MapConversionTargetFormat format) => format switch
    {
        MapConversionTargetFormat.AlphaWdt053 => true,
        MapConversionTargetFormat.LkAdtV18 => true,
        MapConversionTargetFormat.MopSplitAdt => false,
        _ => false,
    };

    public static string GetUnavailableReason(MapConversionTargetFormat format) => format switch
    {
        MapConversionTargetFormat.MopSplitAdt =>
            "The native split ADT writer is not implemented yet; no split input may be routed through LkAdtWriter.",
        _ => $"No writer is registered for {GetDisplayName(format)}.",
    };

    public static MapConversionValidationResult Validate(
        MapConversionSourceFormat source,
        MapConversionTargetFormat target)
    {
        if (!HasWriter(target))
        {
            return new MapConversionValidationResult(
                IsSupported: false,
                IsLossy: false,
                Error: GetUnavailableReason(target),
                Warnings: []);
        }

        if (source == MapConversionSourceFormat.AlphaWdt053
            && target == MapConversionTargetFormat.AlphaWdt053)
        {
            return new MapConversionValidationResult(
                IsSupported: true,
                IsLossy: false,
                Error: null,
                Warnings: []);
        }

        if (source == MapConversionSourceFormat.AlphaWdt053
            && target == MapConversionTargetFormat.LkAdtV18)
        {
            return new MapConversionValidationResult(
                IsSupported: true,
                IsLossy: true,
                Error: null,
                Warnings:
                [
                    "Alpha 0.5.3 data is being normalized into the LK v18 monolithic ADT contract; Alpha-only fields may be reduced."
                ]);
        }

        if (source == MapConversionSourceFormat.SplitAdtFamily
            && target == MapConversionTargetFormat.AlphaWdt053)
        {
            return new MapConversionValidationResult(
                IsSupported: true,
                IsLossy: true,
                Error: null,
                Warnings:
                [
                    "Split ADT data is being flattened into an Alpha 0.5.3 monolithic WDT; companion bands and modern-only state cannot survive unchanged."
                ]);
        }

        if (source == MapConversionSourceFormat.SplitAdtFamily
            && target == MapConversionTargetFormat.LkAdtV18)
        {
            return new MapConversionValidationResult(
                IsSupported: true,
                IsLossy: true,
                Error: null,
                Warnings:
                [
                    "Split ADT data is being down-converted to one LK v18 monolithic ADT per tile; companion bands and modern-only state are not represented by LkAdtWriter."
                ]);
        }

        return new MapConversionValidationResult(
            IsSupported: false,
            IsLossy: false,
            Error: $"The route from {source} to {GetDisplayName(target)} is not registered.",
            Warnings: []);
    }
}
