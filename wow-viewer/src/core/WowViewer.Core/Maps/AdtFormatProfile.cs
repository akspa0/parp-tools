namespace WowViewer.Core.Maps;

public enum TerrainAlphaDecodeMode
{
    LegacySequential,
    LichKingStrict,
    Cataclysm400
}

public enum TerrainLiquidProfile
{
    InlineMclqOnly,
    Mh2oWithMclqFallback,
    Mh2oOnly
}

public sealed class AdtFormatProfile
{
    public required string ProfileId { get; init; }
    public required int McinEntrySize { get; init; }
    public required int MclqLayerStride { get; init; }
    public required int MclqTileFlagsOffset { get; init; }
    public required int MddfRecordSize { get; init; }
    public required int ModfRecordSize { get; init; }
    public required bool UseMhdrOffsetsOnly { get; init; }
    public required bool EnableMh2oFallbackWhenNoMclq { get; init; }
    public required uint BigAlphaFlagsMask { get; init; }
    public required bool PreferTex0ForTextureData { get; init; }
    public required bool PreferObj0ForPlacementData { get; init; }
    public required bool UseMcnkHeaderAlphaSize { get; init; }
    public required bool UseMcnkHeaderShadowSize { get; init; }
    public required TerrainAlphaDecodeMode AlphaDecodeMode { get; init; }
    public required TerrainLiquidProfile LiquidProfile { get; init; }

    public AdtMcalDecodeProfile DecodeProfile => AlphaDecodeMode switch
    {
        TerrainAlphaDecodeMode.LegacySequential => AdtMcalDecodeProfile.LegacySequential,
        TerrainAlphaDecodeMode.LichKingStrict => AdtMcalDecodeProfile.LichKingStrict,
        TerrainAlphaDecodeMode.Cataclysm400 => AdtMcalDecodeProfile.Cataclysm400,
        _ => AdtMcalDecodeProfile.LichKingStrict
    };
}

public static class AdtFormatProfiles
{
    private const int Pre310Major = 3;
    private const int Pre310Minor = 0;
    private const int Pre310Patch = 1;

    public static readonly AdtFormatProfile AdtProfile060070Baseline = new()
    {
        ProfileId = "AdtProfile_060_070_Baseline",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x2D4,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = false,
        EnableMh2oFallbackWhenNoMclq = true,
        BigAlphaFlagsMask = 0,
        PreferTex0ForTextureData = false,
        PreferObj0ForPlacementData = false,
        UseMcnkHeaderAlphaSize = false,
        UseMcnkHeaderShadowSize = false,
        AlphaDecodeMode = TerrainAlphaDecodeMode.LegacySequential,
        LiquidProfile = TerrainLiquidProfile.InlineMclqOnly
    };

    public static readonly AdtFormatProfile AdtProfile33512340 = new()
    {
        ProfileId = "AdtProfile_335_12340",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = true,
        BigAlphaFlagsMask = 0x4u | 0x80u,
        PreferTex0ForTextureData = false,
        PreferObj0ForPlacementData = false,
        UseMcnkHeaderAlphaSize = true,
        UseMcnkHeaderShadowSize = true,
        AlphaDecodeMode = TerrainAlphaDecodeMode.LichKingStrict,
        LiquidProfile = TerrainLiquidProfile.Mh2oWithMclqFallback
    };

    public static readonly AdtFormatProfile AdtProfile3018303 = new()
    {
        ProfileId = "AdtProfile_301_8303",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = true,
        BigAlphaFlagsMask = 0x4u | 0x80u,
        PreferTex0ForTextureData = false,
        PreferObj0ForPlacementData = false,
        UseMcnkHeaderAlphaSize = true,
        UseMcnkHeaderShadowSize = true,
        AlphaDecodeMode = TerrainAlphaDecodeMode.LichKingStrict,
        LiquidProfile = TerrainLiquidProfile.Mh2oWithMclqFallback
    };

    public static readonly AdtFormatProfile AdtProfile0703694 = new()
    {
        ProfileId = "AdtProfile_070_3694",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x2D4,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false,
        BigAlphaFlagsMask = 0,
        PreferTex0ForTextureData = false,
        PreferObj0ForPlacementData = false,
        UseMcnkHeaderAlphaSize = false,
        UseMcnkHeaderShadowSize = false,
        AlphaDecodeMode = TerrainAlphaDecodeMode.LegacySequential,
        LiquidProfile = TerrainLiquidProfile.InlineMclqOnly
    };

    public static readonly AdtFormatProfile AdtProfile0803734 = new()
    {
        ProfileId = "AdtProfile_080_3734",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x2D4,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false,
        BigAlphaFlagsMask = 0,
        PreferTex0ForTextureData = false,
        PreferObj0ForPlacementData = false,
        UseMcnkHeaderAlphaSize = false,
        UseMcnkHeaderShadowSize = false,
        AlphaDecodeMode = TerrainAlphaDecodeMode.LegacySequential,
        LiquidProfile = TerrainLiquidProfile.InlineMclqOnly
    };

    public static readonly AdtFormatProfile AdtProfile0903807 = new()
    {
        ProfileId = "AdtProfile_090_3807",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false,
        BigAlphaFlagsMask = 0,
        PreferTex0ForTextureData = false,
        PreferObj0ForPlacementData = false,
        UseMcnkHeaderAlphaSize = false,
        UseMcnkHeaderShadowSize = false,
        AlphaDecodeMode = TerrainAlphaDecodeMode.LegacySequential,
        LiquidProfile = TerrainLiquidProfile.InlineMclqOnly
    };

    public static readonly AdtFormatProfile AdtProfile40xUnknown = new()
    {
        ProfileId = "AdtProfile_40x_Unknown",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = true,
        BigAlphaFlagsMask = 0x4u | 0x80u,
        PreferTex0ForTextureData = true,
        PreferObj0ForPlacementData = true,
        UseMcnkHeaderAlphaSize = true,
        UseMcnkHeaderShadowSize = true,
        AlphaDecodeMode = TerrainAlphaDecodeMode.Cataclysm400,
        LiquidProfile = TerrainLiquidProfile.Mh2oOnly
    };

    public static AdtFormatProfile Resolve(string? buildVersion)
    {
        if (string.IsNullOrWhiteSpace(buildVersion))
            return AdtProfile060070Baseline;

        if (ClientBuildKey.TryParse(buildVersion, out var key))
        {
            if (key.Major == 3 && key.Minor == 0 && key.Patch == 1)
                return AdtProfile3018303;

            if (key.Major == 0 && key.Minor == 7)
                return AdtProfile0703694;

            if (key.Major == 0 && key.Minor == 8)
                return AdtProfile0803734;

            if (key.Major == 0 && key.Minor == 9)
                return AdtProfile0903807;

            if (key.Major == 3 && key.Minor == 3 && key.Patch == 5)
                return AdtProfile33512340;

            if (key.Major == 4)
                return AdtProfile40xUnknown;

            if (key.Major == 0 && (key.Minor == 6 || key.Minor == 7))
                return AdtProfile060070Baseline;
        }

        return AdtProfile060070Baseline;
    }

    public static bool IsPreRelease301Build(string? buildVersion) =>
        ClientBuildKey.TryParse(buildVersion, out var key)
        && key.Major == Pre310Major
        && key.Minor == Pre310Minor
        && key.Patch == Pre310Patch;
}