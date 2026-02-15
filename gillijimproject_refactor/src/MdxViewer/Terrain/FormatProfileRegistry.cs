namespace MdxViewer.Terrain;

public sealed class AdtProfile
{
    public required string ProfileId { get; init; }
    public required int McinEntrySize { get; init; }
    public required int MclqLayerStride { get; init; }
    public required int MclqTileFlagsOffset { get; init; }
    public required int MddfRecordSize { get; init; }
    public required int ModfRecordSize { get; init; }
    public required bool UseMhdrOffsetsOnly { get; init; }
    public required bool EnableMh2oFallbackWhenNoMclq { get; init; }
}

public sealed class WmoProfile
{
    public required string ProfileId { get; init; }
    public required bool StrictGroupChunkOrder { get; init; }
    public required bool EnableMliqGroupLiquids { get; init; }
    public required bool EnablePortalOptionalBlocks { get; init; }
}

public sealed class MdxProfile
{
    public required string ProfileId { get; init; }
    public required bool RequiresMdlxMagic { get; init; }
    public required int TextureRecordSize { get; init; }
    public required bool TextureSectionSizeStrict { get; init; }
    public required bool GeosetHardFailIfMissing { get; init; }
}

public enum ModelRootMagic
{
    MD20
}

public sealed class M2Profile
{
    public required string ProfileId { get; init; }
    public required ModelRootMagic RequiredRootMagic { get; init; }
    public required int MinSupportedVersion { get; init; }
    public required int MaxSupportedVersion { get; init; }
    public required bool UseTypedOffsetCountTable { get; init; }
    public required bool StrictSpanValidation { get; init; }
    public required int VersionSplitThreshold { get; init; }
    public required int SkinLikeAStride { get; init; }
    public required int SkinLikeBStride { get; init; }
    public required int EffectLikeAStride { get; init; }
    public required int EffectLikeBStride { get; init; }
}

public static class FormatProfileRegistry
{
    public static readonly AdtProfile AdtProfile060070Baseline = new()
    {
        ProfileId = "AdtProfile_060_070_Baseline",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x2D4,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = false,
        EnableMh2oFallbackWhenNoMclq = true
    };

    public static readonly AdtProfile AdtProfile0913810 = new()
    {
        ProfileId = "AdtProfile_091_3810",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly AdtProfile AdtProfile0103892Provisional = new()
    {
        ProfileId = "AdtProfile_010_3892_Provisional",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly AdtProfile AdtProfile33512340 = new()
    {
        ProfileId = "AdtProfile_335_12340",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly AdtProfile AdtProfile3018303 = new()
    {
        ProfileId = "AdtProfile_301_8303",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly AdtProfile AdtProfile0703694 = new()
    {
        ProfileId = "AdtProfile_070_3694",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x2D4,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly AdtProfile AdtProfile0803734 = new()
    {
        ProfileId = "AdtProfile_080_3734",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x2D4,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly AdtProfile AdtProfile0903807 = new()
    {
        ProfileId = "AdtProfile_090_3807",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly WmoProfile WmoProfile0913810 = new()
    {
        ProfileId = "WmoProfile_091_3810",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile0103892Provisional = new()
    {
        ProfileId = "WmoProfile_010_3892_Provisional",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile3018303 = new()
    {
        ProfileId = "WmoProfile_301_8303",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile33512340 = new()
    {
        ProfileId = "WmoProfile_335_12340",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile0703694 = new()
    {
        ProfileId = "WmoProfile_070_3694",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile0803734 = new()
    {
        ProfileId = "WmoProfile_080_3734",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile0903807 = new()
    {
        ProfileId = "WmoProfile_090_3807",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile090xUnknown = new()
    {
        ProfileId = "WmoProfile_090x_Unknown",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile060070Baseline = new()
    {
        ProfileId = "WmoProfile_060_070_Baseline",
        StrictGroupChunkOrder = false,
        EnableMliqGroupLiquids = false,
        EnablePortalOptionalBlocks = false
    };

    public static readonly MdxProfile MdxProfile0913810 = new()
    {
        ProfileId = "MdxProfile_091_3810",
        RequiresMdlxMagic = true,
        TextureRecordSize = 0x10C,
        TextureSectionSizeStrict = true,
        GeosetHardFailIfMissing = false
    };

    public static readonly MdxProfile MdxProfile0103892Provisional = new()
    {
        ProfileId = "MdxProfile_010_3892_Provisional",
        RequiresMdlxMagic = true,
        TextureRecordSize = 0x10C,
        TextureSectionSizeStrict = true,
        GeosetHardFailIfMissing = false
    };

    public static readonly MdxProfile MdxProfile0703694 = new()
    {
        ProfileId = "MdxProfile_070_3694",
        RequiresMdlxMagic = true,
        TextureRecordSize = 0x10C,
        TextureSectionSizeStrict = true,
        GeosetHardFailIfMissing = false
    };

    public static readonly MdxProfile MdxProfile0803734Provisional = new()
    {
        ProfileId = "MdxProfile_080_3734_Provisional",
        RequiresMdlxMagic = true,
        TextureRecordSize = 0x10C,
        TextureSectionSizeStrict = true,
        GeosetHardFailIfMissing = false
    };

    public static readonly MdxProfile MdxProfile0903807Provisional = new()
    {
        ProfileId = "MdxProfile_090_3807_Provisional",
        RequiresMdlxMagic = true,
        TextureRecordSize = 0x10C,
        TextureSectionSizeStrict = true,
        GeosetHardFailIfMissing = false
    };

    public static readonly MdxProfile MdxProfile090xUnknown = new()
    {
        ProfileId = "MdxProfile_090x_Unknown",
        RequiresMdlxMagic = true,
        TextureRecordSize = 0x10C,
        TextureSectionSizeStrict = true,
        GeosetHardFailIfMissing = false
    };

    public static readonly MdxProfile MdxProfile060070Baseline = new()
    {
        ProfileId = "MdxProfile_060_070_Baseline",
        RequiresMdlxMagic = true,
        TextureRecordSize = 0x10C,
        TextureSectionSizeStrict = false,
        GeosetHardFailIfMissing = false
    };

    public static readonly M2Profile M2Profile3018303 = new()
    {
        ProfileId = "M2Profile_301_8303",
        RequiredRootMagic = ModelRootMagic.MD20,
        MinSupportedVersion = 0x104,
        MaxSupportedVersion = 0x108,
        UseTypedOffsetCountTable = true,
        StrictSpanValidation = true,
        VersionSplitThreshold = 0x108,
        SkinLikeAStride = 0x70,
        SkinLikeBStride = 0x2C,
        EffectLikeAStride = 0xD4,
        EffectLikeBStride = 0x7C
    };

    public static readonly M2Profile M2Profile33512340 = new()
    {
        ProfileId = "M2Profile_335_12340",
        RequiredRootMagic = ModelRootMagic.MD20,
        MinSupportedVersion = 0x108,
        MaxSupportedVersion = 0x108,
        UseTypedOffsetCountTable = true,
        StrictSpanValidation = true,
        VersionSplitThreshold = 0x108,
        SkinLikeAStride = 0x70,
        SkinLikeBStride = 0x2C,
        EffectLikeAStride = 0xD4,
        EffectLikeBStride = 0x7C
    };

    public static readonly M2Profile M2Profile30xUnknown = new()
    {
        ProfileId = "M2Profile_30x_Unknown",
        RequiredRootMagic = ModelRootMagic.MD20,
        MinSupportedVersion = 0x104,
        MaxSupportedVersion = 0x108,
        UseTypedOffsetCountTable = true,
        StrictSpanValidation = true,
        VersionSplitThreshold = 0x108,
        SkinLikeAStride = 0x70,
        SkinLikeBStride = 0x2C,
        EffectLikeAStride = 0xD4,
        EffectLikeBStride = 0x7C
    };

    public static readonly AdtProfile AdtProfile090xUnknown = new()
    {
        ProfileId = "AdtProfile_090x_Unknown",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly AdtProfile AdtProfile080xUnknown = new()
    {
        ProfileId = "AdtProfile_080x_Unknown",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x2D4,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly AdtProfile AdtProfile30xUnknown = new()
    {
        ProfileId = "AdtProfile_30x_Unknown",
        McinEntrySize = 0x10,
        MclqLayerStride = 0x324,
        MclqTileFlagsOffset = 0x290,
        MddfRecordSize = 0x24,
        ModfRecordSize = 0x40,
        UseMhdrOffsetsOnly = true,
        EnableMh2oFallbackWhenNoMclq = false
    };

    public static readonly WmoProfile WmoProfile080xUnknown = new()
    {
        ProfileId = "WmoProfile_080x_Unknown",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly WmoProfile WmoProfile30xUnknown = new()
    {
        ProfileId = "WmoProfile_30x_Unknown",
        StrictGroupChunkOrder = true,
        EnableMliqGroupLiquids = true,
        EnablePortalOptionalBlocks = true
    };

    public static readonly MdxProfile MdxProfile080xUnknown = new()
    {
        ProfileId = "MdxProfile_080x_Unknown",
        RequiresMdlxMagic = true,
        TextureRecordSize = 0x10C,
        TextureSectionSizeStrict = true,
        GeosetHardFailIfMissing = false
    };

    public static AdtProfile ResolveAdtProfile(string? buildVersion)
    {
        if (string.Equals(buildVersion, "0.10.3892", StringComparison.OrdinalIgnoreCase))
            return AdtProfile0103892Provisional;

        if (string.Equals(buildVersion, "0.9.1.3810", StringComparison.OrdinalIgnoreCase))
            return AdtProfile0913810;

        if (string.Equals(buildVersion, "3.3.5.12340", StringComparison.OrdinalIgnoreCase))
            return AdtProfile33512340;

        if (string.Equals(buildVersion, "3.0.1.8303", StringComparison.OrdinalIgnoreCase))
            return AdtProfile3018303;

        if (string.Equals(buildVersion, "0.7.0.3694", StringComparison.OrdinalIgnoreCase))
            return AdtProfile0703694;

        if (string.Equals(buildVersion, "0.8.0.3734", StringComparison.OrdinalIgnoreCase))
            return AdtProfile0803734;

        if (string.Equals(buildVersion, "0.9.0.3807", StringComparison.OrdinalIgnoreCase))
            return AdtProfile0903807;

        if (TryParseBuild(buildVersion, out int major, out int minor, out _, out _))
        {
            if (major == 0 && minor == 9)
                return AdtProfile090xUnknown;

            if (major == 3 && minor == 0)
                return AdtProfile30xUnknown;

            if (major == 0 && minor == 8)
                return AdtProfile080xUnknown;

            if (major == 0 && (minor == 6 || minor == 7))
                return AdtProfile060070Baseline;
        }

        return AdtProfile060070Baseline;
    }

    public static WmoProfile ResolveWmoProfile(string? buildVersion)
    {
        if (string.Equals(buildVersion, "0.10.3892", StringComparison.OrdinalIgnoreCase))
            return WmoProfile0103892Provisional;

        if (string.Equals(buildVersion, "0.9.1.3810", StringComparison.OrdinalIgnoreCase))
            return WmoProfile0913810;

        if (string.Equals(buildVersion, "3.3.5.12340", StringComparison.OrdinalIgnoreCase))
            return WmoProfile33512340;

        if (string.Equals(buildVersion, "3.0.1.8303", StringComparison.OrdinalIgnoreCase))
            return WmoProfile3018303;

        if (string.Equals(buildVersion, "0.7.0.3694", StringComparison.OrdinalIgnoreCase))
            return WmoProfile0703694;

        if (string.Equals(buildVersion, "0.8.0.3734", StringComparison.OrdinalIgnoreCase))
            return WmoProfile0803734;

        if (string.Equals(buildVersion, "0.9.0.3807", StringComparison.OrdinalIgnoreCase))
            return WmoProfile0903807;

        if (TryParseBuild(buildVersion, out int major, out int minor, out _, out _))
        {
            if (major == 0 && minor == 9)
                return WmoProfile090xUnknown;

            if (major == 3 && minor == 0)
                return WmoProfile30xUnknown;

            if (major == 0 && minor == 8)
                return WmoProfile080xUnknown;

            if (major == 0 && (minor == 6 || minor == 7))
                return WmoProfile060070Baseline;
        }

        return WmoProfile060070Baseline;
    }

    public static MdxProfile ResolveMdxProfile(string? buildVersion)
    {
        if (string.Equals(buildVersion, "0.10.3892", StringComparison.OrdinalIgnoreCase))
            return MdxProfile0103892Provisional;

        if (string.Equals(buildVersion, "0.9.1.3810", StringComparison.OrdinalIgnoreCase))
            return MdxProfile0913810;

        if (string.Equals(buildVersion, "0.7.0.3694", StringComparison.OrdinalIgnoreCase))
            return MdxProfile0703694;

        if (string.Equals(buildVersion, "0.8.0.3734", StringComparison.OrdinalIgnoreCase))
            return MdxProfile0803734Provisional;

        if (string.Equals(buildVersion, "0.9.0.3807", StringComparison.OrdinalIgnoreCase))
            return MdxProfile0903807Provisional;

        if (TryParseBuild(buildVersion, out int major, out int minor, out _, out _))
        {
            if (major == 0 && minor == 9)
                return MdxProfile090xUnknown;

            if (major == 0 && minor == 8)
                return MdxProfile080xUnknown;

            if (major == 0 && (minor == 6 || minor == 7))
                return MdxProfile060070Baseline;
        }

        return MdxProfile060070Baseline;
    }

    public static M2Profile? ResolveModelProfile(string? buildVersion)
    {
        if (string.Equals(buildVersion, "3.3.5.12340", StringComparison.OrdinalIgnoreCase))
            return M2Profile33512340;

        if (string.Equals(buildVersion, "3.0.1.8303", StringComparison.OrdinalIgnoreCase))
            return M2Profile3018303;

        if (TryParseBuild(buildVersion, out int major, out int minor, out _, out _))
        {
            if (major == 3 && minor == 0)
                return M2Profile30xUnknown;
        }

        return null;
    }

    private static bool TryParseBuild(string? buildVersion, out int major, out int minor, out int patch, out int build)
    {
        major = 0;
        minor = 0;
        patch = 0;
        build = 0;

        if (string.IsNullOrWhiteSpace(buildVersion))
            return false;

        string[] parts = buildVersion.Split('.');
        if (parts.Length < 4)
            return false;

        return int.TryParse(parts[0], out major)
            && int.TryParse(parts[1], out minor)
            && int.TryParse(parts[2], out patch)
            && int.TryParse(parts[3], out build);
    }
}
