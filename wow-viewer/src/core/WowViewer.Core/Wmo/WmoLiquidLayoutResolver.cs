namespace WowViewer.Core.Wmo;

public enum WmoLiquidCoordinateFamily
{
    Unknown,
    LegacyV14,
    LegacyV16,
    StandardV17Plus,
}

public static class WmoLiquidLayoutResolver
{
    public static WmoLiquidCoordinateFamily ResolveCoordinateFamily(uint? wmoVersion, string? buildVersion = null)
    {
        if (wmoVersion == 14)
            return WmoLiquidCoordinateFamily.LegacyV14;

        if (wmoVersion == 16)
            return WmoLiquidCoordinateFamily.LegacyV16;

        if (wmoVersion >= 17)
            return WmoLiquidCoordinateFamily.StandardV17Plus;

        if (!TryParseBuild(buildVersion, out int major, out int minor, out _, out _))
            return WmoLiquidCoordinateFamily.Unknown;

        if (major == 0 && minor <= 5)
            return WmoLiquidCoordinateFamily.LegacyV14;

        if (major == 0 && minor == 6)
            return WmoLiquidCoordinateFamily.LegacyV16;

        return WmoLiquidCoordinateFamily.StandardV17Plus;
    }

    public static int GetBaselineRotationQuarterTurns(uint? wmoVersion, string? buildVersion = null)
    {
        _ = ResolveCoordinateFamily(wmoVersion, buildVersion);

        // Keep the shared baseline neutral. Runtime auto-fit handles the current
        // validated assets more reliably than a hard-coded quarter turn, and the
        // version family is still surfaced explicitly for future per-family rules.
        return 0;
    }

    private static bool TryParseBuild(string? buildVersion, out int major, out int minor, out int patch, out int build)
    {
        major = 0;
        minor = 0;
        patch = 0;
        build = 0;

        if (string.IsNullOrWhiteSpace(buildVersion))
            return false;

        string[] parts = buildVersion.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 3 || parts.Length > 4)
            return false;

        if (!int.TryParse(parts[0], out major)
            || !int.TryParse(parts[1], out minor)
            || !int.TryParse(parts[2], out patch))
        {
            return false;
        }

        if (parts.Length == 4 && !int.TryParse(parts[3], out build))
            return false;

        return true;
    }
}