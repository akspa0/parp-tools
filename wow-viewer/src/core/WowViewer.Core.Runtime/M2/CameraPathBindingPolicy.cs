namespace WowViewer.Core.Runtime.M2;

/// <summary>
/// Compares camera-path provenance without making formatting differences look like
/// a map change. A path may only be reused when its map identity is equivalent;
/// build labels are normalized only for formatting differences.
/// </summary>
public static class CameraPathBindingPolicy
{
    public static bool AreEquivalent(
        string? pathMapName,
        string? pathBuildVersion,
        string? currentMapName,
        string? currentBuildVersion)
        => AreEquivalentMapNames(pathMapName, currentMapName)
            && AreEquivalentBuildVersions(pathBuildVersion, currentBuildVersion);

    public static bool AreEquivalentMapNames(string? left, string? right)
    {
        string normalizedLeft = NormalizeMapName(left);
        string normalizedRight = NormalizeMapName(right);
        if (normalizedLeft.Length == 0 || normalizedRight.Length == 0)
            return IsMissingMapName(left) && IsMissingMapName(right);

        return string.Equals(normalizedLeft, normalizedRight, StringComparison.Ordinal);
    }

    public static bool AreEquivalentBuildVersions(string? left, string? right)
    {
        string normalizedLeft = NormalizeBuildVersion(left);
        string normalizedRight = NormalizeBuildVersion(right);

        // Missing provenance is handled only when the operator starts a path
        // (the viewer binds it to the active client then). During playback a
        // build becoming unavailable must stop rather than turn into a
        // wildcard match.
        if (normalizedLeft.Length == 0 || normalizedRight.Length == 0)
            return IsMissingBuildVersion(left) && IsMissingBuildVersion(right);

        return string.Equals(normalizedLeft, normalizedRight, StringComparison.Ordinal);
    }

    public static string NormalizeMapName(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;

        string normalized = value.Trim().Trim('"', '\'');
        if (IsMissingMapName(normalized))
            return string.Empty;

        normalized = normalized.Replace('\\', '/').TrimEnd('/');
        int lastSeparator = normalized.LastIndexOf('/');
        if (lastSeparator >= 0)
            normalized = normalized[(lastSeparator + 1)..];

        if (normalized.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
            normalized = normalized[..^4];

        return normalized.Trim().ToUpperInvariant();
    }

    public static string NormalizeBuildVersion(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;

        string normalized = value.Trim().Trim('"', '\'');
        if (IsMissingBuildVersion(normalized))
            return string.Empty;

        if (normalized.StartsWith("build:", StringComparison.OrdinalIgnoreCase))
            normalized = normalized[6..].Trim();
        else if (normalized.StartsWith("version:", StringComparison.OrdinalIgnoreCase))
            normalized = normalized[8..].Trim();
        else if (normalized.StartsWith("build=", StringComparison.OrdinalIgnoreCase))
            normalized = normalized[6..].Trim();
        else if (normalized.StartsWith("version=", StringComparison.OrdinalIgnoreCase))
            normalized = normalized[8..].Trim();

        if (normalized.StartsWith('v'))
            normalized = normalized[1..];

        normalized = normalized.Replace('_', '.');
        string[] components = normalized.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (components.Length == 0)
            return string.Empty;

        for (int index = 0; index < components.Length; index++)
        {
            if (uint.TryParse(components[index], out uint component))
                components[index] = component.ToString(System.Globalization.CultureInfo.InvariantCulture);
            else
                components[index] = components[index].ToUpperInvariant();
        }

        return string.Join('.', components);
    }

    public static bool IsMissingMapName(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return true;

        string normalized = value.Trim().Trim('"', '\'');
        return IsMissingMapToken(normalized);
    }

    public static bool IsMissingBuildVersion(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return true;

        string normalized = value.Trim().Trim('"', '\'');
        return IsMissingBuildToken(normalized);
    }

    private static bool IsMissingMapToken(string value)
        => value.Equals("unknown", StringComparison.OrdinalIgnoreCase)
            || value.Equals("unknown_map", StringComparison.OrdinalIgnoreCase)
            || value.Equals("standalone", StringComparison.OrdinalIgnoreCase)
            || value.Equals("none", StringComparison.OrdinalIgnoreCase);

    private static bool IsMissingBuildToken(string value)
        => value.Equals("unknown", StringComparison.OrdinalIgnoreCase)
            || value.Equals("unknown_build", StringComparison.OrdinalIgnoreCase)
            || value.Equals("standalone", StringComparison.OrdinalIgnoreCase)
            || value.Equals("none", StringComparison.OrdinalIgnoreCase);
}
