using System;

namespace WoWRollback.AdtModule.Analysis;

internal static class WowPath
{
    public static string Normalize(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) return string.Empty;
        var p = path.Replace('\u005C', '/').Trim();
        while (p.Contains("//", StringComparison.Ordinal)) p = p.Replace("//", "/", StringComparison.Ordinal);
        return p.ToLowerInvariant();
    }
}
