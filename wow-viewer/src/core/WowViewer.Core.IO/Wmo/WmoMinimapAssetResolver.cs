namespace WowViewer.Core.IO.Wmo;

/// <summary>
/// Resolves WMO stem names to full asset paths by searching the MPQ file list
/// for .wmo root files matching the stem name.
/// </summary>
public static class WmoMinimapAssetResolver
{
    /// <summary>
    /// Build a mapping from WMO stem names to their full asset paths.
    /// Scans the archive catalog for .wmo root files and extracts the stem
    /// (basename without extension) as the key.
    /// </summary>
    /// <param name="catalog">The MPQ archive catalog with loaded archives and listfile.</param>
    /// <returns>Dictionary mapping stem name (lowercased) to full asset path.</returns>
    public static Dictionary<string, string> BuildStemToAssetPathMap(Files.IArchiveCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var knownFiles = catalog.GetAllKnownFiles();

        foreach (string path in knownFiles)
        {
            if (!path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                continue;

            // Extract stem: "World\wmo\dungeon\deadmines\deadmines.wmo" → "deadmines"
            string filename = Path.GetFileNameWithoutExtension(path);
            if (string.IsNullOrEmpty(filename))
                continue;

            // Use case-insensitive key; keep the first match (most specific path wins)
            string key = filename.ToLowerInvariant();
            if (!map.ContainsKey(key))
            {
                map[key] = path;
            }
        }

        return map;
    }

    /// <summary>
    /// Resolve a WMO stem name to its full asset path.
    /// Returns null if no matching .wmo root file is found.
    /// </summary>
    public static string? ResolveStemToAssetPath(
        Dictionary<string, string> stemMap,
        string wmoStem)
    {
        if (string.IsNullOrEmpty(wmoStem))
            return null;

        string key = wmoStem.ToLowerInvariant();
        return stemMap.TryGetValue(key, out string? assetPath) ? assetPath : null;
    }
}
