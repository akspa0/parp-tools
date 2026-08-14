namespace WowViewer.Core.Lit;

/// <summary>
/// Resolves all LIT profiles stored directly in a map's client folder while preserving the
/// conventional source order used by the viewer.
/// </summary>
public static class LitSourcePathResolver
{
    public static IReadOnlyList<string> Resolve(
        IEnumerable<string> knownLitPaths,
        string mapName)
    {
        ArgumentNullException.ThrowIfNull(knownLitPaths);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);

        string[] conventionalPaths =
        {
            $"World\\{mapName}\\lights.lit",
            $"World\\Maps\\{mapName}\\lights.lit",
            $"World\\{mapName}\\areatest.lit",
            $"World\\Maps\\{mapName}\\areatest.lit",
            $"World\\{mapName}\\light.lit",
            $"World\\Maps\\{mapName}\\light.lit",
        };

        var result = new List<string>(conventionalPaths.Length);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string path in conventionalPaths)
            Add(path);

        foreach (string path in knownLitPaths
                     .Where(path => IsDirectMapLitPath(path, mapName))
                     .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
        {
            Add(path);
        }

        return result;

        void Add(string path)
        {
            string normalized = path.Replace('/', '\\').TrimStart('\\');
            if (seen.Add(normalized))
                result.Add(normalized);
        }
    }

    private static bool IsDirectMapLitPath(string path, string mapName)
    {
        if (string.IsNullOrWhiteSpace(path)
            || !path.EndsWith(".lit", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        string[] segments = path.Replace('/', '\\')
            .TrimStart('\\')
            .Split('\\', StringSplitOptions.RemoveEmptyEntries);

        if (segments.Length < 3
            || !segments[^1].EndsWith(".lit", StringComparison.OrdinalIgnoreCase)
            || !segments[^2].Equals(mapName, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return segments.Length == 3
            ? segments[0].Equals("World", StringComparison.OrdinalIgnoreCase)
            : segments.Length == 4
                && segments[0].Equals("World", StringComparison.OrdinalIgnoreCase)
                && segments[1].Equals("Maps", StringComparison.OrdinalIgnoreCase);
    }
}
