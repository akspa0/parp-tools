namespace WoWMapConverter.Core.Services;

/// <summary>
/// Loads and queries community listfiles for asset path resolution.
/// Supports both community listfile (all versions) and LK 3.x specific listfile.
/// </summary>
public class ListfileService
{
    private readonly Dictionary<string, string> _pathLookup = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _filenameLookup = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _lkPaths = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Load the community listfile CSV.
    /// Format: fileDataId;path (semicolon-separated)
    /// </summary>
    public void LoadCommunityListfile(string csvPath)
    {
        if (!File.Exists(csvPath))
            throw new FileNotFoundException($"Community listfile not found: {csvPath}");

        foreach (var line in File.ReadLines(csvPath))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var parts = line.Split(';', 2);
            if (parts.Length >= 2)
            {
                var path = parts[1].Trim();
                var normalized = NormalizePath(path);
                var filename = Path.GetFileName(normalized);

                _pathLookup[normalized] = path;
                
                // Store by filename for fuzzy matching
                if (!string.IsNullOrEmpty(filename))
                {
                    _filenameLookup.TryAdd(filename, path);
                }
            }
        }
    }

    /// <summary>
    /// Load the LK 3.x listfile (text format, one path per line).
    /// </summary>
    public void LoadLkListfile(string txtPath)
    {
        if (!File.Exists(txtPath))
            return;

        foreach (var line in File.ReadLines(txtPath))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var normalized = NormalizePath(line.Trim());
            _lkPaths.Add(normalized);
        }
    }

    /// <summary>
    /// Check if a path exists in the LK listfile.
    /// </summary>
    public bool ExistsInLk(string path)
    {
        return _lkPaths.Contains(NormalizePath(path));
    }

    /// <summary>
    /// Try to find a matching path in the community listfile.
    /// Returns the canonical path if found.
    /// </summary>
    public string? FindPath(string path)
    {
        var normalized = NormalizePath(path);
        if (_pathLookup.TryGetValue(normalized, out var canonical))
            return canonical;
        return null;
    }

    /// <summary>
    /// Try to find a path by filename only (fuzzy matching).
    /// </summary>
    public string? FindByFilename(string filename)
    {
        var name = Path.GetFileName(filename);
        if (_filenameLookup.TryGetValue(name, out var canonical))
            return canonical;
        return null;
    }

    /// <summary>
    /// Fix an asset path using listfile lookups.
    /// Returns the fixed path or original if not found.
    /// </summary>
    public string FixAssetPath(string path, bool fuzzy = false)
    {
        // Try exact match first
        var found = FindPath(path);
        if (found != null)
            return found;

        // Try fuzzy match by filename
        if (fuzzy)
        {
            found = FindByFilename(path);
            if (found != null)
                return found;
        }

        return path;
    }

    /// <summary>
    /// Normalize a path for consistent lookups.
    /// </summary>
    private static string NormalizePath(string path)
    {
        return path.Replace('/', '\\').ToLowerInvariant().Trim();
    }

    /// <summary>
    /// Get total paths loaded from community listfile.
    /// </summary>
    public int CommunityPathCount => _pathLookup.Count;

    /// <summary>
    /// Get total paths loaded from LK listfile.
    /// </summary>
    public int LkPathCount => _lkPaths.Count;
    /// <summary>
    /// Export all known paths to a flat text file (for use with StormLib).
    /// </summary>
    public void ExportToFile(string outputPath)
    {
        using var writer = new StreamWriter(outputPath);
        // Prioritize community listfile
        foreach (var path in _pathLookup.Values)
        {
            writer.WriteLine(path);
        }
        // Add LK paths (might contain duplicates, StormLib handles it)
        foreach (var path in _lkPaths)
        {
             writer.WriteLine(path);
        }
    }
}
