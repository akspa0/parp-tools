using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Provides minimap tiles from loose BLP files on the file system.
/// This is the legacy/default behavior for WoWRollback.
/// </summary>
public sealed class LooseFileMinimapProvider : IMinimapProvider
{
    // Internal storage matching MinimapLocator's structure
    private readonly Dictionary<string, Dictionary<string, Dictionary<(int Row, int Col), MinimapTileInfo>>> _versionMapTiles;

    private LooseFileMinimapProvider(
        Dictionary<string, Dictionary<string, Dictionary<(int, int), MinimapTileInfo>>> versionMapTiles)
    {
        _versionMapTiles = versionMapTiles;
    }

    /// <summary>
    /// Creates a provider by scanning a root directory for version subdirectories containing minimap files.
    /// </summary>
    /// <param name="rootDirectory">Root directory to scan.</param>
    /// <param name="versions">List of version identifiers to load.</param>
    /// <returns>Configured provider.</returns>
    public static LooseFileMinimapProvider Build(string rootDirectory, IReadOnlyList<string> versions)
    {
        var comparer = StringComparer.OrdinalIgnoreCase;
        var versionMapTiles = new Dictionary<string, Dictionary<string, Dictionary<(int, int), MinimapTileInfo>>>(comparer);

        if (string.IsNullOrWhiteSpace(rootDirectory) || versions.Count == 0)
            return new LooseFileMinimapProvider(versionMapTiles);

        var testDataRoot = DetectTestDataRoot(rootDirectory);
        var versionDirectories = ResolveVersionDirectories(rootDirectory, versions);

        foreach (var (versionKey, versionDirectory) in versionDirectories.OrderBy(kvp => kvp.Key, comparer))
        {
            try
            {
                LoadVersion(versionKey, versionDirectory, testDataRoot, versionMapTiles);
            }
            catch
            {
                // Ignore individual version failures; viewer can fall back to placeholders.
            }
        }

        return new LooseFileMinimapProvider(versionMapTiles);
    }

    public Task<Stream?> OpenTileAsync(string version, string mapName, int tileX, int tileY)
    {
        // Note: MinimapLocator uses (Row, Col) which maps to (TileY, TileX)
        if (!_versionMapTiles.TryGetValue(version, out var mapTiles))
            return Task.FromResult<Stream?>(null);

        if (!mapTiles.TryGetValue(mapName, out var tiles))
            return Task.FromResult<Stream?>(null);

        if (!tiles.TryGetValue((tileY, tileX), out var tileInfo))
            return Task.FromResult<Stream?>(null);

        try
        {
            var stream = File.OpenRead(tileInfo.SourcePath);
            return Task.FromResult<Stream?>(stream);
        }
        catch
        {
            return Task.FromResult<Stream?>(null);
        }
    }

    public IEnumerable<string> EnumerateMaps(string version)
    {
        if (!_versionMapTiles.TryGetValue(version, out var mapTiles))
            return Enumerable.Empty<string>();

        return mapTiles.Keys;
    }

    public IEnumerable<(int TileX, int TileY)> EnumerateTiles(string version, string mapName)
    {
        if (!_versionMapTiles.TryGetValue(version, out var mapTiles))
            yield break;

        if (!mapTiles.TryGetValue(mapName, out var tiles))
            yield break;

        foreach (var (row, col) in tiles.Keys)
        {
            // Convert (Row, Col) back to (TileX, TileY)
            yield return (col, row);
        }
    }

    // --- Helper methods adapted from MinimapLocator ---

    private static void LoadVersion(
        string versionKey,
        string versionDirectory,
        string? testDataRoot,
        Dictionary<string, Dictionary<string, Dictionary<(int, int), MinimapTileInfo>>> versionMapTiles)
    {
        var minimapRoot = FindMinimapRoot(versionDirectory) ?? ResolveSharedMinimapRoot(versionKey, testDataRoot);
        if (minimapRoot is null) return;

        var allEntries = new List<MinimapEntry>();
        var trsFiles = GetCandidateTrsFiles(minimapRoot).ToList();

        foreach (var trsPath in trsFiles)
        {
            try
            {
                allEntries.AddRange(ParseTrsFile(trsPath, minimapRoot));
            }
            catch
            {
                // Ignore parse errors
            }
        }

        if (allEntries.Count == 0)
        {
            allEntries.AddRange(ScanMinimapDirectory(minimapRoot));
        }

        if (!versionMapTiles.TryGetValue(versionKey, out var mapTiles))
        {
            mapTiles = new Dictionary<string, Dictionary<(int, int), MinimapTileInfo>>(StringComparer.OrdinalIgnoreCase);
            versionMapTiles[versionKey] = mapTiles;
        }

        var checksumLookup = new Dictionary<(string Map, int X, int Y), string>();

        foreach (var entry in allEntries)
        {
            var resolvedPath = ResolveEntryPath(entry, versionDirectory, versionKey, testDataRoot);
            if (resolvedPath is null || !File.Exists(resolvedPath)) continue;

            if (!mapTiles.TryGetValue(entry.MapName, out var tiles))
            {
                tiles = new Dictionary<(int, int), MinimapTileInfo>();
                mapTiles[entry.MapName] = tiles;
            }

            var key = (entry.TileRow, entry.TileCol);
            var checksumKey = (entry.MapName, entry.TileCol, entry.TileRow);

            if (!checksumLookup.TryGetValue(checksumKey, out var checksum))
            {
                checksum = ComputeFileMd5(resolvedPath);
                checksumLookup[checksumKey] = checksum;
                if (!tiles.ContainsKey(key))
                    tiles[key] = new MinimapTileInfo(resolvedPath, entry.TileCol, entry.TileRow, versionKey, false);
            }
            else
            {
                var currentChecksum = ComputeFileMd5(resolvedPath);
                if (!string.Equals(checksum, currentChecksum, StringComparison.OrdinalIgnoreCase))
                {
                    tiles[key] = new MinimapTileInfo(resolvedPath, entry.TileCol, entry.TileRow, versionKey, true);
                    checksumLookup[checksumKey] = currentChecksum;
                }
            }
        }
    }

    private static string? FindMinimapRoot(string versionDirectory)
    {
        string Combine(params string[] parts) => Path.Combine(parts);

        var preferred = new[]
        {
            Combine(versionDirectory, "tree", "World", "Textures", "Minimap"),
            Combine(versionDirectory, "tree", "world", "textures", "minimap"),
            Combine(versionDirectory, "tree", "Textures", "Minimap"),
            Combine(versionDirectory, "tree", "textures", "Minimap"),
            Combine(versionDirectory, "World", "Textures", "Minimap"),
            Combine(versionDirectory, "world", "textures", "minimap"),
            Combine(versionDirectory, "Textures", "Minimap"),
            Combine(versionDirectory, "textures", "minimap"),
            Combine(versionDirectory, "tree", "World", "Minimaps"),
            Combine(versionDirectory, "tree", "world", "minimaps"),
            Combine(versionDirectory, "World", "Minimaps"),
            Combine(versionDirectory, "world", "minimaps")
        };

        foreach (var path in preferred)
        {
            if (Directory.Exists(path)) return path;
        }

        try
        {
            return Directory.EnumerateDirectories(versionDirectory, "*", SearchOption.AllDirectories)
                .FirstOrDefault(dir =>
                {
                    var name = Path.GetFileName(dir);
                    var parent = Path.GetDirectoryName(dir);
                    if (name.Equals("minimap", StringComparison.OrdinalIgnoreCase))
                    {
                        return parent is not null && Path.GetFileName(parent).Equals("textures", StringComparison.OrdinalIgnoreCase);
                    }
                    if (name.Equals("Minimaps", StringComparison.OrdinalIgnoreCase))
                    {
                        return parent is not null && Path.GetFileName(parent).Equals("World", StringComparison.OrdinalIgnoreCase);
                    }
                    return false;
                });
        }
        catch
        {
            return null;
        }
    }

    private static IEnumerable<string> GetCandidateTrsFiles(string minimapRoot)
    {
        var candidates = new List<string>();
        var directTrs = Path.Combine(minimapRoot, "md5translate.trs");
        if (File.Exists(directTrs)) candidates.Add(directTrs);

        var directTxt = Path.Combine(minimapRoot, "md5translate.txt");
        if (File.Exists(directTxt)) candidates.Add(directTxt);

        if (candidates.Count > 0) return candidates;

        try
        {
            return Directory.EnumerateFiles(minimapRoot, "md5translate.*", SearchOption.AllDirectories)
                .Where(path => path.EndsWith(".trs", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".txt", StringComparison.OrdinalIgnoreCase))
                .ToList();
        }
        catch
        {
            return Array.Empty<string>();
        }
    }

    private static IEnumerable<MinimapEntry> ParseTrsFile(string trsPath, string minimapRoot)
    {
        var entries = new List<MinimapEntry>();
        var baseDir = Path.GetDirectoryName(trsPath);
        if (string.IsNullOrEmpty(baseDir)) baseDir = minimapRoot;

        string? currentMap = null;
        foreach (var raw in File.ReadLines(trsPath))
        {
            var line = raw.Trim();
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;

            if (line.StartsWith("dir:", StringComparison.OrdinalIgnoreCase))
            {
                currentMap = line.Substring(4).Trim();
                continue;
            }

            if (currentMap is null) continue;

            var parts = line.Split('\t');
            if (parts.Length != 2) continue;

            var left = parts[0].Trim();
            var right = parts[1].Trim();

            static bool HasMapStem(string value)
                => value.Contains("map", StringComparison.OrdinalIgnoreCase) &&
                   (value.EndsWith(".blp", StringComparison.OrdinalIgnoreCase) ||
                    value.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                    value.EndsWith(".svg", StringComparison.OrdinalIgnoreCase));

            string MapSide(string value) => HasMapStem(value) ? value : string.Empty;

            var mapCandidate = MapSide(left);
            var actualCandidate = mapCandidate.Length > 0 ? right : left;
            if (mapCandidate.Length == 0)
            {
                mapCandidate = MapSide(right);
                actualCandidate = mapCandidate.Length > 0 ? left : right;
            }

            if (mapCandidate.Length == 0) continue;

            var stem = Path.GetFileNameWithoutExtension(mapCandidate);
            if (!stem.StartsWith("map", StringComparison.OrdinalIgnoreCase)) continue;
            var coords = stem.Substring(3).Split('_');
            if (coords.Length != 2) continue;
            if (!int.TryParse(coords[0], out var tileX)) continue;
            if (!int.TryParse(coords[1], out var tileY)) continue;

            var relativePath = actualCandidate.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
            var fullPath = Path.Combine(baseDir!, relativePath);
            entries.Add(new MinimapEntry(currentMap, tileY, tileX, fullPath, actualCandidate));
        }

        return entries;
    }

    private static IEnumerable<MinimapEntry> ScanMinimapDirectory(string minimapRoot)
    {
        var entries = new List<MinimapEntry>();
        try
        {
            foreach (var mapDir in Directory.EnumerateDirectories(minimapRoot))
            {
                var mapName = Path.GetFileName(mapDir);
                foreach (var blp in Directory.EnumerateFiles(mapDir, "*.blp", SearchOption.TopDirectoryOnly))
                {
                    var stem = Path.GetFileNameWithoutExtension(blp);
                    if (!stem.StartsWith("map", StringComparison.OrdinalIgnoreCase)) continue;
                    var coords = stem.Substring(3).Split('_');
                    if (coords.Length != 2) continue;
                    if (!int.TryParse(coords[0], out var tileX)) continue;
                    if (!int.TryParse(coords[1], out var tileY)) continue;
                    var relative = Path.GetRelativePath(minimapRoot, blp);
                    entries.Add(new MinimapEntry(mapName, tileY, tileX, blp, relative));
                }
            }
        }
        catch
        {
            // ignore directory enumeration errors
        }

        return entries;
    }

    private static string? DetectTestDataRoot(string rootDirectory)
    {
        try
        {
            var fullPath = Path.GetFullPath(rootDirectory);
            if (Path.GetFileName(fullPath).Equals("test_data", StringComparison.OrdinalIgnoreCase) && Directory.Exists(fullPath))
                return fullPath;

            var current = new DirectoryInfo(fullPath);
            for (var depth = 0; depth < 5 && current is not null; depth++, current = current.Parent)
            {
                var candidate = Path.Combine(current.FullName, "test_data");
                if (Directory.Exists(candidate))
                    return candidate;
            }
        }
        catch
        {
            // ignored
        }

        return null;
    }

    private static Dictionary<string, string> ResolveVersionDirectories(string rootDirectory, IReadOnlyList<string> versions)
    {
        var comparer = StringComparer.OrdinalIgnoreCase;
        var map = new Dictionary<string, string>(comparer);
        foreach (var versionIdentifier in versions)
        {
            if (string.IsNullOrWhiteSpace(versionIdentifier)) continue;

            var resolved = ResolveVersionDirectory(rootDirectory, versionIdentifier);
            if (resolved is null) continue;

            var folderName = Path.GetFileName(Path.TrimEndingDirectorySeparator(resolved));
            if (!map.ContainsKey(folderName))
            {
                map[folderName] = resolved;
            }
        }
        return map;
    }

    private static string? ResolveVersionDirectory(string rootDirectory, string versionIdentifier)
    {
        var directPath = Path.Combine(rootDirectory, versionIdentifier);
        if (Directory.Exists(directPath))
            return directPath;

        try
        {
            var candidates = Directory.GetDirectories(rootDirectory)
                .Where(dir =>
                {
                    var name = Path.GetFileName(dir);
                    if (name.Equals(versionIdentifier, StringComparison.OrdinalIgnoreCase))
                        return true;
                    return name.StartsWith(versionIdentifier + ".", StringComparison.OrdinalIgnoreCase);
                })
                .OrderBy(dir => dir, StringComparer.OrdinalIgnoreCase)
                .ToList();

            return candidates.Count switch
            {
                0 => null,
                _ => candidates[0]
            };
        }
        catch
        {
            return null;
        }
    }

    private static string? ResolveEntryPath(MinimapEntry entry, string versionDirectory, string versionKey, string? testDataRoot)
    {
        static string Normalize(string path)
        {
            var normalized = path
                .Replace('\\', Path.DirectorySeparatorChar)
                .Replace('/', Path.DirectorySeparatorChar);
            return normalized.TrimStart(Path.DirectorySeparatorChar).TrimEnd(Path.DirectorySeparatorChar);
        }

        if (File.Exists(entry.FullPath))
            return entry.FullPath;
        var originalNormalized = Normalize(entry.OriginalPath);

        var candidate = Path.Combine(versionDirectory, originalNormalized);
        if (File.Exists(candidate))
            return candidate;

        if (testDataRoot is not null)
        {
            var sharedRoot = ResolveSharedMinimapRoot(versionKey, testDataRoot);
            if (sharedRoot is not null)
            {
                candidate = Path.Combine(sharedRoot, originalNormalized);
                if (File.Exists(candidate))
                    return candidate;
            }

            foreach (var alias in EnumerateVersionAliases(versionKey))
            {
                var aliasRoot = Path.Combine(testDataRoot, alias);
                candidate = Path.Combine(aliasRoot, originalNormalized);
                if (File.Exists(candidate))
                    return candidate;
            }

            candidate = Path.Combine(testDataRoot, originalNormalized);
            if (File.Exists(candidate))
                return candidate;
        }

        try
        {
            var parent = Directory.GetParent(versionDirectory);
            if (parent is not null)
            {
                candidate = Path.Combine(parent.FullName, originalNormalized);
                if (File.Exists(candidate))
                    return candidate;
            }
        }
        catch
        {
            // ignore IO exceptions during fallback
        }

        return null;
    }

    private static string? ResolveSharedMinimapRoot(string versionKey, string? testDataRoot)
    {
        if (testDataRoot is null) return null;

        foreach (var alias in EnumerateVersionAliases(versionKey))
        {
            foreach (var candidate in EnumerateMinimapRootCandidates(Path.Combine(testDataRoot, alias)))
            {
                if (Directory.Exists(candidate))
                    return candidate;
            }
        }

        foreach (var fallback in EnumerateMinimapRootCandidates(testDataRoot))
        {
            if (Directory.Exists(fallback))
                return fallback;
        }

        return null;
    }

    private static IEnumerable<string> EnumerateMinimapRootCandidates(string basePath)
    {
        yield return Path.Combine(basePath, "tree", "World", "Textures", "Minimap");
        yield return Path.Combine(basePath, "tree", "world", "textures", "minimap");
        yield return Path.Combine(basePath, "tree", "Textures", "Minimap");
        yield return Path.Combine(basePath, "tree", "textures", "Minimap");
        yield return Path.Combine(basePath, "World", "Textures", "Minimap");
        yield return Path.Combine(basePath, "world", "textures", "minimap");
        yield return Path.Combine(basePath, "Textures", "Minimap");
        yield return Path.Combine(basePath, "textures", "minimap");
    }

    private static IEnumerable<string> EnumerateVersionAliases(string versionKey)
    {
        if (string.IsNullOrWhiteSpace(versionKey)) yield break;

        var normalized = versionKey.Trim();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        string initial = normalized.Replace('/', '.');
        if (seen.Add(initial)) yield return initial;

        var parts = initial.Split('.', StringSplitOptions.RemoveEmptyEntries);
        for (var length = parts.Length - 1; length >= 1; length--)
        {
            var alias = string.Join('.', parts.Take(length));
            if (seen.Add(alias)) yield return alias;
        }
    }

    private static string ComputeFileMd5(string path)
    {
        using var stream = File.OpenRead(path);
        using var md5 = System.Security.Cryptography.MD5.Create();
        var hash = md5.ComputeHash(stream);
        return BitConverter.ToString(hash).Replace("-", string.Empty);
    }

    // --- Helper records ---

    private readonly record struct MinimapEntry(string MapName, int TileRow, int TileCol, string FullPath, string OriginalPath);
    private readonly record struct MinimapTileInfo(string SourcePath, int TileX, int TileY, string Version, bool IsAlternate);
}
