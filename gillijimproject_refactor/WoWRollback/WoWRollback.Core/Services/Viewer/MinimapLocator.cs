using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WoWRollback.Core.Services.Viewer;

internal sealed class MinimapLocator
{
    private readonly Dictionary<string, Dictionary<(int Row, int Col), string>> _mapTiles;

    private MinimapLocator(Dictionary<string, Dictionary<(int, int), string>> mapTiles)
    {
        _mapTiles = mapTiles;
    }

    public static MinimapLocator Build(string rootDirectory, IReadOnlyList<string> versions)
    {
        var comparer = StringComparer.OrdinalIgnoreCase;
        var mapTiles = new Dictionary<string, Dictionary<(int, int), string>>(comparer);

        if (string.IsNullOrWhiteSpace(rootDirectory) || versions.Count == 0)
            return new MinimapLocator(mapTiles);

        var versionDirectories = ResolveVersionDirectories(rootDirectory, versions);
        foreach (var (_, versionDirectory) in versionDirectories.OrderBy(kvp => kvp.Key, comparer))
        {
            try
            {
                LoadVersion(versionDirectory, mapTiles);
            }
            catch
            {
                // Ignore individual version failures; viewer can fall back to placeholders.
            }
        }

        return new MinimapLocator(mapTiles);
    }

    public bool TryOpen(string map, int tileRow, int tileCol, out Stream? stream)
    {
        stream = null;
        if (!_mapTiles.TryGetValue(map, out var tiles))
            return false;

        if (!tiles.TryGetValue((tileRow, tileCol), out var path) || string.IsNullOrWhiteSpace(path))
            return false;

        try
        {
            stream = File.OpenRead(path);
            return true;
        }
        catch
        {
            stream = null;
            return false;
        }
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

    private static void LoadVersion(string versionDirectory, Dictionary<string, Dictionary<(int, int), string>> mapTiles)
    {
        var minimapRoot = FindMinimapRoot(versionDirectory);
        if (minimapRoot is null) return;

        var entries = new List<MinimapEntry>();

        foreach (var trsPath in GetCandidateTrsFiles(minimapRoot))
        {
            try
            {
                entries.AddRange(ParseTrsFile(trsPath, minimapRoot));
            }
            catch
            {
                // Ignore malformed TRS; fallback to directory scan below.
            }
        }

        if (entries.Count == 0)
        {
            entries.AddRange(ScanMinimapDirectory(minimapRoot));
        }

        foreach (var entry in entries)
        {
            if (!File.Exists(entry.FullPath)) continue;
            if (!mapTiles.TryGetValue(entry.MapName, out var tiles))
            {
                tiles = new Dictionary<(int, int), string>();
                mapTiles[entry.MapName] = tiles;
            }

            var key = (entry.TileRow, entry.TileCol);
            if (!tiles.ContainsKey(key))
            {
                tiles[key] = entry.FullPath;
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
            Combine(versionDirectory, "World", "Textures", "Minimap"),
            Combine(versionDirectory, "world", "textures", "minimap")
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
                    if (!name.Equals("minimap", StringComparison.OrdinalIgnoreCase)) return false;
                    var parent = Path.GetDirectoryName(dir);
                    return parent is not null && Path.GetFileName(parent).Equals("textures", StringComparison.OrdinalIgnoreCase);
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

            string MapSide(string value) => value.Contains("map", StringComparison.OrdinalIgnoreCase) && value.Contains(".blp", StringComparison.OrdinalIgnoreCase)
                ? value
                : string.Empty;

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
            entries.Add(new MinimapEntry(currentMap, tileY, tileX, fullPath));
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
                    entries.Add(new MinimapEntry(mapName, tileY, tileX, blp));
                }
            }
        }
        catch
        {
            // ignore directory enumeration errors
        }

        return entries;
    }

    private readonly record struct MinimapEntry(string MapName, int TileRow, int TileCol, string FullPath);
}
