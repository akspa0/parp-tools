using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DBCD;
using DBCD.Providers;

namespace WoWRollback.Core.Services.Dbc;

/// <summary>
/// Resolves canonical map directory names using Map.dbc across multiple versions.
/// Provides normalization from friendly/common names to the internal directory used by minimaps.
/// </summary>
public sealed class MapDirectoryResolver
{
    private readonly string _rootDirectory;
    private readonly IReadOnlyList<string> _versions;
    private readonly StringComparer _cmp = StringComparer.OrdinalIgnoreCase;

    // versionKey -> (alias/name -> directory)
    private readonly Dictionary<string, Dictionary<string, string>> _byVersion = new(StringComparer.OrdinalIgnoreCase);

    public MapDirectoryResolver(string rootDirectory, IReadOnlyList<string> versions)
    {
        _rootDirectory = rootDirectory;
        _versions = versions;
        BuildIndex();
    }

    /// <summary>
    /// Normalize any provided name (friendly or directory) to a canonical directory, probing all known versions.
    /// If no match is found, returns the original value.
    /// </summary>
    public string Normalize(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return name;
        foreach (var kvp in _byVersion)
        {
            var map = kvp.Value;
            if (map.TryGetValue(name, out var dir)) return dir;
        }
        return name;
    }

    private void BuildIndex()
    {
        foreach (var version in _versions)
        {
            try
            {
                var dirMap = LoadVersion(version);
                if (dirMap.Count > 0)
                    _byVersion[version] = dirMap;
            }
            catch
            {
                // non-fatal per-version
            }
        }
    }

    private Dictionary<string, string> LoadVersion(string versionKey)
    {
        var result = new Dictionary<string, string>(_cmp);
        var versionDir = ResolveVersionDirectory(_rootDirectory, versionKey);
        if (versionDir is null) return result;

        var dbcDir = FindDbcDirectory(versionDir);
        if (dbcDir is null) return result;

        try
        {
            var dbcProvider = new FilesystemDBCProvider(dbcDir);
            var dbdProvider = new GithubDBDProvider(true);
            var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

            IDBCDStorage storage;
            try { storage = dbcd.Load("Map", versionKey, Locale.None); }
            catch { storage = dbcd.Load("Map", null, Locale.None); }

            foreach (var pair in (IEnumerable<KeyValuePair<int, DBCDRow>>)storage)
            {
                var row = pair.Value;
                var dir = FirstString(row, "m_Directory", "Directory");
                if (string.IsNullOrWhiteSpace(dir)) continue;
                var name = FirstString(row, "MapName_lang", "m_MapName_lang", "MapName");
                var internalName = FirstString(row, "InternalName", "m_InternalName");

                // Index by multiple aliases to the directory
                result[dir] = dir;
                if (!string.IsNullOrWhiteSpace(name)) result[name] = dir;
                if (!string.IsNullOrWhiteSpace(internalName)) result[internalName] = dir;
            }
        }
        catch
        {
            // ignore DBCD errors
        }

        return result;
    }

    private static string? ResolveVersionDirectory(string rootDirectory, string versionIdentifier)
    {
        var directPath = Path.Combine(rootDirectory, versionIdentifier);
        if (Directory.Exists(directPath)) return directPath;
        try
        {
            var candidates = Directory.GetDirectories(rootDirectory)
                .Where(d => Path.GetFileName(d).StartsWith(versionIdentifier + ".", StringComparison.OrdinalIgnoreCase) ||
                            Path.GetFileName(d).Equals(versionIdentifier, StringComparison.OrdinalIgnoreCase))
                .OrderBy(d => d, StringComparer.OrdinalIgnoreCase)
                .ToList();
            return candidates.FirstOrDefault();
        }
        catch { return null; }
    }

    private static string? FindDbcDirectory(string versionDirectory)
    {
        string Combine(params string[] parts) => Path.Combine(parts);
        var candidates = new[]
        {
            Combine(versionDirectory, "tree", "DBFilesClient"),
            Combine(versionDirectory, "DBFilesClient"),
            versionDirectory
        };
        foreach (var c in candidates)
        {
            try
            {
                if (Directory.Exists(c) && Directory.GetFiles(c, "Map.dbc", SearchOption.AllDirectories).Any())
                    return c;
            }
            catch { }
        }
        try
        {
            return Directory.EnumerateDirectories(versionDirectory, "*", SearchOption.AllDirectories)
                .FirstOrDefault(d => string.Equals(Path.GetFileName(d), "DBFilesClient", StringComparison.OrdinalIgnoreCase));
        }
        catch { return null; }
    }

    private static string? FirstString(DBCDRow row, params string[] cols)
    {
        foreach (var c in cols)
        {
            try
            {
                var val = row[c];
                if (val is null) continue;
                var s = val.ToString();
                if (!string.IsNullOrWhiteSpace(s)) return s.Trim();
            }
            catch { /* missing column */ }
        }
        return null;
    }
}
