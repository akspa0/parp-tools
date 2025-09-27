using System.Text.RegularExpressions;
using System.Runtime.CompilerServices;
using System.Linq;

namespace WoWRollback.Core.Services;

public static class BuildTagResolver
{
    private static readonly Regex VersionPattern = new("^\\d+\\.\\d+\\.\\d+(?:\\.\\d+)?$", RegexOptions.Compiled);
    private static readonly object CacheLock = new();
    private static IReadOnlyDictionary<string, string>? _versionToBuild;

    public static string ResolveForPath(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return "unknown_build";
        }

        var (versionOnly, buildFull) = ExtractVersionSegments(path);
        if (!string.IsNullOrEmpty(buildFull))
        {
            return Sanitize(buildFull);
        }

        if (!string.IsNullOrEmpty(versionOnly))
        {
            var fromDb = ResolveFromDb(versionOnly, path);
            if (!string.IsNullOrEmpty(fromDb))
            {
                return Sanitize(fromDb);
            }

            return Sanitize(versionOnly);
        }

        return "unknown_build";
    }

    private static (string? VersionOnly, string? BuildFull) ExtractVersionSegments(string path)
    {
        string? versionOnly = null;
        string? buildFull = null;

        var separators = new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar };
        var segments = path.Split(separators, StringSplitOptions.RemoveEmptyEntries);
        foreach (var segment in segments)
        {
            if (!VersionPattern.IsMatch(segment))
            {
                continue;
            }

            if (segment.Count(c => c == '.') >= 3)
            {
                buildFull = segment;
                break;
            }

            versionOnly = segment;
        }

        return (versionOnly, buildFull);
    }

    private static string? ResolveFromDb(string version, string referencePath)
    {
        var map = LoadBuildMap(referencePath);
        if (map.TryGetValue(version, out var build))
        {
            return build;
        }

        return null;
    }

    private static IReadOnlyDictionary<string, string> LoadBuildMap(string referencePath)
    {
        lock (CacheLock)
        {
            if (_versionToBuild is not null)
            {
                return _versionToBuild;
            }

            var searchStart = Directory.Exists(referencePath)
                ? referencePath
                : Path.GetDirectoryName(referencePath);

            var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var dir = !string.IsNullOrEmpty(searchStart) ? new DirectoryInfo(searchStart) : new DirectoryInfo(Directory.GetCurrentDirectory());

            while (dir is not null)
            {
                var candidate = Path.Combine(dir.FullName, "lib", "WoWDBDefs", "definitions", "AreaTable.dbd");
                if (File.Exists(candidate))
                {
                    foreach (var line in File.ReadLines(candidate))
                    {
                        var trimmed = line.Trim();
                        if (!trimmed.StartsWith("BUILD", StringComparison.OrdinalIgnoreCase))
                        {
                            continue;
                        }

                        var parts = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length < 2)
                        {
                            continue;
                        }

                        var build = parts[1];
                        if (!VersionPattern.IsMatch(build))
                        {
                            continue;
                        }

                        var tokens = build.Split('.');
                        if (tokens.Length < 3)
                        {
                            continue;
                        }

                        var versionKey = string.Join('.', tokens.Take(3));
                        if (!dict.ContainsKey(versionKey))
                        {
                            dict[versionKey] = build;
                        }
                    }

                    break;
                }

                dir = dir.Parent;
            }

            _versionToBuild = dict;
            return _versionToBuild;
        }
    }

    private static string Sanitize(string value)
    {
        foreach (var c in Path.GetInvalidFileNameChars())
        {
            value = value.Replace(c, '_');
        }

        return value;
    }
}
