using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core;

public sealed class MultiListfileResolver
{
    private readonly Dictionary<string, int> _primary; // normalized wow paths -> id
    private readonly Dictionary<string, int> _secondary;

    public MultiListfileResolver(ListfileLoader? primaryLk, ListfileLoader? secondaryCommunity)
    {
        _primary = primaryLk is null ? new(StringComparer.OrdinalIgnoreCase) : Build(primaryLk);
        _secondary = secondaryCommunity is null ? new(StringComparer.OrdinalIgnoreCase) : Build(secondaryCommunity);
    }

    private MultiListfileResolver(Dictionary<string, int> primary, Dictionary<string, int> secondary)
    {
        _primary = primary;
        _secondary = secondary;
    }

    public static MultiListfileResolver FromFiles(string? lkListfilePath, string? communityListfilePath)
    {
        var primary = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var secondary = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        if (!string.IsNullOrWhiteSpace(lkListfilePath) && File.Exists(lkListfilePath))
        {
            var dict = LoadAnyListfile(lkListfilePath);
            foreach (var kv in dict)
            {
                if (!primary.ContainsKey(kv.Key)) primary[kv.Key] = kv.Value;
            }
        }
        if (!string.IsNullOrWhiteSpace(communityListfilePath) && File.Exists(communityListfilePath))
        {
            var dict = LoadAnyListfile(communityListfilePath);
            foreach (var kv in dict)
            {
                if (!secondary.ContainsKey(kv.Key)) secondary[kv.Key] = kv.Value;
            }
        }

        return new MultiListfileResolver(primary, secondary);
    }

    private static Dictionary<string, int> LoadAnyListfile(string path)
    {
        // CSV with id;path or plain TXT with just paths
        var dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var ext = Path.GetExtension(path);
        var isTxt = ext.Equals(".txt", StringComparison.OrdinalIgnoreCase);

        if (isTxt)
        {
            int nextId = 1_000_000_000; // synthetic ids
            foreach (var line in File.ReadLines(path))
            {
                var t = line.Trim();
                if (t.Length == 0 || t.StartsWith("#")) continue;
                var norm = WowPath.Normalize(t);
                if (!dict.ContainsKey(norm)) dict[norm] = nextId++;
            }
        }
        else
        {
            var loader = new ListfileLoader();
            loader.Load(path);
            foreach (var kv in loader.IdByPathNormalized)
            {
                var norm = WowPath.Normalize(kv.Key);
                if (!dict.ContainsKey(norm)) dict[norm] = kv.Value;
            }
        }

        return dict;
    }

    private static Dictionary<string, int> Build(ListfileLoader loader)
    {
        var dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var kv in loader.IdByPathNormalized)
        {
            // Convert listfile's forward-slash canonical path to wow-normalized (lower + backslashes)
            var norm = WowPath.Normalize(kv.Key);
            if (!dict.ContainsKey(norm)) dict[norm] = kv.Value;
        }
        return dict;
    }

    public bool Exists(string path)
    {
        var norm = WowPath.Normalize(path);
        return _primary.ContainsKey(norm) || _secondary.ContainsKey(norm);
    }

    public bool TryGetIdByPath(string path, out int id)
    {
        var norm = WowPath.Normalize(path);
        if (_primary.TryGetValue(norm, out id)) return true;
        if (_secondary.TryGetValue(norm, out id)) return true;
        id = 0;
        return false;
    }

    public string? FindSimilar(string path, IEnumerable<string> allowedExtensions)
    {
        var norm = WowPath.Normalize(path);
        var targetName = Path.GetFileName(norm);
        var targetNameNoExt = Path.GetFileNameWithoutExtension(norm);
        var allowed = new HashSet<string>(allowedExtensions.Select(e => e.ToLowerInvariant()));

        // Search primary by exact basename, filtered by allowed ext
        var primaryMatch = _primary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Where(p => string.Equals(Path.GetFileName(p), targetName, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(p => PathSimilarity(norm, p))
            .FirstOrDefault();
        if (primaryMatch is not null) return primaryMatch;

        // Try basename without ext equality
        primaryMatch = _primary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Where(p => string.Equals(Path.GetFileNameWithoutExtension(p), targetNameNoExt, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(p => PathSimilarity(norm, p))
            .FirstOrDefault();
        if (primaryMatch is not null) return primaryMatch;

        // Fallback to secondary
        var secondaryMatch = _secondary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Where(p => string.Equals(Path.GetFileName(p), targetName, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(p => PathSimilarity(norm, p))
            .FirstOrDefault();
        if (secondaryMatch is not null) return secondaryMatch;

        secondaryMatch = _secondary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Where(p => string.Equals(Path.GetFileNameWithoutExtension(p), targetNameNoExt, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(p => PathSimilarity(norm, p))
            .FirstOrDefault();
        if (secondaryMatch is not null) return secondaryMatch;

        return null;
    }

    private static double PathSimilarity(string a, string b)
    {
        // Jaccard on path segments as a simple heuristic
        var segA = a.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var segB = b.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var inter = segA.Intersect(segB, StringComparer.OrdinalIgnoreCase).Count();
        var union = segA.Union(segB, StringComparer.OrdinalIgnoreCase).Count();
        return union == 0 ? 0 : (double)inter / union;
    }
}
