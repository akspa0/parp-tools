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

        // Heuristic: common prefix variants (e.g., AZ_ + name)
        var variants = new[] { $"AZ_{targetNameNoExt}", $"A_{targetNameNoExt}", $"{targetNameNoExt}_A" };
        primaryMatch = _primary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Where(p => variants.Any(v => string.Equals(Path.GetFileNameWithoutExtension(p), v, StringComparison.OrdinalIgnoreCase)))
            .OrderByDescending(p => PathSimilarity(norm, p))
            .FirstOrDefault();
        if (primaryMatch is not null) return primaryMatch;

        // Fuzzy basename similarity over primary
        var primaryFuzzy = _primary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Select(p => new { Path = p, Score = BasenameSimilarity(targetNameNoExt, Path.GetFileNameWithoutExtension(p)) })
            .Where(x => x.Score >= 0.70)
            .OrderByDescending(x => x.Score)
            .ThenByDescending(x => PathSimilarity(norm, x.Path))
            .Select(x => x.Path)
            .FirstOrDefault();
        if (primaryFuzzy is not null) return primaryFuzzy;

        // Secondary: exact filename
        var secondaryMatch = _secondary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Where(p => string.Equals(Path.GetFileName(p), targetName, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(p => PathSimilarity(norm, p))
            .FirstOrDefault();
        if (secondaryMatch is not null) return secondaryMatch;

        // Secondary: exact filename without ext
        secondaryMatch = _secondary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Where(p => string.Equals(Path.GetFileNameWithoutExtension(p), targetNameNoExt, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(p => PathSimilarity(norm, p))
            .FirstOrDefault();
        if (secondaryMatch is not null) return secondaryMatch;

        // Secondary: prefix variants
        secondaryMatch = _secondary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Where(p => variants.Any(v => string.Equals(Path.GetFileNameWithoutExtension(p), v, StringComparison.OrdinalIgnoreCase)))
            .OrderByDescending(p => PathSimilarity(norm, p))
            .FirstOrDefault();
        if (secondaryMatch is not null) return secondaryMatch;

        // Secondary: fuzzy basename similarity
        var secondaryFuzzy = _secondary.Keys
            .Where(p => allowed.Contains(Path.GetExtension(p).ToLowerInvariant()))
            .Select(p => new { Path = p, Score = BasenameSimilarity(targetNameNoExt, Path.GetFileNameWithoutExtension(p)) })
            .Where(x => x.Score >= 0.70)
            .OrderByDescending(x => x.Score)
            .ThenByDescending(x => PathSimilarity(norm, x.Path))
            .Select(x => x.Path)
            .FirstOrDefault();
        if (secondaryFuzzy is not null) return secondaryFuzzy;

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

    private static double BasenameSimilarity(string a, string b)
    {
        if (string.IsNullOrWhiteSpace(a) || string.IsNullOrWhiteSpace(b)) return 0.0;
        var sa = Sanitize(a);
        var sb = Sanitize(b);
        if (sa.Equals(sb, StringComparison.OrdinalIgnoreCase)) return 1.0;
        if (sa.Contains(sb, StringComparison.OrdinalIgnoreCase) || sb.Contains(sa, StringComparison.OrdinalIgnoreCase)) return 0.9;
        // Levenshtein-based normalized similarity
        int dist = Levenshtein(sa.ToLowerInvariant(), sb.ToLowerInvariant());
        int maxLen = Math.Max(sa.Length, sb.Length);
        if (maxLen == 0) return 0.0;
        return 1.0 - ((double)dist / maxLen);
    }

    private static string Sanitize(string s)
    {
        // remove non-alphanumeric and collapse
        var chars = s.Where(ch => char.IsLetterOrDigit(ch)).ToArray();
        return new string(chars);
    }

    private static int Levenshtein(string a, string b)
    {
        int n = a.Length;
        int m = b.Length;
        if (n == 0) return m;
        if (m == 0) return n;
        var d = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++) d[i, 0] = i;
        for (int j = 0; j <= m; j++) d[0, j] = j;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);
            }
        }
        return d[n, m];
    }

    public bool ContainsPrimary(string path)
    {
        var norm = WowPath.Normalize(path);
        return _primary.ContainsKey(norm);
    }

    public bool ContainsSecondary(string path)
    {
        var norm = WowPath.Normalize(path);
        return _secondary.ContainsKey(norm);
    }
}
