using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core;

public sealed class ListfileLoader
{
    public IReadOnlyDictionary<int, string> ById => _byId;
    public IReadOnlyDictionary<string, int> IdByPathNormalized => _idByPath;

    private readonly Dictionary<int, string> _byId = new();
    private readonly Dictionary<string, int> _idByPath = new(StringComparer.OrdinalIgnoreCase);

    public static string NormalizePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) return string.Empty;
        var p = path.Replace('\\', '/');
        // WoW assets are generally case-insensitive
        return p.Trim();
    }

    public void Load(string csvPath)
    {
        if (!File.Exists(csvPath)) throw new FileNotFoundException("Listfile not found", csvPath);
        foreach (var line in File.ReadLines(csvPath))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0) continue;
            var semi = trimmed.IndexOf(';');
            if (semi <= 0 || semi >= trimmed.Length - 1) continue;

            var idStr = trimmed.Substring(0, semi).Trim();
            var path = NormalizePath(trimmed.Substring(semi + 1));
            if (!int.TryParse(idStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var id)) continue;

            _byId[id] = path;
            if (!_idByPath.ContainsKey(path))
            {
                _idByPath[path] = id;
            }
        }
    }

    public bool TryGetIdByPath(string path, out int id)
    {
        var norm = NormalizePath(path);
        // direct
        if (_idByPath.TryGetValue(norm, out id)) return true;

        // fuzzy: allow different case and backslash usage already normalized; try m2/mdx swap
        var swapExt = SwapM2Mdx(norm);
        if (swapExt is not null && _idByPath.TryGetValue(swapExt, out id)) return true;

        id = 0;
        return false;
    }

    private static string? SwapM2Mdx(string norm)
    {
        if (norm.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
            return norm.Substring(0, norm.Length - 3) + ".mdx";
        if (norm.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
            return norm.Substring(0, norm.Length - 4) + ".m2";
        return null;
    }
}
