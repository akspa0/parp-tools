using System;
using System.Collections.Generic;
using System.IO;

namespace AlphaWdtAnalyzer.Core.Assets;

public sealed class AssetInventory
{
    private readonly HashSet<string> _paths = new(StringComparer.OrdinalIgnoreCase);

    public AssetInventory(IEnumerable<string>? roots)
    {
        if (roots is null) return;
        foreach (var root in roots)
        {
            if (string.IsNullOrWhiteSpace(root)) continue;
            if (!Directory.Exists(root)) continue;
            IndexRoot(root);
        }
    }

    private void IndexRoot(string root)
    {
        var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".wmo", ".mdx", ".blp" };
        foreach (var file in Directory.EnumerateFiles(root, "*.*", SearchOption.AllDirectories))
        {
            var ext = Path.GetExtension(file);
            if (!exts.Contains(ext)) continue;
            var rel = GetRelativeNormalized(root, file);
            if (rel.Length == 0) continue;
            _paths.Add(rel);
        }
    }

    public bool Exists(string path)
    {
        var norm = WowPath.Normalize(path);
        return _paths.Contains(norm);
    }

    private static string GetRelativeNormalized(string root, string full)
    {
        try
        {
            var rel = Path.GetRelativePath(root, full);
            return WowPath.Normalize(rel);
        }
        catch
        {
            return string.Empty;
        }
    }
}
