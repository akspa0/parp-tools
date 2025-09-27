using DBCD;
using DBCD.Providers;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;

namespace DBCTool.V2.IO;

public static class DbdcHelper
{
    public static IDBCDStorage LoadTable(string table, string canonicalBuild, string dbcDir, FilesystemDBDProvider dbdProvider, DBCD.Locale locale)
    {
        var provider = new FilesystemDBCProvider(dbcDir, useCache: true);
        var dbcd = new DBCD.DBCD(provider, dbdProvider);
        try { return dbcd.Load(table, canonicalBuild, locale); }
        catch { return dbcd.Load(table, canonicalBuild, DBCD.Locale.None); }
    }

    public static string DetectIdColumn(IDBCDStorage storage)
    {
        try
        {
            var cols = storage.AvailableColumns ?? Array.Empty<string>();
            string[] prefers = new[] { "ID", "Id", "MapID", "MapId", "m_ID" };
            foreach (var p in prefers)
            {
                var match = cols.FirstOrDefault(x => string.Equals(x, p, StringComparison.OrdinalIgnoreCase));
                if (!string.IsNullOrEmpty(match)) return match;
            }
            var anyId = cols.FirstOrDefault(x => x.EndsWith("ID", StringComparison.OrdinalIgnoreCase));
            return anyId ?? string.Empty;
        }
        catch { return string.Empty; }
    }

    public static string DetectColumn(IDBCDStorage storage, params string[] preferred)
    {
        try
        {
            var cols = storage.AvailableColumns ?? Array.Empty<string>();
            foreach (var c in preferred)
            {
                var match = cols.FirstOrDefault(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase));
                if (!string.IsNullOrEmpty(match)) return match;
            }
            var any = cols.FirstOrDefault(x => x.IndexOf("name", StringComparison.OrdinalIgnoreCase) >= 0);
            return any ?? (preferred.Length > 0 ? preferred[0] : string.Empty);
        }
        catch { return preferred.Length > 0 ? preferred[0] : string.Empty; }
    }

    public static string FirstNonEmpty(params string[] vals)
    {
        foreach (var v in vals)
        {
            if (!string.IsNullOrWhiteSpace(v)) return v;
        }
        return string.Empty;
    }

    public static T SafeField<T>(DBCDRow row, string col)
    {
        try { return row.Field<T>(col); } catch { return default!; }
    }

    public static string DirToken(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return string.Empty;
        var parts = s.Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Length > 0 ? parts[^1] : s.Trim();
    }

    public static string Norm(string s) => (s ?? string.Empty).Trim().ToLowerInvariant();

    // Stronger normalization for matching across builds: lowercased, remove punctuation/spaces,
    // and apply known alias corrections.
    private static readonly Dictionary<string, string> NameAliases = new(StringComparer.OrdinalIgnoreCase)
    {
        // Alpha → LK spelling corrections
        ["aszhara"] = "azshara",
        ["shadowfang"] = "shadowfang keep",
        ["south seas"] = "south seas unused",
    };

    // Additional aliases provided via config/runtime
    private static readonly Dictionary<string, string> ExtraAliases = new(StringComparer.OrdinalIgnoreCase);

    public static string NormKey(string s)
    {
        var n = Norm(s);
        if (string.IsNullOrEmpty(n)) return string.Empty;
        // Drop leading article 'the '
        if (n.StartsWith("the ")) n = n.Substring(4).TrimStart();
        if (ExtraAliases.TryGetValue(n, out var mapped)) n = mapped;
        else if (NameAliases.TryGetValue(n, out mapped)) n = mapped;
        var sb = new StringBuilder(n.Length);
        foreach (var ch in n)
        {
            if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) sb.Append(ch);
            // drop spaces, punctuation, etc.
        }
        return sb.ToString();
    }

    public static void AddNameAliases(IEnumerable<KeyValuePair<string, string>> pairs)
    {
        if (pairs is null) return;
        foreach (var kv in pairs)
        {
            var k = (kv.Key ?? string.Empty).Trim();
            var v = (kv.Value ?? string.Empty).Trim();
            if (k.Length == 0 || v.Length == 0) continue;
            ExtraAliases[k.ToLowerInvariant()] = v.ToLowerInvariant();
        }
    }

    public static void AddNameAlias(string from, string to) => AddNameAliases(new[] { new KeyValuePair<string, string>(from, to) });

    public static string Csv(string s)
    {
        if (s is null) return string.Empty;
        if (s.IndexOfAny(new[] { '"', ',', '\n', '\r' }) >= 0) return '"' + s.Replace("\"", "\"\"") + '"';
        return s;
    }

    public static Dictionary<int, string> BuildMapNames(IDBCDStorage mapStorage)
    {
        string idCol = DetectIdColumn(mapStorage);
        var dict = new Dictionary<int, string>();
        foreach (var k in mapStorage.Keys)
        {
            var row = mapStorage[k];
            string dirRaw = SafeField<string>(row, "Directory");
            string dirTok = DirToken(dirRaw);
            string name = FirstNonEmpty(
                SafeField<string>(row, "MapName_lang"),
                SafeField<string>(row, "MapName"),
                SafeField<string>(row, "InternalName"),
                dirTok
            );
            int mapId = !string.IsNullOrWhiteSpace(idCol) ? SafeField<int>(row, idCol) : k;
            dict[mapId] = name ?? string.Empty;
        }
        return dict;
    }
}
