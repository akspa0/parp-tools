using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core.Dbc;

public sealed class AreaRow
{
    public int Id { get; init; }
    public string Name { get; init; } = string.Empty;
}

public sealed class AreaCrosswalk
{
    public List<(AreaRow Alpha, AreaRow Lk, string Match)> Matches { get; } = new();
    public List<AreaRow> UnmatchedAlpha { get; } = new();
    public List<AreaRow> UnmatchedLk { get; } = new();
}

public static class AreaTableAnalyzer
{
    public static AreaCrosswalk Compare(string alphaAreaDbc, string lkAreaDbc)
    {
        var alpha = Load(alphaAreaDbc);
        var lk = Load(lkAreaDbc);

        var res = new AreaCrosswalk();

        // Index by normalized name
        string Norm(string s) => NormalizeName(s);

        var lkByNorm = lk.GroupBy(a => Norm(a.Name))
                         .ToDictionary(g => g.Key, g => g.ToList());

        var matchedLk = new HashSet<int>();

        foreach (var a in alpha)
        {
            var key = Norm(a.Name);
            if (string.IsNullOrWhiteSpace(key))
            {
                res.UnmatchedAlpha.Add(a);
                continue;
            }

            if (lkByNorm.TryGetValue(key, out var cands) && cands.Count > 0)
            {
                // choose first candidate for now
                var best = cands.FirstOrDefault(l => !matchedLk.Contains(l.Id)) ?? cands[0];
                matchedLk.Add(best.Id);
                var matchType = string.Equals(a.Name, best.Name, StringComparison.Ordinal)
                    ? "exact"
                    : (string.Equals(a.Name, best.Name, StringComparison.OrdinalIgnoreCase) ? "case_only" : "normalized");
                res.Matches.Add((a, best, matchType));
            }
            else
            {
                res.UnmatchedAlpha.Add(a);
            }
        }

        // Unmatched LK
        foreach (var l in lk)
        {
            if (!matchedLk.Contains(l.Id)) res.UnmatchedLk.Add(l);
        }

        return res;
    }

    private static List<AreaRow> Load(string path)
    {
        var tbl = RawDbcParser.Parse(path);
        var rows = new List<AreaRow>(tbl.RecordCount);
        int idIdx = 0; // heuristic: first field is ID in many DBCs
        foreach (var row in tbl.Rows)
        {
            var name = FirstNonEmpty(row.GuessedStrings) ?? string.Empty;
            var id = row.Fields.Length > idIdx ? (int)row.Fields[idIdx] : -1;
            rows.Add(new AreaRow { Id = id, Name = name });
        }
        return rows;
    }

    private static string? FirstNonEmpty(string?[] arr)
    {
        foreach (var s in arr)
        {
            if (!string.IsNullOrWhiteSpace(s)) return s!.Trim();
        }
        return null;
    }

    public static string NormalizeName(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return string.Empty;
        var span = s.AsSpan().ToString().Trim().ToLowerInvariant();
        // remove characters that often vary between eras
        var filtered = new string(span.Where(ch => char.IsLetterOrDigit(ch) || ch == '/' || ch == '_').ToArray());
        // collapse multiple underscores
        while (filtered.Contains("__", StringComparison.Ordinal)) filtered = filtered.Replace("__", "_", StringComparison.Ordinal);
        return filtered;
    }
}
