using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4EdgeIdentityRow(
    string Stratum, byte RawType, bool SourceCk24Zero,
    long Links, long UniqueReverse, long MultipleReverse, long NoReverse,
    long UniqueReverseSameValue, long UniqueReverseDifferentValue,
    long NextRecordSameValue, long NextRecordDifferentValue,
    long ReusedValueLinks, long ReusedValueDistinctEdges);

public sealed record Pm4EdgeIdentityReport(
    int Files, long Links, long UniqueReversePairs, long MultipleReverseLinks, long NoReverseLinks,
    IReadOnlyList<Pm4EdgeIdentityRow> Rows);

/// <summary>
/// Tests whether raw MSLK +0x04 behaves as a connection/edge identifier. Reverse records are
/// resolved independently of the field under test; equality alone is never called object identity.
/// </summary>
public static class Pm4EdgeIdentityProbe
{
    public static Pm4EdgeIdentityReport Analyze(IReadOnlyList<Pm4MapAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var files = new Dictionary<(int X, int Y), Pm4MapAuditInput>();
        foreach (var f in input)
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(f.FileId, out int x, out int y))
                throw new ArgumentException("File IDs must retain tile coordinates.");
            if (!files.TryAdd((x, y), f)) throw new ArgumentException("Duplicate tile in one map.");
        }

        // Resolve every directed record to a global (source surface, target surface) key.
        var directed = new List<Directed>();
        foreach (var f in files.Values)
        {
            Pm4CoordinateService.TryParseTileCoordinates(f.FileId, out int sx, out int sy);
            var c = f.Document.KnownChunks;
            for (int owner = 0; owner < c.Msur.Count; owner++)
            {
                var s = c.Msur[owner];
                long end = (long)s._0x18 + s.AttributeMask;
                if (end > c.Mslk.Count) throw new InvalidDataException("Invalid proposed MSUR/MSLK window.");
                for (long k = s._0x18; k < end; k++)
                {
                    var link = c.Mslk[(int)k];
                    var tile = ((int)(link.LinkId & 255), (int)((link.LinkId >> 8) & 255));
                    files.TryGetValue(tile, out var targetFile);
                    bool targetValid = targetFile is not null
                        && link.RefIndex < targetFile.Document.KnownChunks.Msur.Count;
                    directed.Add(new Directed(f.FileId, owner, (int)k, link,
                        targetFile?.FileId, targetValid ? link.RefIndex : null, targetValid, s.Ck24 == 0));
                }
            }
        }

        // Reverse lookup is keyed only by independently resolved endpoints, never by +0x04.
        var byEndpoints = directed.Where(d => d.TargetValid)
            .GroupBy(d => (d.SourceFile, d.SourceSurface, d.TargetFile, d.TargetSurface))
            .ToDictionary(g => g.Key, g => g.ToArray());
        var byFileAndIndex = directed.GroupBy(d => (d.SourceFile, d.LinkIndex))
            .ToDictionary(g => g.Key, g => g.First());
        var rows = new Dictionary<(string, byte, bool), long[]>();
        long uniqueReverse = 0, multipleReverse = 0, noReverse = 0;
        foreach (var d in directed)
        {
            Directed[]? reverse = null;
            if (d.TargetValid && byEndpoints.TryGetValue((d.TargetFile!, d.TargetSurface!.Value, d.SourceFile, d.SourceSurface), out var candidates))
                reverse = candidates;
            string stratum = reverse is null ? "no-reverse" : reverse.Length == 1 ? "unique-reverse" : "multiple-reverse";
            if (stratum == "unique-reverse") uniqueReverse++;
            else if (stratum == "multiple-reverse") multipleReverse++;
            else noReverse++;
            var rowKey = (stratum, d.Link.TypeFlags, d.SourceCk24Zero);
            if (!rows.TryGetValue(rowKey, out var n)) rows[rowKey] = n = new long[10];
            n[0]++;
            if (stratum == "unique-reverse")
            {
                n[1]++;
                if (reverse![0].Link.GroupObjectId == d.Link.GroupObjectId) n[2]++;
                else n[3]++;
            }
            else if (stratum == "multiple-reverse") n[4]++;
            else n[5]++;

            // Wrong-next-record control: the following record in the same file, if any.
            if (byFileAndIndex.TryGetValue((d.SourceFile, d.LinkIndex + 1), out var next))
            {
                if (next.Link.GroupObjectId == d.Link.GroupObjectId) n[6]++;
                else n[7]++;
            }
        }

        // Per-file reuse of +0x04 across distinct directed source-edge keys.
        foreach (var fileGroup in directed.GroupBy(d => d.SourceFile))
        {
            foreach (var valueGroup in fileGroup.GroupBy(d => d.Link.GroupObjectId))
            {
                var edges = valueGroup.Select(d => (d.SourceSurface, d.Link.Subtype)).Distinct().Count();
                if (edges <= 1) continue;
                var rowKey = ("reused-value", valueGroup.First().Link.TypeFlags, valueGroup.First().SourceCk24Zero);
                if (!rows.TryGetValue(rowKey, out var n)) rows[rowKey] = n = new long[10];
                n[8] += valueGroup.Count();
                n[9] += edges;
            }
        }

        return new Pm4EdgeIdentityReport(files.Count, directed.Count, uniqueReverse, multipleReverse, noReverse,
            rows.OrderBy(p => p.Key.Item1, StringComparer.Ordinal).ThenBy(p => p.Key.Item2).ThenBy(p => p.Key.Item3)
                .Select(p => new Pm4EdgeIdentityRow(p.Key.Item1, p.Key.Item2, p.Key.Item3,
                    p.Value[0], p.Value[1], p.Value[4], p.Value[5], p.Value[2], p.Value[3],
                    p.Value[6], p.Value[7], p.Value[8], p.Value[9])).ToArray());
    }

    private sealed record Directed(
        string SourceFile, int SourceSurface, int LinkIndex,
        Pm4MslkEntry Link, string? TargetFile, int? TargetSurface, bool TargetValid, bool SourceCk24Zero);
}
