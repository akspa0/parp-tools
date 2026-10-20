using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4CrossTileLinkRow(
    string AddressHypothesis, bool DifferentTile, byte RawType, bool NegativePathStart,
    long Links, long MissingTargetFile, long ReferenceOutOfRange, long Resolved,
    long SharedBoundaryEdge, long Reciprocal, long SelfReference,
    long CrossTileWithLocalAlternative, long PairedAddressSharedEdge, long PairedLocalSharedEdge,
    long PairedAddressReciprocal, long PairedLocalReciprocal);

/// <summary>Tests packed-tile address order with geometry/back-links; no renderer/matcher changes.</summary>
public static class Pm4CrossTileLinkProbe
{
    public static IReadOnlyList<Pm4CrossTileLinkRow> Analyze(IReadOnlyList<Pm4MapAuditInput> input)
    {
        var files = input.OrderBy(f => f.FileId, StringComparer.Ordinal).ToArray();
        var byTile = new Dictionary<(int X, int Y), int>();
        var offsets = new int[files.Length];
        var coordinates = new (int X, int Y)[files.Length];
        int totalSurfaces = 0;
        var polygons = new List<Vector3[]>();
        var surfaceFiles = new List<int>();
        for (int f = 0; f < files.Length; f++)
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(files[f].FileId, out int x, out int y))
                throw new ArgumentException("Every input needs filename tile coordinates.");
            if (!byTile.TryAdd((x, y), f)) throw new ArgumentException("Duplicate tile coordinates in one map.");
            coordinates[f] = (x, y);
            offsets[f] = totalSurfaces;
            var c = files[f].Document.KnownChunks;
            totalSurfaces += c.Msur.Count;
            for (int s = 0; s < c.Msur.Count; s++)
            {
                var raw = c.Msur[s];
                var points = new List<Vector3>();
                bool valid = raw.IndexCount >= 3 && (long)raw.MsviFirstIndex + raw.IndexCount <= c.Msvi.Count;
                if (valid)
                    for (long i = raw.MsviFirstIndex; i < (long)raw.MsviFirstIndex + raw.IndexCount; i++)
                    {
                        uint vi = c.Msvi[(int)i];
                        if (vi >= c.Msvt.Count) { valid = false; break; }
                        Vector3 p = c.Msvt[(int)vi];
                        if (!float.IsFinite(p.X) || !float.IsFinite(p.Y) || !float.IsFinite(p.Z)) { valid = false; break; }
                        points.Add(p);
                    }
                polygons.Add(valid ? points.ToArray() : []);
                surfaceFiles.Add(f);
            }
        }

        var result = new List<Pm4CrossTileLinkRow>();
        var localPairs = new HashSet<(int Source, int Target)>();
        foreach (string mode in new[] { "source-file-control", "swapped-byte-control", "low-byte-X-next-byte-Y" })
        {
            var counters = new Dictionary<(bool Cross, byte Type, bool Anchor), long[]>();
            var edges = new List<(int Source, int Target, byte Type, bool Anchor, int? LocalAlternative)>();
            foreach (var (file, f) in files.Select((file, i) => (file, i)))
            {
                var c = file.Document.KnownChunks;
                for (int s = 0; s < c.Msur.Count; s++)
                {
                    var surface = c.Msur[s];
                    long end = (long)surface._0x18 + surface.AttributeMask;
                    if (end > c.Mslk.Count) throw new InvalidDataException("Invalid proposed link window; cannot assert complete link-address probe.");
                    for (long k = surface._0x18; k < end; k++)
                    {
                        var link = c.Mslk[(int)k];
                        int low = (int)(link.LinkId & 255), high = (int)((link.LinkId >> 8) & 255);
                        var tile = mode == "source-file-control" ? coordinates[f]
                            : mode == "low-byte-X-next-byte-Y" ? (low, high) : (high, low);
                        var key = (tile != coordinates[f], link.TypeFlags, link.MspiFirstIndex < 0);
                        if (!counters.TryGetValue(key, out var n)) counters[key] = n = new long[12];
                        n[0]++;
                        if (!byTile.TryGetValue(tile, out int targetFile)) { n[1]++; continue; }
                        if (link.RefIndex >= files[targetFile].Document.KnownChunks.Msur.Count) { n[2]++; continue; }
                        int source = offsets[f] + s, target = offsets[targetFile] + link.RefIndex;
                        n[3]++;
                        if (source == target) n[6]++;
                        bool shared = SharesEdge(polygons[source], polygons[target]);
                        if (shared) n[4]++;
                        int? local = null;
                        if (mode == "low-byte-X-next-byte-Y" && targetFile != f && link.RefIndex < c.Msur.Count)
                        {
                            local = offsets[f] + link.RefIndex;
                            n[7]++;
                            if (shared) n[8]++;
                            if (SharesEdge(polygons[source], polygons[local.Value])) n[9]++;
                        }
                        edges.Add((source, target, link.TypeFlags, link.MspiFirstIndex < 0, local));
                    }
                }
            }
            var pairs = edges.Select(e => (e.Source, e.Target)).ToHashSet();
            if (mode == "source-file-control") localPairs = pairs;
            foreach (var e in edges)
            {
                var n = counters[(surfaceFiles[e.Source] != surfaceFiles[e.Target], e.Type, e.Anchor)];
                if (e.Source != e.Target && pairs.Contains((e.Target, e.Source)))
                    n[5]++;
                if (e.LocalAlternative.HasValue)
                {
                    if (pairs.Contains((e.Target, e.Source))) n[10]++;
                    if (e.Source != e.LocalAlternative.Value && localPairs.Contains((e.LocalAlternative.Value, e.Source))) n[11]++;
                }
            }
            foreach (var p in counters.OrderBy(p => p.Key.Cross).ThenBy(p => p.Key.Type).ThenBy(p => p.Key.Anchor))
                result.Add(new Pm4CrossTileLinkRow(mode, p.Key.Cross, p.Key.Type, p.Key.Anchor,
                    p.Value[0], p.Value[1], p.Value[2], p.Value[3], p.Value[4], p.Value[5], p.Value[6],
                    p.Value[7], p.Value[8], p.Value[9], p.Value[10], p.Value[11]));
        }
        return result;
    }

    private static bool SharesEdge(Vector3[] a, Vector3[] b)
    {
        const float squared = 0.0001f; // 0.01 yard tolerance, as the local geometry probe.
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
            {
                Vector3 p = a[i], q = a[(i + 1) % a.Length], r = b[j], s = b[(j + 1) % b.Length];
                if (Vector3.DistanceSquared(p, q) <= squared || Vector3.DistanceSquared(r, s) <= squared) continue;
                if ((Vector3.DistanceSquared(p, r) <= squared && Vector3.DistanceSquared(q, s) <= squared)
                    || (Vector3.DistanceSquared(p, s) <= squared && Vector3.DistanceSquared(q, r) <= squared)) return true;
            }
        return false;
    }
}
