using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4FieldDomainFit(string Scope, string Domain, long Values, long InRange);
public sealed record Pm4ReverseLinkFit(
    string Hypothesis, byte RawType, bool SourceCk24Zero, long Values, long MissingTargetTile,
    long TargetSurfaceOutOfRange, long CandidateInRange, long InTargetSurfaceWindow,
    long BackReferenceMatches, long WindowAndBackReferenceMatches);
public sealed record Pm4FieldDomainReport(
    int Files, long Links, int DistinctRawValues, long ZeroValues, long MaximumWordValues,
    long NonemptySurfaceWindows, long ConstantValueWindows,
    IReadOnlyList<Pm4FieldDomainFit> RangeFits, IReadOnlyList<Pm4ReverseLinkFit> ReverseFits);

/// <summary>
/// Tests raw MSLK +0x04 against index and reverse-link alternatives. Range fits are explicitly
/// not semantic matches. Does not rename or alter the underlying reader/matcher contracts.
/// </summary>
public static class Pm4FieldDomainProbe
{
    public static Pm4FieldDomainReport Analyze(IReadOnlyList<Pm4MapAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var files = input.OrderBy(f => f.FileId, StringComparer.Ordinal).ToArray();
        var byTile = new Dictionary<(int X, int Y), Pm4MapAuditInput>();
        foreach (var f in files)
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(f.FileId, out int x, out int y))
                throw new ArgumentException("Map input IDs need explicit tile coordinates.");
            if (!byTile.TryAdd((x, y), f)) throw new ArgumentException("Duplicate tile in map input.");
        }
        var values = new HashSet<uint>();
        var ranges = new Dictionary<(string Scope, string Domain), long[]>();
        var reverse = new Dictionary<(string Mode, byte Type, bool Zero), long[]>();
        long links = 0, zeros = 0, maxWords = 0, windows = 0, constant = 0;
        foreach (var f in files)
        {
            var c = f.Document.KnownChunks;
            Pm4CoordinateService.TryParseTileCoordinates(f.FileId, out int sourceX, out int sourceY);
            foreach (var link in c.Mslk)
            {
                uint value = link.GroupObjectId;
                links++; values.Add(value);
                if (value == 0) zeros++;
                if (value == uint.MaxValue) maxWords++;
                Range("source-file", c, value);
                var targetTile = ((int)(link.LinkId & 255), (int)((link.LinkId >> 8) & 255));
                if (byTile.TryGetValue(targetTile, out var t)) Range("addressed-file", t.Document.KnownChunks, value);
            }
            for (int owner = 0; owner < c.Msur.Count; owner++)
            {
                var s = c.Msur[owner];
                long end = (long)s._0x18 + s.AttributeMask;
                if (end > c.Mslk.Count) throw new InvalidDataException("Proposed surface/link window is invalid.");
                if (s.AttributeMask == 0) continue;
                windows++;
                uint first = c.Mslk[(int)s._0x18].GroupObjectId;
                bool same = true;
                for (long k = s._0x18; k < end; k++)
                {
                    var link = c.Mslk[(int)k];
                    same &= link.GroupObjectId == first;
                    foreach (string mode in new[] { "addressed-raw-index", "addressed-index-plus-1-control", "addressed-index-plus-17-control", "source-file-control" })
                    {
                        var key = (mode, link.TypeFlags, s.Ck24 == 0);
                        if (!reverse.TryGetValue(key, out var n)) reverse[key] = n = new long[7];
                        n[0]++;
                        var tile = mode == "source-file-control" ? (sourceX, sourceY)
                            : ((int)(link.LinkId & 255), (int)((link.LinkId >> 8) & 255));
                        if (!byTile.TryGetValue(tile, out var targetFile)) { n[1]++; continue; }
                        var tc = targetFile.Document.KnownChunks;
                        if (link.RefIndex >= tc.Msur.Count) { n[2]++; continue; }
                        ulong candidateIndex = link.GroupObjectId;
                        if (mode == "addressed-index-plus-1-control") candidateIndex++;
                        if (mode == "addressed-index-plus-17-control") candidateIndex += 17;
                        if (candidateIndex >= (ulong)tc.Mslk.Count) continue;
                        n[3]++;
                        var targetSurface = tc.Msur[link.RefIndex];
                        bool inWindow = candidateIndex >= targetSurface._0x18
                            && candidateIndex < (ulong)targetSurface._0x18 + targetSurface.AttributeMask;
                        if (inWindow) n[4]++;
                        var candidate = tc.Mslk[(int)candidateIndex];
                        bool back = candidate.RefIndex == owner && (candidate.LinkId & 255) == sourceX
                            && ((candidate.LinkId >> 8) & 255) == sourceY;
                        if (back) n[5]++;
                        if (back && inWindow) n[6]++;
                    }
                }
                if (same) constant++;
            }
        }
        return new Pm4FieldDomainReport(files.Length, links, values.Count, zeros, maxWords, windows, constant,
            ranges.OrderBy(p => p.Key.Scope, StringComparer.Ordinal).ThenBy(p => p.Key.Domain, StringComparer.Ordinal)
                .Select(p => new Pm4FieldDomainFit(p.Key.Scope, p.Key.Domain, p.Value[0], p.Value[1])).ToArray(),
            reverse.OrderBy(p => p.Key.Mode, StringComparer.Ordinal).ThenBy(p => p.Key.Type).ThenBy(p => p.Key.Zero)
                .Select(p => new Pm4ReverseLinkFit(p.Key.Mode, p.Key.Type, p.Key.Zero,
                    p.Value[0], p.Value[1], p.Value[2], p.Value[3], p.Value[4], p.Value[5], p.Value[6])).ToArray());

        void Range(string scope, Pm4KnownChunkSet c, uint value)
        {
            foreach (var domain in new (string Name, int Count)[]
            {
                ("MSUR", c.Msur.Count), ("MSLK", c.Mslk.Count), ("MSVI", c.Msvi.Count),
                ("MSVT", c.Msvt.Count), ("MSPI", c.Mspi.Count), ("MSPV", c.Mspv.Count),
                ("MSCN", c.Mscn.Count), ("MPRL", c.Mprl.Count), ("MPRR", c.Mprr.Count),
            })
            {
                var key = (scope, domain.Name);
                if (!ranges.TryGetValue(key, out var n)) ranges[key] = n = new long[2];
                n[0]++;
                if (value < domain.Count) n[1]++;
            }
        }
    }
}
