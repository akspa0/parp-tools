using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4PointReferenceRow(
    string Candidate, byte RawType, bool SourceCk24Zero, long Links, long CandidateAvailable,
    long TargetAvailable, long SourceVertexTight, long SourceBoundaryTight, long SourceBoundaryLoose,
    long TargetBoundaryTight, long BothBoundariesTight, long AuxiliaryBoundaryTight, long SourceCentroidTight);
public sealed record Pm4SubtypeEdgeRow(
    string Hypothesis, byte RawType, bool SourceCk24Zero, long Links, long TargetAvailable,
    long ValidOrdinal, long MatchesOtherBoundary, long AuxiliaryAvailable,
    long MatchesAuxiliaryEdge, long EndpointsOnAuxiliaryBoundary,
    long ValidVerticalPlane, long AuxiliaryInSelectedVerticalPlane);
public sealed record Pm4PointReferenceReport(
    int Files, long WindowLinks, long MissingTargetLinks, long InvalidTargetReferences,
    float TightTolerance, float LooseTolerance,
    IReadOnlyList<Pm4PointReferenceRow> PointRows, IReadOnlyList<Pm4SubtypeEdgeRow> SubtypeRows);

/// <summary>Read-only spatial tests of raw +0x04 and an independent raw subtype edge-ordinal hypothesis.</summary>
public static class Pm4PointReferenceProbe
{
    private static readonly string[] Candidates =
    [
        "source-MSCN", "source-MSCN-plus-1-control", "source-MSCN-plus-17-control", "source-MSCN-hashed-control",
        "source-MSVI-to-MSVT", "source-MSVI-plus-17-control", "source-MSVT", "source-MSPV", "source-MSPI-to-MSPV",
        "addressed-MSCN", "source-MPRL-raw", "source-MPRL-ZXY",
    ];
    private static readonly string[] Ordinals =
    ["source-zero-based", "source-one-based", "source-shift-1-control", "target-zero-based", "target-one-based", "target-shift-1-control"];

    public static Pm4PointReferenceReport Analyze(IReadOnlyList<Pm4MapAuditInput> input, bool includePointCandidates = true)
    {
        ArgumentNullException.ThrowIfNull(input);
        var files = new Dictionary<(int X, int Y), CachedFile>();
        foreach (var file in input)
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(file.FileId, out int x, out int y))
                throw new ArgumentException("File IDs must retain tile coordinates.");
            if (!files.TryAdd((x, y), Cache(file.Document.KnownChunks))) throw new ArgumentException("Duplicate tile in one map.");
        }
        var pointRows = new Dictionary<(string Name, byte Type, bool Zero), long[]>();
        var ordinalRows = new Dictionary<(string Name, byte Type, bool Zero), long[]>();
        long total = 0, missing = 0, badTarget = 0;
        const float tight = 0.01f, loose = 0.25f;
        foreach (var file in files.Values)
        {
            var c = file.Chunks;
            for (int owner = 0; owner < c.Msur.Count; owner++)
            {
                var surface = c.Msur[owner];
                long end = (long)surface._0x18 + surface.AttributeMask;
                if (end > c.Mslk.Count) throw new InvalidDataException("Invalid proposed MSUR/MSLK window.");
                Vector3[] source = file.Polygons[owner];
                for (long k = surface._0x18; k < end; k++)
                {
                    total++;
                    var link = c.Mslk[(int)k];
                    files.TryGetValue(((int)(link.LinkId & 255), (int)((link.LinkId >> 8) & 255)), out var targetFile);
                    Vector3[]? target = null;
                    if (targetFile is null) missing++;
                    else if (link.RefIndex >= targetFile.Polygons.Length) badTarget++;
                    else target = targetFile.Polygons[link.RefIndex];
                    Vector3[] auxiliary = Path(c, link);
                    foreach (string mode in includePointCandidates ? Candidates : Array.Empty<string>())
                    {
                        var key = (mode, link.TypeFlags, surface.Ck24 == 0);
                        if (!pointRows.TryGetValue(key, out var n)) pointRows[key] = n = new long[10];
                        n[0]++;
                        if (target is not null) n[2]++;
                        Vector3? candidate = Candidate(mode, c, targetFile?.Chunks, link.GroupObjectId, (uint)owner, (uint)k);
                        if (!candidate.HasValue || !Finite(candidate.Value)) continue;
                        n[1]++;
                        Vector3 p = candidate.Value;
                        bool sourceTight = BoundarySquared(p, source) <= tight * tight;
                        bool targetTight = target is not null && BoundarySquared(p, target) <= tight * tight;
                        if (source.Any(v => Vector3.DistanceSquared(v, p) <= tight * tight)) n[3]++;
                        if (sourceTight) n[4]++;
                        if (BoundarySquared(p, source) <= loose * loose) n[5]++;
                        if (targetTight) n[6]++;
                        if (sourceTight && targetTight) n[7]++;
                        if (BoundarySquared(p, auxiliary) <= tight * tight) n[8]++;
                        if (Vector3.DistanceSquared(p, file.Centroids[owner]) <= tight * tight) n[9]++;
                    }
                    foreach (string mode in Ordinals)
                    {
                        var key = (mode, link.TypeFlags, surface.Ck24 == 0);
                        if (!ordinalRows.TryGetValue(key, out var n)) ordinalRows[key] = n = new long[9];
                        n[0]++;
                        bool useTarget = mode.StartsWith("target", StringComparison.Ordinal);
                        if (target is not null) n[1]++;
                        if (useTarget && target is null) continue;
                        Vector3[] polygon = useTarget ? target! : source;
                        Vector3[]? other = useTarget ? source : target;
                        int ordinal = link.Subtype;
                        if (mode.Contains("one-based", StringComparison.Ordinal)) ordinal--;
                        if (ordinal < 0 || ordinal >= polygon.Length) continue;
                        if (mode.Contains("shift-1", StringComparison.Ordinal)) ordinal = (ordinal + 1) % polygon.Length;
                        n[2]++;
                        Vector3 a = polygon[ordinal], b = polygon[(ordinal + 1) % polygon.Length];
                        if (other != null && EdgeMatches(a, b, other, tight * tight)) n[3]++;
                        if (auxiliary.Length >= 2)
                        {
                            n[4]++;
                            if (EdgeMatches(a, b, auxiliary, tight * tight)) n[5]++;
                            if (BoundarySquared(a, auxiliary) <= tight * tight && BoundarySquared(b, auxiliary) <= tight * tight) n[6]++;
                            float dx = b.X - a.X, dy = b.Y - a.Y, lengthSquared = dx * dx + dy * dy;
                            if (lengthSquared > 1e-12f)
                            {
                                n[7]++;
                                bool inPlane = true;
                                foreach (var p in auxiliary)
                                {
                                    float cross = (p.X - a.X) * dy - (p.Y - a.Y) * dx;
                                    if (cross * cross > tight * tight * lengthSquared) { inPlane = false; break; }
                                }
                                if (inPlane) n[8]++;
                            }
                        }
                    }
                }
            }
        }
        return new Pm4PointReferenceReport(files.Count, total, missing, badTarget, tight, loose,
            pointRows.OrderBy(p => p.Key.Name, StringComparer.Ordinal).ThenBy(p => p.Key.Type).ThenBy(p => p.Key.Zero)
                .Select(p => new Pm4PointReferenceRow(p.Key.Name, p.Key.Type, p.Key.Zero, p.Value[0], p.Value[1], p.Value[2],
                    p.Value[3], p.Value[4], p.Value[5], p.Value[6], p.Value[7], p.Value[8], p.Value[9])).ToArray(),
            ordinalRows.OrderBy(p => p.Key.Name, StringComparer.Ordinal).ThenBy(p => p.Key.Type).ThenBy(p => p.Key.Zero)
                .Select(p => new Pm4SubtypeEdgeRow(p.Key.Name, p.Key.Type, p.Key.Zero, p.Value[0], p.Value[1], p.Value[2], p.Value[3],
                    p.Value[4], p.Value[5], p.Value[6], p.Value[7], p.Value[8])).ToArray());
    }

    private static Vector3? Candidate(string mode, Pm4KnownChunkSet c, Pm4KnownChunkSet? target, uint raw, uint owner, uint linkIndex)
    {
        ulong i = raw;
        if (mode.Contains("plus-1-", StringComparison.Ordinal)) i++;
        if (mode.Contains("plus-17-", StringComparison.Ordinal)) i += 17;
        if (mode == "source-MSCN-hashed-control")
        {
            if (c.Mscn.Count == 0) return null;
            i = unchecked(owner * 2654435761u ^ linkIndex * 2246822519u ^ 3266489917u) % (uint)c.Mscn.Count;
        }
        if (mode.StartsWith("source-MSCN", StringComparison.Ordinal)) return At(c.Mscn, i);
        if (mode.StartsWith("source-MSVI", StringComparison.Ordinal)) return i < (ulong)c.Msvi.Count ? At(c.Msvt, c.Msvi[(int)i]) : null;
        if (mode == "source-MSVT") return At(c.Msvt, i);
        if (mode == "source-MSPV") return At(c.Mspv, i);
        if (mode == "source-MSPI-to-MSPV") return i < (ulong)c.Mspi.Count ? At(c.Mspv, c.Mspi[(int)i]) : null;
        if (mode == "addressed-MSCN") return target is null ? null : At(target.Mscn, i);
        if (i >= (ulong)c.Mprl.Count) return null;
        Vector3 p = c.Mprl[(int)i].Position;
        // ZXY matches the existing helper's MPRL-to-MSVT raw-frame composition; tested, not presumed.
        return mode == "source-MPRL-ZXY" ? new Vector3(p.Z, p.X, p.Y) : p;
    }

    private static Vector3? At(IReadOnlyList<Vector3> points, ulong index) => index < (ulong)points.Count ? points[(int)index] : null;
    private static bool Finite(Vector3 p) => float.IsFinite(p.X) && float.IsFinite(p.Y) && float.IsFinite(p.Z);

    private static CachedFile Cache(Pm4KnownChunkSet c)
    {
        var polygons = new Vector3[c.Msur.Count][];
        var centroids = new Vector3[c.Msur.Count];
        for (int i = 0; i < polygons.Length; i++)
        {
            var s = c.Msur[i];
            if ((long)s.MsviFirstIndex + s.IndexCount > c.Msvi.Count) throw new InvalidDataException("Invalid surface geometry window.");
            var p = new List<Vector3>();
            for (int j = 0; j < s.IndexCount; j++)
            {
                uint vi = c.Msvi[(int)s.MsviFirstIndex + j];
                if (vi >= c.Msvt.Count || !Finite(c.Msvt[(int)vi])) throw new InvalidDataException("Invalid surface vertex.");
                p.Add(c.Msvt[(int)vi]);
            }
            polygons[i] = p.ToArray();
            centroids[i] = p.Count == 0 ? new Vector3(float.NaN) : p.Aggregate(Vector3.Zero, (a, b) => a + b) / p.Count;
        }
        return new CachedFile(c, polygons, centroids);
    }

    private static Vector3[] Path(Pm4KnownChunkSet c, Pm4MslkEntry l)
    {
        if (l.MspiFirstIndex < 0 || (long)l.MspiFirstIndex + l.MspiIndexCount > c.Mspi.Count) return [];
        var p = new List<Vector3>();
        for (int i = 0; i < l.MspiIndexCount; i++)
        {
            uint vi = c.Mspi[l.MspiFirstIndex + i];
            if (vi >= c.Mspv.Count || !Finite(c.Mspv[(int)vi])) return [];
            p.Add(c.Mspv[(int)vi]);
        }
        return p.ToArray();
    }

    private static float BoundarySquared(Vector3 p, Vector3[] boundary)
    {
        float best = float.PositiveInfinity;
        for (int i = 0; i < boundary.Length; i++)
        {
            Vector3 a = boundary[i], delta = boundary[(i + 1) % boundary.Length] - a;
            float length = delta.LengthSquared();
            if (length <= 1e-12f) continue;
            float t = Math.Clamp(Vector3.Dot(p - a, delta) / length, 0, 1);
            best = Math.Min(best, Vector3.DistanceSquared(p, a + t * delta));
        }
        return best;
    }

    private static bool EdgeMatches(Vector3 a, Vector3 b, Vector3[] other, float squared)
    {
        if (Vector3.DistanceSquared(a, b) <= squared) return false;
        for (int j = 0; j < other.Length; j++)
        {
            Vector3 p = other[j], q = other[(j + 1) % other.Length];
            if ((Vector3.DistanceSquared(a, p) <= squared && Vector3.DistanceSquared(b, q) <= squared)
                || (Vector3.DistanceSquared(a, q) <= squared && Vector3.DistanceSquared(b, p) <= squared)) return true;
        }
        return false;
    }

    private sealed record CachedFile(Pm4KnownChunkSet Chunks, Vector3[][] Polygons, Vector3[] Centroids);
}
