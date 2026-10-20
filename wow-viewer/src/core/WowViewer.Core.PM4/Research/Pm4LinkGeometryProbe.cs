using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4LinkGeometryRow(
    string Relation, string Ck24Relation, byte RawType, byte RawSubtype, bool NegativePathStart,
    long Count, long SharedVertex, long SharedBoundaryEdge, long SelfReference,
    long ValidPathWindows, long PathTouchesOwner, long PathTouchesReference,
    long PathSharesOwnerEdge, long PathSharesReferenceEdge);

public sealed record Pm4LinkReferenceException(int OwnerSurface, int LinkIndex, Pm4MslkEntry Raw);

public sealed record Pm4LinkGeometryResult(
    int Surfaces, int Links, int InvalidPolygonWindows, int InvalidLinkWindows,
    int InvalidSurfaceReferences, IReadOnlyList<Pm4LinkGeometryRow> Rows,
    IReadOnlyList<Pm4LinkReferenceException> OutOfRangeReferences);

/// <summary>
/// Geometry check of the proposed MSUR-window/MSLK-reference relation, with two wrong-target
/// controls. Does not interpret group/type names or assign object boundaries.
/// </summary>
public static class Pm4LinkGeometryProbe
{
    public static Pm4LinkGeometryResult Analyze(Pm4ResearchDocument document, float epsilon = 0.01f)
    {
        ArgumentNullException.ThrowIfNull(document);
        if (!float.IsFinite(epsilon) || epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        var c = document.KnownChunks;
        var polygons = new Vector3[c.Msur.Count][];
        int invalidPolygons = 0, invalidWindows = 0, invalidReferences = 0;
        for (int i = 0; i < polygons.Length; i++)
        {
            var s = c.Msur[i];
            long end = (long)s.MsviFirstIndex + s.IndexCount;
            var points = new List<Vector3>();
            bool valid = end <= c.Msvi.Count && s.IndexCount >= 3;
            if (valid)
                for (long k = s.MsviFirstIndex; k < end; k++)
                {
                    uint vi = c.Msvi[(int)k];
                    if (vi >= c.Msvt.Count) { valid = false; break; }
                    Vector3 p = c.Msvt[(int)vi];
                    if (!float.IsFinite(p.X) || !float.IsFinite(p.Y) || !float.IsFinite(p.Z)) { valid = false; break; }
                    points.Add(p);
                }
            polygons[i] = valid ? points.ToArray() : [];
            if (!valid) invalidPolygons++;
        }
        var rows = new Dictionary<(string Relation, string Ck, byte Type, byte Subtype, bool Anchor), long[]>();
        var exceptions = new List<Pm4LinkReferenceException>();
        for (int i = 0; i < c.Msur.Count; i++)
        {
            var s = c.Msur[i];
            long end = (long)s._0x18 + s.AttributeMask;
            if (end > c.Mslk.Count) { invalidWindows++; continue; }
            for (long k = s._0x18; k < end; k++)
            {
                var link = c.Mslk[(int)k];
                int target = link.RefIndex;
                if (target >= c.Msur.Count)
                {
                    invalidReferences++;
                    exceptions.Add(new Pm4LinkReferenceException(i, (int)k, link));
                    continue;
                }
                Vector3[]? path = null;
                if (link.MspiFirstIndex >= 0 && link.MspiIndexCount >= 2
                    && (long)link.MspiFirstIndex + link.MspiIndexCount <= c.Mspi.Count)
                {
                    var points = new List<Vector3>();
                    for (int j = 0; j < link.MspiIndexCount; j++)
                    {
                        uint vi = c.Mspi[link.MspiFirstIndex + j];
                        if (vi >= c.Mspv.Count) { points.Clear(); break; }
                        Vector3 p = c.Mspv[(int)vi];
                        if (!float.IsFinite(p.X) || !float.IsFinite(p.Y) || !float.IsFinite(p.Z)) { points.Clear(); break; }
                        points.Add(p);
                    }
                    if (points.Count >= 2) path = points.ToArray();
                }
                var ownerContact = path is null ? (Vertex: false, Edge: false) : Compare(polygons[i], path, epsilon);
                Measure("stored-reference", target);
                Measure("offset-17-control", (target + 17) % c.Msur.Count);
                // Deterministic within-file wrong target, independent of actual RefIndex.
                uint hash = unchecked((uint)i * 2654435761u ^ (uint)k * 2246822519u ^ 3266489917u);
                Measure("hashed-target-control", (int)(hash % (uint)c.Msur.Count));

                void Measure(string relation, int other)
                {
                    uint a = s.Ck24, b = c.Msur[other].Ck24;
                    string ck = a == 0 && b == 0 ? "both-zero" : a == b ? "same-nonzero" : "different";
                    var key = (relation, ck, link.TypeFlags, link.Subtype, link.MspiFirstIndex < 0);
                    if (!rows.TryGetValue(key, out long[]? counts)) rows[key] = counts = new long[9];
                    counts[0]++;
                    if (i == other) counts[3]++;
                    var measured = Compare(polygons[i], polygons[other], epsilon);
                    if (measured.Vertex) counts[1]++;
                    if (measured.Edge) counts[2]++;
                    if (path != null)
                    {
                        counts[4]++;
                        if (ownerContact.Vertex) counts[5]++;
                        if (ownerContact.Edge) counts[7]++;
                        var referenceContact = Compare(polygons[other], path, epsilon);
                        if (referenceContact.Vertex) counts[6]++;
                        if (referenceContact.Edge) counts[8]++;
                    }
                }
            }
        }
        return new Pm4LinkGeometryResult(c.Msur.Count, c.Mslk.Count, invalidPolygons, invalidWindows, invalidReferences,
            rows.OrderBy(p => p.Key.Relation, StringComparer.Ordinal).ThenBy(p => p.Key.Ck, StringComparer.Ordinal)
                .ThenBy(p => p.Key.Type).ThenBy(p => p.Key.Subtype).ThenBy(p => p.Key.Anchor)
                .Select(p => new Pm4LinkGeometryRow(p.Key.Relation, p.Key.Ck, p.Key.Type, p.Key.Subtype, p.Key.Anchor,
                    p.Value[0], p.Value[1], p.Value[2], p.Value[3], p.Value[4], p.Value[5], p.Value[6], p.Value[7], p.Value[8])).ToArray(), exceptions);
    }

    private static (bool Vertex, bool Edge) Compare(Vector3[] a, Vector3[] b, float epsilon)
    {
        float squared = epsilon * epsilon;
        bool vertex = false;
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
            {
                bool same = Vector3.DistanceSquared(a[i], b[j]) <= squared;
                vertex |= same;
                if (Vector3.DistanceSquared(a[i], a[(i + 1) % a.Length]) <= squared
                    || Vector3.DistanceSquared(b[j], b[(j + 1) % b.Length]) <= squared)
                    continue;
                if ((same && Vector3.DistanceSquared(a[(i + 1) % a.Length], b[(j + 1) % b.Length]) <= squared)
                    || (Vector3.DistanceSquared(a[i], b[(j + 1) % b.Length]) <= squared
                        && Vector3.DistanceSquared(a[(i + 1) % a.Length], b[j]) <= squared))
                    return (true, true);
            }
        return (vertex, false);
    }
}
