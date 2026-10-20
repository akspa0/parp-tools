using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

// A read-only reproduction of the resolver's rounded-cell zero-bucket connectivity.
// These are candidate components, NOT confirmed M2 objects. Terrain may survive here.
internal static class SparseM2Probe
{
    public static object Measure(Pm4KnownChunkSet chunks, IReadOnlyList<Placement> placements)
    {
        int[] indices = Enumerable.Range(0, chunks.Msur.Count)
            .Where(i => chunks.Msur[i].PackedParams == 0).ToArray();
        var parent = indices.ToDictionary(i => i, i => i);
        var cells = new Dictionary<(int, int, int), int>();
        int Find(int i)
        {
            int root = i;
            while (parent[root] != root)
                root = parent[root];
            while (parent[i] != i)
            {
                int next = parent[i];
                parent[i] = root;
                i = next;
            }
            return root;
        }

        foreach (int i in indices)
        {
            var s = chunks.Msur[i];
            long end = (long)s.MsviFirstIndex + s.IndexCount;
            if (end > chunks.Msvi.Count)
                throw new InvalidDataException("Unkeyed surface index window exceeds MSVI.");
            for (long k = s.MsviFirstIndex; k < end; k++)
            {
                uint vi = chunks.Msvi[(int)k];
                if (vi >= chunks.Msvt.Count)
                    throw new InvalidDataException("Unkeyed vertex index exceeds MSVT.");
                Vector3 v = chunks.Msvt[(int)vi];
                var cell = ((int)MathF.Round(v.X / 0.25f), (int)MathF.Round(v.Y / 0.25f), (int)MathF.Round(v.Z / 0.25f));
                if (cells.TryGetValue(cell, out int other))
                    parent[Find(i)] = Find(other);
                else
                    cells[cell] = i;
            }
        }

        var m2s = placements.Where(p => p.Kind == "M2").ToArray();
        var components = new List<Component>();
        foreach (var component in indices.GroupBy(Find).OrderBy(g => g.Min()))
        {
            var surfaces = component.Select(i => chunks.Msur[i]).ToArray();
            var raw = Pm4PlacementMath.CollectSurfaceVertices(chunks.Msvt, chunks.Msvi, surfaces);
            if (raw.Count == 0)
                continue;
            Vector3[] points = raw.Select(Pm4CoordinateService.Pm4LocalToAdtPlacement).ToArray();
            Vector3 min = points.Aggregate(Vector3.Min);
            Vector3 max = points.Aggregate(Vector3.Max);
            Vector3 centroid = points.Aggregate(Vector3.Zero, (a, b) => a + b) / points.Length;
            var fp = Pm4SurfaceCorrelationExtractor.ExtractFromPm4Group(
                chunks.Msvt, chunks.Msvi, surfaces, 0, 0x40, $"component-{component.Min()}", "unkeyed", "m2");
            var fine = Pm4SurfaceCorrelationExtractor.ExtractFromPm4Group(
                chunks.Msvt, chunks.Msvi, surfaces, 0, 0x40, $"component-{component.Min()}", "unkeyed", "m2", 0.25f, 0.25f);
            components.Add(new Component(component.Min(), surfaces.Length, fp?.TriangleCount ?? 0,
                fp?.TriangleHistogram.Count ?? 0, min, max, centroid,
                HistogramKey(fp), HistogramKey(fine)));
        }

        Vector3 lo = components.Count == 0 ? Vector3.Zero : components.Select(c => c.Min).Aggregate(Vector3.Min);
        Vector3 hi = components.Count == 0 ? Vector3.Zero : components.Select(c => c.Max).Aggregate(Vector3.Max);
        var measurements = components.Select(c =>
        {
            var xy = new Vector2(c.Centroid.X, c.Centroid.Y);
            var reflected = new Vector2(lo.X + hi.X - xy.X, lo.Y + hi.Y - xy.Y);
            var nearest = m2s.Select(p => new
            {
                p.ModelPath, p.UniqueId, Distance = Vector2.Distance(xy, new Vector2(p.Position.X, p.Position.Y)),
            }).OrderBy(p => p.Distance).Take(3).ToArray();
            float? controlDistance = m2s.Length == 0 ? null : m2s.Min(p => Vector2.Distance(reflected, new Vector2(p.Position.X, p.Position.Y)));
            return new
            {
                c.Id, c.Surfaces, c.Triangles, c.HistogramBins,
                BoundsSize = new[] { c.Max.X - c.Min.X, c.Max.Y - c.Min.Y, c.Max.Z - c.Min.Z },
                Centroid = new[] { c.Centroid.X, c.Centroid.Y, c.Centroid.Z },
                NearestAuthoredM2 = nearest,
                M2OriginsWithin24 = m2s.Count(p => Vector2.Distance(xy, new Vector2(p.Position.X, p.Position.Y)) <= 24),
                ReflectedCentroidNearestM2Distance = controlDistance,
                DefaultHistogramPeers = components.Count(p => p.Histogram == c.Histogram),
                FineHistogramPeers = components.Count(p => p.FineHistogram == c.FineHistogram),
            };
        }).ToArray();
        return new
        {
            Method = "PackedParams == 0 only; shared rounded XYZ cells at 0.25 yards, as current resolver. Not a Euclidean-epsilon weld.",
            Limits = "Includes terrain/unknowns. Centroid is not a model pivot. Nearest-origin proximity and reflected control are diagnostics, not object identities.",
            ComponentCount = measurements.Length,
            AtMostTwoTriangles = measurements.Count(c => c.Triangles <= 2),
            AtMostFourTriangles = measurements.Count(c => c.Triangles <= 4),
            NonemptyDefaultFingerprints = components.Count(c => c.Triangles > 0),
            DistinctDefaultHistograms = components.Where(c => c.Triangles > 0).Select(c => c.Histogram).Distinct().Count(),
            DistinctFineHistograms = components.Where(c => c.Triangles > 0).Select(c => c.FineHistogram).Distinct().Count(),
            WithM2OriginWithin24 = measurements.Count(c => c.M2OriginsWithin24 > 0),
            ReflectedWithM2OriginWithin24 = measurements.Count(c => c.ReflectedCentroidNearestM2Distance <= 24),
            Components = measurements,
        };
    }

    private static string HistogramKey(SurfaceCorrelationFingerprint? fp) => fp is null ? "empty" :
        string.Join("|", fp.TriangleHistogram.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}:{p.Value}"));

    private sealed record Component(int Id, int Surfaces, int Triangles, int HistogramBins,
        Vector3 Min, Vector3 Max, Vector3 Centroid, string Histogram, string FineHistogram);
}
