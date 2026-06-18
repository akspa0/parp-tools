using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.PM4.Services;

public static class Pm4SurfaceCorrelationExtractor
{
    public static SurfaceCorrelationFingerprint? ExtractFromPm4Group(
        IReadOnlyList<Vector3> msvt,
        IReadOnlyList<uint> msvi,
        IReadOnlyList<Pm4MsurEntry> surfaces,
        uint ck24,
        byte ck24Type,
        string assetId,
        string assetPath,
        string assetKind,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalAlignmentBinSize = 0.0f,
        float planarOffsetBinSize = 0.0f)
    {
        ArgumentNullException.ThrowIfNull(msvt);
        ArgumentNullException.ThrowIfNull(msvi);
        ArgumentNullException.ThrowIfNull(surfaces);

        if (surfaces.Count == 0 || msvt.Count < 3)
            return null;

        List<TriangleFeature> triangles = new(surfaces.Count * 2);
        HashSet<int> usedVertices = new();

        foreach (Pm4MsurEntry surface in surfaces)
        {
            int first = checked((int)surface.MsviFirstIndex);
            int count = surface.IndexCount;
            if (count < 3 || first + count > msvi.Count)
                continue;

            List<int> localVerts = new(count);
            for (int i = 0; i < count; i++)
            {
                int vi = checked((int)msvi[first + i]);
                if ((uint)vi < (uint)msvt.Count)
                    localVerts.Add(vi);
            }

            if (localVerts.Count < 3)
                continue;

            foreach (int vi in localVerts)
                usedVertices.Add(vi);

            for (int i = 1; i < localVerts.Count - 1; i++)
            {
                Vector3 a = msvt[localVerts[0]];
                Vector3 b = msvt[localVerts[i]];
                Vector3 c = msvt[localVerts[i + 1]];

                if (!float.IsFinite(a.X) || !float.IsFinite(b.X) || !float.IsFinite(c.X))
                    continue;

                triangles.Add(TriangleFeature.FromTriangle(a, b, c));
            }
        }

        if (triangles.Count == 0)
            return null;

        (Vector3 dominantNormal, Vector3 centroid) = ComputeGroupFrame(triangles);

        Dictionary<string, int> histogram = BuildHistogram(
            triangles, dominantNormal, centroid, edgeBinSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize);
        float totalArea = triangles.Sum(static t => t.Area);

        return new SurfaceCorrelationFingerprint(
            assetId, assetPath, assetKind, ck24Type,
            surfaces.Count, triangles.Count, usedVertices.Count,
            histogram, $"pm4-ck24-0x{ck24:X6}")
        {
            TotalTriangleArea = totalArea,
            MeanTriangleArea = totalArea / triangles.Count,
        };
    }

    public static SurfaceCorrelationFingerprint? ExtractFromWmoGroup(
        IReadOnlyList<Vector3> movt,
        IReadOnlyList<ushort> movi,
        IReadOnlyList<WmoGroupFaceMaterialDetail>? faceMaterials,
        int groupIndex,
        string wmoPath,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalAlignmentBinSize = 0.0f,
        float planarOffsetBinSize = 0.0f)
    {
        ArgumentNullException.ThrowIfNull(movt);
        ArgumentNullException.ThrowIfNull(movi);

        if (movt.Count < 3 || movi.Count < 3)
            return null;

        int triangleCount = movi.Count / 3;
        List<TriangleFeature> triangles = new(triangleCount);
        HashSet<int> usedVertices = new();

        for (int i = 0; i < triangleCount; i++)
        {
            int i0 = movi[i * 3];
            int i1 = movi[i * 3 + 1];
            int i2 = movi[i * 3 + 2];

            if ((uint)i0 >= (uint)movt.Count || (uint)i1 >= (uint)movt.Count || (uint)i2 >= (uint)movt.Count)
                continue;

            Vector3 a = movt[i0];
            Vector3 b = movt[i1];
            Vector3 c = movt[i2];

            if (!float.IsFinite(a.X) || !float.IsFinite(b.X) || !float.IsFinite(c.X))
                continue;

            float e0 = Vector3.Distance(a, b);
            float e1 = Vector3.Distance(b, c);
            float e2 = Vector3.Distance(c, a);

            if (e0 < 0.001f || e1 < 0.001f || e2 < 0.001f)
                continue;

            triangles.Add(TriangleFeature.FromTriangle(a, b, c));
            usedVertices.Add(i0);
            usedVertices.Add(i1);
            usedVertices.Add(i2);
        }

        if (triangles.Count == 0)
            return null;

        int surfaceCount = faceMaterials?.Count ?? triangleCount;

        (Vector3 dominantNormal, Vector3 centroid) = ComputeGroupFrame(triangles);

        Dictionary<string, int> histogram = BuildHistogram(
            triangles, dominantNormal, centroid, edgeBinSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize);
        float totalArea = triangles.Sum(static t => t.Area);

        return new SurfaceCorrelationFingerprint(
            $"{wmoPath}#group{groupIndex}",
            wmoPath,
            "wmo",
            0x42,
            surfaceCount,
            triangles.Count,
            usedVertices.Count,
            histogram,
            $"wmo-group-{groupIndex}")
        {
            TotalTriangleArea = totalArea,
            MeanTriangleArea = totalArea / triangles.Count,
        };
    }

    public static SurfaceCorrelationFingerprint? MergeWmoGroups(
        IReadOnlyList<SurfaceCorrelationFingerprint> groupFingerprints,
        string wmoPath,
        float binSize = 1.0f)
    {
        if (groupFingerprints.Count == 0)
            return null;

        int totalTriangles = groupFingerprints.Sum(static f => f.TriangleCount);
        int totalVertices = groupFingerprints.Sum(static f => f.VertexCount);
        int totalSurfaces = groupFingerprints.Sum(static f => f.SurfaceCount);

        Dictionary<string, int> histogram = new();
        float totalArea = 0;

        foreach (SurfaceCorrelationFingerprint g in groupFingerprints)
        {
            foreach (var kv in g.TriangleHistogram)
            {
                histogram[kv.Key] = histogram.TryGetValue(kv.Key, out int v) ? v + kv.Value : kv.Value;
            }
            totalArea += g.TotalTriangleArea;
        }

        return new SurfaceCorrelationFingerprint(
            wmoPath,
            wmoPath,
            "wmo",
            0x42,
            totalSurfaces,
            totalTriangles,
            totalVertices,
            histogram,
            "wmo-root-merged")
        {
            TotalTriangleArea = totalArea,
            MeanTriangleArea = totalArea / totalTriangles,
        };
    }

    private static (Vector3 DominantNormal, Vector3 Centroid) ComputeGroupFrame(IReadOnlyList<TriangleFeature> triangles)
    {
        Vector3 normalSum = Vector3.Zero;
        Vector3 centroidSum = Vector3.Zero;
        float weightSum = 0;

        foreach (TriangleFeature t in triangles)
        {
            normalSum += t.Normal * t.Area;
            centroidSum += t.Centroid * t.Area;
            weightSum += t.Area;
        }

        Vector3 dominantNormal = normalSum.LengthSquared() > 0.0001f
            ? Vector3.Normalize(normalSum)
            : Vector3.UnitZ;

        Vector3 centroid = weightSum > 0.0001f
            ? centroidSum / weightSum
            : Vector3.Zero;

        return (dominantNormal, centroid);
    }

    private static Dictionary<string, int> BuildHistogram(
        IReadOnlyList<TriangleFeature> triangles,
        Vector3 dominantNormal,
        Vector3 centroid,
        float edgeBinSize,
        float areaBinSize,
        float normalAlignmentBinSize,
        float planarOffsetBinSize)
    {
        Dictionary<string, int> histogram = new();
        foreach (TriangleFeature t in triangles)
        {
            string key = t.Key(dominantNormal, centroid, edgeBinSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize).HistogramKey;
            histogram[key] = histogram.TryGetValue(key, out int v) ? v + 1 : 1;
        }

        return histogram;
    }
}
