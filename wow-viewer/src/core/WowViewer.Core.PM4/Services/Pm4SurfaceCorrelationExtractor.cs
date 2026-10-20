using System.Numerics;
using WowViewer.Core.M2;
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
        return ExtractFromWmoGroupWithFilter(
            movt, movi, faceMaterials, groupIndex, wmoPath,
            edgeBinSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize,
            static _ => true);
    }

    public static SurfaceCorrelationFingerprint? ExtractFromWmoCollisionGroup(
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
        return ExtractFromWmoGroupWithFilter(
            movt, movi, faceMaterials, groupIndex, wmoPath,
            edgeBinSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize,
            IsCollidableFace);
    }

    private static bool IsCollidableFace(WmoGroupFaceMaterialDetail? material)
    {
        if (material is null)
            return false;

        const byte CollisionFace = 0x08;
        const byte RenderFace = 0x20;
        const byte NoCollide = 0x04;

        byte flags = material.Flags;
        return (flags & CollisionFace) != 0 || ((flags & RenderFace) != 0 && (flags & NoCollide) == 0);
    }

    private static SurfaceCorrelationFingerprint? ExtractFromWmoGroupWithFilter(
        IReadOnlyList<Vector3> movt,
        IReadOnlyList<ushort> movi,
        IReadOnlyList<WmoGroupFaceMaterialDetail>? faceMaterials,
        int groupIndex,
        string wmoPath,
        float edgeBinSize,
        float areaBinSize,
        float normalAlignmentBinSize,
        float planarOffsetBinSize,
        Func<WmoGroupFaceMaterialDetail?, bool> faceFilter)
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
            if (faceMaterials is not null && i < faceMaterials.Count && !faceFilter(faceMaterials[i]))
                continue;

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

    public static SurfaceCorrelationFingerprint? ExtractFromM2Collision(
        IReadOnlyList<Vector3> collisionVertices,
        IReadOnlyList<ushort> collisionIndices,
        string m2Path,
        float minNormalZ = 0.5f,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalAlignmentBinSize = 0.0f,
        float planarOffsetBinSize = 0.0f)
    {
        return ExtractFromM2TopSurfaces(
            collisionVertices,
            collisionIndices,
            m2Path,
            minNormalZ,
            edgeBinSize,
            areaBinSize,
            normalAlignmentBinSize,
            planarOffsetBinSize,
            sourceLabel: "m2-collision-top-surfaces");
    }

    public static SurfaceCorrelationFingerprint? ExtractFromM2Model(
        M2ModelDocument model,
        string m2Path,
        float minNormalZ = 0.5f,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalAlignmentBinSize = 0.0f,
        float planarOffsetBinSize = 0.0f)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (model.HasCollisionMesh)
        {
            var fp = ExtractFromM2Collision(
                model.CollisionVertices,
                model.CollisionTriangles,
                m2Path,
                minNormalZ,
                edgeBinSize,
                areaBinSize,
                normalAlignmentBinSize,
                planarOffsetBinSize);

            if (fp is not null)
                return fp;
        }

        return null;
    }

    public static SurfaceCorrelationFingerprint? ExtractFromM2TopSurfaces(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<ushort> indices,
        string m2Path,
        float minNormalZ = 0.5f,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalAlignmentBinSize = 0.0f,
        float planarOffsetBinSize = 0.0f,
        string sourceLabel = "m2-top-surfaces")
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(indices);

        if (vertices.Count < 3 || indices.Count < 3)
            return null;

        int triangleCount = indices.Count / 3;
        List<TriangleFeature> triangles = new(triangleCount);
        HashSet<int> usedVertices = new();

        for (int i = 0; i < triangleCount; i++)
        {
            int i0 = indices[i * 3];
            int i1 = indices[i * 3 + 1];
            int i2 = indices[i * 3 + 2];

            if ((uint)i0 >= (uint)vertices.Count || (uint)i1 >= (uint)vertices.Count || (uint)i2 >= (uint)vertices.Count)
                continue;

            Vector3 a = vertices[i0];
            Vector3 b = vertices[i1];
            Vector3 c = vertices[i2];

            if (!float.IsFinite(a.X) || !float.IsFinite(b.X) || !float.IsFinite(c.X))
                continue;

            Vector3 cross = Vector3.Cross(b - a, c - a);
            float crossLength = cross.Length();
            if (crossLength < 0.0001f)
                continue;

            Vector3 normal = cross / crossLength;
            if (normal.Z < minNormalZ)
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

        (Vector3 dominantNormal, Vector3 centroid) = ComputeGroupFrame(triangles);

        Dictionary<string, int> histogram = BuildHistogram(
            triangles, dominantNormal, centroid, edgeBinSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize);
        float totalArea = triangles.Sum(static t => t.Area);

        return new SurfaceCorrelationFingerprint(
            m2Path,
            m2Path,
            "m2",
            0x40,
            triangles.Count,
            triangles.Count,
            usedVertices.Count,
            histogram,
            sourceLabel)
        {
            TotalTriangleArea = totalArea,
            MeanTriangleArea = totalArea / triangles.Count,
        };
    }

    public static SurfaceCorrelationFingerprint? ExtractFromM2TopSurfaces(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<uint> indices,
        string m2Path,
        float minNormalZ = 0.5f,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalAlignmentBinSize = 0.0f,
        float planarOffsetBinSize = 0.0f,
        string sourceLabel = "m2-render-top-surfaces")
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(indices);

        if (vertices.Count < 3 || indices.Count < 3)
            return null;

        int triangleCount = indices.Count / 3;
        List<TriangleFeature> triangles = new(triangleCount);
        HashSet<int> usedVertices = new();

        for (int i = 0; i < triangleCount; i++)
        {
            uint i0 = indices[i * 3];
            uint i1 = indices[i * 3 + 1];
            uint i2 = indices[i * 3 + 2];

            if (i0 >= (uint)vertices.Count || i1 >= (uint)vertices.Count || i2 >= (uint)vertices.Count)
                continue;

            Vector3 a = vertices[(int)i0];
            Vector3 b = vertices[(int)i1];
            Vector3 c = vertices[(int)i2];

            if (!float.IsFinite(a.X) || !float.IsFinite(b.X) || !float.IsFinite(c.X))
                continue;

            Vector3 cross = Vector3.Cross(b - a, c - a);
            float crossLength = cross.Length();
            if (crossLength < 0.0001f)
                continue;

            Vector3 normal = cross / crossLength;
            if (normal.Z < minNormalZ)
                continue;

            float e0 = Vector3.Distance(a, b);
            float e1 = Vector3.Distance(b, c);
            float e2 = Vector3.Distance(c, a);

            if (e0 < 0.001f || e1 < 0.001f || e2 < 0.001f)
                continue;

            triangles.Add(TriangleFeature.FromTriangle(a, b, c));
            usedVertices.Add((int)i0);
            usedVertices.Add((int)i1);
            usedVertices.Add((int)i2);
        }

        if (triangles.Count == 0)
            return null;

        (Vector3 dominantNormal, Vector3 centroid) = ComputeGroupFrame(triangles);

        Dictionary<string, int> histogram = BuildHistogram(
            triangles, dominantNormal, centroid, edgeBinSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize);
        float totalArea = triangles.Sum(static t => t.Area);

        return new SurfaceCorrelationFingerprint(
            m2Path,
            m2Path,
            "m2",
            0x40,
            triangles.Count,
            triangles.Count,
            usedVertices.Count,
            histogram,
            sourceLabel)
        {
            TotalTriangleArea = totalArea,
            MeanTriangleArea = totalArea / triangles.Count,
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
