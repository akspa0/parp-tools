using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4FingerprintExtractor
{
    public const float DegenerateEigenvalueRatio = 0.95f;

    public static Pm4FingerprintRecord? ExtractFromGeometry(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<int> indices,
        int surfaceCount,
        byte ck24Type,
        IReadOnlyDictionary<byte, int> typeFlagsProfile,
        string assetId,
        string assetPath,
        string assetKind,
        int groupCount = 1,
        string sourceLabel = "")
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(indices);

        if (vertices.Count < 3 || indices.Count < 3)
            return null;

        List<Vector3> referencedVerts = ExtractReferencedVertices(vertices, indices);
        if (referencedVerts.Count < 3)
            return null;

        List<Vector3> normalizedPoints = PcaNormalize(referencedVerts, out bool isNearSymmetric);
        if (normalizedPoints.Count < 3)
            return null;

        Vector3 boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 boundsMax = new(float.MinValue, float.MinValue, float.MinValue);
        Vector3 center = Vector3.Zero;
        foreach (Vector3 p in normalizedPoints)
        {
            boundsMin = Vector3.Min(boundsMin, p);
            boundsMax = Vector3.Max(boundsMax, p);
            center += p;
        }
        center /= normalizedPoints.Count;

        List<Vector2> footprintPoints = new(normalizedPoints.Count);
        foreach (Vector3 p in normalizedPoints)
            footprintPoints.Add(new Vector2(p.X, p.Y));

        Vector2[] hull = Pm4CorrelationMath.BuildConvexHull(footprintPoints);
        if (hull.Length < 3)
            return null;

        float footprintArea = Pm4CorrelationMath.ComputeFootprintArea(hull);
        if (footprintArea <= 0f)
            return null;

        float spanX = boundsMax.X - boundsMin.X;
        float spanY = boundsMax.Y - boundsMin.Y;
        float spanZ = boundsMax.Z - boundsMin.Z;
        float[] sortedDims = [spanX, spanY, spanZ];
        Array.Sort(sortedDims);

        Dictionary<byte, int> resolvedTypeFlags = typeFlagsProfile is not null
            ? new Dictionary<byte, int>(typeFlagsProfile)
            : [];

        List<HullPoint2> hullPoints = new(hull.Length);
        foreach (Vector2 p in hull)
            hullPoints.Add(HullPoint2.FromVector2(p));

        string sourceLabelResolved = string.IsNullOrWhiteSpace(sourceLabel)
            ? (isNearSymmetric ? "pca-near-symmetric" : "pca-normalized")
            : sourceLabel + (isNearSymmetric ? "|near-symmetric" : "");

        return new Pm4FingerprintRecord(
            assetId,
            assetPath,
            assetKind,
            ck24Type,
            surfaceCount,
            vertices.Count,
            indices.Count,
            groupCount,
            sortedDims[0],
            sortedDims[1],
            sortedDims[2],
            Bounds3Serial.FromBounds(boundsMin, boundsMax),
            new HullPoint2(center.X, center.Y),
            hullPoints,
            footprintArea,
            resolvedTypeFlags,
            sourceLabelResolved);
    }

    public static Pm4FingerprintRecord? ExtractFromTriangles(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<int> triangleIndices,
        int surfaceCount,
        byte ck24Type,
        IReadOnlyDictionary<byte, int> typeFlagsProfile,
        string assetId,
        string assetPath,
        string assetKind,
        int groupCount = 1,
        string sourceLabel = "")
    {
        return ExtractFromGeometry(vertices, triangleIndices, surfaceCount, ck24Type, typeFlagsProfile, assetId, assetPath, assetKind, groupCount, sourceLabel);
    }

    public static IReadOnlyList<Vector2> FlipHull(IReadOnlyList<Vector2> hull, bool flipX, bool flipY)
    {
        Vector2[] flipped = new Vector2[hull.Count];
        for (int i = 0; i < hull.Count; i++)
        {
            Vector2 p = hull[i];
            flipped[i] = new Vector2(
                flipX ? -p.X : p.X,
                flipY ? -p.Y : p.Y);
        }

        return Pm4CorrelationMath.BuildConvexHull(flipped);
    }

    private static List<Vector3> ExtractReferencedVertices(IReadOnlyList<Vector3> vertices, IReadOnlyList<int> indices)
    {
        HashSet<int> referencedIndices = [];
        for (int i = 0; i < indices.Count; i++)
        {
            int idx = indices[i];
            if ((uint)idx < (uint)vertices.Count)
                referencedIndices.Add(idx);
        }

        List<Vector3> points = new(referencedIndices.Count);
        foreach (int idx in referencedIndices)
        {
            Vector3 v = vertices[idx];
            if (float.IsFinite(v.X) && float.IsFinite(v.Y) && float.IsFinite(v.Z))
                points.Add(v);
        }

        return points;
    }

    private static List<Vector3> PcaNormalize(List<Vector3> points, out bool isNearSymmetric)
    {
        isNearSymmetric = false;

        if (points.Count < 3)
            return [];

        Vector3 centroid = Vector3.Zero;
        foreach (Vector3 p in points)
            centroid += p;
        centroid /= points.Count;

        List<Vector3> centered = new(points.Count);
        for (int i = 0; i < points.Count; i++)
            centered.Add(points[i] - centroid);

        float varX = 0f, varY = 0f, covXY = 0f;
        for (int i = 0; i < centered.Count; i++)
        {
            Vector3 p = centered[i];
            varX += p.X * p.X;
            varY += p.Y * p.Y;
            covXY += p.X * p.Y;
        }
        varX /= centered.Count;
        varY /= centered.Count;
        covXY /= centered.Count;

        float trace = varX + varY;
        float det = varX * varY - covXY * covXY;
        float discriminant = MathF.Max(0f, trace * trace - 4f * det);
        float sqrtDisc = MathF.Sqrt(discriminant);
        float lambda1 = (trace + sqrtDisc) * 0.5f;
        float lambda2 = (trace - sqrtDisc) * 0.5f;

        if (lambda1 <= 0f)
            return [];

        if (lambda2 > 0f && lambda2 / lambda1 >= DegenerateEigenvalueRatio)
            isNearSymmetric = true;

        Vector2 principalAxis;
        if (MathF.Abs(covXY) > 1e-10f)
        {
            principalAxis = new Vector2(lambda1 - varY, covXY);
        }
        else
        {
            principalAxis = varX >= varY ? new Vector2(1f, 0f) : new Vector2(0f, 1f);
        }

        float axisLength = MathF.Sqrt(principalAxis.X * principalAxis.X + principalAxis.Y * principalAxis.Y);
        if (axisLength < 1e-10f)
            return [];

        principalAxis /= axisLength;

        float cos = principalAxis.X;
        float sin = principalAxis.Y;

        List<Vector3> rotated = new(centered.Count);
        for (int i = 0; i < centered.Count; i++)
        {
            Vector3 p = centered[i];
            rotated.Add(new Vector3(
                cos * p.X + sin * p.Y,
                -sin * p.X + cos * p.Y,
                p.Z));
        }

        return rotated;
    }
}
