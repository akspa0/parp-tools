using System.Numerics;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// Full 3D geometric shape signature capturing the true 3D surface geometry of a PM4 object
/// or model (vertical cross-section profile, mass distribution / 3D inertia, surface normals,
/// and sampled 3D point cloud).
/// Replaces simplistic 2D footprint bounding boxes with real 3D shape comparison.
/// </summary>
public sealed class Pm4Geometry3DShapeSignature
{
    public const int VerticalSliceBands = 5;
    public const int SamplePointTarget = 36;

    /// <summary>Normalized 3D surface points in local object space (centered at origin, scaled to max extent 1.0).</summary>
    public IReadOnlyList<Vector3> NormalizedSamplePoints { get; }

    /// <summary>Maximum radius at each vertical height slice (bottom to top, 5 bands).</summary>
    public float[] VerticalSliceRadii { get; }

    /// <summary>Point density fraction in each vertical height slice (bottom to top, 5 bands).</summary>
    public float[] VerticalSliceDensities { get; }

    /// <summary>Normalized principal moments of inertia (Ix, Iy, Iz) / sum(I).</summary>
    public Vector3 InertiaTensor { get; }

    /// <summary>Distribution of surface normals: X = upward/downward (floors/ceilings), Y = horizontal (walls), Z = angled (ramps/slopes/hulls).</summary>
    public Vector3 NormalDistribution { get; }

    /// <summary>3D bounding extents in unscaled local space.</summary>
    public Vector3 LocalExtents { get; }

    public float Length { get; }
    public float Width { get; }
    public float Height { get; }

    public Pm4Geometry3DShapeSignature(
        IReadOnlyList<Vector3> samplePoints,
        float[] verticalSliceRadii,
        float[] verticalSliceDensities,
        Vector3 inertiaTensor,
        Vector3 normalDistribution,
        Vector3 localExtents)
    {
        NormalizedSamplePoints = samplePoints;
        VerticalSliceRadii = verticalSliceRadii;
        VerticalSliceDensities = verticalSliceDensities;
        InertiaTensor = inertiaTensor;
        NormalDistribution = normalDistribution;
        LocalExtents = localExtents;

        Length = MathF.Max(localExtents.X, localExtents.Y);
        Width = MathF.Min(localExtents.X, localExtents.Y);
        Height = MathF.Max(0.01f, localExtents.Z);
    }

    /// <summary>
    /// Builds a full 3D shape signature from a 3D triangle mesh and optional line segments in local space.
    /// </summary>
    public static Pm4Geometry3DShapeSignature FromGeometry(
        IReadOnlyList<(Vector3 A, Vector3 B, Vector3 C)> triangles,
        IReadOnlyList<(Vector3 From, Vector3 To)>? lines = null)
    {
        var allPoints = new List<Vector3>(triangles.Count * 3 + (lines?.Count ?? 0) * 2);
        var normals = new List<Vector3>(triangles.Count);

        for (int i = 0; i < triangles.Count; i++)
        {
            var (a, b, c) = triangles[i];
            allPoints.Add(a);
            allPoints.Add(b);
            allPoints.Add(c);

            Vector3 ab = b - a;
            Vector3 ac = c - a;
            Vector3 cross = Vector3.Cross(ab, ac);
            float len = cross.Length();
            if (len > 1e-4f)
                normals.Add(cross / len);
        }

        if (lines != null)
        {
            for (int i = 0; i < lines.Count; i++)
            {
                allPoints.Add(lines[i].From);
                allPoints.Add(lines[i].To);
            }
        }

        if (allPoints.Count == 0)
            return FromBounds(Vector3.Zero, Vector3.One);

        // 1. Compute bounds and center of mass
        Vector3 bMin = new(float.MaxValue);
        Vector3 bMax = new(float.MinValue);
        Vector3 sum = Vector3.Zero;

        for (int i = 0; i < allPoints.Count; i++)
        {
            Vector3 p = allPoints[i];
            bMin = Vector3.Min(bMin, p);
            bMax = Vector3.Max(bMax, p);
            sum += p;
        }

        Vector3 center = sum / allPoints.Count;
        Vector3 extents = bMax - bMin;
        float maxExtent = MathF.Max(extents.X, MathF.Max(extents.Y, MathF.Max(0.01f, extents.Z)));
        float invMaxExtent = 1.0f / maxExtent;

        // 2. Normal distribution
        float horizCount = 0;
        float vertCount = 0;
        float angledCount = 0;

        for (int i = 0; i < normals.Count; i++)
        {
            float absZ = MathF.Abs(normals[i].Z);
            if (absZ > 0.80f) horizCount++;
            else if (absZ < 0.25f) vertCount++;
            else angledCount++;
        }

        float totalNormals = MathF.Max(1f, normals.Count);
        Vector3 normalDist = new(horizCount / totalNormals, vertCount / totalNormals, angledCount / totalNormals);

        // 3. Vertical slices (5 bands)
        float[] sliceRadii = new float[VerticalSliceBands];
        float[] sliceDensities = new float[VerticalSliceBands];
        float totalHeight = MathF.Max(0.01f, extents.Z);

        for (int i = 0; i < allPoints.Count; i++)
        {
            Vector3 p = allPoints[i];
            float normalizedZ = Math.Clamp((p.Z - bMin.Z) / totalHeight, 0f, 0.999f);
            int band = (int)(normalizedZ * VerticalSliceBands);
            sliceDensities[band]++;

            float planarRadius = MathF.Sqrt((p.X - center.X) * (p.X - center.X) + (p.Y - center.Y) * (p.Y - center.Y));
            if (planarRadius > sliceRadii[band])
                sliceRadii[band] = planarRadius;
        }

        float maxRadius = 0.01f;
        for (int b = 0; b < VerticalSliceBands; b++)
            if (sliceRadii[b] > maxRadius) maxRadius = sliceRadii[b];

        for (int b = 0; b < VerticalSliceBands; b++)
        {
            sliceRadii[b] /= maxRadius;
            sliceDensities[b] /= allPoints.Count;
        }

        // 4. Moments of Inertia
        float ixx = 0, iyy = 0, izz = 0;
        for (int i = 0; i < allPoints.Count; i++)
        {
            Vector3 rel = (allPoints[i] - center) * invMaxExtent;
            ixx += rel.Y * rel.Y + rel.Z * rel.Z;
            iyy += rel.X * rel.X + rel.Z * rel.Z;
            izz += rel.X * rel.X + rel.Y * rel.Y;
        }

        float totalInertia = MathF.Max(1e-4f, ixx + iyy + izz);
        Vector3 inertiaTensor = new(ixx / totalInertia, iyy / totalInertia, izz / totalInertia);

        // 5. Representative normalized sample points for Chamfer matching
        var samplePoints = new List<Vector3>(SamplePointTarget);
        int stride = Math.Max(1, allPoints.Count / SamplePointTarget);
        for (int i = 0; i < allPoints.Count && samplePoints.Count < SamplePointTarget; i += stride)
        {
            samplePoints.Add((allPoints[i] - center) * invMaxExtent);
        }

        return new Pm4Geometry3DShapeSignature(
            samplePoints,
            sliceRadii,
            sliceDensities,
            inertiaTensor,
            normalDist,
            extents);
    }

    /// <summary>
    /// Fallback signature from bounding box when no mesh triangles exist.
    /// </summary>
    public static Pm4Geometry3DShapeSignature FromBounds(Vector3 min, Vector3 max)
    {
        Vector3 extents = max - min;
        float[] radii = [1f, 1f, 1f, 1f, 1f];
        float[] densities = [0.2f, 0.2f, 0.2f, 0.2f, 0.2f];
        Vector3 inertia = new(0.333f, 0.333f, 0.334f);
        Vector3 normals = new(0.333f, 0.333f, 0.334f);

        Vector3 center = (min + max) * 0.5f;
        float maxExtent = MathF.Max(extents.X, MathF.Max(extents.Y, MathF.Max(0.01f, extents.Z)));
        float invMax = 1f / maxExtent;

        var samplePoints = new List<Vector3>(8)
        {
            (new Vector3(min.X, min.Y, min.Z) - center) * invMax,
            (new Vector3(max.X, min.Y, min.Z) - center) * invMax,
            (new Vector3(min.X, max.Y, min.Z) - center) * invMax,
            (new Vector3(max.X, max.Y, min.Z) - center) * invMax,
            (new Vector3(min.X, min.Y, max.Z) - center) * invMax,
            (new Vector3(max.X, min.Y, max.Z) - center) * invMax,
            (new Vector3(min.X, max.Y, max.Z) - center) * invMax,
            (new Vector3(max.X, max.Y, max.Z) - center) * invMax,
        };

        return new Pm4Geometry3DShapeSignature(samplePoints, radii, densities, inertia, normals, extents);
    }

    /// <summary>
    /// Compares two 3D geometry shape signatures across 3D vertical slice profile,
    /// 3D inertia moments, surface normal distributions, and 3D point Chamfer distance.
    /// Rotates around Z to find the best rotational correspondence.
    /// </summary>
    public static float Compare3D(
        Pm4Geometry3DShapeSignature a,
        Pm4Geometry3DShapeSignature b,
        out float bestYawOffsetDegrees)
    {
        bestYawOffsetDegrees = 0f;

        // 1. Vertical profile distance
        float dProfile = 0;
        float dDensity = 0;
        for (int i = 0; i < VerticalSliceBands; i++)
        {
            dProfile += MathF.Abs(a.VerticalSliceRadii[i] - b.VerticalSliceRadii[i]);
            dDensity += MathF.Abs(a.VerticalSliceDensities[i] - b.VerticalSliceDensities[i]);
        }
        dProfile /= VerticalSliceBands;
        dDensity /= VerticalSliceBands;

        // 2. Inertia tensor distance
        float dInertia = Vector3.Distance(a.InertiaTensor, b.InertiaTensor);

        // 3. Normal distribution distance
        float dNormals = Vector3.Distance(a.NormalDistribution, b.NormalDistribution);

        // 4. 3D Chamfer point-cloud distance across 8 candidate yaw angles
        float minChamfer = float.MaxValue;
        float bestYaw = 0f;

        float[] candidateYaws = [0f, 45f, 90f, 135f, 180f, 225f, 270f, 315f];

        for (int y = 0; y < candidateYaws.Length; y++)
        {
            float yawDeg = candidateYaws[y];
            float rad = yawDeg * (MathF.PI / 180f);
            float cos = MathF.Cos(rad);
            float sin = MathF.Sin(rad);

            // Compute bidirectional Chamfer distance
            float sumDistA = 0;
            for (int i = 0; i < a.NormalizedSamplePoints.Count; i++)
            {
                Vector3 pa = a.NormalizedSamplePoints[i];
                float nearestSq = float.MaxValue;

                for (int j = 0; j < b.NormalizedSamplePoints.Count; j++)
                {
                    Vector3 pb = b.NormalizedSamplePoints[j];
                    Vector3 rotB = new(pb.X * cos - pb.Y * sin, pb.X * sin + pb.Y * cos, pb.Z);
                    float dSq = Vector3.DistanceSquared(pa, rotB);
                    if (dSq < nearestSq) nearestSq = dSq;
                }
                sumDistA += MathF.Sqrt(nearestSq);
            }

            float sumDistB = 0;
            for (int j = 0; j < b.NormalizedSamplePoints.Count; j++)
            {
                Vector3 pb = b.NormalizedSamplePoints[j];
                Vector3 rotB = new(pb.X * cos - pb.Y * sin, pb.X * sin + pb.Y * cos, pb.Z);
                float nearestSq = float.MaxValue;

                for (int i = 0; i < a.NormalizedSamplePoints.Count; i++)
                {
                    float dSq = Vector3.DistanceSquared(a.NormalizedSamplePoints[i], rotB);
                    if (dSq < nearestSq) nearestSq = dSq;
                }
                sumDistB += MathF.Sqrt(nearestSq);
            }

            float chamfer = (sumDistA / Math.Max(1, a.NormalizedSamplePoints.Count) +
                             sumDistB / Math.Max(1, b.NormalizedSamplePoints.Count)) * 0.5f;

            if (chamfer < minChamfer)
            {
                minChamfer = chamfer;
                bestYaw = yawDeg;
            }
        }

        bestYawOffsetDegrees = bestYaw;

        // 5. Combine into comprehensive 3D shape similarity score
        // Chamfer distance is usually in [0..0.5] for matching objects
        float chamferNorm = Math.Clamp(minChamfer * 2.5f, 0f, 1f);

        double score = 1.0 - (
            0.40 * chamferNorm +
            0.25 * dProfile +
            0.15 * dInertia +
            0.10 * dDensity +
            0.10 * dNormals);

        return (float)Math.Clamp(score, 0.0, 1.0);
    }
}

