using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Matches PM4 pathfinding geometry to WMO collision geometry to derive placement transforms.
/// This enables reconstruction of MODF placement data from PM4 files.
/// </summary>
public sealed class Pm4WmoGeometryMatcher
{
    public record GeometryStats(
        int VertexCount,
        int FaceCount,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        Vector3 Centroid,
        Vector3 Dimensions,
        Matrix4x4 CovarianceMatrix,
        Vector3[] PrincipalAxes,
        float[] PrincipalExtents);

    public record PlacementTransform(
        Vector3 Position,
        Vector3 Rotation,  // Euler angles in degrees
        float Scale,
        float MatchConfidence);

    /// <summary>
    /// Load vertices from an OBJ file.
    /// </summary>
    public List<Vector3> LoadObjVertices(string objPath)
    {
        var vertices = new List<Vector3>();
        
        foreach (var line in File.ReadLines(objPath))
        {
            if (line.StartsWith("v "))
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 4)
                {
                    float x = float.Parse(parts[1]);
                    float y = float.Parse(parts[2]);
                    float z = float.Parse(parts[3]);
                    vertices.Add(new Vector3(x, y, z));
                }
            }
        }
        
        return vertices;
    }

    /// <summary>
    /// Load vertices from OBJ content string (in-memory).
    /// </summary>
    public List<Vector3> LoadObjVerticesFromText(string objContent)
    {
        var vertices = new List<Vector3>();
        using var reader = new StringReader(objContent);
        
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (line.StartsWith("v "))
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 4)
                {
                    if (float.TryParse(parts[1], out float x) &&
                        float.TryParse(parts[2], out float y) &&
                        float.TryParse(parts[3], out float z))
                    {
                        vertices.Add(new Vector3(x, y, z));
                    }
                }
            }
        }
        
        return vertices;
    }

    /// <summary>
    /// Compute geometric statistics for a point cloud.
    /// </summary>
    public GeometryStats ComputeStats(List<Vector3> vertices)
    {
        if (vertices.Count == 0)
            throw new ArgumentException("No vertices provided");

        // Compute bounds and centroid
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        var sum = Vector3.Zero;

        foreach (var v in vertices)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
            sum += v;
        }

        var centroid = sum / vertices.Count;
        var dimensions = max - min;

        // Compute covariance matrix
        float cxx = 0, cyy = 0, czz = 0;
        float cxy = 0, cxz = 0, cyz = 0;

        foreach (var v in vertices)
        {
            var d = v - centroid;
            cxx += d.X * d.X;
            cyy += d.Y * d.Y;
            czz += d.Z * d.Z;
            cxy += d.X * d.Y;
            cxz += d.X * d.Z;
            cyz += d.Y * d.Z;
        }

        int n = vertices.Count;
        var covariance = new Matrix4x4(
            cxx / n, cxy / n, cxz / n, 0,
            cxy / n, cyy / n, cyz / n, 0,
            cxz / n, cyz / n, czz / n, 0,
            0, 0, 0, 1);

        // Compute principal axes via power iteration (simplified)
        var (axes, extents) = ComputePrincipalAxes(vertices, centroid);

        return new GeometryStats(
            vertices.Count,
            0, // Face count not computed here
            min,
            max,
            centroid,
            dimensions,
            covariance,
            axes,
            extents);
    }

    /// <summary>
    /// Compute principal axes using power iteration on covariance matrix.
    /// </summary>
    private (Vector3[] axes, float[] extents) ComputePrincipalAxes(List<Vector3> vertices, Vector3 centroid)
    {
        // Center the points
        var centered = vertices.Select(v => v - centroid).ToList();

        // Compute covariance matrix elements
        double cxx = 0, cyy = 0, czz = 0;
        double cxy = 0, cxz = 0, cyz = 0;

        foreach (var v in centered)
        {
            cxx += v.X * v.X;
            cyy += v.Y * v.Y;
            czz += v.Z * v.Z;
            cxy += v.X * v.Y;
            cxz += v.X * v.Z;
            cyz += v.Y * v.Z;
        }

        int n = centered.Count;
        cxx /= n; cyy /= n; czz /= n;
        cxy /= n; cxz /= n; cyz /= n;

        // Power iteration for dominant eigenvector
        var axes = new Vector3[3];
        var extents = new float[3];

        // First principal axis
        axes[0] = PowerIteration(cxx, cxy, cxz, cyy, cyz, czz);
        extents[0] = ComputeExtentAlongAxis(centered, axes[0]);

        // Second principal axis (orthogonal to first)
        // Deflate matrix and iterate again
        var (cxx2, cxy2, cxz2, cyy2, cyz2, czz2) = DeflateMatrix(cxx, cxy, cxz, cyy, cyz, czz, axes[0]);
        axes[1] = PowerIteration(cxx2, cxy2, cxz2, cyy2, cyz2, czz2);
        extents[1] = ComputeExtentAlongAxis(centered, axes[1]);

        // Third axis is cross product
        axes[2] = Vector3.Normalize(Vector3.Cross(axes[0], axes[1]));
        extents[2] = ComputeExtentAlongAxis(centered, axes[2]);

        return (axes, extents);
    }

    private Vector3 PowerIteration(double cxx, double cxy, double cxz, double cyy, double cyz, double czz, int iterations = 50)
    {
        var v = Vector3.Normalize(new Vector3(1, 1, 1));

        for (int i = 0; i < iterations; i++)
        {
            var newV = new Vector3(
                (float)(cxx * v.X + cxy * v.Y + cxz * v.Z),
                (float)(cxy * v.X + cyy * v.Y + cyz * v.Z),
                (float)(cxz * v.X + cyz * v.Y + czz * v.Z));
            
            if (newV.Length() < 0.0001f)
                break;
                
            v = Vector3.Normalize(newV);
        }

        return v;
    }

    private (double, double, double, double, double, double) DeflateMatrix(
        double cxx, double cxy, double cxz, double cyy, double cyz, double czz, Vector3 axis)
    {
        // Remove contribution of axis from covariance matrix
        double lambda = cxx * axis.X * axis.X + cyy * axis.Y * axis.Y + czz * axis.Z * axis.Z +
                       2 * cxy * axis.X * axis.Y + 2 * cxz * axis.X * axis.Z + 2 * cyz * axis.Y * axis.Z;

        return (
            cxx - lambda * axis.X * axis.X,
            cxy - lambda * axis.X * axis.Y,
            cxz - lambda * axis.X * axis.Z,
            cyy - lambda * axis.Y * axis.Y,
            cyz - lambda * axis.Y * axis.Z,
            czz - lambda * axis.Z * axis.Z);
    }

    private float ComputeExtentAlongAxis(List<Vector3> points, Vector3 axis)
    {
        float min = float.MaxValue;
        float max = float.MinValue;

        foreach (var p in points)
        {
            float proj = Vector3.Dot(p, axis);
            min = Math.Min(min, proj);
            max = Math.Max(max, proj);
        }

        return max - min;
    }

    /// <summary>
    /// Find the transformation that aligns WMO geometry to PM4 geometry.
    /// </summary>
    /// <param name="forceUnitScale">If true, forces scale to 1.0 (WMOs cannot be scaled)</param>
    public PlacementTransform FindAlignment(GeometryStats pm4Stats, GeometryStats wmoStats, bool forceUnitScale = true)
    {
        Console.WriteLine("\n=== Geometric Alignment Analysis ===\n");

        // Compare dimensions
        Console.WriteLine("Dimensions comparison:");
        Console.WriteLine($"  PM4: {pm4Stats.Dimensions.X:F1} x {pm4Stats.Dimensions.Y:F1} x {pm4Stats.Dimensions.Z:F1}");
        Console.WriteLine($"  WMO: {wmoStats.Dimensions.X:F1} x {wmoStats.Dimensions.Y:F1} x {wmoStats.Dimensions.Z:F1}");

        // Compare principal extents
        Console.WriteLine("\nPrincipal extents:");
        Console.WriteLine($"  PM4: {pm4Stats.PrincipalExtents[0]:F1}, {pm4Stats.PrincipalExtents[1]:F1}, {pm4Stats.PrincipalExtents[2]:F1}");
        Console.WriteLine($"  WMO: {wmoStats.PrincipalExtents[0]:F1}, {wmoStats.PrincipalExtents[1]:F1}, {wmoStats.PrincipalExtents[2]:F1}");

        // Compare principal axes
        Console.WriteLine("\nPrincipal axes:");
        for (int i = 0; i < 3; i++)
        {
            var pm4Axis = pm4Stats.PrincipalAxes[i];
            var wmoAxis = wmoStats.PrincipalAxes[i];
            float dot = Math.Abs(Vector3.Dot(pm4Axis, wmoAxis));
            Console.WriteLine($"  Axis {i}: PM4=({pm4Axis.X:F3}, {pm4Axis.Y:F3}, {pm4Axis.Z:F3}) WMO=({wmoAxis.X:F3}, {wmoAxis.Y:F3}, {wmoAxis.Z:F3}) dot={dot:F3}");
        }

        // WMOs cannot be scaled - they are always 1.0
        // For M2/MDX models, scale can vary
        float scale = 1.0f;
        var pm4Extents = pm4Stats.PrincipalExtents.OrderByDescending(x => x).ToArray();
        var wmoExtents = wmoStats.PrincipalExtents.OrderByDescending(x => x).ToArray();
        
        float computedScale = 1.0f;
        if (wmoExtents[0] > 0.001f)
        {
            computedScale = pm4Extents[0] / wmoExtents[0];
        }

        if (forceUnitScale)
        {
            scale = 1.0f;
            Console.WriteLine($"\nScale: 1.0 (WMO - forced unit scale, computed was {computedScale:F4})");
        }
        else
        {
            scale = computedScale;
            Console.WriteLine($"\nEstimated scale: {scale:F4}");
        }

        // ROTATION: Zero for now - heading calculation was producing wrong orientations
        // TODO: Investigate why principal axis rotation doesn't match MODF rotation format
        var eulerDegrees = new Vector3(0, 0, 0);
        Console.WriteLine($"Estimated rotation: ({eulerDegrees.X:F1}°, {eulerDegrees.Y:F1}°, {eulerDegrees.Z:F1}°) [ZEROED]");

        // Compute translation using bounding box centers
        // BB center is more accurate than vertex centroid because WMO origin is typically
        // near the geometric center of the bounding box, not the average of walkable vertices
        var pm4BoundsCenter = (pm4Stats.BoundsMin + pm4Stats.BoundsMax) / 2;
        var wmoBoundsCenter = (wmoStats.BoundsMin + wmoStats.BoundsMax) / 2;
        var translation = pm4BoundsCenter - (wmoBoundsCenter * scale);
        Console.WriteLine($"Estimated translation: ({translation.X:F1}, {translation.Y:F1}, {translation.Z:F1})");

        // Compute match confidence
        // For WMOs (unit scale): confidence based on how close extents match at 1:1
        // For M2s (variable scale): confidence based on ratio consistency
        float confidence;
        if (forceUnitScale)
        {
            // For WMOs: extents should match closely at 1:1 scale
            // Compute how well the extents match (normalized difference)
            float extentDiff1 = Math.Abs(pm4Extents[0] - wmoExtents[0]) / Math.Max(pm4Extents[0], wmoExtents[0]);
            float extentDiff2 = Math.Abs(pm4Extents[1] - wmoExtents[1]) / Math.Max(pm4Extents[1], wmoExtents[1]);
            float extentDiff3 = Math.Abs(pm4Extents[2] - wmoExtents[2]) / Math.Max(pm4Extents[2], wmoExtents[2]);
            float avgDiff = (extentDiff1 + extentDiff2 + extentDiff3) / 3;
            
            // Confidence decreases as difference increases
            // 0% diff = 100% confidence, 50% diff = 0% confidence
            confidence = Math.Max(0, 1 - avgDiff * 2);
        }
        else
        {
            // For M2s: ratios should be consistent (same scale across all axes)
            float extentRatio1 = pm4Extents[0] / wmoExtents[0];
            float extentRatio2 = pm4Extents[1] / wmoExtents[1];
            float extentRatio3 = pm4Extents[2] / wmoExtents[2];
            float avgRatio = (extentRatio1 + extentRatio2 + extentRatio3) / 3;
            float ratioVariance = ((extentRatio1 - avgRatio) * (extentRatio1 - avgRatio) +
                                  (extentRatio2 - avgRatio) * (extentRatio2 - avgRatio) +
                                  (extentRatio3 - avgRatio) * (extentRatio3 - avgRatio)) / 3;
            confidence = Math.Max(0, 1 - ratioVariance * 10);
        }
        Console.WriteLine($"Match confidence: {confidence:P1}");

        return new PlacementTransform(translation, eulerDegrees, scale, confidence);
    }

    private Quaternion FindRotationBetweenAxes(Vector3[] fromAxes, Vector3[] toAxes)
    {
        // Build rotation matrix from axis correspondence
        // This is simplified - assumes axes are orthonormal
        
        // Try to match axes by finding best correspondence
        var bestRotation = Quaternion.Identity;
        float bestScore = -1;

        // Try different axis permutations and signs
        int[] perms = { 0, 1, 2 };
        int[] signs = { 1, -1 };

        foreach (var p0 in new[] { 0, 1, 2 })
        foreach (var p1 in new[] { 0, 1, 2 })
        foreach (var p2 in new[] { 0, 1, 2 })
        foreach (var s0 in signs)
        foreach (var s1 in signs)
        foreach (var s2 in signs)
        {
            if (p0 == p1 || p1 == p2 || p0 == p2) continue;

            var from0 = fromAxes[p0] * s0;
            var from1 = fromAxes[p1] * s1;
            var from2 = fromAxes[p2] * s2;

            // Compute rotation that maps from -> to
            var rot = ComputeRotationMatrix(from0, from1, from2, toAxes[0], toAxes[1], toAxes[2]);
            
            // Score this rotation
            float score = 0;
            for (int i = 0; i < 3; i++)
            {
                var rotated = Vector3.Transform(fromAxes[i], rot);
                score += Math.Abs(Vector3.Dot(rotated, toAxes[i]));
            }

            if (score > bestScore)
            {
                bestScore = score;
                bestRotation = rot;
            }
        }

        return bestRotation;
    }

    private Quaternion ComputeRotationMatrix(Vector3 from0, Vector3 from1, Vector3 from2,
                                             Vector3 to0, Vector3 to1, Vector3 to2)
    {
        // Build matrices and compute rotation
        // Simplified: use first axis alignment
        var axis = Vector3.Cross(from0, to0);
        if (axis.Length() < 0.0001f)
        {
            // Axes are parallel or anti-parallel
            if (Vector3.Dot(from0, to0) < 0)
                return Quaternion.CreateFromAxisAngle(from1, MathF.PI);
            return Quaternion.Identity;
        }

        axis = Vector3.Normalize(axis);
        float angle = MathF.Acos(Math.Clamp(Vector3.Dot(from0, to0), -1, 1));
        return Quaternion.CreateFromAxisAngle(axis, angle);
    }

    private Vector3 QuaternionToEulerDegrees(Quaternion q)
    {
        // Convert quaternion to Euler angles (in degrees)
        float sinr_cosp = 2 * (q.W * q.X + q.Y * q.Z);
        float cosr_cosp = 1 - 2 * (q.X * q.X + q.Y * q.Y);
        float roll = MathF.Atan2(sinr_cosp, cosr_cosp);

        float sinp = 2 * (q.W * q.Y - q.Z * q.X);
        float pitch = MathF.Abs(sinp) >= 1 ? MathF.CopySign(MathF.PI / 2, sinp) : MathF.Asin(sinp);

        float siny_cosp = 2 * (q.W * q.Z + q.X * q.Y);
        float cosy_cosp = 1 - 2 * (q.Y * q.Y + q.Z * q.Z);
        float yaw = MathF.Atan2(siny_cosp, cosy_cosp);

        return new Vector3(
            roll * 180 / MathF.PI,
            pitch * 180 / MathF.PI,
            yaw * 180 / MathF.PI);
    }

    /// <summary>
    /// Transform WMO vertices using the computed placement and export to OBJ.
    /// </summary>
    public void ExportTransformedWmo(string wmoObjPath, PlacementTransform transform, string outputPath)
    {
        var rotation = Quaternion.CreateFromYawPitchRoll(
            transform.Rotation.Z * MathF.PI / 180,
            transform.Rotation.Y * MathF.PI / 180,
            transform.Rotation.X * MathF.PI / 180);

        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("# WMO transformed to PM4 world space");
        sw.WriteLine($"# Position: {transform.Position}");
        sw.WriteLine($"# Rotation: {transform.Rotation}");
        sw.WriteLine($"# Scale: {transform.Scale}");
        sw.WriteLine();

        foreach (var line in File.ReadLines(wmoObjPath))
        {
            if (line.StartsWith("v "))
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 4)
                {
                    float x = float.Parse(parts[1]);
                    float y = float.Parse(parts[2]);
                    float z = float.Parse(parts[3]);

                    var v = new Vector3(x, y, z);
                    v *= transform.Scale;
                    v = Vector3.Transform(v, rotation);
                    v += transform.Position;

                    sw.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6}");
                }
            }
            else
            {
                sw.WriteLine(line);
            }
        }

        Console.WriteLine($"[INFO] Exported transformed WMO to: {outputPath}");
    }

    /// <summary>
    /// Full analysis pipeline: load both OBJs, compute stats, find alignment, export transformed WMO.
    /// </summary>
    public PlacementTransform AnalyzeAndAlign(string pm4ObjPath, string wmoObjPath, string? outputTransformedPath = null)
    {
        Console.WriteLine("Loading PM4 geometry...");
        var pm4Verts = LoadObjVertices(pm4ObjPath);
        var pm4Stats = ComputeStats(pm4Verts);
        Console.WriteLine($"  {pm4Verts.Count} vertices, bounds: {pm4Stats.BoundsMin} to {pm4Stats.BoundsMax}");

        Console.WriteLine("Loading WMO geometry...");
        var wmoVerts = LoadObjVertices(wmoObjPath);
        var wmoStats = ComputeStats(wmoVerts);
        Console.WriteLine($"  {wmoVerts.Count} vertices, bounds: {wmoStats.BoundsMin} to {wmoStats.BoundsMax}");

        var transform = FindAlignment(pm4Stats, wmoStats);

        if (!string.IsNullOrEmpty(outputTransformedPath))
        {
            ExportTransformedWmo(wmoObjPath, transform, outputTransformedPath);
        }

        return transform;
    }
}
