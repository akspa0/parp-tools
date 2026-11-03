using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using WowViewer.Core.PM4.Footprint;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// A candidate asset indexed for direct geometric matching.
/// </summary>
public sealed record Pm4ModelCandidate(
    string ModelPath,
    string ModelName,
    string AssetKind,
    Vector3 LocalBoundsMin,
    Vector3 LocalBoundsMax,
    Pm4GeometricShapeDescriptor Descriptor)
{
    public static Pm4ModelCandidate Create(string modelPath, string assetKind, Vector3 localMin, Vector3 localMax)
    {
        string name = Path.GetFileName(modelPath);
        var desc = Pm4GeometricShapeDescriptor.FromBounds(localMin, localMax);
        return new Pm4ModelCandidate(modelPath, name, assetKind, localMin, localMax, desc);
    }
}

/// <summary>
/// Result of matching a PM4 3D footprint prism to a candidate asset.
/// </summary>
public sealed record Pm4DirectMatchResult(
    Pm4ModelCandidate Candidate,
    double Score,
    float SolvedScale,
    Vector3 SolvedRotation,
    Vector3 SolvedPosition,
    string Explanation);

/// <summary>
/// Pure algorithmic zero-donor asset matcher that identifies authentic 3D models directly
/// from PM4 footprint prisms without requiring donor ADTs or Rosetta cross-reference maps.
/// </summary>
public static class Pm4DirectModelMatcher
{
    /// <summary>
    /// Matches a PM4 footprint prism against a library of candidate model bounds.
    /// </summary>
    public static List<Pm4DirectMatchResult> MatchPrism(
        Pm4FootprintPrism prism,
        IReadOnlyList<Pm4ModelCandidate> candidates,
        int maxResults = 5)
    {
        ArgumentNullException.ThrowIfNull(prism);
        if (candidates == null || candidates.Count == 0)
            return [];

        var prismDesc = Pm4GeometricShapeDescriptor.FromPrism(prism);
        var results = new List<Pm4DirectMatchResult>();

        for (int i = 0; i < candidates.Count; i++)
        {
            Pm4ModelCandidate cand = candidates[i];
            Pm4GeometricShapeDescriptor candDesc = cand.Descriptor;

            // 1. Compute scale-invariant aspect ratio differences
            float diffXY = MathF.Abs(prismDesc.AspectXY - candDesc.AspectXY) / MathF.Max(prismDesc.AspectXY, candDesc.AspectXY);
            float diffZH = MathF.Abs(prismDesc.AspectZH - candDesc.AspectZH) / MathF.Max(prismDesc.AspectZH, candDesc.AspectZH);

            // 2. Solve scale factor
            float candPlanar = MathF.Sqrt(candDesc.Length * candDesc.Length + candDesc.Width * candDesc.Width);
            float prismPlanar = MathF.Sqrt(prismDesc.Length * prismDesc.Length + prismDesc.Width * prismDesc.Width);
            float rawScale = candPlanar > 0.01f ? prismPlanar / candPlanar : 1.0f;
            float scale = Math.Clamp(rawScale, 0.35f, 3.0f);

            // Size mismatch penalty:
            // Authentic ADT doodads and structures do not deviate by 5x-10x in scale.
            // When rawScale diverges significantly from 1.0 (e.g. matching a 2-yard bench to a 50-yard bridge),
            // penalize the score heavily so absurd scale-stretched matches are eliminated.
            float scalePenalty = 0f;
            if (rawScale > 2.2f)
                scalePenalty = MathF.Min(1.0f, (rawScale - 2.2f) / 1.5f);
            else if (rawScale < 0.45f)
                scalePenalty = MathF.Min(1.0f, (0.45f - rawScale) / 0.25f);

            if (rawScale > 4.0f || rawScale < 0.2f)
                scalePenalty = 1.0f;

            double score = Math.Max(0.0, 1.0 - (0.5 * diffXY + 0.5 * diffZH) - (0.5 * scalePenalty));

            // Small doodad vs giant structure penalty:
            // A 20+ yard footprint is architectural (bridge, building, wall); small M2 props (candPlanar < 4.0f) should not match it.
            if (prismPlanar > 20.0f && candPlanar < 4.0f && cand.AssetKind.Equals("M2", StringComparison.OrdinalIgnoreCase))
            {
                score *= 0.5;
            }

            // 3. Solve Yaw angle (aligning candidate local orientation to PM4 principal axis)
            float yawDegrees = prism.YawDegrees;
            float candDx = MathF.Abs(cand.LocalBoundsMax.X - cand.LocalBoundsMin.X);
            float candDy = MathF.Abs(cand.LocalBoundsMax.Y - cand.LocalBoundsMin.Y);
            if (candDy > candDx * 1.15f)
            {
                yawDegrees += 90f;
                while (yawDegrees > 180f) yawDegrees -= 360f;
                while (yawDegrees < -180f) yawDegrees += 360f;
            }

            // 4. Solve Placement Position using WC3-heritage asymmetric pivot offset
            // Candidate origin (0,0,0) is offset from candidate bounding center by candDesc.LocalOriginOffset.
            // When scaled and rotated, this places the authored entrance/doorstep/pivot at the authentic position.
            float rotZRad = yawDegrees * (MathF.PI / 180f);
            Matrix4x4 rotMatrix = Matrix4x4.CreateRotationZ(MathF.PI) * Matrix4x4.CreateRotationZ(rotZRad);
            Vector3 rotatedOriginOffset = Vector3.Transform(candDesc.LocalOriginOffset * scale, rotMatrix);

            float posX = prism.Centroid.X + rotatedOriginOffset.X;
            float posY = prism.Centroid.Y + rotatedOriginOffset.Y;

            // 5. Solve vertical placement (Z):
            // Doodads (M2): align top surface to prism.TopZ: P_Z = TopZ - (LocalBoundsMax.Z * scale)
            // WMOs: align base to prism.BaseZ: P_Z = BaseZ - (LocalBoundsMin.Z * scale)
            float posZ;
            if (cand.AssetKind.Equals("M2", StringComparison.OrdinalIgnoreCase))
            {
                float modelTopZ = cand.LocalBoundsMax.Z > 0.01f ? cand.LocalBoundsMax.Z : candDesc.Height;
                posZ = prism.TopZ - (modelTopZ * scale);
            }
            else
            {
                posZ = prism.BaseZ - (cand.LocalBoundsMin.Z * scale);
            }

            Vector3 solvedPos = new(posX, posY, posZ);
            Vector3 solvedRot = new(0f, 0f, yawDegrees);

            string explanation = $"AspectXY: {candDesc.AspectXY:F2} (prism: {prismDesc.AspectXY:F2}), " +
                                 $"AspectZH: {candDesc.AspectZH:F2} (prism: {prismDesc.AspectZH:F2}), scale: {scale:F2}";

            results.Add(new Pm4DirectMatchResult(
                cand,
                score,
                scale,
                solvedRot,
                solvedPos,
                explanation));
        }

        results.Sort(static (a, b) => b.Score.CompareTo(a.Score));
        if (results.Count > maxResults)
            results.RemoveRange(maxResults, results.Count - maxResults);

        return results;
    }
}

