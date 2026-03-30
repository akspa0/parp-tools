using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4CorrelationMath
{
    public static IReadOnlyList<Pm4CorrelationObjectState> BuildObjectStatesFromGeometry(IReadOnlyList<Pm4CorrelationGeometryInput> inputs, int maxFootprintSamples = 192)
    {
        List<Pm4CorrelationObjectInput> resolvedInputs = new(inputs.Count);

        for (int index = 0; index < inputs.Count; index++)
        {
            Pm4CorrelationGeometryInput input = inputs[index];
            resolvedInputs.Add(new Pm4CorrelationObjectInput(
                input.TileX,
                input.TileY,
                input.GroupKey,
                input.Object,
                BuildWorldGeometryPoints(input.Lines, input.Triangles, input.GeometryTransform),
                Vector3.Transform(Vector3.Zero, input.GeometryTransform)));
        }

        return BuildObjectStates(resolvedInputs, maxFootprintSamples);
    }

    public static IReadOnlyList<Pm4CorrelationObjectState> BuildObjectStates(IReadOnlyList<Pm4CorrelationObjectInput> inputs, int maxFootprintSamples = 192)
    {
        int resolvedMaxFootprintSamples = Math.Max(1, maxFootprintSamples);
        List<Pm4CorrelationObjectState> states = new(inputs.Count);

        for (int index = 0; index < inputs.Count; index++)
        {
            Pm4CorrelationObjectInput input = inputs[index];
            ComputeBounds(input.WorldGeometryPoints, input.EmptyGeometryCenter, out Vector3 boundsMin, out Vector3 boundsMax, out Vector3 center);
            Vector2[] footprintHull = BuildFootprintHull(input.WorldGeometryPoints, resolvedMaxFootprintSamples);
            float footprintArea = ComputeFootprintArea(footprintHull);
            states.Add(new Pm4CorrelationObjectState(
                input.TileX,
                input.TileY,
                input.GroupKey,
                input.Object,
                boundsMin,
                boundsMax,
                center,
                footprintHull,
                footprintArea));
        }

        return states;
    }

    public static Pm4CorrelationMetrics EvaluateMetrics(
        Vector3 referenceBoundsMin,
        Vector3 referenceBoundsMax,
        Vector3 referenceCenter,
        IReadOnlyList<Vector2> referenceFootprintHull,
        float referenceFootprintArea,
        Vector3 candidateBoundsMin,
        Vector3 candidateBoundsMax,
        Vector3 candidateCenter,
        IReadOnlyList<Vector2> candidateFootprintHull,
        float candidateFootprintArea)
    {
        float planarGap = ComputePlanarAabbGap(referenceBoundsMin, referenceBoundsMax, candidateBoundsMin, candidateBoundsMax);
        float verticalGap = ComputeAxisGap(referenceBoundsMin.Z, referenceBoundsMax.Z, candidateBoundsMin.Z, candidateBoundsMax.Z);
        float centerDistance = Vector3.Distance(referenceCenter, candidateCenter);
        float planarOverlapRatio = ComputePlanarOverlapRatio(referenceBoundsMin, referenceBoundsMax, candidateBoundsMin, candidateBoundsMax);
        float volumeOverlapRatio = ComputeAabbOverlapRatio(referenceBoundsMin, referenceBoundsMax, candidateBoundsMin, candidateBoundsMax);
        float footprintOverlapRatio = ComputeConvexFootprintOverlapRatio(referenceFootprintHull, candidateFootprintHull, referenceFootprintArea, candidateFootprintArea);
        float footprintAreaRatio = ComputeFootprintAreaRatio(referenceFootprintArea, candidateFootprintArea);
        float footprintDistance = ComputeSymmetricFootprintDistance(referenceFootprintHull, candidateFootprintHull);

        return new Pm4CorrelationMetrics(
            planarGap,
            verticalGap,
            centerDistance,
            planarOverlapRatio,
            volumeOverlapRatio,
            footprintOverlapRatio,
            footprintAreaRatio,
            footprintDistance);
    }

    public static int CompareCandidateScores(Pm4CorrelationCandidateScore left, Pm4CorrelationCandidateScore right)
    {
        int compareSameTile = right.SameTile.CompareTo(left.SameTile);
        if (compareSameTile != 0)
            return compareSameTile;

        int compareFootprintOverlap = right.Metrics.FootprintOverlapRatio.CompareTo(left.Metrics.FootprintOverlapRatio);
        if (compareFootprintOverlap != 0)
            return compareFootprintOverlap;

        int comparePlanarOverlap = right.Metrics.PlanarOverlapRatio.CompareTo(left.Metrics.PlanarOverlapRatio);
        if (comparePlanarOverlap != 0)
            return comparePlanarOverlap;

        int compareFootprintArea = right.Metrics.FootprintAreaRatio.CompareTo(left.Metrics.FootprintAreaRatio);
        if (compareFootprintArea != 0)
            return compareFootprintArea;

        int compareVolumeOverlap = right.Metrics.VolumeOverlapRatio.CompareTo(left.Metrics.VolumeOverlapRatio);
        if (compareVolumeOverlap != 0)
            return compareVolumeOverlap;

        int compareFootprintDistance = left.Metrics.FootprintDistance.CompareTo(right.Metrics.FootprintDistance);
        if (compareFootprintDistance != 0)
            return compareFootprintDistance;

        int comparePlanarGap = left.Metrics.PlanarGap.CompareTo(right.Metrics.PlanarGap);
        if (comparePlanarGap != 0)
            return comparePlanarGap;

        int compareVerticalGap = left.Metrics.VerticalGap.CompareTo(right.Metrics.VerticalGap);
        if (compareVerticalGap != 0)
            return compareVerticalGap;

        return left.Metrics.CenterDistance.CompareTo(right.Metrics.CenterDistance);
    }

    public static Vector2[] BuildTransformedFootprintHull(IReadOnlyList<Vector3> sourcePoints, in Matrix4x4 transform)
    {
        if (sourcePoints.Count == 0)
            return [];

        List<Vector2> projected = new(sourcePoints.Count);
        for (int index = 0; index < sourcePoints.Count; index++)
        {
            Vector3 transformed = Vector3.Transform(sourcePoints[index], transform);
            projected.Add(new Vector2(transformed.X, transformed.Y));
        }

        return BuildConvexHull(projected);
    }

    public static Vector2[] BuildFootprintHull(IReadOnlyList<Vector3> worldPoints, int maxSamples = 192)
    {
        if (worldPoints.Count == 0)
            return [];

        int resolvedMaxSamples = Math.Max(1, maxSamples);
        int stride = Math.Max(1, worldPoints.Count / resolvedMaxSamples);
        List<Vector2> points = new(Math.Min(worldPoints.Count, resolvedMaxSamples));

        for (int index = 0; index < worldPoints.Count; index++)
        {
            if (points.Count >= resolvedMaxSamples)
                break;

            if (index % stride != 0)
                continue;

            Vector3 point = worldPoints[index];
            if (!float.IsFinite(point.X) || !float.IsFinite(point.Y))
                continue;

            points.Add(new Vector2(point.X, point.Y));
        }

        return BuildConvexHull(points);
    }

    public static float ComputeFootprintArea(IReadOnlyList<Vector2> polygon)
    {
        return ComputePolygonArea(polygon);
    }

    private static float ComputeFootprintAreaRatio(float areaA, float areaB)
    {
        float maxArea = MathF.Max(areaA, areaB);
        if (maxArea <= 0f)
            return 0f;

        return MathF.Min(areaA, areaB) / maxArea;
    }

    private static float ComputeConvexFootprintOverlapRatio(IReadOnlyList<Vector2> hullA, IReadOnlyList<Vector2> hullB, float areaA, float areaB)
    {
        if (hullA.Count < 3 || hullB.Count < 3)
            return 0f;

        float minArea = MathF.Min(areaA, areaB);
        if (minArea <= 0f)
            return 0f;

        List<Vector2> normalizedHullA = NormalizePolygonWinding(hullA);
        List<Vector2> normalizedHullB = NormalizePolygonWinding(hullB);
        List<Vector2> intersection = ClipConvexPolygon(normalizedHullA, normalizedHullB);
        if (intersection.Count < 3)
            return 0f;

        float ratio = ComputePolygonArea(intersection) / minArea;
        return Math.Clamp(ratio, 0f, 1f);
    }

    private static void ComputeBounds(IReadOnlyList<Vector3> points, Vector3 emptyGeometryCenter, out Vector3 boundsMin, out Vector3 boundsMax, out Vector3 center)
    {
        boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        bool hasBounds = false;

        for (int index = 0; index < points.Count; index++)
        {
            Vector3 point = points[index];
            if (!float.IsFinite(point.X) || !float.IsFinite(point.Y) || !float.IsFinite(point.Z))
                continue;

            boundsMin = Vector3.Min(boundsMin, point);
            boundsMax = Vector3.Max(boundsMax, point);
            hasBounds = true;
        }

        if (!hasBounds)
        {
            center = emptyGeometryCenter;
            boundsMin = emptyGeometryCenter;
            boundsMax = emptyGeometryCenter;
            return;
        }

        center = (boundsMin + boundsMax) * 0.5f;
    }

    private static List<Vector3> BuildWorldGeometryPoints(
        IReadOnlyList<Pm4GeometryLineSegment> lines,
        IReadOnlyList<Pm4GeometryTriangle> triangles,
        in Matrix4x4 geometryTransform)
    {
        int totalPointCount = lines.Count * 2 + triangles.Count * 3;
        List<Vector3> points = new(Math.Max(totalPointCount, 0));

        for (int index = 0; index < lines.Count; index++)
        {
            Pm4GeometryLineSegment line = lines[index];
            points.Add(Vector3.Transform(line.From, geometryTransform));
            points.Add(Vector3.Transform(line.To, geometryTransform));
        }

        for (int index = 0; index < triangles.Count; index++)
        {
            Pm4GeometryTriangle triangle = triangles[index];
            points.Add(Vector3.Transform(triangle.A, geometryTransform));
            points.Add(Vector3.Transform(triangle.B, geometryTransform));
            points.Add(Vector3.Transform(triangle.C, geometryTransform));
        }

        return points;
    }

    private static Vector2[] BuildConvexHull(IReadOnlyList<Vector2> points)
    {
        if (points.Count == 0)
            return [];

        List<Vector2> sorted = points
            .Where(static point => float.IsFinite(point.X) && float.IsFinite(point.Y))
            .Distinct()
            .OrderBy(static point => point.X)
            .ThenBy(static point => point.Y)
            .ToList();

        if (sorted.Count <= 2)
            return sorted.ToArray();

        static float Cross(in Vector2 origin, in Vector2 a, in Vector2 b)
        {
            return (a.X - origin.X) * (b.Y - origin.Y) - (a.Y - origin.Y) * (b.X - origin.X);
        }

        List<Vector2> lower = new(sorted.Count);
        for (int index = 0; index < sorted.Count; index++)
        {
            Vector2 point = sorted[index];
            while (lower.Count >= 2 && Cross(lower[^2], lower[^1], point) <= 0f)
                lower.RemoveAt(lower.Count - 1);

            lower.Add(point);
        }

        List<Vector2> upper = new(sorted.Count);
        for (int index = sorted.Count - 1; index >= 0; index--)
        {
            Vector2 point = sorted[index];
            while (upper.Count >= 2 && Cross(upper[^2], upper[^1], point) <= 0f)
                upper.RemoveAt(upper.Count - 1);

            upper.Add(point);
        }

        lower.RemoveAt(lower.Count - 1);
        upper.RemoveAt(upper.Count - 1);
        lower.AddRange(upper);
        return NormalizePolygonWinding(lower).ToArray();
    }

    private static List<Vector2> ClipConvexPolygon(IReadOnlyList<Vector2> subjectPolygon, IReadOnlyList<Vector2> clipPolygon)
    {
        List<Vector2> output = NormalizePolygonWinding(subjectPolygon);
        if (output.Count == 0)
            return output;

        List<Vector2> normalizedClipPolygon = NormalizePolygonWinding(clipPolygon);

        for (int edgeIndex = 0; edgeIndex < normalizedClipPolygon.Count; edgeIndex++)
        {
            Vector2 clipStart = normalizedClipPolygon[edgeIndex];
            Vector2 clipEnd = normalizedClipPolygon[(edgeIndex + 1) % normalizedClipPolygon.Count];
            List<Vector2> input = output;
            output = new List<Vector2>();
            if (input.Count == 0)
                break;

            Vector2 start = input[^1];
            bool startInside = IsInsideClipEdge(start, clipStart, clipEnd);
            for (int index = 0; index < input.Count; index++)
            {
                Vector2 end = input[index];
                bool endInside = IsInsideClipEdge(end, clipStart, clipEnd);

                if (endInside)
                {
                    if (!startInside)
                        output.Add(ComputeLineIntersection(start, end, clipStart, clipEnd));

                    output.Add(end);
                }
                else if (startInside)
                {
                    output.Add(ComputeLineIntersection(start, end, clipStart, clipEnd));
                }

                start = end;
                startInside = endInside;
            }

            output = RemoveDuplicatePolygonPoints(output);
        }

        return NormalizePolygonWinding(output);
    }

    private static bool IsInsideClipEdge(Vector2 point, Vector2 edgeStart, Vector2 edgeEnd)
    {
        float cross = (edgeEnd.X - edgeStart.X) * (point.Y - edgeStart.Y)
            - (edgeEnd.Y - edgeStart.Y) * (point.X - edgeStart.X);
        return cross >= -0.0001f;
    }

    private static Vector2 ComputeLineIntersection(Vector2 a0, Vector2 a1, Vector2 b0, Vector2 b1)
    {
        float ax = a1.X - a0.X;
        float ay = a1.Y - a0.Y;
        float bx = b1.X - b0.X;
        float by = b1.Y - b0.Y;
        float denominator = ax * by - ay * bx;
        if (MathF.Abs(denominator) < 0.0001f)
            return a1;

        float t = ((b0.X - a0.X) * by - (b0.Y - a0.Y) * bx) / denominator;
        return new Vector2(a0.X + ax * t, a0.Y + ay * t);
    }

    private static float ComputeSymmetricFootprintDistance(IReadOnlyList<Vector2> hullA, IReadOnlyList<Vector2> hullB)
    {
        if (hullA.Count == 0 || hullB.Count == 0)
            return float.PositiveInfinity;

        return (ComputeMeanNearestFootprintDistance(hullA, hullB) + ComputeMeanNearestFootprintDistance(hullB, hullA)) * 0.5f;
    }

    private static float ComputeMeanNearestFootprintDistance(IReadOnlyList<Vector2> source, IReadOnlyList<Vector2> target)
    {
        if (source.Count == 0 || target.Count == 0)
            return float.PositiveInfinity;

        float totalDistance = 0f;
        for (int sourceIndex = 0; sourceIndex < source.Count; sourceIndex++)
        {
            float bestDistanceSquared = float.PositiveInfinity;
            for (int targetIndex = 0; targetIndex < target.Count; targetIndex++)
            {
                float distanceSquared = Vector2.DistanceSquared(source[sourceIndex], target[targetIndex]);
                if (distanceSquared < bestDistanceSquared)
                    bestDistanceSquared = distanceSquared;
            }

            totalDistance += MathF.Sqrt(bestDistanceSquared);
        }

        return totalDistance / source.Count;
    }

    private static float ComputeAxisGap(float minA, float maxA, float minB, float maxB)
    {
        if (maxA < minB)
            return minB - maxA;

        if (maxB < minA)
            return minA - maxB;

        return 0f;
    }

    private static float ComputeOverlapLength(float minA, float maxA, float minB, float maxB)
    {
        return MathF.Max(0f, MathF.Min(maxA, maxB) - MathF.Max(minA, minB));
    }

    private static float ComputePlanarAabbGap(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB)
    {
        float dx = ComputeAxisGap(minA.X, maxA.X, minB.X, maxB.X);
        float dy = ComputeAxisGap(minA.Y, maxA.Y, minB.Y, maxB.Y);
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    private static float ComputePlanarOverlapRatio(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB)
    {
        float overlapX = ComputeOverlapLength(minA.X, maxA.X, minB.X, maxB.X);
        float overlapY = ComputeOverlapLength(minA.Y, maxA.Y, minB.Y, maxB.Y);
        float areaA = MathF.Max(0f, maxA.X - minA.X) * MathF.Max(0f, maxA.Y - minA.Y);
        float areaB = MathF.Max(0f, maxB.X - minB.X) * MathF.Max(0f, maxB.Y - minB.Y);
        float minArea = MathF.Min(areaA, areaB);
        if (minArea <= 0f)
            return 0f;

        return (overlapX * overlapY) / minArea;
    }

    private static float ComputeAabbOverlapRatio(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB)
    {
        float overlapX = ComputeOverlapLength(minA.X, maxA.X, minB.X, maxB.X);
        float overlapY = ComputeOverlapLength(minA.Y, maxA.Y, minB.Y, maxB.Y);
        float overlapZ = ComputeOverlapLength(minA.Z, maxA.Z, minB.Z, maxB.Z);
        float volumeA = MathF.Max(0f, maxA.X - minA.X) * MathF.Max(0f, maxA.Y - minA.Y) * MathF.Max(0f, maxA.Z - minA.Z);
        float volumeB = MathF.Max(0f, maxB.X - minB.X) * MathF.Max(0f, maxB.Y - minB.Y) * MathF.Max(0f, maxB.Z - minB.Z);
        float minVolume = MathF.Min(volumeA, volumeB);
        if (minVolume <= 0f)
            return 0f;

        return (overlapX * overlapY * overlapZ) / minVolume;
    }

    private static float ComputePolygonArea(IReadOnlyList<Vector2> polygon)
    {
        return MathF.Abs(ComputeSignedPolygonArea(polygon));
    }

    private static float ComputeSignedPolygonArea(IReadOnlyList<Vector2> polygon)
    {
        if (polygon.Count < 3)
            return 0f;

        float twiceArea = 0f;
        for (int index = 0; index < polygon.Count; index++)
        {
            Vector2 current = polygon[index];
            Vector2 next = polygon[(index + 1) % polygon.Count];
            twiceArea += current.X * next.Y - next.X * current.Y;
        }

        return twiceArea * 0.5f;
    }

    private static List<Vector2> NormalizePolygonWinding(IReadOnlyList<Vector2> polygon)
    {
        List<Vector2> normalized = RemoveDuplicatePolygonPoints(polygon);
        if (normalized.Count >= 3 && ComputeSignedPolygonArea(normalized) < 0f)
            normalized.Reverse();

        return normalized;
    }

    private static List<Vector2> RemoveDuplicatePolygonPoints(IReadOnlyList<Vector2> polygon)
    {
        const float epsilon = 0.0001f;
        List<Vector2> cleaned = new(polygon.Count);

        for (int index = 0; index < polygon.Count; index++)
        {
            Vector2 point = polygon[index];
            if (cleaned.Count == 0 || Vector2.DistanceSquared(cleaned[^1], point) > epsilon * epsilon)
                cleaned.Add(point);
        }

        if (cleaned.Count > 1 && Vector2.DistanceSquared(cleaned[0], cleaned[^1]) <= epsilon * epsilon)
            cleaned.RemoveAt(cleaned.Count - 1);

        return cleaned;
    }
}