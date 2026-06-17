using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4FingerprintExtractorTests
{
    private static readonly byte Ck24TypeWmo = 0x42;
    private static readonly IReadOnlyDictionary<byte, int> EmptyTypeFlags = new Dictionary<byte, int>();

    [Fact]
    public void ExtractFromGeometry_KnownBox_ProducesCorrectSortedDimensions()
    {
        List<Vector3> vertices =
        [
            new(0f, 0f, 0f),
            new(10f, 0f, 0f),
            new(10f, 20f, 0f),
            new(0f, 20f, 0f),
            new(0f, 0f, 30f),
            new(10f, 0f, 30f),
            new(10f, 20f, 30f),
            new(0f, 20f, 30f),
        ];
        int[] indices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

        Pm4FingerprintRecord? fingerprint = Pm4FingerprintExtractor.ExtractFromGeometry(
            vertices, indices, surfaceCount: 4, ck24Type: Ck24TypeWmo,
            typeFlagsProfile: EmptyTypeFlags,
            assetId: "test-box", assetPath: "test.box", assetKind: "wmo");

        Assert.NotNull(fingerprint);
        Assert.Equal(10f, fingerprint.SortedDim0, 1f);
        Assert.Equal(20f, fingerprint.SortedDim1, 1f);
        Assert.Equal(30f, fingerprint.SortedDim2, 1f);
        Assert.Equal(8, fingerprint.VertexCount);
        Assert.Equal(12, fingerprint.IndexCount);
        Assert.Equal(4, fingerprint.SurfaceCount);
        Assert.True(fingerprint.FootprintArea > 150f);
        Assert.True(fingerprint.FootprintArea < 250f);
        Assert.True(fingerprint.NormalizedFootprintHull.Count >= 4);
    }

    [Fact]
    public void ExtractFromGeometry_RotatedBox_ProducesSameHullAsOriginal()
    {
        List<Vector3> originalVertices =
        [
            new(0f, 0f, 0f),
            new(10f, 0f, 0f),
            new(10f, 20f, 0f),
            new(0f, 20f, 0f),
            new(0f, 0f, 30f),
            new(10f, 0f, 30f),
            new(10f, 20f, 30f),
            new(0f, 20f, 30f),
        ];
        int[] indices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

        float angle = MathF.PI / 4f;
        float cos = MathF.Cos(angle);
        float sin = MathF.Sin(angle);
        List<Vector3> rotatedVertices = originalVertices
            .Select(v => new Vector3(
                v.X * cos - v.Y * sin + 100f,
                v.X * sin + v.Y * cos + 200f,
                v.Z + 50f))
            .ToList();

        Pm4FingerprintRecord? originalFp = Pm4FingerprintExtractor.ExtractFromGeometry(
            originalVertices, indices, 4, Ck24TypeWmo, EmptyTypeFlags,
            "box-orig", "orig.box", "wmo");
        Pm4FingerprintRecord? rotatedFp = Pm4FingerprintExtractor.ExtractFromGeometry(
            rotatedVertices, indices, 4, Ck24TypeWmo, EmptyTypeFlags,
            "box-rot", "rot.box", "wmo");

        Assert.NotNull(originalFp);
        Assert.NotNull(rotatedFp);

        Assert.Equal(originalFp.SortedDim0, rotatedFp.SortedDim0, 0.5f);
        Assert.Equal(originalFp.SortedDim1, rotatedFp.SortedDim1, 0.5f);
        Assert.Equal(originalFp.SortedDim2, rotatedFp.SortedDim2, 0.5f);
        Assert.Equal(originalFp.FootprintArea, rotatedFp.FootprintArea, 2f);

        float overlap = ComputeHullOverlap(
            originalFp.NormalizedFootprintHull,
            rotatedFp.NormalizedFootprintHull);
        Assert.True(overlap >= 0.95f, $"Rotation invariance failed: overlap={overlap:F3}");
    }

    [Fact]
    public void ExtractFromGeometry_LShape_ProducesCorrectHull()
    {
        List<Vector3> vertices =
        [
            new(0f, 0f, 0f),
            new(30f, 0f, 0f),
            new(30f, 10f, 0f),
            new(10f, 10f, 0f),
            new(10f, 20f, 0f),
            new(0f, 20f, 0f),
        ];
        int[] indices = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5];

        Pm4FingerprintRecord? fingerprint = Pm4FingerprintExtractor.ExtractFromGeometry(
            vertices, indices, 4, Ck24TypeWmo, EmptyTypeFlags,
            "l-shape", "l.shape", "wmo");

        Assert.NotNull(fingerprint);
        Assert.True(fingerprint.NormalizedFootprintHull.Count >= 4);
        Assert.True(fingerprint.FootprintArea > 500f);
        Assert.True(fingerprint.FootprintArea < 700f);
    }

    [Fact]
    public void ExtractFromGeometry_DegenerateLessThan3Vertices_ReturnsNull()
    {
        List<Vector3> vertices = [new(0f, 0f, 0f), new(1f, 0f, 0f)];
        int[] indices = [0, 1, 0];

        Pm4FingerprintRecord? fingerprint = Pm4FingerprintExtractor.ExtractFromGeometry(
            vertices, indices, 1, Ck24TypeWmo, EmptyTypeFlags,
            "degenerate", "deg.box", "wmo");

        Assert.Null(fingerprint);
    }

    [Fact]
    public void ExtractFromGeometry_CollinearPoints_ReturnsNull()
    {
        List<Vector3> vertices =
        [
            new(0f, 0f, 0f),
            new(5f, 0f, 0f),
            new(10f, 0f, 0f),
        ];
        int[] indices = [0, 1, 2];

        Pm4FingerprintRecord? fingerprint = Pm4FingerprintExtractor.ExtractFromGeometry(
            vertices, indices, 1, Ck24TypeWmo, EmptyTypeFlags,
            "collinear", "col.box", "wmo");

        Assert.Null(fingerprint);
    }

    [Fact]
    public void ExtractFromGeometry_NearSymmetricSquare_FlagsNearSymmetric()
    {
        List<Vector3> vertices =
        [
            new(0f, 0f, 0f),
            new(10f, 0f, 0f),
            new(10f, 10f, 0f),
            new(0f, 10f, 0f),
            new(0f, 0f, 5f),
            new(10f, 0f, 5f),
            new(10f, 10f, 5f),
            new(0f, 10f, 5f),
        ];
        int[] indices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

        Pm4FingerprintRecord? fingerprint = Pm4FingerprintExtractor.ExtractFromGeometry(
            vertices, indices, 4, Ck24TypeWmo, EmptyTypeFlags,
            "square", "square.box", "wmo");

        Assert.NotNull(fingerprint);
        Assert.Contains("near-symmetric", fingerprint.SourceLabel);
    }

    [Fact]
    public void ExtractFromGeometry_NonSymmetricRect_NoNearSymmetricFlag()
    {
        List<Vector3> vertices =
        [
            new(0f, 0f, 0f),
            new(40f, 0f, 0f),
            new(40f, 10f, 0f),
            new(0f, 10f, 0f),
            new(0f, 0f, 20f),
            new(40f, 0f, 20f),
            new(40f, 10f, 20f),
            new(0f, 10f, 20f),
        ];
        int[] indices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

        Pm4FingerprintRecord? fingerprint = Pm4FingerprintExtractor.ExtractFromGeometry(
            vertices, indices, 4, Ck24TypeWmo, EmptyTypeFlags,
            "rect", "rect.box", "wmo");

        Assert.NotNull(fingerprint);
        Assert.DoesNotContain("near-symmetric", fingerprint.SourceLabel);
    }

    [Fact]
    public void FlipHull_ProducesValidHullForAllFlipCombinations()
    {
        List<Vector2> hull =
        [
            new(0f, 0f),
            new(10f, 0f),
            new(10f, 5f),
            new(0f, 5f),
        ];

        IReadOnlyList<Vector2> flipNone = Pm4FingerprintExtractor.FlipHull(hull, false, false);
        IReadOnlyList<Vector2> flipX = Pm4FingerprintExtractor.FlipHull(hull, true, false);
        IReadOnlyList<Vector2> flipY = Pm4FingerprintExtractor.FlipHull(hull, false, true);
        IReadOnlyList<Vector2> flipXY = Pm4FingerprintExtractor.FlipHull(hull, true, true);

        Assert.True(flipNone.Count >= 4);
        Assert.True(flipX.Count >= 4);
        Assert.True(flipY.Count >= 4);
        Assert.True(flipXY.Count >= 4);

        Assert.All(flipX, p => Assert.True(p.X <= 0f));
        Assert.All(flipY, p => Assert.True(p.Y <= 0f));
    }

    [Fact]
    public void ExtractFromGeometry_RotatedLShape_MatchesOriginalHull()
    {
        List<Vector3> originalVertices =
        [
            new(0f, 0f, 0f),
            new(30f, 0f, 0f),
            new(30f, 10f, 0f),
            new(10f, 10f, 0f),
            new(10f, 20f, 0f),
            new(0f, 20f, 0f),
        ];
        int[] indices = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5];

        float angle = MathF.PI / 3f;
        float cos = MathF.Cos(angle);
        float sin = MathF.Sin(angle);
        List<Vector3> rotatedVertices = originalVertices
            .Select(v => new Vector3(
                v.X * cos - v.Y * sin + 500f,
                v.X * sin + v.Y * cos + 300f,
                v.Z))
            .ToList();

        Pm4FingerprintRecord? originalFp = Pm4FingerprintExtractor.ExtractFromGeometry(
            originalVertices, indices, 4, Ck24TypeWmo, EmptyTypeFlags,
            "l-orig", "l.shape", "wmo");
        Pm4FingerprintRecord? rotatedFp = Pm4FingerprintExtractor.ExtractFromGeometry(
            rotatedVertices, indices, 4, Ck24TypeWmo, EmptyTypeFlags,
            "l-rot", "l.shape", "wmo");

        Assert.NotNull(originalFp);
        Assert.NotNull(rotatedFp);

        float overlap = ComputeHullOverlap(
            originalFp.NormalizedFootprintHull,
            rotatedFp.NormalizedFootprintHull);
        Assert.True(overlap >= 0.90f, $"L-shape rotation invariance failed: overlap={overlap:F3}");
    }

    private static float ComputeHullOverlap(IReadOnlyList<HullPoint2> hullA, IReadOnlyList<HullPoint2> hullB)
    {
        List<Vector2> a = hullA.Select(static p => p.AsVector2()).ToList();
        List<Vector2> b = hullB.Select(static p => p.AsVector2()).ToList();

        Vector2[] bestFlipped = a.ToArray();
        float bestOverlap = EvaluateFootprintOverlap(bestFlipped, b);

        foreach ((bool fx, bool fy) in new[] { (true, false), (false, true), (true, true) })
        {
            Vector2[] flipped = new Vector2[a.Count];
            for (int i = 0; i < a.Count; i++)
            {
                flipped[i] = new Vector2(
                    fx ? -a[i].X : a[i].X,
                    fy ? -a[i].Y : a[i].Y);
            }
            float overlap = EvaluateFootprintOverlap(flipped, b);
            if (overlap > bestOverlap)
            {
                bestOverlap = overlap;
                bestFlipped = flipped;
            }
        }

        return bestOverlap;
    }

    private static float EvaluateFootprintOverlap(IReadOnlyList<Vector2> hullA, IReadOnlyList<Vector2> hullB)
    {
        float areaA = ComputePolygonArea(hullA);
        float areaB = ComputePolygonArea(hullB);
        float minArea = MathF.Min(areaA, areaB);
        if (minArea <= 0f)
            return 0f;

        List<Vector2> intersection = ClipConvex(hullA, hullB);
        if (intersection.Count < 3)
            return 0f;

        return Math.Clamp(ComputePolygonArea(intersection) / minArea, 0f, 1f);
    }

    private static float ComputePolygonArea(IReadOnlyList<Vector2> polygon)
    {
        if (polygon.Count < 3)
            return 0f;
        float twiceArea = 0f;
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 current = polygon[i];
            Vector2 next = polygon[(i + 1) % polygon.Count];
            twiceArea += current.X * next.Y - next.X * current.Y;
        }
        return MathF.Abs(twiceArea * 0.5f);
    }

    private static List<Vector2> ClipConvex(IReadOnlyList<Vector2> subject, IReadOnlyList<Vector2> clip)
    {
        List<Vector2> output = new(subject);
        for (int edge = 0; edge < clip.Count; edge++)
        {
            Vector2 clipStart = clip[edge];
            Vector2 clipEnd = clip[(edge + 1) % clip.Count];
            List<Vector2> input = output;
            output = new List<Vector2>();
            if (input.Count == 0)
                break;

            Vector2 start = input[^1];
            bool startInside = IsInside(start, clipStart, clipEnd);
            for (int i = 0; i < input.Count; i++)
            {
                Vector2 end = input[i];
                bool endInside = IsInside(end, clipStart, clipEnd);
                if (endInside)
                {
                    if (!startInside)
                        output.Add(LineIntersection(start, end, clipStart, clipEnd));
                    output.Add(end);
                }
                else if (startInside)
                {
                    output.Add(LineIntersection(start, end, clipStart, clipEnd));
                }
                start = end;
                startInside = endInside;
            }
        }
        return output;
    }

    private static bool IsInside(Vector2 point, Vector2 edgeStart, Vector2 edgeEnd)
    {
        float cross = (edgeEnd.X - edgeStart.X) * (point.Y - edgeStart.Y)
            - (edgeEnd.Y - edgeStart.Y) * (point.X - edgeStart.X);
        return cross >= -0.0001f;
    }

    private static Vector2 LineIntersection(Vector2 a0, Vector2 a1, Vector2 b0, Vector2 b1)
    {
        float ax = a1.X - a0.X;
        float ay = a1.Y - a0.Y;
        float bx = b1.X - b0.X;
        float by = b1.Y - b0.Y;
        float denom = ax * by - ay * bx;
        if (MathF.Abs(denom) < 0.0001f)
            return a1;
        float t = ((b0.X - a0.X) * by - (b0.Y - a0.Y) * bx) / denom;
        return new Vector2(a0.X + ax * t, a0.Y + ay * t);
    }
}
