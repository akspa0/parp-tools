using System.Numerics;

namespace Pm4Research.Core;

public static class Pm4ResearchMsurGeometryAnalyzer
{
    private const float DotStrongThreshold = 0.95f;
    private const float DotModerateThreshold = 0.75f;
    private const float UnitNormalTolerance = 0.1f;
    private const float MagnitudeEpsilon = 1e-5f;

    public static Pm4MsurGeometryReport AnalyzeDirectory(string inputDirectory)
    {
        List<Pm4ResearchFile> files = Directory
            .EnumerateFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        int analyzedSurfaceCount = 0;
        int degenerateSurfaceCount = 0;
        int unitLikeStoredNormalCount = 0;
        int strongAlignmentCount = 0;
        int moderateAlignmentCount = 0;
        int weakAlignmentCount = 0;
        int positiveAlignmentCount = 0;
        int negativeAlignmentCount = 0;
        float storedMagnitudeSum = 0f;
        float absoluteDotSum = 0f;

        var storedMagnitudeBuckets = new Dictionary<string, int>(StringComparer.Ordinal);
        var alignmentBuckets = new Dictionary<string, int>(StringComparer.Ordinal);
        var ck24TypeStrongCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var ck24TypeWeakCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var heightAccumulator = CreateHeightAccumulators();
        List<Pm4MsurGeometryExample> examples = new();

        foreach (Pm4ResearchFile file in files)
        {
            (int? tileX, int? tileY) = TryParseTileCoordinates(file.SourcePath);
            IReadOnlyList<uint> msvi = file.KnownChunks.Msvi;
            IReadOnlyList<Vector3> msvt = file.KnownChunks.Msvt;

            for (int surfaceIndex = 0; surfaceIndex < file.KnownChunks.Msur.Count; surfaceIndex++)
            {
                Pm4MsurEntry surface = file.KnownChunks.Msur[surfaceIndex];
                List<Vector3> vertices = CollectSurfaceVertices(surface, msvi, msvt);
                if (vertices.Count < 3)
                {
                    degenerateSurfaceCount++;
                    continue;
                }

                Vector3 geometryNormal = ComputeNewellNormal(vertices);
                float geometryMagnitude = geometryNormal.Length();
                if (geometryMagnitude <= MagnitudeEpsilon)
                {
                    degenerateSurfaceCount++;
                    continue;
                }

                analyzedSurfaceCount++;

                geometryNormal /= geometryMagnitude;
                Vector3 centroid = ComputeCentroid(vertices);
                float storedMagnitude = surface.Normal.Length();
                storedMagnitudeSum += storedMagnitude;
                AddCount(storedMagnitudeBuckets, BucketStoredMagnitude(storedMagnitude));

                if (MathF.Abs(storedMagnitude - 1f) <= UnitNormalTolerance)
                    unitLikeStoredNormalCount++;

                float signedDot = 0f;
                float absoluteDot = 0f;
                float storedPlaneDistance = float.NaN;

                if (storedMagnitude > MagnitudeEpsilon)
                {
                    Vector3 storedNormal = surface.Normal / storedMagnitude;
                    signedDot = Vector3.Dot(storedNormal, geometryNormal);
                    absoluteDot = MathF.Abs(signedDot);
                    storedPlaneDistance = Vector3.Dot(storedNormal, centroid);
                    absoluteDotSum += absoluteDot;

                    if (signedDot >= 0)
                        positiveAlignmentCount++;
                    else
                        negativeAlignmentCount++;

                    if (absoluteDot >= DotStrongThreshold)
                    {
                        strongAlignmentCount++;
                        AddCount(ck24TypeStrongCounts, $"0x{surface.Ck24Type:X2}");
                    }
                    else if (absoluteDot >= DotModerateThreshold)
                    {
                        moderateAlignmentCount++;
                    }
                    else
                    {
                        weakAlignmentCount++;
                        AddCount(ck24TypeWeakCounts, $"0x{surface.Ck24Type:X2}");
                    }

                    AddCount(alignmentBuckets, BucketAlignment(absoluteDot));
                }
                else
                {
                    weakAlignmentCount++;
                    AddCount(alignmentBuckets, "zero-stored-normal");
                    AddCount(ck24TypeWeakCounts, $"0x{surface.Ck24Type:X2}");
                }

                float geometricPlaneDistance = Vector3.Dot(geometryNormal, centroid);
                UpdateHeight(heightAccumulator["centroid.x"], surface.Height, centroid.X);
                UpdateHeight(heightAccumulator["centroid.y"], surface.Height, centroid.Y);
                UpdateHeight(heightAccumulator["centroid.z"], surface.Height, centroid.Z);
                UpdateHeight(heightAccumulator["geomPlane.+"], surface.Height, geometricPlaneDistance);
                UpdateHeight(heightAccumulator["geomPlane.-"], surface.Height, -geometricPlaneDistance);
                if (!float.IsNaN(storedPlaneDistance))
                {
                    UpdateHeight(heightAccumulator["storedPlane.+"], surface.Height, storedPlaneDistance);
                    UpdateHeight(heightAccumulator["storedPlane.-"], surface.Height, -storedPlaneDistance);
                }

                examples.Add(new Pm4MsurGeometryExample(
                    file.SourcePath,
                    tileX,
                    tileY,
                    surfaceIndex,
                    surface.Ck24,
                    surface.Ck24Type,
                    surface.Ck24ObjectId,
                    surface.IndexCount,
                    storedMagnitude,
                    signedDot,
                    absoluteDot,
                    surface.Height,
                    geometricPlaneDistance,
                    storedPlaneDistance,
                    centroid));
            }
        }

        List<Pm4HeightCandidateSummary> heightCandidates = heightAccumulator
            .OrderBy(static pair => pair.Key)
            .Select(static pair => pair.Value.ToSummary(pair.Key))
            .OrderBy(static summary => summary.MeanAbsoluteError)
            .ToList();

        List<Pm4FieldDistribution> distributions =
        [
            BuildDistribution("MSUR.StoredNormalMagnitude", storedMagnitudeBuckets, null, "Distribution of raw stored-normal magnitudes before normalization."),
            BuildDistribution("MSUR.StoredNormalAlignment", alignmentBuckets, null, "Absolute dot between stored normal and geometry-derived surface normal."),
            BuildDistribution("MSUR.StrongAlignment.Ck24Type", ck24TypeStrongCounts, null, "CK24 type bytes on surfaces where stored-vs-geometry normals align strongly."),
            BuildDistribution("MSUR.WeakAlignment.Ck24Type", ck24TypeWeakCounts, null, "CK24 type bytes on surfaces where stored-vs-geometry normals align weakly or stored normals collapse."),
        ];

        List<Pm4MsurGeometryExample> bestAligned = examples
            .OrderByDescending(static item => item.AbsoluteDot)
            .ThenBy(static item => item.SourcePath)
            .Take(24)
            .ToList();

        List<Pm4MsurGeometryExample> worstAligned = examples
            .OrderBy(static item => item.AbsoluteDot)
            .ThenBy(static item => item.SourcePath)
            .Take(24)
            .ToList();

        List<string> notes =
        [
            "This report measures geometry-derived evidence against the current MSUR Normal/Height naming without assuming any viewer/world transform.",
            "Strong normal alignment supports the idea that bytes 4..15 encode a plane-like direction, but it does not by itself prove final semantic naming or sign conventions.",
            "Height candidate scoring is deliberately comparative; it is intended to show whether the stored float behaves more like an axis value or a plane-distance term.",
        ];

        return new Pm4MsurGeometryReport(
            inputDirectory,
            files.Count,
            analyzedSurfaceCount,
            degenerateSurfaceCount,
            unitLikeStoredNormalCount,
            strongAlignmentCount,
            moderateAlignmentCount,
            weakAlignmentCount,
            positiveAlignmentCount,
            negativeAlignmentCount,
            analyzedSurfaceCount > 0 ? storedMagnitudeSum / analyzedSurfaceCount : 0f,
            analyzedSurfaceCount > 0 ? absoluteDotSum / analyzedSurfaceCount : 0f,
            heightCandidates,
            distributions,
            bestAligned,
            worstAligned,
            notes);
    }

    private static Dictionary<string, HeightAccumulator> CreateHeightAccumulators()
    {
        return new Dictionary<string, HeightAccumulator>(StringComparer.Ordinal)
        {
            ["centroid.x"] = new(),
            ["centroid.y"] = new(),
            ["centroid.z"] = new(),
            ["geomPlane.+"] = new(),
            ["geomPlane.-"] = new(),
            ["storedPlane.+"] = new(),
            ["storedPlane.-"] = new(),
        };
    }

    private static void UpdateHeight(HeightAccumulator accumulator, float actual, float candidate)
    {
        if (float.IsNaN(candidate) || float.IsInfinity(candidate))
            return;

        float error = MathF.Abs(actual - candidate);
        accumulator.TotalError += error;
        accumulator.Count++;
        if (error <= 0.1f)
            accumulator.FitsWithinPointOne++;
        if (error <= 1f)
            accumulator.FitsWithinOne++;
        if (error <= 4f)
            accumulator.FitsWithinFour++;
    }

    private static List<Vector3> CollectSurfaceVertices(Pm4MsurEntry surface, IReadOnlyList<uint> msvi, IReadOnlyList<Vector3> msvt)
    {
        List<Vector3> vertices = new(surface.IndexCount);
        int firstIndex = (int)surface.MsviFirstIndex;
        int endExclusive = Math.Min(firstIndex + surface.IndexCount, msvi.Count);
        for (int idx = Math.Max(0, firstIndex); idx < endExclusive; idx++)
        {
            int vertexIndex = (int)msvi[idx];
            if ((uint)vertexIndex >= (uint)msvt.Count)
                continue;

            vertices.Add(msvt[vertexIndex]);
        }

        return vertices;
    }

    private static Vector3 ComputeNewellNormal(IReadOnlyList<Vector3> vertices)
    {
        Vector3 normal = Vector3.Zero;
        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 current = vertices[i];
            Vector3 next = vertices[(i + 1) % vertices.Count];
            normal.X += (current.Y - next.Y) * (current.Z + next.Z);
            normal.Y += (current.Z - next.Z) * (current.X + next.X);
            normal.Z += (current.X - next.X) * (current.Y + next.Y);
        }

        return normal;
    }

    private static Vector3 ComputeCentroid(IReadOnlyList<Vector3> vertices)
    {
        Vector3 sum = Vector3.Zero;
        for (int i = 0; i < vertices.Count; i++)
            sum += vertices[i];

        return sum / vertices.Count;
    }

    private static string BucketStoredMagnitude(float magnitude)
    {
        if (magnitude <= MagnitudeEpsilon)
            return "zero";
        if (magnitude < 0.5f)
            return "(0,0.5)";
        if (magnitude < 0.9f)
            return "[0.5,0.9)";
        if (magnitude <= 1.1f)
            return "[0.9,1.1]";
        if (magnitude <= 1.5f)
            return "(1.1,1.5]";
        return ">1.5";
    }

    private static string BucketAlignment(float absoluteDot)
    {
        if (absoluteDot >= DotStrongThreshold)
            return ">=0.95";
        if (absoluteDot >= DotModerateThreshold)
            return "[0.75,0.95)";
        return "<0.75";
    }

    private static Pm4FieldDistribution BuildDistribution(string field, Dictionary<string, int> counts, string? range, string? notes)
    {
        return new Pm4FieldDistribution(
            field,
            counts.Values.Sum(),
            counts.Count,
            range,
            counts
                .OrderByDescending(static kv => kv.Value)
                .ThenBy(static kv => kv.Key)
                .Take(12)
                .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
                .ToList(),
            notes);
    }

    private static void AddCount(Dictionary<string, int> counts, string key)
    {
        counts.TryGetValue(key, out int existing);
        counts[key] = existing + 1;
    }

    private static (int? TileX, int? TileY) TryParseTileCoordinates(string? sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
            return (null, null);

        string fileName = Path.GetFileNameWithoutExtension(sourcePath);
        string[] parts = fileName.Split('_', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 2)
            return (null, null);

        if (!int.TryParse(parts[^2], out int tileX) || !int.TryParse(parts[^1], out int tileY))
            return (null, null);

        return (tileX, tileY);
    }

    private sealed class HeightAccumulator
    {
        public float TotalError { get; set; }
        public int Count { get; set; }
        public int FitsWithinPointOne { get; set; }
        public int FitsWithinOne { get; set; }
        public int FitsWithinFour { get; set; }

        public Pm4HeightCandidateSummary ToSummary(string candidate)
        {
            return new Pm4HeightCandidateSummary(
                candidate,
                Count > 0 ? TotalError / Count : float.PositiveInfinity,
                FitsWithinPointOne,
                FitsWithinOne,
                FitsWithinFour);
        }
    }
}