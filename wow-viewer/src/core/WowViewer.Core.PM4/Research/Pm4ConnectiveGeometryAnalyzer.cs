using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Measures what the MSLK -> MSPI -> MSPV path windows encode, and whether MSUR._0x18 -> MSCN
/// behaves like a per-object exterior boundary link.
/// </summary>
/// <remarks>
/// This exists because the legacy indices-vs-triangles mode counters in
/// <see cref="Pm4ResearchUnknownsAnalyzer"/> cannot discriminate: trianglesMode tests
/// <c>3*first + 3*count &lt;= mspiCount</c> while indicesMode tests <c>first + count &lt;= mspiCount</c>,
/// and the former implies the latter for every non-negative input. Its trianglesOnly bucket is
/// therefore zero by construction rather than by evidence. Those counters are left untouched — they
/// are a published baseline — and the geometric measurements here are the replacement.
/// </remarks>
public static class Pm4ConnectiveGeometryAnalyzer
{
    /// <summary>Perpendicular distance below which points count as collinear or coplanar.</summary>
    private const float FlatnessEpsilon = 0.05f;

    /// <summary>Triangle area below which a triple counts as degenerate.</summary>
    private const float DegenerateAreaEpsilon = 1e-4f;

    /// <summary>Spatial-hash cell size for testing whether two vertex streams meet.</summary>
    private const float CoincidenceEpsilon = 0.25f;

    public static Pm4ConnectiveGeometryReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4ResearchDocument> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        return Analyze(resolvedDirectory, files);
    }

    public static Pm4ConnectiveGeometryReport AnalyzeFile(string path)
        => Analyze(path, [Pm4ResearchReader.ReadFile(path)]);

    /// <summary>
    /// Analyzes already-read documents. Public so the discriminator's power can be demonstrated on
    /// constructed cases before any corpus claim is made — a measurement that cannot separate the
    /// interpretations it is testing is not evidence, which is exactly how the legacy
    /// indices-vs-triangles counters failed.
    /// </summary>
    public static Pm4ConnectiveGeometryReport Analyze(string inputDirectory, IReadOnlyList<Pm4ResearchDocument> files)
    {
        int nonEmptyFileCount = 0;
        int totalMslkEntries = 0;
        int activeWindows = 0;
        int negativeFirstIndexEntries = 0;
        int zeroCountEntries = 0;
        long totalWindowIndices = 0;
        int minWindowSize = int.MaxValue;
        int maxWindowSize = 0;

        Dictionary<int, int> sizeHistogram = [];
        Dictionary<string, FamilyAccumulator> families = new(StringComparer.Ordinal);
        TopologyAccumulator overallTopology = new();
        OrientationAccumulator windowOrientation = new();
        OrientationAccumulator surfaceOrientation = new();
        CoincidenceAccumulator coincidence = new();

        int filesWithMscn = 0;
        long msurToMscnFits = 0;
        long msurToMscnMisses = 0;
        long totalMscnPoints = 0;
        long totalMsvtPoints = 0;
        long distinctMscnReferenced = 0;
        long mscnPointsUnreferenced = 0;

        foreach (Pm4ResearchDocument file in files)
        {
            Pm4KnownChunkSet chunks = file.KnownChunks;
            if (chunks.Msur.Count > 0 || chunks.Mslk.Count > 0 || chunks.Mprl.Count > 0)
                nonEmptyFileCount++;

            int mspiCount = chunks.Mspi.Count;
            int mspvCount = chunks.Mspv.Count;
            int mscnCount = chunks.Mscn.Count;

            foreach (Pm4MslkEntry link in chunks.Mslk)
            {
                totalMslkEntries++;

                string familyKey = $"type=0x{link.TypeFlags:X2} subtype={link.Subtype}";
                if (!families.TryGetValue(familyKey, out FamilyAccumulator? family))
                {
                    family = new FamilyAccumulator(familyKey, link.TypeFlags, link.Subtype);
                    families.Add(familyKey, family);
                }

                family.TotalEntries++;
                family.FilePaths.Add(file.SourcePath ?? string.Empty);

                if (link.MspiFirstIndex < 0)
                {
                    negativeFirstIndexEntries++;
                    family.NegativeFirstIndexEntries++;
                    continue;
                }

                if (link.MspiIndexCount == 0)
                {
                    zeroCountEntries++;
                    continue;
                }

                int first = link.MspiFirstIndex;
                int count = link.MspiIndexCount;
                if (first + count > mspiCount)
                    continue;

                activeWindows++;
                family.ActiveWindows++;
                totalWindowIndices += count;
                family.WindowSizeSum += count;
                minWindowSize = Math.Min(minWindowSize, count);
                maxWindowSize = Math.Max(maxWindowSize, count);

                AddCount(sizeHistogram, count);
                AddCount(family.SizeHistogram, count);

                MeasureWindow(chunks, first, count, mspvCount, overallTopology, family.Topology, windowOrientation);
            }

            foreach (Pm4MsurEntry surface in chunks.Msur)
                surfaceOrientation.Add(surface.Normal);

            MeasureStreamCoincidence(chunks, coincidence);

            if (mscnCount > 0)
            {
                filesWithMscn++;
                totalMscnPoints += mscnCount;
                totalMsvtPoints += chunks.Msvt.Count;

                HashSet<uint> referenced = [];
                foreach (Pm4MsurEntry surface in chunks.Msur)
                {
                    if (surface.MscnRefIndex < (uint)mscnCount)
                    {
                        msurToMscnFits++;
                        referenced.Add(surface.MscnRefIndex);
                    }
                    else
                    {
                        msurToMscnMisses++;
                    }
                }

                distinctMscnReferenced += referenced.Count;
                mscnPointsUnreferenced += mscnCount - referenced.Count;
            }
        }

        Pm4MslkWindowPopulation population = new(
            totalMslkEntries,
            activeWindows,
            negativeFirstIndexEntries,
            zeroCountEntries,
            totalWindowIndices,
            activeWindows == 0 ? 0d : (double)totalWindowIndices / activeWindows,
            minWindowSize == int.MaxValue ? 0 : minWindowSize,
            maxWindowSize);

        Pm4MscnLinkageSummary mscnLinkage = new(
            filesWithMscn,
            msurToMscnFits,
            msurToMscnMisses,
            totalMscnPoints,
            distinctMscnReferenced,
            mscnPointsUnreferenced,
            totalMscnPoints == 0 ? 0d : (double)distinctMscnReferenced / totalMscnPoints,
            totalMsvtPoints == 0 ? 0d : (double)totalMscnPoints / totalMsvtPoints);

        IReadOnlyList<Pm4WindowFamilySummary> familySummaries = families.Values
            .OrderByDescending(static family => family.ActiveWindows)
            .ThenBy(static family => family.FamilyKey, StringComparer.Ordinal)
            .Select(static family => family.ToReport())
            .ToList();

        // The surface mesh picks the reference axis; the path quads are then measured against it.
        int surfaceAxis = surfaceOrientation.DominantAxis();
        Pm4FaceOrientationSummary windowOrientationReport = windowOrientation.ToReport("MSPV/MSPI path windows", surfaceAxis);
        Pm4FaceOrientationSummary surfaceOrientationReport = surfaceOrientation.ToReport("MSUR surface normals", surfaceAxis);

        return new Pm4ConnectiveGeometryReport(
            inputDirectory,
            files.Count,
            nonEmptyFileCount,
            population,
            BuildHistogram(sizeHistogram, 24),
            overallTopology.ToReport(),
            windowOrientationReport,
            surfaceOrientationReport,
            coincidence.ToReport(),
            familySummaries,
            mscnLinkage,
            BuildNotes(population, overallTopology, mscnLinkage, windowOrientationReport, surfaceOrientationReport, surfaceAxis));
    }

    /// <summary>
    /// Measures one window's geometry. Every test here is one a polyline and a triangle run
    /// would answer differently.
    /// </summary>
    private static void MeasureWindow(
        Pm4KnownChunkSet chunks,
        int first,
        int count,
        int mspvCount,
        TopologyAccumulator overall,
        TopologyAccumulator family,
        OrientationAccumulator orientation)
    {
        overall.WindowsMeasured++;
        family.WindowsMeasured++;

        if (count % 3 == 0)
        {
            overall.MultipleOfThreeWindows++;
            family.MultipleOfThreeWindows++;
        }

        uint firstVertex = chunks.Mspi[first];
        uint lastVertex = chunks.Mspi[first + count - 1];
        if (count > 2 && firstVertex == lastVertex)
        {
            overall.ClosedWindows++;
            family.ClosedWindows++;
        }

        HashSet<uint> distinct = [];
        List<Vector3> points = new(count);
        for (int i = 0; i < count; i++)
        {
            uint vertexIndex = chunks.Mspi[first + i];
            distinct.Add(vertexIndex);
            if (vertexIndex < (uint)mspvCount)
                points.Add(chunks.Mspv[(int)vertexIndex]);
        }

        if (distinct.Count < count)
        {
            overall.WindowsWithDuplicateVertices++;
            family.WindowsWithDuplicateVertices++;
        }

        // A triangle run read as one would produce few degenerate triples; a polyline read as
        // triangles produces them constantly.
        for (int i = 0; i + 2 < points.Count; i += 3)
        {
            double area = TriangleArea(points[i], points[i + 1], points[i + 2]);
            overall.TriplesTested++;
            family.TriplesTested++;
            if (area < DegenerateAreaEpsilon)
            {
                overall.DegenerateTriples++;
                family.DegenerateTriples++;
            }
        }

        if (points.Count >= 3)
        {
            if (IsCollinear(points))
            {
                overall.CollinearWindows++;
                family.CollinearWindows++;
            }
            else if (IsCoplanar(points))
            {
                overall.CoplanarWindows++;
                family.CoplanarWindows++;
            }

            if (TryComputeNormal(points, out Vector3 normal))
                orientation.Add(normal);
        }
    }

    /// <summary>
    /// Tests whether MSPV points land on MSVT points, via a quantised spatial hash rather than a
    /// pairwise scan. Reported in both directions: walls may terminate on the mesh without the mesh
    /// being fully covered by walls.
    /// </summary>
    private static void MeasureStreamCoincidence(Pm4KnownChunkSet chunks, CoincidenceAccumulator accumulator)
    {
        if (chunks.Mspv.Count == 0 || chunks.Msvt.Count == 0)
            return;

        HashSet<(int, int, int)> meshCells = [];
        foreach (Vector3 vertex in chunks.Msvt)
            meshCells.Add(Quantise(vertex));

        HashSet<(int, int, int)> pathCells = [];
        foreach (Vector3 vertex in chunks.Mspv)
            pathCells.Add(Quantise(vertex));

        foreach (Vector3 vertex in chunks.Mspv)
        {
            accumulator.MspvTested++;
            if (meshCells.Contains(Quantise(vertex)))
                accumulator.MspvCoincident++;
        }

        foreach (Vector3 vertex in chunks.Msvt)
        {
            accumulator.MsvtTested++;
            if (pathCells.Contains(Quantise(vertex)))
                accumulator.MsvtCoincident++;
        }
    }

    private static (int, int, int) Quantise(Vector3 value)
        => ((int)MathF.Round(value.X / CoincidenceEpsilon),
            (int)MathF.Round(value.Y / CoincidenceEpsilon),
            (int)MathF.Round(value.Z / CoincidenceEpsilon));

    /// <summary>First non-degenerate triple's normal, normalized.</summary>
    private static bool TryComputeNormal(IReadOnlyList<Vector3> points, out Vector3 normal)
    {
        Vector3 origin = points[0];
        for (int i = 1; i + 1 < points.Count; i++)
        {
            Vector3 candidate = Vector3.Cross(points[i] - origin, points[i + 1] - origin);
            if (candidate.Length() > 1e-6f)
            {
                normal = Vector3.Normalize(candidate);
                return true;
            }
        }

        normal = Vector3.Zero;
        return false;
    }

    private static double TriangleArea(Vector3 a, Vector3 b, Vector3 c)
        => Vector3.Cross(b - a, c - a).Length() * 0.5;

    /// <summary>Maximum perpendicular distance from the chord through the endpoints.</summary>
    private static bool IsCollinear(IReadOnlyList<Vector3> points)
    {
        Vector3 start = points[0];
        Vector3 axis = points[^1] - start;
        float axisLength = axis.Length();
        if (axisLength < 1e-6f)
            return false;

        Vector3 direction = axis / axisLength;
        for (int i = 1; i < points.Count - 1; i++)
        {
            Vector3 offset = points[i] - start;
            float along = Vector3.Dot(offset, direction);
            if ((offset - (direction * along)).Length() > FlatnessEpsilon)
                return false;
        }

        return true;
    }

    /// <summary>Fits a plane from the first non-degenerate triple and tests the remainder against it.</summary>
    private static bool IsCoplanar(IReadOnlyList<Vector3> points)
    {
        Vector3 origin = points[0];
        Vector3 normal = Vector3.Zero;

        for (int i = 1; i + 1 < points.Count; i++)
        {
            Vector3 candidate = Vector3.Cross(points[i] - origin, points[i + 1] - origin);
            if (candidate.Length() > 1e-6f)
            {
                normal = Vector3.Normalize(candidate);
                break;
            }
        }

        if (normal == Vector3.Zero)
            return false;

        foreach (Vector3 point in points)
        {
            if (Math.Abs(Vector3.Dot(point - origin, normal)) > FlatnessEpsilon)
                return false;
        }

        return true;
    }

    private static IReadOnlyList<Pm4WindowSizeBucket> BuildHistogram(Dictionary<int, int> counts, int take)
    {
        int total = counts.Values.Sum();
        return counts
            .OrderByDescending(static kv => kv.Value)
            .ThenBy(static kv => kv.Key)
            .Take(take)
            .Select(kv => new Pm4WindowSizeBucket(kv.Key, kv.Value, total == 0 ? 0d : (double)kv.Value / total))
            .ToList();
    }

    private static void AddCount(Dictionary<int, int> counts, int key)
    {
        counts.TryGetValue(key, out int existing);
        counts[key] = existing + 1;
    }

    private static IReadOnlyList<string> BuildNotes(
        Pm4MslkWindowPopulation population,
        TopologyAccumulator topology,
        Pm4MscnLinkageSummary mscn,
        Pm4FaceOrientationSummary windows,
        Pm4FaceOrientationSummary surfaces,
        int surfaceAxis)
    {
        List<string> notes =
        [
            "Legacy indicesOnly/trianglesOnly/both counters are NOT reproduced here: trianglesMode implies indicesMode for every non-negative input, so trianglesOnly is zero by construction and carries no topological information. Pm4ResearchUnknownsAnalyzer still publishes them as the historical baseline.",
            $"MSLK entries with MspiFirstIndex < 0: {population.NegativeFirstIndexEntries}. Prior art (PM4Tool/docs/pm4/pm4-analysis-findings.md) reads these as doodad placements rather than geometry; they are counted, never dropped.",
            $"MSUR._0x18 -> MSCN reaches {mscn.ReferencedFraction:P1} of MSCN points; MSCN/MSVT count ratio is {mscn.MscnToMsvtRatio:F2}."
        ];

        if (topology.WindowsMeasured > 0)
        {
            double multipleOfThree = (double)topology.MultipleOfThreeWindows / topology.WindowsMeasured;
            notes.Add($"Windows whose length is a multiple of 3: {multipleOfThree:P1}. A triangle list would concentrate here; a polyline would not.");
        }

        string axisName = surfaceAxis switch { 0 => "X", 1 => "Y", _ => "Z" };
        notes.Add($"MSUR surface normals concentrate on axis {axisName}; that axis is chosen as the reference, not assumed.");

        if (windows.FacesMeasured > 0)
        {
            double perpendicular = (double)windows.NearPerpendicularToDominantAxis / windows.FacesMeasured;
            notes.Add($"Path-window faces near-perpendicular to axis {axisName}: {perpendicular:P1}. High values mean the second stream is vertical relative to the walkable surfaces, i.e. walls to their floors.");
        }

        return notes;
    }

    /// <summary>
    /// Buckets unit normals by their largest component, so face orientation can be reported
    /// without first deciding which axis the format treats as up.
    /// </summary>
    private sealed class OrientationAccumulator
    {
        private const float AxisAlignedThreshold = 0.9f;
        private const float PerpendicularThreshold = 0.1f;

        public long FacesMeasured { get; private set; }

        public long DominantX { get; private set; }

        public long DominantY { get; private set; }

        public long DominantZ { get; private set; }

        public double SumAbsX { get; private set; }

        public double SumAbsY { get; private set; }

        public double SumAbsZ { get; private set; }

        public long NearAxisAligned { get; private set; }

        public void Add(Vector3 normal)
        {
            float length = normal.Length();
            if (length < 1e-6f)
                return;

            Vector3 unit = normal / length;
            float absX = Math.Abs(unit.X);
            float absY = Math.Abs(unit.Y);
            float absZ = Math.Abs(unit.Z);

            FacesMeasured++;
            SumAbsX += absX;
            SumAbsY += absY;
            SumAbsZ += absZ;

            float max = Math.Max(absX, Math.Max(absY, absZ));
            if (max == absX)
                DominantX++;
            else if (max == absY)
                DominantY++;
            else
                DominantZ++;

            if (max >= AxisAlignedThreshold)
                NearAxisAligned++;

            if (absX <= PerpendicularThreshold)
                PerpendicularToX++;
            if (absY <= PerpendicularThreshold)
                PerpendicularToY++;
            if (absZ <= PerpendicularThreshold)
                PerpendicularToZ++;
        }

        /// <summary>
        /// Counts faces whose normal is near-perpendicular to <paramref name="axis"/>, where the
        /// axis is chosen from a different face set. This is the cross-comparison that answers
        /// "walls versus floors" without an up-axis assumption.
        /// </summary>
        public long CountPerpendicularTo(int axis)
            => axis switch
            {
                0 => PerpendicularToX,
                1 => PerpendicularToY,
                _ => PerpendicularToZ
            };

        public long PerpendicularToX { get; private set; }

        public long PerpendicularToY { get; private set; }

        public long PerpendicularToZ { get; private set; }

        /// <summary>Index of the axis this set concentrates on: 0=X, 1=Y, 2=Z.</summary>
        public int DominantAxis()
        {
            long max = Math.Max(DominantX, Math.Max(DominantY, DominantZ));
            return max == DominantX ? 0 : max == DominantY ? 1 : 2;
        }

        public Pm4FaceOrientationSummary ToReport(string name, int perpendicularAxis)
            => new(
                name,
                FacesMeasured,
                DominantX,
                DominantY,
                DominantZ,
                FacesMeasured == 0 ? 0d : SumAbsX / FacesMeasured,
                FacesMeasured == 0 ? 0d : SumAbsY / FacesMeasured,
                FacesMeasured == 0 ? 0d : SumAbsZ / FacesMeasured,
                NearAxisAligned,
                CountPerpendicularTo(perpendicularAxis));
    }

    private sealed class CoincidenceAccumulator
    {
        public long MspvTested { get; set; }

        public long MspvCoincident { get; set; }

        public long MsvtTested { get; set; }

        public long MsvtCoincident { get; set; }

        public Pm4StreamCoincidenceSummary ToReport()
            => new(
                MspvTested,
                MspvCoincident,
                MspvTested == 0 ? 0d : (double)MspvCoincident / MspvTested,
                CoincidenceEpsilon,
                MsvtTested,
                MsvtCoincident,
                MsvtTested == 0 ? 0d : (double)MsvtCoincident / MsvtTested);
    }

    private sealed class TopologyAccumulator
    {
        public int WindowsMeasured { get; set; }

        public int ClosedWindows { get; set; }

        public int MultipleOfThreeWindows { get; set; }

        public int WindowsWithDuplicateVertices { get; set; }

        public int CollinearWindows { get; set; }

        public int CoplanarWindows { get; set; }

        public long TriplesTested { get; set; }

        public long DegenerateTriples { get; set; }

        public Pm4WindowTopologyEvidence ToReport()
            => new(
                WindowsMeasured,
                ClosedWindows,
                MultipleOfThreeWindows,
                WindowsWithDuplicateVertices,
                CollinearWindows,
                CoplanarWindows,
                TriplesTested,
                DegenerateTriples,
                TriplesTested == 0 ? 0d : (double)DegenerateTriples / TriplesTested);
    }

    private sealed class FamilyAccumulator(string familyKey, byte typeFlags, byte subtype)
    {
        public string FamilyKey { get; } = familyKey;

        public byte TypeFlags { get; } = typeFlags;

        public byte Subtype { get; } = subtype;

        public HashSet<string> FilePaths { get; } = new(StringComparer.Ordinal);

        public int TotalEntries { get; set; }

        public int ActiveWindows { get; set; }

        public int NegativeFirstIndexEntries { get; set; }

        public long WindowSizeSum { get; set; }

        public Dictionary<int, int> SizeHistogram { get; } = [];

        public TopologyAccumulator Topology { get; } = new();

        public Pm4WindowFamilySummary ToReport()
        {
            Pm4WindowTopologyEvidence topology = Topology.ToReport();
            int modalSize = SizeHistogram.Count == 0
                ? 0
                : SizeHistogram.OrderByDescending(static kv => kv.Value).ThenBy(static kv => kv.Key).First().Key;

            return new Pm4WindowFamilySummary(
                FamilyKey,
                TypeFlags,
                Subtype,
                FilePaths.Count,
                TotalEntries,
                ActiveWindows,
                NegativeFirstIndexEntries,
                ActiveWindows == 0 ? 0d : (double)WindowSizeSum / ActiveWindows,
                modalSize,
                topology.WindowsMeasured == 0 ? 0d : (double)topology.MultipleOfThreeWindows / topology.WindowsMeasured,
                topology.WindowsMeasured == 0 ? 0d : (double)topology.ClosedWindows / topology.WindowsMeasured,
                topology,
                BuildHistogram(SizeHistogram, 8));
        }
    }
}
