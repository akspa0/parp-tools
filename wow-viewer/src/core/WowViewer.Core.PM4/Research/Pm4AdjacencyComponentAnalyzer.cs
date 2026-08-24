using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Builds connected components from the MSUR adjacency graph and tests them against the CK24
/// grouping, plus the byte slices CK24 decomposes into.
/// </summary>
/// <remarks>
/// <para>
/// This became possible only once <see cref="Pm4MsurWindowAnalyzer"/> established that
/// <c>MSUR._0x18 + MSUR.AttributeMask</c> is a window into MSLK and that <c>MSLK.RefIndex</c> names a
/// neighbouring surface. Before that, the only available notion of "connected" was
/// <see cref="Pm4ComponentIdentityAnalyzer"/>'s geometric vertex weld at epsilon 0.25 — a
/// reconstruction. This one is the format's OWN topology, stored explicitly, with no epsilon and no
/// geometry involved.
/// </para>
/// <para>
/// Purity and distinctness are reported together, always. A field can be 100% pure and still be a
/// class enum that groups nothing — <c>MSUR.GroupKey</c> is exactly that, 100% pure across 9 corpus
/// values. Purity without distinctness is a false-positive generator.
/// </para>
/// </remarks>
public static class Pm4AdjacencyComponentAnalyzer
{
    public static Pm4AdjacencyComponentReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4ResearchDocument> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        return Analyze(resolvedDirectory, files);
    }

    public static Pm4AdjacencyComponentReport AnalyzeFile(string path)
        => Analyze(path, [Pm4ResearchReader.ReadFile(path)]);

    public static Pm4AdjacencyComponentReport Analyze(string inputDirectory, IReadOnlyList<Pm4ResearchDocument> files)
    {
        int nonEmptyFiles = 0;
        long surfaces = 0;
        long components = 0;
        long singletonComponents = 0;
        long ck24Groups = 0;

        long pureComponents = 0;
        long distinctComponents = 0;

        long edges = 0;
        long crossCk24Edges = 0;
        long nonReciprocated = 0;
        long nonReciprocatedCrossCk24 = 0;
        long nonReciprocatedTargetZero = 0;

        var typeValues = new HashSet<uint>();
        var highByteValues = new HashSet<uint>();
        var lowByteValues = new HashSet<uint>();
        var objectIdValues = new HashSet<uint>();
        var ck24Values = new HashSet<uint>();
        var trailerValues = new HashSet<uint>();

        var componentSizes = new Dictionary<int, long>();
        List<Pm4AdjacencyComponentFile> fileReports = [];

        foreach (Pm4ResearchDocument document in files)
        {
            IReadOnlyList<Pm4MsurEntry> msur = document.KnownChunks.Msur;
            IReadOnlyList<Pm4MslkEntry> mslk = document.KnownChunks.Mslk;
            if (msur.Count == 0)
                continue;

            nonEmptyFiles++;
            surfaces += msur.Count;

            // Neighbour sets straight from the stored windows. No geometry, no epsilon.
            var neighbours = new List<int>[msur.Count];
            for (int i = 0; i < msur.Count; i++)
            {
                neighbours[i] = [];
                long start = msur[i]._0x18;
                long end = start + msur[i].AttributeMask;
                if (end > mslk.Count)
                    continue;

                for (long j = start; j < end; j++)
                    neighbours[i].Add(mslk[(int)j].RefIndex);
            }

            var parent = new int[msur.Count];
            for (int i = 0; i < msur.Count; i++)
                parent[i] = i;

            for (int i = 0; i < msur.Count; i++)
            {
                foreach (int n in neighbours[i])
                {
                    if (n >= 0 && n < msur.Count)
                        Union(parent, i, n);
                }
            }

            // Component membership, and the CK24 population inside each.
            var membersByRoot = new Dictionary<int, List<int>>();
            for (int i = 0; i < msur.Count; i++)
            {
                int root = Find(parent, i);
                if (!membersByRoot.TryGetValue(root, out List<int>? bucket))
                {
                    bucket = [];
                    membersByRoot[root] = bucket;
                }
                bucket.Add(i);
            }

            var ck24OwnerCount = new Dictionary<uint, int>();
            foreach (List<int> members in membersByRoot.Values)
            {
                uint first = msur[members[0]].Ck24;
                bool pure = members.All(m => msur[m].Ck24 == first);
                if (pure)
                    ck24OwnerCount[first] = ck24OwnerCount.GetValueOrDefault(first) + 1;
            }

            long filePure = 0;
            long fileDistinct = 0;
            foreach (List<int> members in membersByRoot.Values)
            {
                uint first = msur[members[0]].Ck24;
                bool pure = members.All(m => msur[m].Ck24 == first);
                if (pure)
                {
                    filePure++;
                    if (ck24OwnerCount[first] == 1)
                        fileDistinct++;
                }

                int size = members.Count;
                componentSizes[size] = componentSizes.GetValueOrDefault(size) + 1;
                if (size == 1)
                    singletonComponents++;
            }

            pureComponents += filePure;
            distinctComponents += fileDistinct;
            components += membersByRoot.Count;

            var fileCk24 = new HashSet<uint>();
            foreach (Pm4MsurEntry s in msur)
            {
                fileCk24.Add(s.Ck24);
                ck24Values.Add(s.Ck24);
                typeValues.Add(s.Ck24Type);
                highByteValues.Add(s.Ck24HighByte);
                lowByteValues.Add(s.Ck24LowByte);
                objectIdValues.Add(s.Ck24ObjectId);
                trailerValues.Add(s.PackedParams & 0xFF);
            }
            ck24Groups += fileCk24.Count;

            // Edge-level questions: do connections cross CK24, and what do the
            // non-reciprocated ones look like?
            var neighbourSets = new HashSet<int>[msur.Count];
            for (int i = 0; i < msur.Count; i++)
                neighbourSets[i] = [.. neighbours[i]];

            long fileCross = 0;
            long fileNonRecip = 0;
            for (int i = 0; i < msur.Count; i++)
            {
                foreach (int n in neighbourSets[i])
                {
                    edges++;
                    if (n < 0 || n >= msur.Count)
                        continue;

                    bool cross = msur[i].Ck24 != msur[n].Ck24;
                    if (cross)
                    {
                        crossCk24Edges++;
                        fileCross++;
                    }

                    if (n != i && !neighbourSets[n].Contains(i))
                    {
                        nonReciprocated++;
                        fileNonRecip++;
                        if (cross)
                            nonReciprocatedCrossCk24++;
                        if (n == 0)
                            nonReciprocatedTargetZero++;
                    }
                }
            }

            fileReports.Add(new Pm4AdjacencyComponentFile(
                document.SourcePath ?? "(memory)",
                msur.Count,
                membersByRoot.Count,
                fileCk24.Count,
                filePure,
                fileDistinct,
                fileCross,
                fileNonRecip));
        }

        IReadOnlyList<Pm4ValueFrequency> sizeHistogram = componentSizes
            .OrderByDescending(static kv => kv.Value)
            .ThenBy(static kv => kv.Key)
            .Take(12)
            .Select(static kv => new Pm4ValueFrequency(kv.Key.ToString(), checked((int)kv.Value)))
            .ToList();

        return new Pm4AdjacencyComponentReport(
            inputDirectory,
            files.Count,
            nonEmptyFiles,
            surfaces,
            components,
            singletonComponents,
            ck24Groups,
            pureComponents,
            components == 0 ? 0d : (double)pureComponents / components,
            distinctComponents,
            components == 0 ? 0d : (double)distinctComponents / components,
            edges,
            crossCk24Edges,
            edges == 0 ? 0d : (double)crossCk24Edges / edges,
            nonReciprocated,
            nonReciprocatedCrossCk24,
            nonReciprocatedTargetZero,
            ck24Values.Count,
            typeValues.Count,
            highByteValues.Count,
            lowByteValues.Count,
            objectIdValues.Count,
            trailerValues.Count,
            sizeHistogram,
            fileReports);
    }

    /// <summary>
    /// Dumps the distinct raw <c>MSUR._0x1C</c> values for one file, in hex, with the surface count
    /// behind each. The existing decomposition reads the key as bits 8..31 and dismisses bits 0..7 as
    /// a "padding trailer, not identity" — a claim that is testable by simply looking at whether
    /// those bits vary, and by whether a rival slicing partitions the surfaces identically.
    /// Cardinality alone never settles this: two fields can produce the same number of groups and
    /// group entirely different surfaces, so partition equality is reported, not just counts.
    /// </summary>
    public static Pm4PackedParamsReport DescribePackedParams(string path)
    {
        Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(path);
        IReadOnlyList<Pm4MsurEntry> msur = document.KnownChunks.Msur;

        IReadOnlyList<uint> msvi = document.KnownChunks.Msvi;
        IReadOnlyList<System.Numerics.Vector3> msvt = document.KnownChunks.Msvt;

        var counts = new Dictionary<uint, int>();
        var mins = new Dictionary<uint, System.Numerics.Vector3>();
        var maxs = new Dictionary<uint, System.Numerics.Vector3>();

        foreach (Pm4MsurEntry s in msur)
        {
            counts[s.PackedParams] = counts.GetValueOrDefault(s.PackedParams) + 1;

            // Walk this surface's vertex window so each distinct raw value gets the bounding box of
            // every surface carrying it. If the value is a float describing the object, its geometry
            // is where the correspondence has to show up.
            long start = s.MsviFirstIndex;
            long end = start + s.IndexCount;
            if (end > msvi.Count)
                continue;

            for (long k = start; k < end; k++)
            {
                uint vi = msvi[(int)k];
                if (vi >= msvt.Count)
                    continue;

                System.Numerics.Vector3 p = msvt[(int)vi];
                if (!mins.TryGetValue(s.PackedParams, out System.Numerics.Vector3 lo))
                {
                    mins[s.PackedParams] = p;
                    maxs[s.PackedParams] = p;
                    continue;
                }

                mins[s.PackedParams] = System.Numerics.Vector3.Min(lo, p);
                maxs[s.PackedParams] = System.Numerics.Vector3.Max(maxs[s.PackedParams], p);
            }
        }

        List<Pm4PackedParamsValue> values = counts
            .OrderByDescending(static kv => kv.Value)
            .ThenBy(static kv => kv.Key)
            .Select(kv =>
            {
                mins.TryGetValue(kv.Key, out System.Numerics.Vector3 lo);
                maxs.TryGetValue(kv.Key, out System.Numerics.Vector3 hi);
                System.Numerics.Vector3 size = hi - lo;
                float halfDiagonal = size.Length() / 2f;
                float maxHalfExtent = MathF.Max(size.X, MathF.Max(size.Y, size.Z)) / 2f;

                return new Pm4PackedParamsValue(
                    kv.Key,
                    $"0x{kv.Key:X8}",
                    (byte)(kv.Key >> 24),
                    (byte)(kv.Key >> 16),
                    (byte)(kv.Key >> 8),
                    (byte)kv.Key,
                    kv.Value,
                    BitConverter.UInt32BitsToSingle(kv.Key),
                    size.X,
                    size.Y,
                    size.Z,
                    halfDiagonal,
                    maxHalfExtent,
                    lo.X,
                    lo.Y,
                    lo.Z);
            })
            .ToList();

        // Rival slicings, compared by the partition they induce rather than by cardinality.
        var current = new Dictionary<int, uint>();
        var lowAligned = new Dictionary<int, uint>();
        for (int i = 0; i < msur.Count; i++)
        {
            current[i] = (msur[i].PackedParams >> 8) & 0x00FF_FFFF;
            lowAligned[i] = msur[i].PackedParams & 0x00FF_FFFF;
        }

        bool samePartition = PartitionsMatch(current, lowAligned);

        return new Pm4PackedParamsReport(
            path,
            msur.Count,
            values.Count,
            current.Values.Distinct().Count(),
            lowAligned.Values.Distinct().Count(),
            samePartition,
            values);
    }

    /// <summary>
    /// True when two labellings induce the same grouping of surfaces, regardless of the label values
    /// themselves. This is the question that matters — identical cardinality with different grouping
    /// is the trap.
    /// </summary>
    private static bool PartitionsMatch(Dictionary<int, uint> a, Dictionary<int, uint> b)
    {
        var forward = new Dictionary<uint, uint>();
        var backward = new Dictionary<uint, uint>();

        foreach ((int key, uint left) in a)
        {
            uint right = b[key];
            if (forward.TryGetValue(left, out uint mappedRight) && mappedRight != right)
                return false;
            if (backward.TryGetValue(right, out uint mappedLeft) && mappedLeft != left)
                return false;
            forward[left] = right;
            backward[right] = left;
        }

        return true;
    }

    /// <summary>
    /// Tests whether <c>MSUR._0x1C</c> read as an IEEE-754 float is a Z coordinate in the MSVT frame,
    /// by correlating it against each object's vertical extent corpus-wide.
    /// </summary>
    /// <remarks>
    /// Controls are mandatory and are the point of the method: the value is correlated against the
    /// horizontal extents and the surface count as well. A correlation with Z that is not markedly
    /// stronger than the control correlations proves nothing, because any per-object quantity
    /// correlates weakly with any other. The high-byte population is reported for the same reason —
    /// a uniformly distributed identifier cannot confine 16 values to three exponent bands.
    /// </remarks>
    public static Pm4SurfaceZReport AnalyzeSurfaceZ(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<double> value = [];
        List<double> minZ = [];
        List<double> maxZ = [];
        List<double> midZ = [];
        List<double> minX = [];
        List<double> surfaceCount = [];
        var highBytes = new Dictionary<byte, long>();
        long objects = 0;
        long zeroValued = 0;
        int filesSeen = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(path).KnownChunks;
            IReadOnlyList<Pm4MsurEntry> msur = chunks.Msur;
            IReadOnlyList<uint> msvi = chunks.Msvi;
            IReadOnlyList<System.Numerics.Vector3> msvt = chunks.Msvt;
            if (msur.Count == 0)
                continue;

            filesSeen++;

            var lo = new Dictionary<uint, System.Numerics.Vector3>();
            var hi = new Dictionary<uint, System.Numerics.Vector3>();
            var count = new Dictionary<uint, int>();

            foreach (Pm4MsurEntry s in msur)
            {
                count[s.PackedParams] = count.GetValueOrDefault(s.PackedParams) + 1;
                highBytes[(byte)(s.PackedParams >> 24)] = highBytes.GetValueOrDefault((byte)(s.PackedParams >> 24)) + 1;

                long start = s.MsviFirstIndex;
                long end = start + s.IndexCount;
                if (end > msvi.Count)
                    continue;

                for (long k = start; k < end; k++)
                {
                    uint vi = msvi[(int)k];
                    if (vi >= msvt.Count)
                        continue;

                    System.Numerics.Vector3 p = msvt[(int)vi];
                    if (!lo.TryGetValue(s.PackedParams, out System.Numerics.Vector3 cur))
                    {
                        lo[s.PackedParams] = p;
                        hi[s.PackedParams] = p;
                        continue;
                    }

                    lo[s.PackedParams] = System.Numerics.Vector3.Min(cur, p);
                    hi[s.PackedParams] = System.Numerics.Vector3.Max(hi[s.PackedParams], p);
                }
            }

            foreach ((uint raw, System.Numerics.Vector3 low) in lo)
            {
                objects++;
                if (raw == 0)
                {
                    zeroValued++;
                    continue;
                }

                System.Numerics.Vector3 high = hi[raw];
                value.Add(BitConverter.UInt32BitsToSingle(raw));
                minZ.Add(low.Z);
                maxZ.Add(high.Z);
                midZ.Add((low.Z + high.Z) / 2.0);
                minX.Add(low.X);
                surfaceCount.Add(count[raw]);
            }
        }

        IReadOnlyList<Pm4ValueFrequency> bands = highBytes
            .OrderByDescending(static kv => kv.Value)
            .Select(static kv => new Pm4ValueFrequency($"0x{kv.Key:X2}", checked((int)kv.Value)))
            .ToList();

        return new Pm4SurfaceZReport(
            resolvedDirectory,
            filesSeen,
            objects,
            zeroValued,
            value.Count,
            Correlation(value, minZ),
            Correlation(value, maxZ),
            Correlation(value, midZ),
            Correlation(value, minX),
            Correlation(value, surfaceCount),
            OffsetSpread(value, minZ),
            bands);
    }

    private static double Correlation(List<double> a, List<double> b)
    {
        int n = a.Count;
        if (n < 2)
            return 0d;

        double ma = a.Average();
        double mb = b.Average();
        double sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < n; i++)
        {
            double da = a[i] - ma;
            double db = b[i] - mb;
            sab += da * db;
            saa += da * da;
            sbb += db * db;
        }

        double denom = Math.Sqrt(saa * sbb);
        return denom == 0d ? 0d : sab / denom;
    }

    /// <summary>
    /// Standard deviation of <c>value - minZ</c> against the standard deviation of the value itself.
    /// If the value is a height in the same frame, subtracting the object's floor removes almost all
    /// of the variance and leaves only the per-model origin offset.
    /// </summary>
    private static double OffsetSpread(List<double> value, List<double> minZ)
    {
        if (value.Count < 2)
            return 0d;

        var offsets = new List<double>(value.Count);
        for (int i = 0; i < value.Count; i++)
            offsets.Add(value[i] - minZ[i]);

        return StdDev(offsets) / StdDev(value);
    }

    private static double StdDev(List<double> xs)
    {
        double m = xs.Average();
        double s = xs.Sum(x => (x - m) * (x - m));
        return Math.Sqrt(s / xs.Count);
    }

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    private static void Union(int[] parent, int a, int b)
    {
        int ra = Find(parent, a);
        int rb = Find(parent, b);
        if (ra != rb)
            parent[rb] = ra;
    }
}

public sealed record Pm4AdjacencyComponentFile(
    string SourcePath,
    int SurfaceCount,
    int ComponentCount,
    int Ck24GroupCount,
    long PureComponents,
    long DistinctComponents,
    long CrossCk24Edges,
    long NonReciprocatedEdges);

public sealed record Pm4AdjacencyComponentReport(
    string InputDirectory,
    int FileCount,
    int NonEmptyFileCount,
    long Surfaces,
    long Components,
    long SingletonComponents,
    long Ck24Groups,
    long PureComponents,
    double PureFraction,
    long DistinctComponents,
    double DistinctFraction,
    long Edges,
    long CrossCk24Edges,
    double CrossCk24Fraction,
    long NonReciprocatedEdges,
    long NonReciprocatedCrossCk24,
    long NonReciprocatedTargetZero,
    int DistinctCk24Values,
    int DistinctCk24TypeValues,
    int DistinctCk24HighByteValues,
    int DistinctCk24LowByteValues,
    int DistinctCk24ObjectIdValues,
    int DistinctPackedTrailerValues,
    IReadOnlyList<Pm4ValueFrequency> ComponentSizeHistogram,
    IReadOnlyList<Pm4AdjacencyComponentFile> Files);

public sealed record Pm4PackedParamsValue(
    uint Raw,
    string Hex,
    byte Byte3,
    byte Byte2,
    byte Byte1,
    byte Byte0,
    int SurfaceCount,
    float AsFloat,
    float SizeX,
    float SizeY,
    float SizeZ,
    float HalfDiagonal,
    float MaxHalfExtent,
    float MinX,
    float MinY,
    float MinZ);

public sealed record Pm4PackedParamsReport(
    string SourcePath,
    int SurfaceCount,
    int DistinctRawValues,
    int DistinctCurrentSlice,
    int DistinctLowAlignedSlice,
    bool SlicesInduceSamePartition,
    IReadOnlyList<Pm4PackedParamsValue> Values);

public sealed record Pm4SurfaceZReport(
    string InputDirectory,
    int FilesWithSurfaces,
    long Objects,
    long ZeroValuedObjects,
    int Sampled,
    double CorrelationWithMinZ,
    double CorrelationWithMaxZ,
    double CorrelationWithMidZ,
    double CorrelationWithMinX,
    double CorrelationWithSurfaceCount,
    double OffsetSpreadRatio,
    IReadOnlyList<Pm4ValueFrequency> HighByteBands);
