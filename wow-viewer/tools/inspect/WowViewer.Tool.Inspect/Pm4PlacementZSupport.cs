using System.Numerics;
using System.Text.RegularExpressions;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;


internal sealed record Pm4PlacementZMatch(
    string Pm4Path,
    string AdtPath,
    uint Raw,
    float AsFloat,
    int SurfaceCount,
    string PlacementKind,
    string ModelPath,
    float PlacementZ,
    float Delta);

internal sealed record Pm4PlacementZReport(
    string InputDirectory,
    int Pm4Files,
    int PairedFiles,
    int UnpairedFiles,
    int Objects,
    int ObjectsWithCandidates,
    IReadOnlyList<Pm4PlacementZBucket> RealBuckets,
    IReadOnlyList<Pm4PlacementZBucket> ControlBuckets,
    double MedianAbsDelta,
    double ControlMedianAbsDelta,
    int WmoMatches,
    int M2Matches,
    IReadOnlyList<Pm4PlacementZMatch> Samples);

internal sealed record Pm4PlacementZBucket(double Tolerance, int Count, double Fraction);

/// <summary>
/// Tests whether <c>MSUR._0x1C</c>, read as a float, is the Z of the ADT placement that produced the
/// object — the prediction that follows from it correlating r=0.995 with each object's bounding-box
/// floor.
/// </summary>
/// <remarks>
/// The control is the point. Object Z values within one tile are not uniformly spread, so a naive
/// "nearest placement Z" match scores well by accident. The control keeps the same tile, the same
/// placement set and the same matching rule, and only breaks the correspondence by rotating the
/// object-to-placement assignment by one. A real correspondence survives that; a distributional
/// coincidence does not.
/// </remarks>
internal static partial class Pm4PlacementZSupport
{
    private static readonly double[] Tolerances = [0.001, 0.01, 0.1, 1.0, 5.0];

    [GeneratedRegex(@"_(\d+)_(\d+)\.pm4$", RegexOptions.IgnoreCase)]
    private static partial Regex TilePattern();

    public static Pm4PlacementZReport Run(string pm4Directory, string? adtDirectory, int sampleCount = 12)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);
        string adtRoot = string.IsNullOrWhiteSpace(adtDirectory) ? resolved : adtDirectory;

        int pm4Files = 0, paired = 0, unpaired = 0, objects = 0, withCandidates = 0;
        int wmoMatches = 0, m2Matches = 0;
        var realDeltas = new List<double>();
        var controlDeltas = new List<double>();
        var samples = new List<Pm4PlacementZMatch>();

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            pm4Files++;

            string? adtPath = FindCompanionAdt(pm4Path, adtRoot);
            if (adtPath is null)
            {
                unpaired++;
                continue;
            }

            AdtPlacementCatalog placements;
            try
            {
                placements = AdtPlacementReader.Read(adtPath);
            }
            catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException)
            {
                unpaired++;
                continue;
            }

            var candidates = new List<(string Kind, string Path, Vector3 Position)>();
            foreach (AdtWorldModelPlacement w in placements.WorldModelPlacements)
                candidates.Add(("WMO", w.ModelPath, w.Position));
            foreach (AdtModelPlacement m in placements.ModelPlacements)
                candidates.Add(("M2", m.ModelPath, m.Position));

            if (candidates.Count == 0)
            {
                unpaired++;
                continue;
            }

            paired++;

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            IReadOnlyList<Pm4MsurEntry> msur = chunks.Msur;
            IReadOnlyList<uint> msvi = chunks.Msvi;
            IReadOnlyList<Vector3> msvt = chunks.Msvt;

            // Object extents, converted through the canonical placement transform rather than by
            // hand-flipping axes on a bounding box.
            var lo = new Dictionary<uint, Vector3>();
            var hi = new Dictionary<uint, Vector3>();
            var count = new Dictionary<uint, int>();

            foreach (Pm4MsurEntry s in msur)
            {
                count[s.PackedParams] = count.GetValueOrDefault(s.PackedParams) + 1;

                long start = s.MsviFirstIndex;
                long end = start + s.IndexCount;
                if (end > msvi.Count)
                    continue;

                for (long k = start; k < end; k++)
                {
                    uint vi = msvi[(int)k];
                    if (vi >= msvt.Count)
                        continue;

                    Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(msvt[(int)vi]);
                    if (!lo.TryGetValue(s.PackedParams, out Vector3 cur))
                    {
                        lo[s.PackedParams] = p;
                        hi[s.PackedParams] = p;
                        continue;
                    }

                    lo[s.PackedParams] = Vector3.Min(cur, p);
                    hi[s.PackedParams] = Vector3.Max(hi[s.PackedParams], p);
                }
            }

            var fileMatches = new List<Pm4PlacementZMatch>();
            var fileObjectFloats = new List<float>();

            foreach ((uint raw, Vector3 low) in lo.OrderBy(static kv => kv.Key))
            {
                if (raw == 0)
                    continue;

                objects++;
                Vector3 high = hi[raw];
                float asFloat = BitConverter.UInt32BitsToSingle(raw);

                // Only placements standing inside this object's horizontal footprint are candidates.
                var inside = candidates
                    .Where(c => c.Position.X >= low.X - 1f && c.Position.X <= high.X + 1f
                             && c.Position.Y >= low.Y - 1f && c.Position.Y <= high.Y + 1f)
                    .ToList();

                if (inside.Count == 0)
                    continue;

                withCandidates++;

                (string Kind, string Path, Vector3 Position) best = inside
                    .OrderBy(c => MathF.Abs(asFloat - c.Position.Z))
                    .First();

                float delta = asFloat - best.Position.Z;
                realDeltas.Add(Math.Abs(delta));
                fileObjectFloats.Add(asFloat);

                if (best.Kind == "WMO")
                    wmoMatches++;
                else
                    m2Matches++;

                fileMatches.Add(new Pm4PlacementZMatch(
                    Path.GetFileName(pm4Path),
                    Path.GetFileName(adtPath),
                    raw,
                    asFloat,
                    count[raw],
                    best.Kind,
                    best.Path,
                    best.Position.Z,
                    delta));
            }

            // Control: same tile, same matched placements, correspondence rotated by one.
            for (int i = 0; i < fileMatches.Count && fileMatches.Count > 1; i++)
            {
                float wrongFloat = fileObjectFloats[(i + 1) % fileObjectFloats.Count];
                controlDeltas.Add(Math.Abs(wrongFloat - fileMatches[i].PlacementZ));
            }

            foreach (Pm4PlacementZMatch m in fileMatches)
            {
                if (samples.Count < sampleCount)
                    samples.Add(m);
            }
        }

        return new Pm4PlacementZReport(
            resolved,
            pm4Files,
            paired,
            unpaired,
            objects,
            withCandidates,
            Buckets(realDeltas),
            Buckets(controlDeltas),
            Median(realDeltas),
            Median(controlDeltas),
            wmoMatches,
            m2Matches,
            samples);
    }

    private static IReadOnlyList<Pm4PlacementZBucket> Buckets(List<double> deltas)
        => Tolerances
            .Select(t =>
            {
                int n = deltas.Count(d => d <= t);
                return new Pm4PlacementZBucket(t, n, deltas.Count == 0 ? 0d : (double)n / deltas.Count);
            })
            .ToList();

    private static double Median(List<double> xs)
    {
        if (xs.Count == 0)
            return 0d;

        var sorted = xs.OrderBy(static x => x).ToList();
        return sorted[sorted.Count / 2];
    }

    /// <summary>
    /// PM4 tiles are zero padded (<c>development_01_00.pm4</c>) while the ADTs are not
    /// (<c>development_1_0.adt</c>). Cataclysm-era corpora split placements into <c>_obj0.adt</c>;
    /// 3.3.5-era ones keep them in the monolithic ADT, so both spellings are tried.
    /// </summary>
    private static string? FindCompanionAdt(string pm4Path, string adtDirectory)
    {
        string fileName = Path.GetFileNameWithoutExtension(pm4Path);
        Match match = TilePattern().Match(fileName + ".pm4");
        if (match.Success
            && int.TryParse(match.Groups[1].Value, out int first)
            && int.TryParse(match.Groups[2].Value, out int second))
        {
            string stem = fileName[..match.Index];
            string[] tries =
            [
                Path.Combine(adtDirectory, $"{stem}_{first}_{second}_obj0.adt"),
                Path.Combine(adtDirectory, $"{stem}_{first}_{second}.adt")
            ];

            foreach (string candidate in tries)
            {
                if (File.Exists(candidate))
                    return candidate;
            }
        }

        string literal = Path.Combine(adtDirectory, fileName + "_obj0.adt");
        if (File.Exists(literal))
            return literal;

        literal = Path.Combine(adtDirectory, fileName + ".adt");
        return File.Exists(literal) ? literal : null;
    }
}
