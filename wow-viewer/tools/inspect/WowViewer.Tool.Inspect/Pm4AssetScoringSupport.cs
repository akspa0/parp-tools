using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Scores candidate assets against the box a PM4 object actually occupies, and measures how often the
/// right asset comes out on top.
/// </summary>
/// <remarks>
/// The point is to turn a human's guess about which model belongs somewhere - a hand reconstruction,
/// which is often right about the asset and imprecise about where it sits - into something checkable.
/// So the score deliberately uses <b>shape only</b>. Position never enters it.
///
/// <para>Dimensions are taken as (smaller horizontal, larger horizontal, height), which absorbs the
/// 90-degree rotations that would otherwise swap width for depth. Height is kept separate because it
/// is the axis PM4 recovers almost exactly - <c>BoundsMax.Z</c> lands within 0.028 units - while the
/// horizontal axes run small, PM4 holding only collision-relevant geometry.</para>
///
/// <para><b>Leave-one-out is mandatory here.</b> The library is built from the same MODF rows the test
/// is scored against, so an asset placed once would otherwise have a library entry equal to the very
/// box being matched and would rank first for free. Each placement is removed from its own asset's
/// entry before that asset is scored. Without this the accuracy number is meaningless, and it would
/// look excellent.</para>
///
/// <para>Two controls: ranking assets at random, and ranking by how common an asset is, which is the
/// score to beat for any method claiming to use geometry rather than base rates.</para>
/// </remarks>
internal static class Pm4AssetScoringSupport
{
    public static Pm4AssetScoringReport Analyze(string pm4Directory, string adtDirectory, IReadOnlyList<string>? extraLibraryDirectories = null)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);

        // Library: asset -> the shapes its navmesh box has actually been observed to take.
        //
        // Built from PM4 boxes, NOT from MODF boxes. Scoring a PM4 box against a model box compares
        // two different measurements: a PM4 holds only collision-relevant geometry, so its box runs a
        // median 6.1 units narrower and 10.0 units shorter than the model's. Doing that scored 2.50%
        // top-1, worse than guessing the most common asset. Both sides now come from the same
        // measurement process, so the bias cancels instead of being learned as noise.
        var library = new Dictionary<string, List<Vector3>>(StringComparer.OrdinalIgnoreCase);
        var frequency = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach ((string asset, Vector3 shape, _) in EnumerateMatchedObjects(resolved, adtDirectory))
        {
            if (!library.TryGetValue(asset, out List<Vector3>? shapes))
            {
                shapes = [];
                library[asset] = shapes;
            }

            shapes.Add(shape);
            frequency[asset] = frequency.GetValueOrDefault(asset) + 1;
        }

        int assetsFromPm4 = library.Count;
        long extraShapes = 0;
        var pm4Observed = new HashSet<string>(library.Keys, StringComparer.OrdinalIgnoreCase);

        // Extra library sources widen the CANDIDATE SET only. Their boxes come from MODF, which is a
        // different measurement to a PM4 box - a model box runs a median 6.1 units wider and 10.0
        // taller - so they are corrected by that offset before being pooled. Mixing them in raw is the
        // exact mistake that scored 2.50%, and an uncorrected candidate would be unrankable rather
        // than merely wrong.
        if (extraLibraryDirectories is { Count: > 0 })
        {
            Vector3 correction = new(-MedianOf([.. library.Values.SelectMany(static v => v).Select(static v => v.X)]),
                                     -MedianOf([.. library.Values.SelectMany(static v => v).Select(static v => v.Y)]),
                                     -MedianOf([.. library.Values.SelectMany(static v => v).Select(static v => v.Z)]));
            _ = correction; // shape offsets are applied per-axis below

            foreach (string dir in extraLibraryDirectories)
            {
                if (!Directory.Exists(dir))
                    continue;

                foreach (string adtPath in Directory.EnumerateFiles(dir, "*.adt", SearchOption.AllDirectories))
                {
                    AdtPlacementCatalog catalog;
                    try
                    {
                        catalog = AdtPlacementReader.Read(adtPath);
                    }
                    catch
                    {
                        continue;
                    }

                    foreach (AdtWorldModelPlacement row in catalog.WorldModelPlacements)
                    {
                        string name = Path.GetFileName(row.ModelPath);
                        if (string.IsNullOrWhiteSpace(name))
                            continue;

                        Vector3 modelShape = Shape(row.BoundsMin, row.BoundsMax);
                        Vector3 asNavmesh = new(
                            MathF.Max(modelShape.X - HorizontalOffset, 0.1f),
                            MathF.Max(modelShape.Y - HorizontalOffset, 0.1f),
                            MathF.Max(modelShape.Z - HeightOffset, 0.1f));

                        // ONLY for assets PM4 has never observed. Pooling these into an asset that
                        // already has real navmesh boxes destroyed the score - 48.53% to 15.64% top-1 -
                        // because the model-to-navmesh offset is not the constant the global
                        // correction pretends it is, and 11,021 corrected guesses drown out a handful
                        // of measurements. They earn their place only where the alternative is an
                        // asset that cannot be ranked at all.
                        if (pm4Observed.Contains(name))
                            continue;

                        if (!library.TryGetValue(name, out List<Vector3>? shapes))
                        {
                            shapes = [];
                            library[name] = shapes;
                        }

                        shapes.Add(asNavmesh);
                        extraShapes++;
                        frequency[name] = frequency.GetValueOrDefault(name) + 1;
                    }
                }
            }
        }

        List<string> assets = [.. library.Keys];
        List<string> byFrequency = [.. assets.OrderByDescending(a => frequency[a])];

        long scored = 0;
        long top1 = 0, top3 = 0, top5 = 0, top10 = 0;
        long freqTop1 = 0, freqTop5 = 0;
        double randomTop1Expectation = 0;
        var rankStat = new Stat();
        var shapeBias = new Stat();
        var heightBias = new Stat();
        var samples = new List<Pm4AssetScoringSample>();

        foreach ((string trueName, Vector3 observed, Pm4MatchedContext ctx) in EnumerateMatchedObjects(resolved, adtDirectory))
        {
            if (!library.ContainsKey(trueName))
                continue;

            shapeBias.Add(observed.Y - ctx.TruthShape.Y);
            heightBias.Add(observed.Z - ctx.TruthShape.Z);

            // Rank every asset by shape distance, leaving this observation out of its own entry.
            var ranked = new List<(string Name, double Score)>(assets.Count);
            foreach (string asset in assets)
            {
                Vector3? median = MedianShape(
                    library[asset],
                    asset.Equals(trueName, StringComparison.OrdinalIgnoreCase) ? observed : null);
                if (median is null)
                    continue;

                ranked.Add((asset, ShapeDistance(observed, median.Value)));
            }

            if (ranked.Count == 0)
                continue;

            ranked.Sort(static (a, b) => a.Score.CompareTo(b.Score));

            int rank = ranked.FindIndex(r => r.Name.Equals(trueName, StringComparison.OrdinalIgnoreCase));
            if (rank < 0)
                continue;

            scored++;
            rankStat.Add(rank + 1);
            if (rank < 1) top1++;
            if (rank < 3) top3++;
            if (rank < 5) top5++;
            if (rank < 10) top10++;
            randomTop1Expectation += 1.0 / ranked.Count;

            int freqRank = byFrequency.FindIndex(a => a.Equals(trueName, StringComparison.OrdinalIgnoreCase));
            if (freqRank == 0) freqTop1++;
            if (freqRank < 5) freqTop5++;

            if (samples.Count < 10)
            {
                samples.Add(new Pm4AssetScoringSample(
                    ctx.File, trueName, rank + 1, observed, ctx.TruthShape,
                    string.Join(", ", ranked.Take(3).Select(static r => r.Name))));
            }
        }

        // Held-out TILE cross-validation. Leave-one-out above answers "can shape identify an asset we
        // have seen elsewhere"; this answers the question that actually matters for restoration, which
        // is whether a library carries to a tile contributing nothing to it. Folds are by tile, so no
        // object is ever scored against a library its own tile helped build.
        const int folds = 5;
        var shapesByFold = new Dictionary<string, List<Vector3>[]>(StringComparer.OrdinalIgnoreCase);
        var objectsByFold = new List<(string Asset, Vector3 Shape)>[folds];
        for (int f = 0; f < folds; f++)
            objectsByFold[f] = [];

        var tileFold = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        int nextFold = 0;

        foreach ((string asset, Vector3 shape, Pm4MatchedContext ctx) in EnumerateMatchedObjects(resolved, adtDirectory))
        {
            if (!tileFold.TryGetValue(ctx.File, out int fold))
            {
                fold = nextFold++ % folds;
                tileFold[ctx.File] = fold;
            }

            if (!shapesByFold.TryGetValue(asset, out List<Vector3>[]? perFold))
            {
                perFold = new List<Vector3>[folds];
                for (int f = 0; f < folds; f++)
                    perFold[f] = [];
                shapesByFold[asset] = perFold;
            }

            perFold[fold].Add(shape);
            objectsByFold[fold].Add((asset, shape));
        }

        long hScored = 0, hTop1 = 0, hTop3 = 0, hTop5 = 0, hTop10 = 0, hUnfindable = 0;
        long hFindableScored = 0, hFindableTop1 = 0, hFindableTop5 = 0;
        var hRank = new Stat();

        for (int f = 0; f < folds; f++)
        {
            // Library from every OTHER fold.
            var foldLibrary = new Dictionary<string, Vector3>(StringComparer.OrdinalIgnoreCase);
            foreach ((string asset, List<Vector3>[] perFold) in shapesByFold)
            {
                var pooled = new List<Vector3>();
                for (int g = 0; g < folds; g++)
                {
                    if (g != f)
                        pooled.AddRange(perFold[g]);
                }

                Vector3? median = MedianShape(pooled, null);
                if (median is Vector3 m)
                    foldLibrary[asset] = m;
            }

            if (foldLibrary.Count == 0)
                continue;

            List<string> foldAssets = [.. foldLibrary.Keys];

            foreach ((string trueName, Vector3 observed) in objectsByFold[f])
            {
                hScored++;

                // An asset seen ONLY in the held-out tile cannot be found at any rank. Counting these
                // as ordinary misses would hide the real limit, so they are reported separately.
                bool findable = foldLibrary.ContainsKey(trueName);
                if (!findable)
                {
                    hUnfindable++;
                    continue;
                }

                var ranked = new List<(string Name, double Score)>(foldAssets.Count);
                foreach (string asset in foldAssets)
                    ranked.Add((asset, ShapeDistance(observed, foldLibrary[asset])));

                ranked.Sort(static (a, b) => a.Score.CompareTo(b.Score));
                int rank = ranked.FindIndex(r => r.Name.Equals(trueName, StringComparison.OrdinalIgnoreCase));
                if (rank < 0)
                    continue;

                hRank.Add(rank + 1);
                hFindableScored++;
                if (rank < 1) { hTop1++; hFindableTop1++; }
                if (rank < 3) hTop3++;
                if (rank < 5) { hTop5++; hFindableTop5++; }
                if (rank < 10) hTop10++;
            }
        }

        return new Pm4AssetScoringReport(
            resolved, adtDirectory, assets.Count, scored,
            assetsFromPm4, extraShapes,
            scored == 0 ? 0 : (double)top1 / scored,
            scored == 0 ? 0 : (double)top3 / scored,
            scored == 0 ? 0 : (double)top5 / scored,
            scored == 0 ? 0 : (double)top10 / scored,
            scored == 0 ? 0 : randomTop1Expectation / scored,
            scored == 0 ? 0 : (double)freqTop1 / scored,
            scored == 0 ? 0 : (double)freqTop5 / scored,
            rankStat.ToResult("rank of the true asset"),
            shapeBias.ToResult("PM4 larger-horizontal minus MODF"),
            heightBias.ToResult("PM4 height minus MODF"),
            samples,
            folds,
            hScored,
            hScored == 0 ? 0 : (double)hTop1 / hScored,
            hScored == 0 ? 0 : (double)hTop3 / hScored,
            hScored == 0 ? 0 : (double)hTop5 / hScored,
            hScored == 0 ? 0 : (double)hTop10 / hScored,
            hScored == 0 ? 0 : (double)hUnfindable / hScored,
            hFindableScored,
            hFindableScored == 0 ? 0 : (double)hFindableTop1 / hFindableScored,
            hFindableScored == 0 ? 0 : (double)hFindableTop5 / hFindableScored,
            hRank.ToResult("held-out rank of the true asset"));
    }

    // Measured offsets between a PM4 navmesh box and the model box MODF records.
    private const float HorizontalOffset = 6.064f;
    private const float HeightOffset = 10.044f;

    private static float MedianOf(List<float> v)
    {
        if (v.Count == 0)
            return 0;
        v.Sort();
        return v[v.Count / 2];
    }

    private readonly record struct Pm4MatchedContext(string File, Vector3 TruthShape);

    /// <summary>
    /// Every PM4 object that resolves to a known asset, with the box its own geometry occupies.
    /// </summary>
    /// <remarks>
    /// One enumeration feeds both the library and the scoring pass, so the two cannot silently diverge
    /// in which objects they consider - a difference there would quietly invalidate the leave-one-out.
    /// </remarks>
    private static IEnumerable<(string Asset, Vector3 Shape, Pm4MatchedContext Context)> EnumerateMatchedObjects(
        string resolved, string adtDirectory)
    {
        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtDirectory);
            if (adtPath is null)
                continue;

            AdtPlacementCatalog catalog;
            try
            {
                catalog = AdtPlacementReader.Read(adtPath);
            }
            catch
            {
                continue;
            }

            IReadOnlyList<AdtWorldModelPlacement> modf = catalog.WorldModelPlacements;
            if (modf.Count == 0)
                continue;

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
                continue;

            var byHeight = new Dictionary<uint, AdtWorldModelPlacement>();
            foreach (AdtWorldModelPlacement row in modf)
                byHeight[BitConverter.SingleToUInt32Bits(row.Position.Z)] = row;

            var lo = new Dictionary<uint, Vector3>();
            var hi = new Dictionary<uint, Vector3>();
            foreach (Pm4MsurEntry surface in chunks.Msur)
            {
                if (surface.PackedParams == 0)
                    continue;

                long vs = surface.MsviFirstIndex;
                long ve = vs + surface.IndexCount;
                if (surface.IndexCount == 0 || ve > chunks.Msvi.Count)
                    continue;

                for (long v = vs; v < ve; v++)
                {
                    uint vi = chunks.Msvi[(int)v];
                    if (vi >= chunks.Msvt.Count)
                        continue;

                    Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(chunks.Msvt[(int)vi]);
                    if (!lo.TryGetValue(surface.PackedParams, out Vector3 cur))
                    {
                        lo[surface.PackedParams] = p;
                        hi[surface.PackedParams] = p;
                        continue;
                    }

                    lo[surface.PackedParams] = Vector3.Min(cur, p);
                    hi[surface.PackedParams] = Vector3.Max(hi[surface.PackedParams], p);
                }
            }

            foreach ((uint key, Vector3 min) in lo)
            {
                if (!byHeight.TryGetValue(key, out AdtWorldModelPlacement truth))
                    continue;

                string name = Path.GetFileName(truth.ModelPath);
                if (string.IsNullOrWhiteSpace(name))
                    continue;

                yield return (
                    name,
                    Shape(min, hi[key]),
                    new Pm4MatchedContext(
                        Path.GetFileNameWithoutExtension(pm4Path),
                        Shape(truth.BoundsMin, truth.BoundsMax)));
            }
        }
    }

    /// <summary>
    /// (smaller horizontal, larger horizontal, height) - rotation-insensitive for quarter turns.
    /// </summary>
    private static Vector3 Shape(Vector3 min, Vector3 max)
    {
        float dx = MathF.Abs(max.X - min.X);
        float dy = MathF.Abs(max.Y - min.Y);
        return new Vector3(MathF.Min(dx, dy), MathF.Max(dx, dy), MathF.Abs(max.Z - min.Z));
    }

    /// <summary>
    /// Component-wise median of an asset's observed shapes, optionally removing one occurrence first so
    /// a placement is never matched against itself.
    /// </summary>
    private static Vector3? MedianShape(List<Vector3> shapes, Vector3? exclude)
    {
        List<Vector3> use = shapes;
        if (exclude is Vector3 e)
        {
            use = [.. shapes];
            int at = use.FindIndex(v => Vector3.DistanceSquared(v, e) < 1e-6f);
            if (at >= 0)
                use.RemoveAt(at);
        }

        if (use.Count == 0)
            return null;

        List<float> xs = [.. use.Select(static v => v.X).Order()];
        List<float> ys = [.. use.Select(static v => v.Y).Order()];
        List<float> zs = [.. use.Select(static v => v.Z).Order()];
        return new Vector3(xs[xs.Count / 2], ys[ys.Count / 2], zs[zs.Count / 2]);
    }

    /// <summary>
    /// Relative shape distance. Relative rather than absolute so a hut and a keep are judged on the
    /// same footing instead of the keep dominating every comparison it appears in.
    /// </summary>
    private static double ShapeDistance(Vector3 a, Vector3 b)
    {
        static double Rel(float p, float q)
        {
            float scale = MathF.Max(MathF.Max(MathF.Abs(p), MathF.Abs(q)), 1f);
            return MathF.Abs(p - q) / scale;
        }

        return Rel(a.X, b.X) + Rel(a.Y, b.Y) + (2.0 * Rel(a.Z, b.Z));
    }

    private sealed class Stat
    {
        private readonly List<double> _v = [];

        public void Add(double d) => _v.Add(d);

        public Pm4ErrorStat ToResult(string name)
        {
            if (_v.Count == 0)
                return new Pm4ErrorStat(name, 0, 0, 0, 0, 0);

            _v.Sort();
            return new Pm4ErrorStat(
                name, _v.Count, _v[_v.Count / 2], _v[(int)(_v.Count * 0.90)], _v[^1],
                (double)_v.Count(static x => Math.Abs(x) < 1.0) / _v.Count);
        }
    }
}

internal sealed record Pm4AssetScoringSample(
    string File,
    string TrueAsset,
    int Rank,
    Vector3 ObservedShape,
    Vector3 TruthShape,
    string TopThree);

internal sealed record Pm4AssetScoringReport(
    string Pm4Directory,
    string AdtDirectory,
    int AssetsInLibrary,
    long ObjectsScored,
    int AssetsFromPm4,
    long ExtraLibraryShapes,
    double Top1,
    double Top3,
    double Top5,
    double Top10,
    double RandomTop1Expectation,
    double FrequencyTop1,
    double FrequencyTop5,
    Pm4ErrorStat TrueRank,
    Pm4ErrorStat HorizontalBias,
    Pm4ErrorStat HeightBias,
    IReadOnlyList<Pm4AssetScoringSample> Samples,
    int HoldoutFolds,
    long HoldoutScored,
    double HoldoutTop1,
    double HoldoutTop3,
    double HoldoutTop5,
    double HoldoutTop10,
    double HoldoutUnfindableFraction,
    long HoldoutFindableScored,
    double HoldoutFindableTop1,
    double HoldoutFindableTop5,
    Pm4ErrorStat HoldoutRank);
