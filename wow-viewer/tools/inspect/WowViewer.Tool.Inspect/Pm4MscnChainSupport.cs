using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Tests whether <c>MSCN</c> is a CHAIN of linked points - possibly closing into loops - rather than
/// the unordered node cloud it has been read as, and whether those points coincide with doodads.
/// </summary>
/// <remarks>
/// The standing reading is "a pre-baked node graph, freely positioned, with no index consumer inside
/// the file". That reading was arrived at by testing whether MSCN snaps to a lattice and whether any
/// stream indexes it - neither of which can see ORDER. If consecutive entries are neighbours, the
/// array is its own edge list and needs no index chunk, which would also explain why no consumer was
/// ever found.
///
/// <list type="number">
/// <item><b>Chain</b> - distance between consecutive entries against a same-file random-pair control.
/// A chain has consecutive distances far below chance; a cloud does not.</item>
/// <item><b>Loop</b> - scanning forward from each point for a return to within epsilon of it, against
/// the same search run on a SHUFFLED copy. A shuffle preserves the point set and destroys only the
/// order, so it isolates order as the thing being measured.</item>
/// <item><b>Doodads</b> - nearest MSCN point to each real <c>MDDF</c> row in the paired ADT, against a
/// control of uniformly random positions inside the same tile. This is the decisive one: it is the
/// difference between MSCN being about doodads and MSCN merely being dense enough to be near
/// anything.</item>
/// </list>
///
/// <para>All four axis pairings are measured for the doodad test rather than assumed, because MDDF is
/// on record as having its fields opposite to MSVT and a wrong pairing would read as a clean null.</para>
/// </remarks>
internal static class Pm4MscnChainSupport
{
    public static Pm4MscnChainReport Analyze(string pm4Directory, string? adtDirectory, float epsilon = 0.5f, int maxLoop = 64)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);
        string adtRoot = string.IsNullOrWhiteSpace(adtDirectory) ? resolved : adtDirectory;

        int files = 0, filesWithDoodads = 0;
        long mscnTotal = 0, doodadsTotal = 0;

        var consecutive = new Stat();
        var randomPair = new Stat();
        var loopLengths = new Dictionary<int, long>();
        var shuffledLoopLengths = new Dictionary<int, long>();
        long loopStartsTested = 0, loopsFound = 0, shuffledLoopsFound = 0;

        var nearXY = new Stat(); var nearYX = new Stat();
        var nearXYFlipped = new Stat(); var nearControl = new Stat();

        // If MSCN is a chain of short closed rings, the obvious candidate is that the rings ARE mesh
        // outlines - a second copy of the geometry's boundaries. Distance to the nearest MSVT (floor)
        // and MSPV (wall) vertex tests that, against a control point drawn from the same footprint so
        // that "MSCN is dense" cannot pass for "MSCN is on the mesh".
        var nearMsvt = new Stat(); var nearMspv = new Stat(); var nearMeshControl = new Stat();

        var rng = new Random(12345);

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Mscn.Count < 16)
                continue;

            files++;
            mscnTotal += chunks.Mscn.Count;

            List<Vector3> pts = [.. chunks.Mscn];

            if (chunks.Msvt.Count > 0 || chunks.Mspv.Count > 0)
            {
                Vector3 mlo = pts[0], mhi = pts[0];
                foreach (Vector3 p in pts)
                {
                    mlo = Vector3.Min(mlo, p);
                    mhi = Vector3.Max(mhi, p);
                }

                int mstep = Math.Max(1, pts.Count / 500);
                for (int i = 0; i < pts.Count; i += mstep)
                {
                    if (chunks.Msvt.Count > 0)
                        nearMsvt.Add(Nearest3(chunks.Msvt, pts[i]));
                    if (chunks.Mspv.Count > 0)
                        nearMspv.Add(Nearest3(chunks.Mspv, pts[i]));

                    Vector3 c = new(
                        mlo.X + ((mhi.X - mlo.X) * (float)rng.NextDouble()),
                        mlo.Y + ((mhi.Y - mlo.Y) * (float)rng.NextDouble()),
                        mlo.Z + ((mhi.Z - mlo.Z) * (float)rng.NextDouble()));
                    if (chunks.Msvt.Count > 0)
                        nearMeshControl.Add(Nearest3(chunks.Msvt, c));
                }
            }

            // 1. Chain: consecutive distance against random pairs from the same file.
            for (int i = 1; i < pts.Count; i++)
                consecutive.Add(Vector3.Distance(pts[i], pts[i - 1]));

            for (int i = 0; i < pts.Count; i++)
                randomPair.Add(Vector3.Distance(pts[rng.Next(pts.Count)], pts[rng.Next(pts.Count)]));

            // 2. Loop: forward scan for a return to the start, and the same on a shuffled copy.
            List<Vector3> shuffled = [.. pts];
            for (int i = shuffled.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
            }

            int step = Math.Max(1, pts.Count / 2000);
            for (int i = 0; i < pts.Count; i += step)
            {
                loopStartsTested++;
                int k = FindReturn(pts, i, epsilon, maxLoop);
                if (k > 0)
                {
                    loopsFound++;
                    loopLengths[k] = loopLengths.GetValueOrDefault(k) + 1;
                }

                int ks = FindReturn(shuffled, i, epsilon, maxLoop);
                if (ks > 0)
                {
                    shuffledLoopsFound++;
                    shuffledLoopLengths[ks] = shuffledLoopLengths.GetValueOrDefault(ks) + 1;
                }
            }

            // 3. Doodads.
            string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtRoot);
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

            IReadOnlyList<AdtModelPlacement> mddf = catalog.ModelPlacements;
            if (mddf.Count == 0)
                continue;

            filesWithDoodads++;
            doodadsTotal += mddf.Count;

            // MSCN into the same placement space the MODF comparison used.
            var placed = new List<Vector3>(pts.Count);
            foreach (Vector3 p in pts)
                placed.Add(Pm4CoordinateService.Pm4LocalToAdtPlacement(p));

            Vector3 lo = placed[0], hi = placed[0];
            foreach (Vector3 p in placed)
            {
                lo = Vector3.Min(lo, p);
                hi = Vector3.Max(hi, p);
            }

            foreach (AdtModelPlacement d in mddf)
            {
                Vector3 pos = d.Position;
                nearXY.Add(NearestXy(placed, pos.X, pos.Y));
                nearYX.Add(NearestXy(placed, pos.Y, pos.X));
                nearXYFlipped.Add(NearestXy(placed, Pm4CoordinateService.MapOrigin - pos.X, Pm4CoordinateService.MapOrigin - pos.Y));

                // Control: a random spot inside the same MSCN footprint.
                float cx = lo.X + ((hi.X - lo.X) * (float)rng.NextDouble());
                float cy = lo.Y + ((hi.Y - lo.Y) * (float)rng.NextDouble());
                nearControl.Add(NearestXy(placed, cx, cy));
            }
        }

        return new Pm4MscnChainReport(
            resolved, adtRoot, files, filesWithDoodads, mscnTotal, doodadsTotal, epsilon, maxLoop,
            consecutive.ToResult("consecutive MSCN entries"),
            randomPair.ToResult("CONTROL random pair, same file"),
            loopStartsTested, loopsFound, shuffledLoopsFound,
            loopStartsTested == 0 ? 0 : (double)loopsFound / loopStartsTested,
            loopStartsTested == 0 ? 0 : (double)shuffledLoopsFound / loopStartsTested,
            TopLengths(loopLengths),
            TopLengths(shuffledLoopLengths),
            nearXY.ToResult("MDDF (X,Y) -> nearest MSCN"),
            nearYX.ToResult("MDDF (Y,X) -> nearest MSCN"),
            nearXYFlipped.ToResult("MDDF origin-flipped -> nearest MSCN"),
            nearControl.ToResult("CONTROL random point -> nearest MSCN"),
            nearMsvt.ToResult("MSCN -> nearest MSVT (floor) vertex"),
            nearMspv.ToResult("MSCN -> nearest MSPV (wall) vertex"),
            nearMeshControl.ToResult("CONTROL random point -> nearest MSVT"));
    }

    /// <summary>
    /// Smallest forward step at which the walk returns to within <paramref name="epsilon"/> of its
    /// start, or 0 if it never does inside <paramref name="maxLoop"/>. Steps below 3 are ignored so a
    /// duplicated point is not counted as a loop.
    /// </summary>
    private static int FindReturn(List<Vector3> pts, int start, float epsilon, int maxLoop)
    {
        Vector3 a = pts[start];
        int limit = Math.Min(pts.Count - start - 1, maxLoop);
        for (int k = 3; k <= limit; k++)
        {
            if (Vector3.Distance(a, pts[start + k]) <= epsilon)
                return k;
        }

        return 0;
    }

    private static float Nearest3(IReadOnlyList<Vector3> pts, Vector3 q)
    {
        float best = float.MaxValue;
        foreach (Vector3 p in pts)
        {
            float d = Vector3.DistanceSquared(p, q);
            if (d < best)
                best = d;
        }

        return MathF.Sqrt(best);
    }

    private static float NearestXy(List<Vector3> pts, float x, float y)
    {
        float best = float.MaxValue;
        foreach (Vector3 p in pts)
        {
            float dx = p.X - x, dy = p.Y - y;
            float d = (dx * dx) + (dy * dy);
            if (d < best)
                best = d;
        }

        return MathF.Sqrt(best);
    }

    private static IReadOnlyList<Pm4ValueFrequency> TopLengths(Dictionary<int, long> lengths) =>
        [.. lengths.OrderByDescending(static kv => kv.Value).Take(10)
            .Select(static kv => new Pm4ValueFrequency(kv.Key.ToString(), (int)kv.Value))];

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
                (double)_v.Count(static x => x < 1.0) / _v.Count);
        }
    }
}

internal sealed record Pm4MscnChainReport(
    string Pm4Directory,
    string AdtDirectory,
    int Files,
    int FilesWithDoodads,
    long MscnTotal,
    long DoodadsTotal,
    float Epsilon,
    int MaxLoop,
    Pm4ErrorStat Consecutive,
    Pm4ErrorStat RandomPair,
    long LoopStartsTested,
    long LoopsFound,
    long ShuffledLoopsFound,
    double LoopFraction,
    double ShuffledLoopFraction,
    IReadOnlyList<Pm4ValueFrequency> LoopLengths,
    IReadOnlyList<Pm4ValueFrequency> ShuffledLoopLengths,
    Pm4ErrorStat NearestXy,
    Pm4ErrorStat NearestYx,
    Pm4ErrorStat NearestFlipped,
    Pm4ErrorStat NearestControl,
    Pm4ErrorStat NearestMsvt,
    Pm4ErrorStat NearestMspv,
    Pm4ErrorStat NearestMeshControl);
