using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Characterises <c>MSCN</c> positively: its population relative to the other streams, and how its
/// points sit against the floor mesh (<c>MSVT</c>) and the wall mesh (<c>MSPV</c>).
/// </summary>
/// <remarks>
/// Written because the earlier writeup reduced MSCN to two negatives - "not normals" and "no
/// surviving index consumer" - which understates what is known. MSCN holds positions in the same
/// frame as MSVT and MSPV, and the viewer already draws them at the correct object locations. The
/// open question is what they connect, so this measures coincidence against both meshes rather than
/// asserting a role.
/// </remarks>
public static class Pm4MscnNatureAnalyzer
{
    private const float Epsilon = 0.25f;

    public static Pm4MscnNatureReport AnalyzeDirectory(string inputDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        long files = 0, mscn = 0, msvt = 0, mspv = 0, msur = 0;
        long mscnOnMsvt = 0, mscnOnMspv = 0, mscnOnNeither = 0;
        var perFileMscnPerSurface = new List<double>();

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Mscn.Count == 0 || c.Msur.Count == 0)
                continue;

            files++;
            mscn += c.Mscn.Count;
            msvt += c.Msvt.Count;
            mspv += c.Mspv.Count;
            msur += c.Msur.Count;
            perFileMscnPerSurface.Add((double)c.Mscn.Count / c.Msur.Count);

            var floorGrid = BuildGrid(c.Msvt);
            var wallGrid = BuildGrid(c.Mspv);

            foreach (Vector3 p in c.Mscn)
            {
                bool onFloor = NearAny(floorGrid, c.Msvt, p);
                bool onWall = NearAny(wallGrid, c.Mspv, p);
                if (onFloor) mscnOnMsvt++;
                if (onWall) mscnOnMspv++;
                if (!onFloor && !onWall) mscnOnNeither++;
            }
        }

        return new Pm4MscnNatureReport(
            resolved, files, mscn, msvt, mspv, msur,
            msur == 0 ? 0 : (double)mscn / msur,
            msvt == 0 ? 0 : (double)mscn / msvt,
            mspv == 0 ? 0 : (double)mscn / mspv,
            mscnOnMsvt, mscn == 0 ? 0 : (double)mscnOnMsvt / mscn,
            mscnOnMspv, mscn == 0 ? 0 : (double)mscnOnMspv / mscn,
            mscnOnNeither, mscn == 0 ? 0 : (double)mscnOnNeither / mscn,
            perFileMscnPerSurface.Count == 0 ? 0 : perFileMscnPerSurface.Min(),
            perFileMscnPerSurface.Count == 0 ? 0 : perFileMscnPerSurface.Max());
    }

    /// <summary>
    /// Tests whether a point stream snaps to a regular lattice, against candidate step sizes drawn
    /// from WoW's terrain subdivision. MSVT is measured with the identical test as a control: a
    /// mesh built from model geometry should NOT be gridded, so if MSCN is and MSVT is not, the
    /// lattice belongs to the node network rather than to the coordinate system.
    /// </summary>
    public static Pm4GridSnapReport AnalyzeGridSnap(string inputDirectory, float epsilon = 0.01f, int maxFiles = 60)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        // Tile, chunk (tile/16), cell (chunk/8), half-cell, and plain unit fractions.
        double[] steps = [533.33333, 33.333333, 8.3333333, 4.1666667, 2.0833333, 1.0, 0.5, 0.25];
        var mscnHits = new long[steps.Length];
        var msvtHits = new long[steps.Length];
        long mscnN = 0, msvtN = 0;
        int files = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (files >= maxFiles)
                break;

            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Mscn.Count == 0)
                continue;

            files++;
            foreach (Vector3 p in c.Mscn)
            {
                mscnN += 3;
                for (int i = 0; i < steps.Length; i++)
                    mscnHits[i] += Snapped(p.X, steps[i], epsilon) + Snapped(p.Y, steps[i], epsilon) + Snapped(p.Z, steps[i], epsilon);
            }
            foreach (Vector3 p in c.Msvt)
            {
                msvtN += 3;
                for (int i = 0; i < steps.Length; i++)
                    msvtHits[i] += Snapped(p.X, steps[i], epsilon) + Snapped(p.Y, steps[i], epsilon) + Snapped(p.Z, steps[i], epsilon);
            }
        }

        var rows = new List<Pm4GridSnapRow>();
        for (int i = 0; i < steps.Length; i++)
        {
            rows.Add(new Pm4GridSnapRow(
                steps[i],
                mscnN == 0 ? 0 : (double)mscnHits[i] / mscnN,
                msvtN == 0 ? 0 : (double)msvtHits[i] / msvtN));
        }

        return new Pm4GridSnapReport(resolved, files, mscnN / 3, msvtN / 3, epsilon, rows);
    }

    private static int Snapped(double v, double step, double eps)
    {
        double r = Math.Abs(v / step - Math.Round(v / step)) * step;
        return r <= eps ? 1 : 0;
    }

    /// <summary>
    /// Tests whether a point stream is stored in SPATIAL ORDER, which is what an acceleration
    /// structure looks like and what an append-as-you-go list does not.
    /// </summary>
    /// <remarks>
    /// If MSCN is a lookup map for steering a query to the right neighbourhood, consecutive entries
    /// should be spatially close - tree traversal order, Morton order or a grid sweep all produce
    /// that. If it is simply accumulated while walking objects, consecutive entries are close only
    /// as far as the objects were, which the mesh streams share. So MSVT and MSPV are measured
    /// identically as controls: the claim needs MSCN to be MORE ordered than they are, not merely
    /// ordered.
    ///
    /// The baseline is the mean distance between randomly chosen pairs from the same file, using a
    /// fixed seed so the figure is reproducible. Locality is reported as the ratio of consecutive
    /// distance to that baseline - lower means more spatially ordered.
    /// </remarks>
    public static Pm4SpatialOrderReport AnalyzeSpatialOrder(string inputDirectory, int maxFiles = 80)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);
        var rng = new Random(20260824);
        var mscn = new OrderAccumulator("MSCN");
        var msvt = new OrderAccumulator("MSVT (control)");
        var mspv = new OrderAccumulator("MSPV (control)");
        int files = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (files >= maxFiles)
                break;

            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Mscn.Count < 64)
                continue;

            files++;
            mscn.Observe(c.Mscn, rng);
            msvt.Observe(c.Msvt, rng);
            mspv.Observe(c.Mspv, rng);
        }

        return new Pm4SpatialOrderReport(resolved, files, mscn.ToResult(), msvt.ToResult(), mspv.ToResult());
    }

    private sealed class OrderAccumulator(string name)
    {
        private readonly List<double> _locality = [];
        private readonly List<double> _monotoneX = [];
        public string Name { get; } = name;

        public void Observe(IReadOnlyList<System.Numerics.Vector3> pts, Random rng)
        {
            if (pts.Count < 64)
                return;

            double consecutive = 0;
            int inc = 0;
            for (int i = 0; i + 1 < pts.Count; i++)
            {
                consecutive += Vector3.Distance(pts[i], pts[i + 1]);
                if (pts[i + 1].X >= pts[i].X)
                    inc++;
            }
            consecutive /= pts.Count - 1;

            double random = 0;
            const int samples = 4000;
            for (int i = 0; i < samples; i++)
                random += Vector3.Distance(pts[rng.Next(pts.Count)], pts[rng.Next(pts.Count)]);
            random /= samples;

            if (random > 1e-6)
                _locality.Add(consecutive / random);
            _monotoneX.Add((double)inc / (pts.Count - 1));
        }

        public Pm4SpatialOrderResult ToResult() => new(
            Name,
            _locality.Count,
            _locality.Count == 0 ? 0 : _locality.Average(),
            _monotoneX.Count == 0 ? 0 : _monotoneX.Average());
    }

    /// <summary>
    /// Pairwise coincidence between the three point streams, testing whether PM4 carries parallel
    /// COPIES of one geometry with different attributes attached, or three distinct point sets.
    /// </summary>
    /// <remarks>
    /// The streams are comparable in size and share a frame, which is what a "two or three copies
    /// that interchange properties" reading would predict. Copies would share most of their points.
    /// Distinct roles - a floor mesh, a wall mesh, a node graph - would touch only where they meet.
    /// The rate separates those, and it is reported in BOTH directions because the streams differ in
    /// size and a one-way percentage would hide that.
    /// </remarks>
    public static Pm4StreamOverlapReport AnalyzeStreamOverlap(string inputDirectory, int maxFiles = 60)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);
        long msvt = 0, mspv = 0, mscn = 0;
        long vtInPv = 0, pvInVt = 0, vtInCn = 0, cnInVt = 0, pvInCn = 0, cnInPv = 0;
        int files = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (files >= maxFiles)
                break;

            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Msvt.Count == 0 || c.Mspv.Count == 0)
                continue;

            files++;
            msvt += c.Msvt.Count; mspv += c.Mspv.Count; mscn += c.Mscn.Count;

            var gVt = BuildGrid(c.Msvt);
            var gPv = BuildGrid(c.Mspv);
            var gCn = BuildGrid(c.Mscn);

            foreach (Vector3 p in c.Msvt)
            {
                if (NearAny(gPv, c.Mspv, p)) vtInPv++;
                if (c.Mscn.Count > 0 && NearAny(gCn, c.Mscn, p)) vtInCn++;
            }
            foreach (Vector3 p in c.Mspv)
            {
                if (NearAny(gVt, c.Msvt, p)) pvInVt++;
                if (c.Mscn.Count > 0 && NearAny(gCn, c.Mscn, p)) pvInCn++;
            }
            foreach (Vector3 p in c.Mscn)
            {
                if (NearAny(gVt, c.Msvt, p)) cnInVt++;
                if (NearAny(gPv, c.Mspv, p)) cnInPv++;
            }
        }

        return new Pm4StreamOverlapReport(
            resolved, files, msvt, mspv, mscn,
            msvt == 0 ? 0 : (double)vtInPv / msvt,
            mspv == 0 ? 0 : (double)pvInVt / mspv,
            msvt == 0 ? 0 : (double)vtInCn / msvt,
            mscn == 0 ? 0 : (double)cnInVt / mscn,
            mspv == 0 ? 0 : (double)pvInCn / mspv,
            mscn == 0 ? 0 : (double)cnInPv / mscn);
    }

    private static Dictionary<(int, int, int), List<int>> BuildGrid(IReadOnlyList<Vector3> pts)
    {
        var grid = new Dictionary<(int, int, int), List<int>>();
        for (int i = 0; i < pts.Count; i++)
        {
            var k = Cell(pts[i]);
            if (!grid.TryGetValue(k, out var list))
            {
                list = [];
                grid[k] = list;
            }
            list.Add(i);
        }
        return grid;
    }

    private static (int, int, int) Cell(Vector3 p)
        => ((int)MathF.Floor(p.X / Epsilon), (int)MathF.Floor(p.Y / Epsilon), (int)MathF.Floor(p.Z / Epsilon));

    private static bool NearAny(Dictionary<(int, int, int), List<int>> grid, IReadOnlyList<Vector3> pts, Vector3 p)
    {
        var (cx, cy, cz) = Cell(p);
        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++)
        {
            if (!grid.TryGetValue((cx + dx, cy + dy, cz + dz), out var list))
                continue;
            foreach (int i in list)
            {
                if (Vector3.Distance(pts[i], p) <= Epsilon)
                    return true;
            }
        }
        return false;
    }
}

public sealed record Pm4MscnNatureReport(
    string InputDirectory,
    long Files,
    long MscnPoints,
    long MsvtPoints,
    long MspvPoints,
    long MsurSurfaces,
    double MscnPerSurface,
    double MscnPerMsvt,
    double MscnPerMspv,
    long MscnCoincidentWithMsvt,
    double MscnCoincidentWithMsvtFraction,
    long MscnCoincidentWithMspv,
    double MscnCoincidentWithMspvFraction,
    long MscnCoincidentWithNeither,
    double MscnCoincidentWithNeitherFraction,
    double MinMscnPerSurface,
    double MaxMscnPerSurface);

public sealed record Pm4GridSnapRow(double Step, double MscnSnappedFraction, double MsvtSnappedFraction);

public sealed record Pm4GridSnapReport(
    string InputDirectory,
    int Files,
    long MscnPoints,
    long MsvtPoints,
    float Epsilon,
    IReadOnlyList<Pm4GridSnapRow> Rows);

public sealed record Pm4SpatialOrderResult(string Name, int Files, double LocalityRatio, double AscendingXFraction);

public sealed record Pm4SpatialOrderReport(
    string InputDirectory,
    int Files,
    Pm4SpatialOrderResult Mscn,
    Pm4SpatialOrderResult Msvt,
    Pm4SpatialOrderResult Mspv);

public sealed record Pm4StreamOverlapReport(
    string InputDirectory,
    int Files,
    long MsvtPoints,
    long MspvPoints,
    long MscnPoints,
    double MsvtOnMspv,
    double MspvOnMsvt,
    double MsvtOnMscn,
    double MscnOnMsvt,
    double MspvOnMscn,
    double MscnOnMspv);
