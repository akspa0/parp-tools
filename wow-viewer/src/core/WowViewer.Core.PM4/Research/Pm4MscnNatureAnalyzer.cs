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
