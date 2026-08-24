using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Tests two things about how a PM4 object's surfaces are laid out: whether <c>MSCN</c> holds unit
/// normals or positions, and whether surface subdivision is spatially biased along an axis.
/// </summary>
/// <remarks>
/// <para>
/// The subdivision test exists because objects visibly carry finer detail on one side than the
/// other. A directional bias is the signature of a <b>scan-ordered greedy merge</b>: merging
/// coplanar faces while sweeping along an axis leaves large merged polygons where the sweep began
/// and unmerged remainders where it ended. Light plays no part in a pathfinding mesh, but a sweep
/// direction would produce exactly the asymmetry that looks like directional shading.
/// </para>
/// <para>
/// Both axes are measured, plus an area-weighted control, because "one side has more surfaces" is
/// only meaningful relative to how much of the object's area sits on that side. An object that is
/// simply bigger on one side would otherwise read as biased.
/// </para>
/// </remarks>
public static class Pm4MergeBiasAnalyzer
{
    public static Pm4MergeBiasReport AnalyzeDirectory(string inputDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        long mscnTotal = 0, mscnUnitLength = 0;
        double mscnLenMin = double.MaxValue, mscnLenMax = 0, mscnLenSum = 0;

        var xBias = new List<double>();
        var yBias = new List<double>();
        var xAreaBias = new List<double>();
        int objectsMeasured = 0;
        int filesSeen = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(path).KnownChunks;
            IReadOnlyList<Pm4MsurEntry> msur = chunks.Msur;
            IReadOnlyList<uint> msvi = chunks.Msvi;
            IReadOnlyList<Vector3> msvt = chunks.Msvt;
            if (msur.Count == 0)
                continue;

            filesSeen++;

            foreach (Vector3 n in chunks.Mscn)
            {
                double len = n.Length();
                mscnTotal++;
                mscnLenSum += len;
                if (len < mscnLenMin) mscnLenMin = len;
                if (len > mscnLenMax) mscnLenMax = len;
                if (Math.Abs(len - 1.0) < 0.01)
                    mscnUnitLength++;
            }

            // Per-surface centroid and planar area, grouped by object.
            var byObject = new Dictionary<uint, List<(Vector3 Centroid, double Area)>>();
            foreach (Pm4MsurEntry s in msur)
            {
                long start = s.MsviFirstIndex;
                long end = start + s.IndexCount;
                if (s.IndexCount < 3 || end > msvi.Count)
                    continue;

                var pts = new List<Vector3>(s.IndexCount);
                for (long k = start; k < end; k++)
                {
                    uint vi = msvi[(int)k];
                    if (vi < msvt.Count)
                        pts.Add(msvt[(int)vi]);
                }

                if (pts.Count < 3)
                    continue;

                Vector3 centroid = Vector3.Zero;
                foreach (Vector3 p in pts)
                    centroid += p;
                centroid /= pts.Count;

                // Fan area, matching how MSUR surfaces are triangulated.
                double area = 0;
                for (int i = 1; i + 1 < pts.Count; i++)
                    area += 0.5 * Vector3.Cross(pts[i] - pts[0], pts[i + 1] - pts[0]).Length();

                if (!byObject.TryGetValue(s.PackedParams, out var list))
                {
                    list = [];
                    byObject[s.PackedParams] = list;
                }
                list.Add((centroid, area));
            }

            foreach ((uint raw, var surfaces) in byObject)
            {
                if (raw == 0 || surfaces.Count < 8)
                    continue;

                float minX = surfaces.Min(static s => s.Centroid.X);
                float maxX = surfaces.Max(static s => s.Centroid.X);
                float minY = surfaces.Min(static s => s.Centroid.Y);
                float maxY = surfaces.Max(static s => s.Centroid.Y);
                if (maxX - minX < 1e-3f || maxY - minY < 1e-3f)
                    continue;

                objectsMeasured++;

                float midX = (minX + maxX) / 2f;
                float midY = (minY + maxY) / 2f;

                int lowXCount = surfaces.Count(s => s.Centroid.X < midX);
                int highXCount = surfaces.Count - lowXCount;
                int lowYCount = surfaces.Count(s => s.Centroid.Y < midY);
                int highYCount = surfaces.Count - lowYCount;

                // Signed bias in [-1, 1]: +1 means every surface sits on the high side.
                xBias.Add((highXCount - lowXCount) / (double)surfaces.Count);
                yBias.Add((highYCount - lowYCount) / (double)surfaces.Count);

                // Control: the same split by AREA. If surfaces are merely smaller on one side,
                // count bias will be positive while area bias stays near zero.
                double lowXArea = surfaces.Where(s => s.Centroid.X < midX).Sum(static s => s.Area);
                double highXArea = surfaces.Sum(static s => s.Area) - lowXArea;
                double totalArea = lowXArea + highXArea;
                if (totalArea > 1e-6)
                    xAreaBias.Add((highXArea - lowXArea) / totalArea);
            }
        }

        return new Pm4MergeBiasReport(
            resolved,
            filesSeen,
            mscnTotal,
            mscnUnitLength,
            mscnTotal == 0 ? 0d : (double)mscnUnitLength / mscnTotal,
            mscnTotal == 0 ? 0d : mscnLenMin,
            mscnLenMax,
            mscnTotal == 0 ? 0d : mscnLenSum / mscnTotal,
            objectsMeasured,
            Mean(xBias), StdDev(xBias),
            Mean(yBias), StdDev(yBias),
            Mean(xAreaBias), StdDev(xAreaBias));
    }

    private static double Mean(List<double> xs) => xs.Count == 0 ? 0 : xs.Average();

    private static double StdDev(List<double> xs)
    {
        if (xs.Count < 2)
            return 0;
        double m = xs.Average();
        return Math.Sqrt(xs.Sum(x => (x - m) * (x - m)) / xs.Count);
    }
}

public sealed record Pm4MergeBiasReport(
    string InputDirectory,
    int FilesWithSurfaces,
    long MscnPoints,
    long MscnUnitLengthPoints,
    double MscnUnitLengthFraction,
    double MscnLengthMin,
    double MscnLengthMax,
    double MscnLengthMean,
    int ObjectsMeasured,
    double SurfaceCountBiasX,
    double SurfaceCountBiasXStdDev,
    double SurfaceCountBiasY,
    double SurfaceCountBiasYStdDev,
    double SurfaceAreaBiasX,
    double SurfaceAreaBiasXStdDev);
