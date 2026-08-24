using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Tests whether <c>MSUR._0x00</c> is a surface CLASS — roof, wall, floor — rather than an opaque key.
/// </summary>
/// <remarks>
/// The hypothesis came from looking at the overlay coloured by this field: its values track with
/// exterior roof, exterior wall and interior floor. That is checkable, because those three roles have
/// different geometry and the field knows nothing about geometry:
///
/// <list type="bullet">
/// <item>a floor or a roof has a Z-dominant normal; a wall does not</item>
/// <item>a floor's normal points up, a roof's may point either way, a wall's is horizontal</item>
/// <item>within its own object, a roof sits high and a floor sits low</item>
/// </list>
///
/// So each value's normal orientation and its normalised height inside its own object are measured.
/// If the field is a class, the values separate on those axes. If it is an opaque key, they do not.
/// Height is normalised per object so a tall building and a small prop contribute comparably.
/// </remarks>
public static class Pm4SurfaceClassAnalyzer
{
    public static Pm4SurfaceClassReport AnalyzeDirectory(string inputDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);
        var byValue = new Dictionary<byte, ClassAccumulator>();
        int files = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Msur.Count == 0)
                continue;

            files++;

            // Per-object Z extent, so "high in the object" means the same for a keep and a crate.
            var objMin = new Dictionary<uint, float>();
            var objMax = new Dictionary<uint, float>();
            var centroids = new Dictionary<int, float>();

            for (int i = 0; i < c.Msur.Count; i++)
            {
                Pm4MsurEntry s = c.Msur[i];
                long vs = s.MsviFirstIndex;
                long ve = vs + s.IndexCount;
                if (s.IndexCount == 0 || ve > c.Msvi.Count)
                    continue;

                float zsum = 0; int zn = 0;
                for (long v = vs; v < ve; v++)
                {
                    uint vi = c.Msvi[(int)v];
                    if (vi < c.Msvt.Count) { zsum += c.Msvt[(int)vi].Z; zn++; }
                }
                if (zn == 0)
                    continue;

                float z = zsum / zn;
                centroids[i] = z;
                if (!objMin.TryGetValue(s.PackedParams, out float lo) || z < lo) objMin[s.PackedParams] = z;
                if (!objMax.TryGetValue(s.PackedParams, out float hi) || z > hi) objMax[s.PackedParams] = z;
            }

            for (int i = 0; i < c.Msur.Count; i++)
            {
                if (!centroids.TryGetValue(i, out float z))
                    continue;

                Pm4MsurEntry s = c.Msur[i];
                if (!byValue.TryGetValue(s.GroupKey, out ClassAccumulator? acc))
                {
                    acc = new ClassAccumulator(s.GroupKey);
                    byValue[s.GroupKey] = acc;
                }

                Vector3 n = s.Normal;
                float ax = MathF.Abs(n.X), ay = MathF.Abs(n.Y), az = MathF.Abs(n.Z);
                acc.Surfaces++;
                if (az >= ax && az >= ay) acc.ZDominant++;
                acc.NormalZSum += n.Z;
                if (n.Z > 0.7f) acc.FacingUp++;
                else if (n.Z < -0.7f) acc.FacingDown++;

                float lo = objMin[s.PackedParams], hi = objMax[s.PackedParams];
                if (hi - lo > 0.001f)
                {
                    acc.HeightSum += (z - lo) / (hi - lo);
                    acc.HeightCount++;
                }

                if (s.PackedParams == 0) acc.InZeroPopulation++;
            }
        }

        List<Pm4SurfaceClassResult> results = byValue.Values
            .OrderByDescending(static a => a.Surfaces)
            .Select(static a => a.ToResult())
            .ToList();

        return new Pm4SurfaceClassReport(resolved, files, results);
    }

    private sealed class ClassAccumulator(byte value)
    {
        public byte Value { get; } = value;
        public long Surfaces;
        public long ZDominant;
        public long FacingUp;
        public long FacingDown;
        public double NormalZSum;
        public double HeightSum;
        public long HeightCount;
        public long InZeroPopulation;

        public Pm4SurfaceClassResult ToResult() => new(
            Value, Surfaces,
            Surfaces == 0 ? 0 : (double)ZDominant / Surfaces,
            Surfaces == 0 ? 0 : (double)FacingUp / Surfaces,
            Surfaces == 0 ? 0 : (double)FacingDown / Surfaces,
            Surfaces == 0 ? 0 : NormalZSum / Surfaces,
            HeightCount == 0 ? 0 : HeightSum / HeightCount,
            Surfaces == 0 ? 0 : (double)InZeroPopulation / Surfaces);
    }
}

public sealed record Pm4SurfaceClassResult(
    byte Value,
    long Surfaces,
    double ZDominantFraction,
    double FacingUpFraction,
    double FacingDownFraction,
    double MeanNormalZ,
    double MeanNormalisedHeightInObject,
    double ZeroPopulationFraction);

public sealed record Pm4SurfaceClassReport(
    string InputDirectory,
    int Files,
    IReadOnlyList<Pm4SurfaceClassResult> Classes);
