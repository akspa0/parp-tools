using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Looks for objects that share a horizontal footprint but sit at different heights - a vertical
/// stack of near-copies.
/// </summary>
/// <remarks>
/// Observed in the viewer on sloped terrain: an object that belongs on top of a slope appears
/// repeated down the face of it, several copies at descending heights. If that is what the exporter
/// does, it is a generation rule and spec 184 has to reproduce it; if it is an artefact of how the
/// viewer splits objects, it is not.
///
/// <para>The test pairs objects within one file whose XY boxes overlap heavily (intersection over
/// union above a threshold) but whose placement heights differ. Footprint overlap alone would also
/// catch a building and the floor it stands on, so the Z separation is required too, and stacks are
/// reported with their Z spread so a two-storey building can be told from a slope smear.</para>
/// </remarks>
public static class Pm4VerticalStackAnalyzer
{
    public static Pm4VerticalStackReport Analyze(string inputPath, double minIou = 0.6, float minZGap = 1.0f)
    {
        List<string> files = File.Exists(inputPath)
            ? [inputPath]
            : Directory.EnumerateFiles(Pm4CoordinateService.ResolveMapDirectory(inputPath), "*.pm4", SearchOption.TopDirectoryOnly)
                .OrderBy(Path.GetFileName)
                .ToList();

        int filesScanned = 0, objectsTotal = 0, objectsInStacks = 0, stacks = 0;
        var stackSizes = new Dictionary<int, int>();
        var samples = new List<Pm4VerticalStackSample>();

        foreach (string path in files)
        {
            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Msur.Count == 0)
                continue;

            filesScanned++;

            // One entry per distinct placement height, with its XY box.
            var lo = new Dictionary<uint, Vector3>();
            var hi = new Dictionary<uint, Vector3>();
            foreach (Pm4MsurEntry s in c.Msur)
            {
                if (s.PackedParams == 0)
                    continue;

                long vs = s.MsviFirstIndex;
                long ve = vs + s.IndexCount;
                if (ve > c.Msvi.Count)
                    continue;

                for (long v = vs; v < ve; v++)
                {
                    uint vi = c.Msvi[(int)v];
                    if (vi >= c.Msvt.Count)
                        continue;

                    Vector3 p = c.Msvt[(int)vi];
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

            var keys = lo.Keys.ToList();
            objectsTotal += keys.Count;
            var assigned = new HashSet<uint>();

            for (int i = 0; i < keys.Count; i++)
            {
                if (assigned.Contains(keys[i]))
                    continue;

                var member = new List<uint> { keys[i] };
                for (int j = i + 1; j < keys.Count; j++)
                {
                    if (assigned.Contains(keys[j]))
                        continue;

                    if (Iou(lo[keys[i]], hi[keys[i]], lo[keys[j]], hi[keys[j]]) < minIou)
                        continue;

                    float za = BitConverter.UInt32BitsToSingle(keys[i]);
                    float zb = BitConverter.UInt32BitsToSingle(keys[j]);
                    if (MathF.Abs(za - zb) < minZGap)
                        continue;

                    member.Add(keys[j]);
                }

                if (member.Count < 2)
                    continue;

                stacks++;
                foreach (uint k in member)
                    assigned.Add(k);
                objectsInStacks += member.Count;
                stackSizes[member.Count] = stackSizes.GetValueOrDefault(member.Count) + 1;

                if (samples.Count < 10)
                {
                    List<float> zs = member.Select(static k => BitConverter.UInt32BitsToSingle(k)).OrderBy(static z => z).ToList();
                    samples.Add(new Pm4VerticalStackSample(
                        Path.GetFileNameWithoutExtension(path),
                        member.Count,
                        zs[0],
                        zs[^1],
                        zs[^1] - zs[0],
                        string.Join(", ", zs.Select(static z => z.ToString("F2")))));
                }
            }
        }

        IReadOnlyList<Pm4ValueFrequency> sizes = stackSizes
            .OrderByDescending(static kv => kv.Value)
            .Select(static kv => new Pm4ValueFrequency(kv.Key.ToString(), kv.Value))
            .ToList();

        return new Pm4VerticalStackReport(
            inputPath, filesScanned, objectsTotal, stacks, objectsInStacks,
            objectsTotal == 0 ? 0 : (double)objectsInStacks / objectsTotal,
            sizes, samples);
    }

    private static double Iou(Vector3 aMin, Vector3 aMax, Vector3 bMin, Vector3 bMax)
    {
        float ix = MathF.Max(0, MathF.Min(aMax.X, bMax.X) - MathF.Max(aMin.X, bMin.X));
        float iy = MathF.Max(0, MathF.Min(aMax.Y, bMax.Y) - MathF.Max(aMin.Y, bMin.Y));
        double inter = (double)ix * iy;
        double areaA = (double)(aMax.X - aMin.X) * (aMax.Y - aMin.Y);
        double areaB = (double)(bMax.X - bMin.X) * (bMax.Y - bMin.Y);
        double union = areaA + areaB - inter;
        return union <= 0 ? 0 : inter / union;
    }
}

public sealed record Pm4VerticalStackSample(
    string File,
    int Members,
    float MinZ,
    float MaxZ,
    float ZSpread,
    string Heights);

public sealed record Pm4VerticalStackReport(
    string Input,
    int FilesScanned,
    int ObjectsTotal,
    int Stacks,
    int ObjectsInStacks,
    double ObjectsInStacksFraction,
    IReadOnlyList<Pm4ValueFrequency> StackSizes,
    IReadOnlyList<Pm4VerticalStackSample> Samples);
