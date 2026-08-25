using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Tests whether an object's rotation can be recovered by fitting its <c>MPRL</c> footprint points.
/// </summary>
/// <remarks>
/// The idea being tested: <c>MPRL</c> points lie along the sides of a placed model where it meets the
/// ground, so they carry the model's orientation. Given the real asset one could fit its footprint to
/// them and read off the rotation. That needs the WMO files, which this corpus does not have.
///
/// <para>The method can still be validated without them, because the same claim makes a checkable
/// prediction about <b>pairs</b>. If the point cloud encodes orientation, then two placements of the
/// SAME asset should have footprints differing by exactly the difference of their rotations. So this
/// recovers a relative angle from geometry alone and scores it against the real <c>MODF</c> rotations,
/// which are known for tiles that still have placements.</para>
///
/// <para>Fitting is an exhaustive one-degree sweep minimising mean nearest-neighbour distance between
/// the two centred clouds. Crude, and adequate: the question is whether the signal exists at all, not
/// how fast it can be extracted.</para>
///
/// <para>Which rotation component is the yaw is not assumed - all three are scored, and a control
/// pairs unrelated assets so that "recovered an angle" cannot pass for "recovered the right angle".</para>
/// </remarks>
internal static class Pm4RotationFitSupport
{
    private const int MinPoints = 6;

    public static Pm4RotationFitReport Analyze(string pm4Directory, string adtDirectory, int maxFiles = 200)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);

        // asset -> instances, each a centred footprint cloud plus its recorded rotation.
        var instances = new Dictionary<string, List<Instance>>(StringComparer.OrdinalIgnoreCase);
        int files = 0, objectsWithPoints = 0;

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (files >= maxFiles)
                break;

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

            if (catalog.WorldModelPlacements.Count == 0)
                continue;

            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (c.Msur.Count == 0 || c.Mprl.Count == 0)
                continue;

            files++;

            var byHeight = new Dictionary<uint, AdtWorldModelPlacement>();
            foreach (AdtWorldModelPlacement row in catalog.WorldModelPlacements)
                byHeight[BitConverter.SingleToUInt32Bits(row.Position.Z)] = row;

            var mprl = new List<Vector3>(c.Mprl.Count);
            foreach (Pm4MprlEntry e in c.Mprl)
                mprl.Add(Pm4CoordinateService.MprlToAdtPlacement(e.Position));

            foreach ((uint key, (Vector3 min, Vector3 max)) in ExtractObjects(c))
            {
                if (!byHeight.TryGetValue(key, out AdtWorldModelPlacement truth))
                    continue;

                string asset = Path.GetFileName(truth.ModelPath);
                if (string.IsNullOrWhiteSpace(asset))
                    continue;

                // Footprint points are the MPRL points standing inside this object's own box.
                var points = new List<Vector2>();
                foreach (Vector3 p in mprl)
                {
                    if (p.X >= min.X && p.X <= max.X && p.Y >= min.Y && p.Y <= max.Y)
                        points.Add(new Vector2(p.X, p.Y));
                }

                if (points.Count < MinPoints)
                    continue;

                Vector2 centre = Vector2.Zero;
                foreach (Vector2 p in points)
                    centre += p;
                centre /= points.Count;

                for (int i = 0; i < points.Count; i++)
                    points[i] -= centre;

                objectsWithPoints++;
                if (!instances.TryGetValue(asset, out List<Instance>? list))
                {
                    list = [];
                    instances[asset] = list;
                }

                list.Add(new Instance(points, truth.Rotation));
            }
        }

        // Score each rotation component, plus a mismatched-asset control.
        var errX = new Stat(); var errY = new Stat(); var errZ = new Stat();
        var control = new Stat();

        // Detector power. If the true angle differences are mostly ZERO and the fitter still returns
        // scattered angles, the fitter has failed and the hypothesis was never tested. Both
        // distributions are reported so a null cannot be mistaken for an answer.
        var trueDelta = new Stat();
        var fittedAngles = new Stat();

        // The only pairs that can test rotation recovery are those whose rotations actually DIFFER.
        // Scored separately, because a corpus that places every copy of an asset at the same angle
        // offers no signal and its overall median says nothing either way.
        var rotatedOnly = new Stat();
        long rotatedPairs = 0;
        long pointCountSum = 0, instancesCounted = 0;
        long pairs = 0;
        var rng = new Random(11);
        List<string> assets = [.. instances.Keys];

        foreach ((string asset, List<Instance> list) in instances)
        {
            for (int a = 0; a < list.Count; a++)
            {
                for (int b = a + 1; b < list.Count; b++)
                {
                    if (pairs >= 4000)
                        break;

                    float fitted = FitAngleDegrees(list[a].Points, list[b].Points);
                    pairs++;
                    fittedAngles.Add(fitted <= 180f ? fitted : 360f - fitted);
                    trueDelta.Add(AngleError(0f, list[a].Rotation.Y - list[b].Rotation.Y));
                    pointCountSum += list[a].Points.Count + list[b].Points.Count;
                    instancesCounted += 2;

                    errX.Add(AngleError(fitted, list[a].Rotation.X - list[b].Rotation.X));
                    errY.Add(AngleError(fitted, list[a].Rotation.Y - list[b].Rotation.Y));
                    errZ.Add(AngleError(fitted, list[a].Rotation.Z - list[b].Rotation.Z));

                    double delta = AngleError(0f, list[a].Rotation.Y - list[b].Rotation.Y);
                    if (delta > 5.0)
                    {
                        rotatedPairs++;
                        rotatedOnly.Add(AngleError(fitted, list[a].Rotation.Y - list[b].Rotation.Y));
                    }

                    // Control: fit this instance against an unrelated asset's instance and score it
                    // against the same expected difference. A method that "recovers an angle" for
                    // anything has recovered nothing.
                    string other = assets[rng.Next(assets.Count)];
                    if (!other.Equals(asset, StringComparison.OrdinalIgnoreCase) && instances[other].Count > 0)
                    {
                        Instance far = instances[other][rng.Next(instances[other].Count)];
                        float bogus = FitAngleDegrees(list[a].Points, far.Points);
                        control.Add(AngleError(bogus, list[a].Rotation.Y - list[b].Rotation.Y));
                    }
                }
            }
        }

        return new Pm4RotationFitReport(
            resolved, files, objectsWithPoints,
            instances.Count(static kv => kv.Value.Count > 1),
            pairs,
            errX.ToResult("fitted angle vs MODF Rotation.X difference"),
            errY.ToResult("fitted angle vs MODF Rotation.Y difference"),
            errZ.ToResult("fitted angle vs MODF Rotation.Z difference"),
            control.ToResult("CONTROL fit against an unrelated asset"),
            trueDelta.ToResult("TRUE |Rotation.Y difference| between the pair"),
            fittedAngles.ToResult("FITTED angle magnitude"),
            instancesCounted == 0 ? 0 : (double)pointCountSum / instancesCounted,
            rotatedPairs,
            rotatedOnly.ToResult("fitted vs true, PAIRS THAT ACTUALLY DIFFER"));
    }

    private readonly record struct Instance(List<Vector2> Points, Vector3 Rotation);

    /// <summary>
    /// The rotation, in degrees, that best carries cloud <paramref name="b"/> onto <paramref name="a"/>.
    /// </summary>
    private static float FitAngleDegrees(List<Vector2> a, List<Vector2> b)
    {
        float best = 0;
        double bestScore = double.MaxValue;

        for (int deg = 0; deg < 360; deg++)
        {
            float rad = deg * MathF.PI / 180f;
            float cos = MathF.Cos(rad), sin = MathF.Sin(rad);

            double sum = 0;
            foreach (Vector2 p in a)
            {
                float nearest = float.MaxValue;
                foreach (Vector2 q in b)
                {
                    float qx = (q.X * cos) - (q.Y * sin);
                    float qy = (q.X * sin) + (q.Y * cos);
                    float dx = p.X - qx, dy = p.Y - qy;
                    float d = (dx * dx) + (dy * dy);
                    if (d < nearest)
                        nearest = d;
                }

                sum += Math.Sqrt(nearest);
            }

            if (sum < bestScore)
            {
                bestScore = sum;
                best = deg;
            }
        }

        return best;
    }

    /// <summary>
    /// Smallest absolute difference between two angles in degrees, respecting wraparound.
    /// </summary>
    private static double AngleError(float fittedDegrees, float expectedDegrees)
    {
        double d = Math.Abs(fittedDegrees - expectedDegrees) % 360.0;
        return d > 180.0 ? 360.0 - d : d;
    }

    private static Dictionary<uint, (Vector3 Min, Vector3 Max)> ExtractObjects(Pm4KnownChunkSet chunks)
    {
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

        var result = new Dictionary<uint, (Vector3, Vector3)>();
        foreach ((uint key, Vector3 min) in lo)
            result[key] = (min, hi[key]);

        return result;
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
                (double)_v.Count(static x => x < 10.0) / _v.Count);
        }
    }
}

internal sealed record Pm4RotationFitReport(
    string Pm4Directory,
    int Files,
    int ObjectsWithFootprintPoints,
    int AssetsWithMultipleInstances,
    long PairsFitted,
    Pm4ErrorStat AgainstRotationX,
    Pm4ErrorStat AgainstRotationY,
    Pm4ErrorStat AgainstRotationZ,
    Pm4ErrorStat Control,
    Pm4ErrorStat TrueDelta,
    Pm4ErrorStat FittedAngles,
    double MeanPointsPerInstance,
    long RotatedPairs,
    Pm4ErrorStat RotatedOnly);
