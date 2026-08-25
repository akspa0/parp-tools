using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Measures how much of an <c>MODF</c> row can be reconstructed from a PM4 alone.
/// </summary>
/// <remarks>
/// The join already exists: <c>MSUR._0x1C</c> is bit-identical to <c>MODF.Position.Z</c>, so a PM4
/// object that carries a placement height already names a specific placement row. This asks what else
/// of that row falls out of the geometry - the other two position components, and the bounding box.
///
/// <para>Which PM4 axis answers to which MODF axis is not assumed. All four pairings are measured and
/// all four reported, so the fit can be seen rather than asserted; the axes are known to be swapped
/// somewhere in this format family and picking wrong would produce a confident, wrong exporter.</para>
///
/// <para><b>Control.</b> Every measurement is repeated against a SHUFFLED pairing - the same tile's
/// placements dealt to the wrong objects. Objects in one tile all sit within 533 units of each other,
/// so even nonsense pairings score a bounded error; the real pairing has to beat the shuffle, not
/// merely look small.</para>
/// </remarks>
internal static class Pm4ModfRecoverySupport
{
    public static Pm4ModfRecoveryReport Analyze(string pm4Directory, string? adtDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);
        string adtRoot = string.IsNullOrWhiteSpace(adtDirectory) ? resolved : adtDirectory;

        int filesPaired = 0, filesNoAdt = 0;
        long objectsTotal = 0, objectsMatched = 0, modfRowsTotal = 0;

        var xx = new ErrorAccumulator(); var xy = new ErrorAccumulator();
        var yx = new ErrorAccumulator(); var yy = new ErrorAccumulator();
        var ctlXX = new ErrorAccumulator(); var ctlYY = new ErrorAccumulator();

        var boundsMin = new ErrorAccumulator(); var boundsMax = new ErrorAccumulator();
        var ctlBoundsMin = new ErrorAccumulator();

        // Per axis, because a 3D distance hides the thing worth seeing: the vertical extent can agree
        // to a hundredth of a unit while the horizontal one is metres out, and one summed number
        // reports that as a uniform miss.
        var bMinX = new ErrorAccumulator(); var bMinY = new ErrorAccumulator(); var bMinZ = new ErrorAccumulator();
        var bMaxX = new ErrorAccumulator(); var bMaxY = new ErrorAccumulator(); var bMaxZ = new ErrorAccumulator();
        var ctlMinZ = new ErrorAccumulator();
        long boundsCompared = 0, boundsWithin1 = 0;

        var samples = new List<Pm4ModfRecoverySample>();

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtRoot);
            if (adtPath is null)
            {
                filesNoAdt++;
                continue;
            }

            AdtPlacementCatalog catalog;
            try
            {
                catalog = AdtPlacementReader.Read(adtPath);
            }
            catch
            {
                filesNoAdt++;
                continue;
            }

            IReadOnlyList<AdtWorldModelPlacement> modf = catalog.WorldModelPlacements;
            if (modf.Count == 0)
                continue;

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
                continue;

            filesPaired++;
            modfRowsTotal += modf.Count;

            // One PM4 object per distinct placement height, with the box its geometry occupies.
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

            objectsTotal += lo.Count;

            // Join on the placement height, which is bit-identical between the two files.
            var byHeight = new Dictionary<uint, AdtWorldModelPlacement>();
            foreach (AdtWorldModelPlacement row in modf)
                byHeight[BitConverter.SingleToUInt32Bits(row.Position.Z)] = row;

            List<uint> keys = [.. lo.Keys];
            int shuffleSeed = 0;

            foreach (uint key in keys)
            {
                if (!byHeight.TryGetValue(key, out AdtWorldModelPlacement row))
                    continue;

                objectsMatched++;

                Vector3 min = lo[key], max = hi[key];
                Vector3 centre = (min + max) * 0.5f;

                xx.Add(centre.X - row.Position.X);
                xy.Add(centre.X - row.Position.Y);
                yx.Add(centre.Y - row.Position.X);
                yy.Add(centre.Y - row.Position.Y);

                // Control: this object against a DIFFERENT placement row in the same tile.
                AdtWorldModelPlacement other = modf[(shuffleSeed++ + 1) % modf.Count];
                ctlXX.Add(centre.X - other.Position.X);
                ctlYY.Add(centre.Y - other.Position.Y);

                // MODF carries its own bounding box. If PM4's geometry box is that same box, an
                // exporter gets the extents for free instead of having to derive them from a model.
                boundsCompared++;
                float dMin = Vector3.Distance(min, row.BoundsMin);
                float dMax = Vector3.Distance(max, row.BoundsMax);
                boundsMin.Add(dMin);
                boundsMax.Add(dMax);
                ctlBoundsMin.Add(Vector3.Distance(min, other.BoundsMin));
                bMinX.Add(min.X - row.BoundsMin.X); bMinY.Add(min.Y - row.BoundsMin.Y); bMinZ.Add(min.Z - row.BoundsMin.Z);
                bMaxX.Add(max.X - row.BoundsMax.X); bMaxY.Add(max.Y - row.BoundsMax.Y); bMaxZ.Add(max.Z - row.BoundsMax.Z);
                ctlMinZ.Add(min.Z - other.BoundsMin.Z);
                if (dMin < 1f && dMax < 1f)
                    boundsWithin1++;

                if (samples.Count < 8)
                {
                    samples.Add(new Pm4ModfRecoverySample(
                        Path.GetFileNameWithoutExtension(pm4Path),
                        row.ModelPath,
                        centre,
                        row.Position,
                        min, max,
                        row.BoundsMin, row.BoundsMax));
                }
            }
        }

        return new Pm4ModfRecoveryReport(
            resolved, adtRoot, filesPaired, filesNoAdt,
            objectsTotal, objectsMatched, modfRowsTotal,
            xx.ToResult("pm4.X vs MODF.X"), xy.ToResult("pm4.X vs MODF.Y"),
            yx.ToResult("pm4.Y vs MODF.X"), yy.ToResult("pm4.Y vs MODF.Y"),
            ctlXX.ToResult("CONTROL pm4.X vs wrong row .X"), ctlYY.ToResult("CONTROL pm4.Y vs wrong row .Y"),
            boundsMin.ToResult("|pm4 min - MODF BoundsMin|"),
            boundsMax.ToResult("|pm4 max - MODF BoundsMax|"),
            ctlBoundsMin.ToResult("CONTROL |pm4 min - wrong row BoundsMin|"),
            boundsCompared,
            boundsCompared == 0 ? 0 : (double)boundsWithin1 / boundsCompared,
            [
                bMinX.ToResult("BoundsMin.X"), bMinY.ToResult("BoundsMin.Y"), bMinZ.ToResult("BoundsMin.Z"),
                bMaxX.ToResult("BoundsMax.X"), bMaxY.ToResult("BoundsMax.Y"), bMaxZ.ToResult("BoundsMax.Z"),
                ctlMinZ.ToResult("CONTROL BoundsMin.Z vs wrong row"),
            ],
            samples);
    }

    private sealed class ErrorAccumulator
    {
        private readonly List<double> _values = [];

        public void Add(double v) => _values.Add(Math.Abs(v));

        public Pm4ErrorStat ToResult(string name)
        {
            if (_values.Count == 0)
                return new Pm4ErrorStat(name, 0, 0, 0, 0, 0);

            _values.Sort();
            double median = _values[_values.Count / 2];
            double p90 = _values[(int)(_values.Count * 0.90)];
            int within1 = _values.Count(static v => v < 1.0);
            return new Pm4ErrorStat(name, _values.Count, median, p90, _values[^1], (double)within1 / _values.Count);
        }
    }
}

internal sealed record Pm4ErrorStat(
    string Name,
    int Count,
    double MedianAbsError,
    double P90AbsError,
    double MaxAbsError,
    double FractionWithin1);

internal sealed record Pm4ModfRecoverySample(
    string File,
    string ModelPath,
    Vector3 Pm4Centre,
    Vector3 ModfPosition,
    Vector3 Pm4Min,
    Vector3 Pm4Max,
    Vector3 ModfBoundsMin,
    Vector3 ModfBoundsMax);

internal sealed record Pm4ModfRecoveryReport(
    string Pm4Directory,
    string AdtDirectory,
    int FilesPaired,
    int FilesNoAdt,
    long Pm4ObjectsTotal,
    long Pm4ObjectsMatched,
    long ModfRowsTotal,
    Pm4ErrorStat XvsX,
    Pm4ErrorStat XvsY,
    Pm4ErrorStat YvsX,
    Pm4ErrorStat YvsY,
    Pm4ErrorStat ControlX,
    Pm4ErrorStat ControlY,
    Pm4ErrorStat BoundsMin,
    Pm4ErrorStat BoundsMax,
    Pm4ErrorStat ControlBoundsMin,
    long BoundsCompared,
    double BoundsWithin1Fraction,
    IReadOnlyList<Pm4ErrorStat> BoundsPerAxis,
    IReadOnlyList<Pm4ModfRecoverySample> Samples);
