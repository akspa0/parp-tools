using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4MshdWordStats(
    string Word, long Files, long DistinctValues, long ZeroValues, long MaxValues,
    long MinValue, long MaxValue, double Mean, double Median,
    IReadOnlyList<Pm4MshdValueFrequency> TopValues);

public sealed record Pm4MshdValueFrequency(uint Value, long Count);

public sealed record Pm4MshdWordCorrelation(
    string Word, string Metric, double Pearson, long Samples);

public sealed record Pm4MshdWordFileRow(
    string File, int TileX, int TileY, uint Version, long Bytes,
    uint Field00, uint Field04, uint Field08, uint Field0C, uint Field10, uint Field14, uint Field18, uint Field1C,
    int Surfaces, int Links, int Msvt, int Msvi, int Mspv, int Mspi, int Mscn, int Mprl, int Mprr,
    int Mdbh, int Mdbi, int Mdbf, int Mdos, int Mdsf);

public sealed record Pm4MshdWordReport(
    int Files, IReadOnlyList<Pm4MshdWordStats> Words,
    IReadOnlyList<Pm4MshdWordCorrelation> Correlations, IReadOnlyList<Pm4MshdWordFileRow> Rows);

/// <summary>
/// Correlates each MSHD word with file-level counts and tile coordinates. Correlation is not
/// semantics; a strong count match is a candidate, not a decoded field name.
/// </summary>
public static class Pm4MshdProbe
{
    private static readonly string[] WordNames = ["Field00", "Field04", "Field08", "Field0C", "Field10", "Field14", "Field18", "Field1C"];
    private static readonly string[] Metrics =
    [
        "Surfaces", "Links", "Msvt", "Msvi", "Mspv", "Mspi", "Mscn", "Mprl", "Mprr",
        "Mdbh", "Mdbi", "Mdbf", "Mdos", "Mdsf", "TileX", "TileY", "Bytes",
    ];

    public static Pm4MshdWordReport Analyze(
        IReadOnlyList<Pm4MapAuditInput> input,
        IReadOnlyDictionary<string, long>? fileSizes = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var rows = new List<Pm4MshdWordFileRow>();
        foreach (var f in input.OrderBy(x => x.FileId, StringComparer.Ordinal))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(f.FileId, out int x, out int y))
                throw new ArgumentException("File IDs must retain tile coordinates.");
            var c = f.Document.KnownChunks;
            var h = c.Mshd;
            long bytes = fileSizes is not null && fileSizes.TryGetValue(f.FileId, out long size) ? size : 0;
            rows.Add(new Pm4MshdWordFileRow(f.FileId, x, y, f.Document.Version, bytes,
                h?.Field00 ?? 0, h?.Field04 ?? 0, h?.Field08 ?? 0, h?.Field0C ?? 0,
                h?.Field10 ?? 0, h?.Field14 ?? 0, h?.Field18 ?? 0, h?.Field1C ?? 0,
                c.Msur.Count, c.Mslk.Count, c.Msvt.Count, c.Msvi.Count, c.Mspv.Count, c.Mspi.Count,
                c.Mscn.Count, c.Mprl.Count, c.Mprr.Count,
                c.Mdbh is null ? 0 : 1, c.Mdbi.Count, c.Mdbf.Count, c.Mdos.Count, c.Mdsf.Count));
        }

        var words = new List<Pm4MshdWordStats>();
        for (int w = 0; w < WordNames.Length; w++)
        {
            var values = rows.Select(r => Word(r, w)).ToArray();
            var sorted = values.OrderBy(v => v).ToArray();
            var frequencies = values.GroupBy(v => v).OrderByDescending(g => g.Count()).ThenBy(g => g.Key)
                .Take(8).Select(g => new Pm4MshdValueFrequency(g.Key, g.Count())).ToArray();
            words.Add(new Pm4MshdWordStats(WordNames[w], rows.Count, values.Distinct().Count(),
                values.Count(v => v == 0), values.Count(v => v == uint.MaxValue),
                sorted.Length == 0 ? 0 : sorted[0], sorted.Length == 0 ? 0 : sorted[^1],
                sorted.Length == 0 ? 0 : sorted.Average(v => (double)v),
                sorted.Length == 0 ? 0 : sorted[sorted.Length / 2], frequencies));
        }

        var correlations = new List<Pm4MshdWordCorrelation>();
        for (int w = 0; w < WordNames.Length; w++)
        {
            foreach (string metric in Metrics)
            {
                var xs = rows.Select(r => (double)Word(r, w)).ToArray();
                var ys = rows.Select(r => (double)Metric(r, metric)).ToArray();
                double? p = Pearson(xs, ys);
                if (p.HasValue) correlations.Add(new Pm4MshdWordCorrelation(WordNames[w], metric, p.Value, rows.Count));
            }
        }

        return new Pm4MshdWordReport(rows.Count, words,
            correlations.OrderByDescending(c => Math.Abs(c.Pearson)).ThenBy(c => c.Word).ThenBy(c => c.Metric).ToArray(),
            rows);
    }

    private static uint Word(Pm4MshdWordFileRow r, int index) => index switch
    {
        0 => r.Field00, 1 => r.Field04, 2 => r.Field08, 3 => r.Field0C,
        4 => r.Field10, 5 => r.Field14, 6 => r.Field18, 7 => r.Field1C, _ => 0,
    };

    private static int Metric(Pm4MshdWordFileRow r, string name) => name switch
    {
        "Surfaces" => r.Surfaces, "Links" => r.Links, "Msvt" => r.Msvt, "Msvi" => r.Msvi,
        "Mspv" => r.Mspv, "Mspi" => r.Mspi, "Mscn" => r.Mscn, "Mprl" => r.Mprl, "Mprr" => r.Mprr,
        "Mdbh" => r.Mdbh, "Mdbi" => r.Mdbi, "Mdbf" => r.Mdbf, "Mdos" => r.Mdos, "Mdsf" => r.Mdsf,
        "TileX" => r.TileX, "TileY" => r.TileY, "Bytes" => (int)Math.Min(r.Bytes, int.MaxValue), _ => 0,
    };

    private static double? Pearson(double[] x, double[] y)
    {
        if (x.Length < 2 || x.Length != y.Length) return null;
        double mx = x.Average(), my = y.Average();
        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double dx = x[i] - mx, dy = y[i] - my;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }
        if (sxx <= 1e-12 || syy <= 1e-12) return null;
        return sxy / Math.Sqrt(sxx * syy);
    }
}
