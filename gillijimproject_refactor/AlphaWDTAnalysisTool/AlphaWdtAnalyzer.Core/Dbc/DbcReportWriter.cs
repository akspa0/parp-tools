using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core.Dbc;

public static class DbcReportWriter
{
    public static void Write(string outDir, DbcScanResult scan)
    {
        Directory.CreateDirectory(outDir);

        foreach (var tbl in scan.Tables)
        {
            var csv = Path.Combine(outDir, tbl.Name + ".csv");
            using var sw = new StreamWriter(csv);
            // header: numeric fields then guessed string fields
            var fCount = tbl.Table.FieldCount;
            var numHeaders = Enumerable.Range(0, fCount).Select(i => $"f{i}");
            var strHeaders = Enumerable.Range(0, fCount).Select(i => $"s{i}");
            sw.WriteLine(string.Join(',', numHeaders.Concat(strHeaders)));

            foreach (var row in tbl.Table.Rows)
            {
                var nums = row.Fields.Select(v => v.ToString(CultureInfo.InvariantCulture));
                var strs = row.GuessedStrings.Select(s => EscapeCsv(s ?? string.Empty));
                sw.WriteLine(string.Join(',', nums.Concat(strs)));
            }
        }

        // Optional specialized summaries (best-effort)
        WriteSimpleSummary(outDir, scan, "Light");
        WriteSimpleSummary(outDir, scan, "LiquidType");
        WriteSimpleSummary(outDir, scan, "LiquidMaterial");
    }

    private static void WriteSimpleSummary(string outDir, DbcScanResult scan, string tableName)
    {
        var tbl = scan.Tables.FirstOrDefault(t => t.Name.Equals(tableName, StringComparison.OrdinalIgnoreCase));
        if (tbl is null) return;
        var csv = Path.Combine(outDir, tableName + "_summary.csv");
        using var sw = new StreamWriter(csv);
        // Write first few fields and any non-empty guessed strings to give quick insight
        sw.WriteLine("id,f0,f1,f2,f3,f4,guessed_s0,guessed_s1,guessed_s2");
        int id = 0;
        foreach (var row in tbl.Table.Rows)
        {
            var f0 = row.Fields.Length > 0 ? row.Fields[0].ToString(CultureInfo.InvariantCulture) : string.Empty;
            var f1 = row.Fields.Length > 1 ? row.Fields[1].ToString(CultureInfo.InvariantCulture) : string.Empty;
            var f2 = row.Fields.Length > 2 ? row.Fields[2].ToString(CultureInfo.InvariantCulture) : string.Empty;
            var f3 = row.Fields.Length > 3 ? row.Fields[3].ToString(CultureInfo.InvariantCulture) : string.Empty;
            var f4 = row.Fields.Length > 4 ? row.Fields[4].ToString(CultureInfo.InvariantCulture) : string.Empty;
            var s0 = row.GuessedStrings.Length > 0 ? EscapeCsv(row.GuessedStrings[0] ?? string.Empty) : string.Empty;
            var s1 = row.GuessedStrings.Length > 1 ? EscapeCsv(row.GuessedStrings[1] ?? string.Empty) : string.Empty;
            var s2 = row.GuessedStrings.Length > 2 ? EscapeCsv(row.GuessedStrings[2] ?? string.Empty) : string.Empty;
            sw.WriteLine(string.Join(',', id, f0, f1, f2, f3, f4, s0, s1, s2));
            id++;
        }
    }

    private static string EscapeCsv(string s)
    {
        if (s.Contains('"') || s.Contains(','))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }
}
