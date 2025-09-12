using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

#if USE_DBCD
using DBCD;
using DBCD.Providers;
#endif

namespace AlphaWdtAnalyzer.Core.Dbc;

public static class AreaTableDbcExporter
{
    public static void ExportAlphaAndLkToCsv(string alphaDbcPath, string lkDbcPath, string dbdDefinitionsDir, string outDir)
    {
#if USE_DBCD
        Directory.CreateDirectory(outDir);
        var alphaCsv = Path.Combine(outDir, "AreaTable_Alpha.csv");
        var lkCsv = Path.Combine(outDir, "AreaTable_335.csv");

        ExportOne(alphaDbcPath, dbdDefinitionsDir, TryAlphaBuilds(), alphaCsv);
        ExportOne(lkDbcPath, dbdDefinitionsDir, new[] { "3.3.5.12340" }, lkCsv);
#else
        Console.Error.WriteLine("[AreaTableDbcExporter] DBCD not available in this build. Skipping DBC export.");
#endif
    }

#if USE_DBCD
    private static void ExportOne(string dbcPath, string dbdDefinitionsDir, IEnumerable<string> buildCandidates, string csvPath)
    {
        var dir = Path.GetDirectoryName(dbcPath)!;
        var table = Path.GetFileNameWithoutExtension(dbcPath)!;

        var dbcProvider = new FilesystemDBCProvider(dir);
        var defDir = Directory.Exists(Path.Combine(dbdDefinitionsDir, "definitions"))
            ? Path.Combine(dbdDefinitionsDir, "definitions")
            : dbdDefinitionsDir;
        var dbdProvider = new FilesystemDBDProvider(defDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        Exception? last = null;
        foreach (var build in buildCandidates)
        {
            try
            {
                var storage = dbcd.Load(table, build);
                using var sw = new StreamWriter(csvPath);
                sw.WriteLine("id,name");
                foreach (KeyValuePair<int, DBCDRow> entry in (IEnumerable<KeyValuePair<int, DBCDRow>>)storage)
                {
                    int id = entry.Key;
                    DBCDRow row = entry.Value;
                    var name = TryGetName(row) ?? string.Empty;
                    sw.WriteLine(string.Join(',', id.ToString(CultureInfo.InvariantCulture), Csv(name)));
                }
                return;
            }
            catch (Exception ex)
            {
                last = ex;
            }
        }
        throw last ?? new InvalidOperationException($"Failed to export {table} via DBCD");
    }

    private static IEnumerable<string> TryAlphaBuilds()
    {
        yield return "0.5.5.3494";
        yield return "0.5.3.3368";
    }

    private static string? TryGetName(DBCDRow row)
    {
        string[] fields = new[] { "AreaName_lang", "Name_lang", "AreaName_Lang", "Name_Lang", "AreaName", "Name" };
        foreach (var f in fields)
        {
            try
            {
                object val = row[f];
                if (val is string s && !string.IsNullOrWhiteSpace(s)) return s;
            }
            catch { }
        }
        return null;
    }
#endif

    private static string Csv(string s)
    {
        if (s.Contains('"') || s.Contains(','))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }
}
