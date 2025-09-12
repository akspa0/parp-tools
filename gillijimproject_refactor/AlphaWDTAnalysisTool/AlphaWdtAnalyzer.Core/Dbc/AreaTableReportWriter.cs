using System;
using System.Globalization;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core.Dbc;

public static class AreaTableReportWriter
{
    public static void Write(string outDir, AreaCrosswalk cross)
    {
        Directory.CreateDirectory(outDir);

        var matches = Path.Combine(outDir, "area_matches.csv");
        using (var sw = new StreamWriter(matches))
        {
            sw.WriteLine("alpha_id,alpha_name,lk_id,lk_name,match_type");
            foreach (var m in cross.Matches.OrderBy(m => m.Alpha.Id))
            {
                sw.WriteLine(string.Join(',',
                    m.Alpha.Id.ToString(CultureInfo.InvariantCulture),
                    Csv(m.Alpha.Name),
                    m.Lk.Id.ToString(CultureInfo.InvariantCulture),
                    Csv(m.Lk.Name),
                    Csv(m.Match)));
            }
        }

        var ua = Path.Combine(outDir, "area_unmatched_alpha.csv");
        using (var sw = new StreamWriter(ua))
        {
            sw.WriteLine("alpha_id,alpha_name");
            foreach (var a in cross.UnmatchedAlpha.OrderBy(a => a.Id))
            {
                sw.WriteLine(string.Join(',', a.Id.ToString(CultureInfo.InvariantCulture), Csv(a.Name)));
            }
        }

        var ul = Path.Combine(outDir, "area_unmatched_lk.csv");
        using (var sw = new StreamWriter(ul))
        {
            sw.WriteLine("lk_id,lk_name");
            foreach (var l in cross.UnmatchedLk.OrderBy(l => l.Id))
            {
                sw.WriteLine(string.Join(',', l.Id.ToString(CultureInfo.InvariantCulture), Csv(l.Name)));
            }
        }
    }

    private static string Csv(string s)
    {
        if (s.Contains('"') || s.Contains(','))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }
}
