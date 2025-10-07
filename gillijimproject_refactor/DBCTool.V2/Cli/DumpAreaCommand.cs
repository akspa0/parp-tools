using System.Text;
using DBCD;
using DBCD.Providers;
using static DBCTool.V2.IO.DbdcHelper;

namespace DBCTool.V2.Cli;

public sealed class DumpAreaCommand
{
    public int Run(string dbdDir, string outBase, string localeStr, List<(string build, string dir)> inputs)
    {
        // Resolve inputs for source (0.5.3/0.5.5/0.6.0) and 3.3.5
        string? dir053 = null, dir055 = null, dir060 = null, dir335 = null;
        foreach (var (build, dir) in inputs)
        {
            var alias = ResolveAliasOrInfer(build, dir);
            if (alias == "0.5.3") dir053 = Normalize(dir);
            else if (alias == "0.5.5") dir055 = Normalize(dir);
            else if (alias == "0.6.0") dir060 = Normalize(dir);
            else if (alias == "3.3.5") dir335 = Normalize(dir);
        }
        if (string.IsNullOrEmpty(dir335))
        {
            Console.Error.WriteLine("[dump-area] ERROR: 3.3.5 input is required.");
            return 2;
        }
        var srcDir = dir053 ?? dir055 ?? dir060;
        var srcAlias = dir053 != null ? "0.5.3" : (dir055 != null ? "0.5.5" : (dir060 != null ? "0.6.0" : ""));
        if (string.IsNullOrEmpty(srcDir))
        {
            Console.Error.WriteLine("[dump-area] ERROR: One of 0.5.3/0.5.5/0.6.0 inputs is required.");
            return 2;
        }

        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var locale = ParseLocale(localeStr);

        var storSrc_Area = LoadTable("AreaTable", CanonicalizeBuild(srcAlias), srcDir!, dbdProvider, locale);
        var storTgt_Area = LoadTable("AreaTable", CanonicalizeBuild("3.3.5"), dir335!, dbdProvider, locale);

        var rawDir = Path.Combine(outBase, srcAlias, "raw");
        Directory.CreateDirectory(rawDir);

        DumpAreaTable(Path.Combine(rawDir, $"AreaTable_{srcAlias.Replace('.', '_')}.csv"), storSrc_Area, srcAlias);
        DumpAreaTable(Path.Combine(rawDir, "AreaTable_3_3_5.csv"), storTgt_Area, "3.3.5");

        Console.WriteLine("[dump-area] Wrote raw AreaTable CSVs under raw/");
        return 0;
    }

    private static void DumpAreaTable(string path, IDBCDStorage storage, string alias)
    {
        // Try to detect relevant columns across versions
        string idCol = DetectIdColumn(storage);
        string nameCol = DetectColumn(storage, "AreaName_lang", "AreaName", "Name");
        string parentCol = (alias == "0.5.3" || alias == "0.5.5") ? "ParentAreaNum" : "ParentAreaID";
        string keyCol = (alias == "0.5.3" || alias == "0.5.5") ? "AreaNumber" : (string.IsNullOrWhiteSpace(idCol) ? "ID" : idCol);

        var sb = new StringBuilder();
        sb.AppendLine("row_key,id_or_area,parent,continentId,areaName_lang,areaName,name");
        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int idOrArea = SafeField<int>(row, keyCol);
            int parent = SafeField<int>(row, parentCol);
            int cont = SafeField<int>(row, "ContinentID");
            string nameLang = SafeField<string>(row, "AreaName_lang");
            string areaName = SafeField<string>(row, "AreaName");
            string name = SafeField<string>(row, "Name");
            sb.Append(key.ToString(System.Globalization.CultureInfo.InvariantCulture));
            sb.Append(',');
            sb.Append(idOrArea.ToString(System.Globalization.CultureInfo.InvariantCulture));
            sb.Append(',');
            sb.Append(parent.ToString(System.Globalization.CultureInfo.InvariantCulture));
            sb.Append(',');
            sb.Append(cont.ToString(System.Globalization.CultureInfo.InvariantCulture));
            sb.Append(',');
            sb.Append(Csv(nameLang ?? string.Empty));
            sb.Append(',');
            sb.Append(Csv(areaName ?? string.Empty));
            sb.Append(',');
            sb.AppendLine(Csv(name ?? string.Empty));
        }
        File.WriteAllText(path, sb.ToString(), new UTF8Encoding(true));
    }

    private static string Normalize(string p) => Path.GetFullPath(p);

    private static string ResolveAliasOrInfer(string build, string dir)
    {
        if (!string.IsNullOrWhiteSpace(build)) return build.Trim();
        var tok = (dir ?? string.Empty).ToLowerInvariant();
        if (tok.Contains("0.5.3")) return "0.5.3";
        if (tok.Contains("0.5.5")) return "0.5.5";
        if (tok.Contains("0.6.0")) return "0.6.0";
        if (tok.Contains("3.3.5")) return "3.3.5";
        return build?.Trim() ?? string.Empty;
    }

    private static DBCD.Locale ParseLocale(string s)
    {
        if (Enum.TryParse<DBCD.Locale>(s, ignoreCase: true, out var loc)) return loc;
        return DBCD.Locale.EnUS;
    }

    private static string CanonicalizeBuild(string alias)
    {
        return alias switch
        {
            "0.5.3" => "0.5.3.3368",
            "0.5.5" => "0.5.5.3494",
            "0.6.0" => "0.6.0.3592",
            "3.3.5" => "3.3.5.12340",
            _ => alias
        };
    }
}
