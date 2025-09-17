using System.Globalization;
using DBCD;
using DBCTool.V2.Cli;
using System.Text.Json;

namespace DBCTool.V2.Cli;

internal static class Program
{
    private const string DefaultOutBase = "dbctool_outputs";
    private const string DefaultDbdDir = "lib/WoWDBDefs/definitions";
    private const string DefaultLocale = "enUS";

    private static int Main(string[] args)
    {
        var opts = ParseArgs(args);

        // Merge optional config file defaults
        var cfg = LoadConfig();
        if (cfg is not null)
        {
            if (string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(cfg.DbdDir))
                opts = opts with { DbdDir = cfg.DbdDir };
            if (string.IsNullOrWhiteSpace(opts.OutRoot) && !string.IsNullOrWhiteSpace(cfg.OutRoot))
                opts = opts with { OutRoot = cfg.OutRoot };
            if (string.IsNullOrWhiteSpace(opts.Locale) && !string.IsNullOrWhiteSpace(cfg.Locale))
                opts = opts with { Locale = cfg.Locale };
            if (opts.Inputs.Count == 0 && cfg.Inputs?.Count > 0)
                opts = opts with { Inputs = ParseInputs(cfg.Inputs) };
            if (!opts.CompareAreaV2 && !opts.DumpArea && !opts.QuickCompare && !opts.QuickDump && !string.IsNullOrWhiteSpace(cfg.DefaultCommand))
            {
                var dc = cfg.DefaultCommand!.Trim().ToLowerInvariant();
                if (dc is "compare" or "compare-area" or "compare-area-v2") opts = opts with { CompareAreaV2 = true };
                else if (dc is "dump" or "dump-area") opts = opts with { DumpArea = true };
            }
        }

        // Session folder: dbctool_outputs/session_YYYYMMDD_HHmmss
        var session = $"session_{DateTime.Now:yyyyMMdd_HHmmss}";
        var outBase = Path.Combine(opts.OutRoot ?? DefaultOutBase, session);
        Directory.CreateDirectory(outBase);

        // If only a source alias/flag is provided, default to quick compare (target is always 3.3.5)
        if (!opts.QuickCompare && !opts.CompareAreaV2 && !opts.DumpArea && !opts.QuickDump &&
            (!string.IsNullOrWhiteSpace(opts.SrcAlias) || opts.S53 || opts.S55 || opts.S60))
        {
            opts = opts with { QuickCompare = true };
        }

        // Quick-run shorthands auto-populate missing args
        if (opts.QuickCompare || opts.QuickDump)
        {
            AutoPopulateDefaults(ref opts);
        }

        if (opts.CompareAreaV2 || opts.QuickCompare)
        {
            var cmd = new CompareAreaV2Command();
            return cmd.Run(
                dbdDir: opts.DbdDir ?? TryResolveDbdDir() ?? DefaultDbdDir,
                outBase: outBase,
                localeStr: opts.Locale ?? DefaultLocale,
                inputs: opts.Inputs
            );
        }
        if (opts.DumpArea || opts.QuickDump)
        {
            var cmd = new DumpAreaCommand();
            return cmd.Run(
                dbdDir: opts.DbdDir ?? TryResolveDbdDir() ?? DefaultDbdDir,
                outBase: outBase,
                localeStr: opts.Locale ?? DefaultLocale,
                inputs: opts.Inputs
            );
        }

        PrintHelp();
        return 0;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("DBCTool.V2 - Alpha Area Decode V2 (separate project)");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- \\");
        Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out dbctool_outputs --locale enUS \\");
        Console.WriteLine("    --compare-area-v2 --input 0.5.3=path/to/0.5.3/DBFilesClient --input 3.3.5=path/to/3.3.5/DBFilesClient");
        Console.WriteLine();
        Console.WriteLine("Dump raw AreaTable CSVs for src and 3.3.5:");
        Console.WriteLine("  dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- \\");
        Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out dbctool_outputs --locale enUS \\");
        Console.WriteLine("    --dump-area --input 0.5.3=path/to/0.5.3/DBFilesClient --input 3.3.5=path/to/3.3.5/DBFilesClient");
        Console.WriteLine();
        Console.WriteLine("Quick shorthands (auto-detect paths to minimize typing):");
        Console.WriteLine("  dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --qc");
        Console.WriteLine("  dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --qd");
        Console.WriteLine("(Looks for ../lib/WoWDBDefs/definitions and ../test_data/... by default)");
        Console.WriteLine();
        Console.WriteLine("Select source version (defaults to compare to 3.3.5):");
        Console.WriteLine("  dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --s60");
        Console.WriteLine("  dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --s55");
        Console.WriteLine("  dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --s53");
        Console.WriteLine("  dotnet run --project DBCTool.V2/DBCTool.V2.csproj -- --src 0.6.0");
        Console.WriteLine();
    }

    private sealed record Opts(
        string? DbdDir,
        string? OutRoot,
        string? Locale,
        bool DumpArea,
        bool CompareAreaV2,
        bool QuickDump,
        bool QuickCompare,
        string? SrcAlias,
        bool S53,
        bool S55,
        bool S60,
        List<(string build, string dir)> Inputs
    );

    private static Opts ParseArgs(string[] args)
    {
        string? dbdDir = null, outRoot = null, locale = null;
        bool compareAreaV2 = false;
        bool dumpArea = false;
        bool quickDump = false;
        bool quickCompare = false;
        string? srcAlias = null;
        bool s53 = false, s55 = false, s60 = false;
        var inputs = new List<(string build, string dir)>();

        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a == "--dbd-dir" && i + 1 < args.Length) dbdDir = args[++i];
            else if (a == "--out" && i + 1 < args.Length) outRoot = args[++i];
            else if (a == "--locale" && i + 1 < args.Length) locale = args[++i];
            else if (a == "--compare-area-v2") compareAreaV2 = true;
            else if (a == "--dump-area") dumpArea = true;
            else if (a == "--qc" || a == "--quick-compare") quickCompare = true;
            else if (a == "--qd" || a == "--quick-dump") quickDump = true;
            else if ((a == "--src" || a == "-s") && i + 1 < args.Length) srcAlias = NormalizeAlias(args[++i]);
            else if (a == "--s53") { s53 = true; srcAlias ??= "0.5.3"; }
            else if (a == "--s55") { s55 = true; srcAlias ??= "0.5.5"; }
            else if (a == "--s60") { s60 = true; srcAlias ??= "0.6.0"; }
            else if (a == "--input" && i + 1 < args.Length)
            {
                var spec = args[++i];
                var parts = spec.Split('=', 2, StringSplitOptions.TrimEntries);
                var build = parts.Length == 2 ? parts[0] : "";
                var dir = parts.Length == 2 ? parts[1] : spec;
                inputs.Add((build, dir));
            }
        }
        return new Opts(dbdDir, outRoot, locale, dumpArea, compareAreaV2, quickDump, quickCompare, srcAlias, s53, s55, s60, inputs);
    }

    private static void AutoPopulateDefaults(ref Opts opts)
    {
        // DBD definitions: prefer ../lib/WoWDBDefs/definitions if exists
        opts = opts with { DbdDir = opts.DbdDir ?? TryResolveDbdDir() ?? opts.DbdDir };

        // Inputs: if none supplied, try ../test_data defaults
        if (opts.Inputs.Count == 0)
        {
            var candidates = new List<(string build, string dir)>();
            // Determine preferred source alias based on flags
            var preferred = opts.SrcAlias ?? (opts.S53 ? "0.5.3" : opts.S55 ? "0.5.5" : opts.S60 ? "0.6.0" : null);
            var src053 = TryResolvePath("..\\test_data\\0.5.3\\tree\\DBFilesClient");
            var src055 = TryResolvePath("..\\test_data\\0.5.5\\tree\\DBFilesClient");
            var src060 = TryResolvePath("..\\test_data\\0.6.0\\tree\\DBFilesClient");
            var tgt335 = TryResolvePath("..\\test_data\\3.3.5\\tree\\DBFilesClient");
            // Choose source according to preferred; else first available
            if (preferred == "0.5.3" && !string.IsNullOrEmpty(src053)) candidates.Add(("0.5.3", src053));
            else if (preferred == "0.5.5" && !string.IsNullOrEmpty(src055)) candidates.Add(("0.5.5", src055));
            else if (preferred == "0.6.0" && !string.IsNullOrEmpty(src060)) candidates.Add(("0.6.0", src060));
            else if (!string.IsNullOrEmpty(src053)) candidates.Add(("0.5.3", src053));
            else if (!string.IsNullOrEmpty(src055)) candidates.Add(("0.5.5", src055));
            else if (!string.IsNullOrEmpty(src060)) candidates.Add(("0.6.0", src060));
            if (!string.IsNullOrEmpty(tgt335)) candidates.Add(("3.3.5", tgt335));
            if (candidates.Count > 0)
            {
                opts = opts with { Inputs = candidates };
            }
        }
        else
        {
            // Ensure 3.3.5 target exists; add if missing via auto-resolve
            bool has335 = opts.Inputs.Exists(i => string.Equals(i.build, "3.3.5", StringComparison.OrdinalIgnoreCase));
            if (!has335)
            {
                var tgt335 = TryResolvePath("..\\test_data\\3.3.5\\tree\\DBFilesClient");
                if (!string.IsNullOrEmpty(tgt335))
                {
                    var newInputs = new List<(string build, string dir)>(opts.Inputs) { ("3.3.5", tgt335) };
                    opts = opts with { Inputs = newInputs };
                }
            }
        }
    }

    private static string? TryResolveDbdDir()
    {
        var p1 = TryResolvePath("..\\lib\\WoWDBDefs\\definitions");
        if (!string.IsNullOrEmpty(p1)) return p1;
        var p2 = TryResolvePath("lib\\WoWDBDefs\\definitions");
        if (!string.IsNullOrEmpty(p2)) return p2;
        return null;
    }

    private static string? TryResolvePath(string relative)
    {
        try
        {
            var full = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), relative));
            return Directory.Exists(full) ? full : null;
        }
        catch { return null; }
    }

    private sealed record Config(
        string? DbdDir,
        string? OutRoot,
        string? Locale,
        string? DefaultCommand,
        List<string>? Inputs
    );

    private static Config? LoadConfig()
    {
        try
        {
            // Look for dbctool.v2.json in current dir then parent
            var here = Path.Combine(Directory.GetCurrentDirectory(), "dbctool.v2.json");
            var parent = Path.Combine(Directory.GetParent(Directory.GetCurrentDirectory())?.FullName ?? Directory.GetCurrentDirectory(), "dbctool.v2.json");
            var path = File.Exists(here) ? here : (File.Exists(parent) ? parent : null);
            if (path is null) return null;
            var json = File.ReadAllText(path);
            var cfg = JsonSerializer.Deserialize<Config>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            return cfg;
        }
        catch { return null; }
    }

    private static List<(string build, string dir)> ParseInputs(List<string> specs)
    {
        var list = new List<(string build, string dir)>();
        foreach (var s in specs)
        {
            if (string.IsNullOrWhiteSpace(s)) continue;
            var parts = s.Split('=', 2, StringSplitOptions.TrimEntries);
            if (parts.Length == 2)
            {
                list.Add((parts[0], parts[1]));
            }
            else
            {
                // Infer build from path token
                var tok = s.ToLowerInvariant();
                string build = tok.Contains("0.5.3") ? "0.5.3" : tok.Contains("0.5.5") ? "0.5.5" : tok.Contains("0.6.0") ? "0.6.0" : tok.Contains("3.3.5") ? "3.3.5" : "";
                list.Add((build, s));
            }
        }
        return list;
    }

    private static string NormalizeAlias(string s)
    {
        var t = (s ?? string.Empty).Trim().ToLowerInvariant();
        if (t is "053" or "0.5.3" or "5.3") return "0.5.3";
        if (t is "055" or "0.5.5" or "5.5") return "0.5.5";
        if (t is "060" or "0.6.0" or "6.0" or "0.6") return "0.6.0";
        if (t is "335" or "3.3.5" or "3.3") return "3.3.5";
        return s;
    }
}
