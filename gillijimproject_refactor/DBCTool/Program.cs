using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using DBCD;
using DBCD.Providers;
using DBCTool.Mpq;

namespace DBCTool
{
    internal static class Program
    {
        private const string DefaultOutBase = "out";
        private const string DefaultDbdDir = "lib/WoWDBDefs/definitions";
        private const string DefaultLocale = "enUS";

        private static int Main(string[] args)
        {
            if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
            {
                PrintHelp();
                return 0;
            }

            try
            {
                var (dbdDir, outBase, localeStr, tables, inputs, mpqRoot, mpqArchives, debugMpqTable) = ParseArgs(args);
                if (tables.Count == 0 && string.IsNullOrEmpty(debugMpqTable))
                {
                    Console.Error.WriteLine("ERROR: At least one --table must be specified (e.g., --table AreaTable)");
                    return 2;
                }
                if (inputs.Count == 0)
                {
                    Console.Error.WriteLine("ERROR: At least one --input <build>=<dbcDir|mpq|mpq:<root>> must be specified");
                    return 2;
                }

                var locale = ParseLocale(localeStr);
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);

                foreach (var (build, dbcSpec) in inputs)
                {
                    var subFolderName = $"dbcTool_{build}_{timestamp}";
                    var outDir = Path.Combine(outBase, subFolderName);
                    Directory.CreateDirectory(outDir);

                    Console.WriteLine($"[DBCTool] Exporting tables for build {build} → {outDir}");

                    // Select DBC provider (filesystem or MPQ)
                    IDBCProvider dbcProvider;
                    string? effectiveMpqRoot = null;
                    bool useMpq = IsMpqSpec(dbcSpec);
                    MpqDBCProvider? mpqProv = null;
                    if (useMpq)
                    {
                        effectiveMpqRoot = TryExtractMpqRoot(dbcSpec) ?? mpqRoot;
                        if (mpqArchives.Count > 0)
                        {
                            Console.WriteLine($"  • Using explicit MPQ archives ({mpqArchives.Count})");
                            mpqProv = new MpqDBCProvider(mpqArchives);
                            dbcProvider = mpqProv;
                        }
                        else
                        {
                            if (string.IsNullOrWhiteSpace(effectiveMpqRoot))
                            {
                                Console.Error.WriteLine("ERROR: MPQ mode selected but no --mpq-root provided and no root in input spec (mpq:<root>)");
                                return 3;
                            }
                            Console.WriteLine($"  • Using MPQ root: {effectiveMpqRoot}");
                            mpqProv = MpqDBCProvider.FromRoot(effectiveMpqRoot);
                            dbcProvider = mpqProv;
                        }

                        if (mpqProv != null)
                        {
                            var count = mpqProv.Archives.Count;
                            var sample = string.Join(", ", mpqProv.Archives.Take(8).Select(Path.GetFileName));
                            Console.WriteLine($"  • Discovered {count} MPQ archives{(count > 0 ? ": " + sample + (count > 8 ? ", ..." : string.Empty) : string.Empty)}");
                        }

                        // MPQ Debug probe: show which archives expose the target file, then exit
                        if (!string.IsNullOrEmpty(debugMpqTable))
                        {
                            if (mpqProv == null)
                            {
                                Console.Error.WriteLine("No MPQ provider available for debug probe");
                                return 5;
                            }
                            MpqDebugProbe(mpqProv, debugMpqTable);
                            return 0;
                        }
                    }
                    else
                    {
                        var dbcDir = NormalizePath(dbcSpec);
                        if (!Directory.Exists(dbcDir))
                        {
                            Console.Error.WriteLine($"ERROR: DBC directory not found: {dbcDir}");
                            return 3;
                        }
                        dbcProvider = new FilesystemDBCProvider(dbcDir, useCache: true);
                    }

                    if (!Directory.Exists(dbdDir))
                    {
                        Console.Error.WriteLine($"ERROR: DBD definitions directory not found: {dbdDir}");
                        return 3;
                    }

                    var dbdFsProvider = new FilesystemDBDProvider(dbdDir);
                    var dbcd = new DBCD.DBCD(dbcProvider, dbdFsProvider);

                    foreach (var table in tables)
                    {
                        try
                        {
                            if (!dbdFsProvider.ContainsBuild(table, build))
                            {
                                Console.WriteLine($"  ! Warning: {table}.dbd does not list build {build}. Proceeding; loader may still match by layout hash.");
                            }
                            Console.WriteLine($"  - Loading {table} ...");

                            IDBCDStorage storage;
                            bool usedFallback = false;
                            try
                            {
                                storage = dbcd.Load(table, build, locale);
                            }
                            catch (Exception ex) when (locale != DBCD.Locale.None)
                            {
                                Console.WriteLine($"    Load failed with locale {localeStr} ({ex.GetType().Name}). Retrying with Locale.None to work around locstring mask alignment...");
                                usedFallback = true;
                                storage = dbcd.Load(table, build, DBCD.Locale.None);
                            }

                            Console.WriteLine($"    Loaded {table}: layout=0x{storage.LayoutHash:X8}, rows={storage.Count}{(usedFallback ? " (Locale=None)" : string.Empty)}");
                            var outPath = Path.Combine(outDir, $"{table}.csv");
                            CsvExporter.WriteCsv(storage, outPath);
                            Console.WriteLine($"    Wrote {outPath}");
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"  ! Failed to export {table} for build {build}: {ex}");
                            if (ex.InnerException != null)
                            {
                                Console.Error.WriteLine($"    Inner: {ex.InnerException}");
                            }
                            return 4;
                        }
                    }
                }

                Console.WriteLine("[DBCTool] Done.");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Unhandled error: {ex}");
                return 1;
            }
        }

        private static void MpqDebugProbe(MpqDBCProvider mpqProv, string table)
        {
            var candidates = new[]
            {
                $"DBFilesClient\\{table}.dbc",
                $"DBFilesClient/{table}.dbc"
            };

            Console.WriteLine($"[MPQ-DEBUG] Probing {table} in {mpqProv.Archives.Count} archives...");
            foreach (var ap in mpqProv.Archives)
            {
                if (!StormLib.SFileOpenArchive(ap, 0, 0, out var h))
                {
                    Console.WriteLine($"  [open-failed] {ap}");
                    continue;
                }
                try
                {
                    foreach (var p in candidates)
                    {
                        bool visible = StormLib.SFileHasFile(h, p);
                        if (visible)
                        {
                            Console.WriteLine($"  [direct-visible] {Path.GetFileName(ap)} → {p}");
                        }
                    }
                }
                finally
                {
                    StormLib.SFileCloseArchive(h);
                }
            }
            Console.WriteLine("[MPQ-DEBUG] Probe complete.");
        }

        private static (string dbdDir, string outBase, string locale, List<string> tables, List<(string build, string dir)> inputs, string? mpqRoot, List<string> mpqArchives, string debugMpqTable) ParseArgs(string[] args)
        {
            string dbdDir = DefaultDbdDir;
            string outBase = DefaultOutBase;
            string locale = DefaultLocale;
            string? mpqRoot = null;
            string debugMpqTable = string.Empty;
            var tables = new List<string>();
            var inputs = new List<(string build, string dir)>();
            var mpqArchives = new List<string>();

            for (int i = 0; i < args.Length; i++)
            {
                var a = args[i];
                switch (a)
                {
                    case "--dbd-dir":
                        dbdDir = RequireValue(args, ref i, a);
                        break;
                    case "--out":
                        outBase = RequireValue(args, ref i, a);
                        break;
                    case "--locale":
                        locale = RequireValue(args, ref i, a);
                        break;
                    case "--table":
                        var t = RequireValue(args, ref i, a);
                        foreach (var part in t.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                        {
                            if (!string.IsNullOrWhiteSpace(part))
                                tables.Add(part);
                        }
                        break;
                    case "--input":
                        var spec = RequireValue(args, ref i, a);
                        // format: <build>=<dir|mpq|mpq:<root>>
                        var eq = spec.IndexOf('=');
                        if (eq <= 0 || eq >= spec.Length - 1)
                            throw new ArgumentException($"Invalid --input format: {spec}. Expected <build>=<dbcDir|mpq|mpq:<root>>");
                        var build = spec.Substring(0, eq).Trim();
                        var dir = spec.Substring(eq + 1).Trim();
                        inputs.Add((build, dir));
                        break;
                    case "--mpq-root":
                        mpqRoot = RequireValue(args, ref i, a);
                        break;
                    case "--mpq-archive":
                        mpqArchives.Add(RequireValue(args, ref i, a));
                        break;
                    case "--debug-mpq-file":
                        debugMpqTable = RequireValue(args, ref i, a);
                        break;
                    default:
                        // ignore unknowns to keep CLI simple
                        break;
                }
            }

            return (NormalizePath(dbdDir), NormalizePath(outBase), locale, tables, inputs, mpqRoot != null ? NormalizePath(mpqRoot) : null, mpqArchives.Select(NormalizePath).ToList(), debugMpqTable);
        }

        private static string RequireValue(string[] args, ref int i, string flag)
        {
            if (i + 1 >= args.Length)
                throw new ArgumentException($"Missing value for {flag}");
            i++;
            return args[i];
        }

        private static string NormalizePath(string p)
        {
            return Path.IsPathRooted(p) ? p : Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), p));
        }

        private static DBCD.Locale ParseLocale(string s)
        {
            if (Enum.TryParse<DBCD.Locale>(s, ignoreCase: true, out var loc))
                return loc;
            return DBCD.Locale.EnUS;
        }

        private static bool IsMpqSpec(string spec)
        {
            return string.Equals(spec, "mpq", StringComparison.OrdinalIgnoreCase) || spec.StartsWith("mpq:", StringComparison.OrdinalIgnoreCase);
        }

        private static string? TryExtractMpqRoot(string spec)
        {
            const string prefix = "mpq:";
            if (spec.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                var root = spec.Substring(prefix.Length).Trim();
                return string.IsNullOrWhiteSpace(root) ? null : NormalizePath(root);
            }
            return null;
        }

        private static void PrintHelp()
        {
            Console.WriteLine("DBCTool - Export WoW DBC tables to CSV using DBCD + WoWDBDefs");
            Console.WriteLine();
            Console.WriteLine("Usage (filesystem mode):");
            Console.WriteLine("  dotnet run --project DBCTool -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions ");
            Console.WriteLine("    --out out ");
            Console.WriteLine("    --locale enUS ");
            Console.WriteLine("    --table AreaTable [--table Map] ");
            Console.WriteLine("    --input 3.3.5.12340=test_data/3.3.5/tree/DBFilesClient ");
            Console.WriteLine();
            Console.WriteLine("Usage (MPQ mode, auto):");
            Console.WriteLine("  dotnet run --project DBCTool -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS ");
            Console.WriteLine("    --table Map ");
            Console.WriteLine("    --mpq-root C:/WoW-3.3.5 ");
            Console.WriteLine("    --input 3.3.5.12340=mpq");
            Console.WriteLine();
            Console.WriteLine("Usage (MPQ mode, explicit archives):");
            Console.WriteLine("  dotnet run --project DBCTool -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS ");
            Console.WriteLine("    --table Map ");
            Console.WriteLine("    --mpq-archive C:/WoW/Data/common.MPQ --mpq-archive C:/WoW/Data/lichking.MPQ ");
            Console.WriteLine("    --mpq-archive C:/WoW/Data/patch.MPQ --mpq-archive C:/WoW/Data/patch-2.MPQ ");
            Console.WriteLine("    --mpq-archive C:/WoW/Data/enUS/patch-enUS.MPQ ");
            Console.WriteLine("    --input 3.3.5.12340=mpq");
            Console.WriteLine();
            Console.WriteLine("Debug MPQ (probe which archive exposes a table):");
            Console.WriteLine("  dotnet run -- --mpq-root C:/WoW-3.3.5 --input 3.3.5.12340=mpq --debug-mpq-file AreaTable");
            Console.WriteLine();
        }
    }
}
