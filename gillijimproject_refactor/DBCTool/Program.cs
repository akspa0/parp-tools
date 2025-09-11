using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using DBCD;
using DBCD.Providers;

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
                var (dbdDir, outBase, localeStr, inputs, buildOverride, compareArea) = ParseArgs(args);

                if (inputs.Count == 0)
                {
                    Console.Error.WriteLine("ERROR: At least one --input must be specified");
                    return 2;
                }

                var locale = ParseLocale(localeStr);

                foreach (var (build, dbcSpec) in inputs)
                {
                    // Determine alias/canonical build and stable out folder (no timestamp)
                    var alias = ResolveAliasOrInfer(build, dbcSpec, buildOverride: string.Empty);
                    var canonicalBuild = !string.IsNullOrWhiteSpace(alias) ? CanonicalizeBuild(alias) : build;
                    var subFolderName = !string.IsNullOrWhiteSpace(alias)
                        ? alias
                        : (!string.IsNullOrWhiteSpace(ResolveAlias(build)) ? ResolveAlias(build) : (string.IsNullOrWhiteSpace(build) ? "unknown" : build));
                    var outDir = Path.Combine(outBase, subFolderName);
                    Directory.CreateDirectory(outDir);

                    Console.WriteLine($"[DBCTool] Exporting tables for build {subFolderName} → {outDir}");

                    var dbcDir = NormalizePath(dbcSpec);
                    if (!Directory.Exists(dbcDir))
                    {
                        Console.Error.WriteLine($"ERROR: DBC directory not found: {dbcDir}");
                        return 3;
                    }

                    var dbcProvider = new FilesystemDBCProvider(dbcDir, useCache: true);

                    if (!Directory.Exists(dbdDir))
                    {
                        Console.Error.WriteLine($"ERROR: DBD definitions directory not found: {dbdDir}");
                        return 3;
                    }

                    var dbdFsProvider = new FilesystemDBDProvider(dbdDir);
                    var dbcd = new DBCD.DBCD(dbcProvider, dbdFsProvider);

                    var tables = new List<string>();
                    if (compareArea)
                    {
                        tables.Add("AreaTable");
                    }

                    foreach (var table in tables)
                    {
                        try
                        {
                            if (!dbdFsProvider.ContainsBuild(table, canonicalBuild))
                            {
                                Console.WriteLine($"  ! Warning: {table}.dbd does not list build {canonicalBuild}. Proceeding; loader may still match by layout hash.");
                            }
                            Console.WriteLine($"  - Loading {table} ...");

                            IDBCDStorage storage;
                            bool usedFallback = false;
                            try
                            {
                                storage = dbcd.Load(table, canonicalBuild, locale);
                            }
                            catch (Exception ex) when (locale != DBCD.Locale.None)
                            {
                                Console.WriteLine($"    Load failed with locale {localeStr} ({ex.GetType().Name}). Retrying with Locale.None to work around locstring mask alignment...");
                                usedFallback = true;
                                storage = dbcd.Load(table, canonicalBuild, DBCD.Locale.None);
                            }

                            Console.WriteLine($"    Loaded {table}: layout=0x{storage.LayoutHash:X8}, rows={storage.Count}{(usedFallback ? " (Locale=None)" : string.Empty)}");
                            var outPath = Path.Combine(outDir, $"{table}.csv");
                            CsvExporter.WriteCsv(storage, outPath);
                            Console.WriteLine($"    Wrote {outPath}");
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"  ! Failed to export {table} for build {canonicalBuild}: {ex}");
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

        private static int MpqList(string archivePath, string mask)
        {
            Console.WriteLine("[MPQ-LIST] MPQ support has been removed; filesystem mode only.");
            return 0;
        }

        private static int MpqTestOpen(string archivePath, string mpqPath)
        {
            Console.WriteLine("[MPQ-TEST] MPQ support has been removed; filesystem mode only.");
            return 0;
        }

        private static void MpqDebugProbe(object _mpqProvIgnored, string _tableIgnored)
        {
            Console.WriteLine("[MPQ-DEBUG] MPQ support has been removed; filesystem mode only.");
        }

        private static (string dbdDir, string outBase, string locale, List<(string build, string dir)> inputs, string buildOverride, bool compareArea) ParseArgs(string[] args)
        {
            string dbdDir = DefaultDbdDir;
            string outBase = DefaultOutBase;
            string locale = DefaultLocale;
            string buildOverride = string.Empty;
            bool compareArea = false;
            var inputs = new List<(string build, string dir)>();

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
                    case "--input":
                        var spec = RequireValue(args, ref i, a);
                        var eq = spec.IndexOf('=');
                        if (eq > 0 && eq < spec.Length - 1)
                        {
                            var b = spec.Substring(0, eq).Trim();
                            var d = spec.Substring(eq + 1).Trim();
                            inputs.Add((b, d));
                        }
                        else
                        {
                            // Bare directory: infer alias from path later
                            inputs.Add((string.Empty, spec));
                        }
                        break;
                    case "--build":
                        buildOverride = RequireValue(args, ref i, a);
                        break;
                    case "--compare-area":
                        compareArea = true;
                        break;
                    default:
                        break;
                }
            }

            return (NormalizePath(dbdDir), NormalizePath(outBase), locale, inputs.Select(t => (t.build, NormalizePath(t.dir))).ToList(), buildOverride, compareArea);
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

        private static byte[] ReadAllUnknown(IntPtr _hFile, int _maxLimit) => Array.Empty<byte>();

        private static string ResolveAliasOrInfer(string buildOrAlias, string dbcSpec, string buildOverride)
        {
            if (!string.IsNullOrWhiteSpace(buildOverride))
                return ResolveAlias(buildOverride);
            if (!string.IsNullOrWhiteSpace(buildOrAlias))
                return ResolveAlias(buildOrAlias);
            return InferAliasFromPath(dbcSpec);
        }

        private static string CanonicalizeBuild(string alias)
        {
            return alias switch
            {
                "0.5.3" => "0.5.3.3368",
                "0.5.5" => "0.5.5.3494",
                "3.3.5" => "3.3.5.12340",
                _ => alias
            };
        }

        private static string ResolveAlias(string buildOrAlias)
        {
            if (string.IsNullOrWhiteSpace(buildOrAlias)) return string.Empty;
            var s = buildOrAlias.Trim();
            if (s.StartsWith("0.5.3")) return "0.5.3";
            if (s.StartsWith("0.5.5")) return "0.5.5";
            if (s.StartsWith("3.3.5")) return "3.3.5";
            return string.Empty;
        }

        private static string InferAliasFromPath(string path)
        {
            var p = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar).ToLowerInvariant();
            if (p.Contains("0.5.3")) return "0.5.3";
            if (p.Contains("0.5.5")) return "0.5.5";
            if (p.Contains("3.3.5")) return "3.3.5";
            return string.Empty;
        }

        private static void PrintHelp()
        {
            Console.WriteLine("DBCTool - Export WoW DBC tables to CSV using DBCD + WoWDBDefs (filesystem only)");
            Console.WriteLine();
            Console.WriteLine("Usage (with build alias):");
            Console.WriteLine("  dotnet run -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS ");
            Console.WriteLine("    --table AreaTable [--table Map] ");
            Console.WriteLine("    --input 3.3.5=path/to/3.3.5/DBFilesClient");
            Console.WriteLine();
            Console.WriteLine("Usage (bare directory; build inferred from path tokens 0.5.3|0.5.5|3.3.5):");
            Console.WriteLine("  dotnet run -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS ");
            Console.WriteLine("    --table AreaTable ");
            Console.WriteLine("    --input path/to/0.5.3/DBFilesClient");
            Console.WriteLine();
        }
    }
}
