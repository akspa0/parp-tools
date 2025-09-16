using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
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
                var (dbdDir, outBase, localeStr, tables, inputs, buildOverride, compareArea, compareAreaV2, exportAll, srcAliasFlag, srcBuildFlag, tgtBuildFlag, exportRemap, applyRemap, disallowDoNotUse) = ParseArgs(args);

                if (inputs.Count == 0)
                {
                    Console.Error.WriteLine("ERROR: At least one --input must be specified");
                    return 2;
                }

                var locale = ParseLocale(localeStr);
                if (compareArea)
                {
                    return CompareAreas(dbdDir, outBase, locale, inputs, buildOverride, srcAliasFlag, srcBuildFlag, tgtBuildFlag, exportRemap, applyRemap, disallowDoNotUse);
                }
                if (compareAreaV2)
                {
                    return CompareAreasV2(dbdDir, outBase, locale, inputs, buildOverride, srcAliasFlag, srcBuildFlag, tgtBuildFlag, disallowDoNotUse);
                }
                if (!exportAll && (tables == null || tables.Count == 0))
                {
                    Console.Error.WriteLine("ERROR: At least one --table must be specified (or use --all)");
                    return 2;
                }

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

                    var tablesLocal = exportAll ? DiscoverTables(dbcDir, dbdDir, dbdFsProvider, alias, canonicalBuild) : new List<string>(tables);
                    if (compareArea && !tablesLocal.Contains("AreaTable", StringComparer.OrdinalIgnoreCase))
                        tablesLocal.Add("AreaTable");

                    Console.WriteLine($"  - Tables to export: {tablesLocal.Count}");

                    int okCount = 0, failCount = 0;
                    foreach (var table in tablesLocal)
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
                            okCount++;
                        }
                        catch (Exception ex)
                        {
                            failCount++;
                            Console.Error.WriteLine($"  ! Failed to export {table} for build {canonicalBuild}: {ex.Message}");
                            if (!exportAll)
                            {
                                if (ex.InnerException != null) Console.Error.WriteLine($"    Inner: {ex.InnerException.Message}");
                                return 4;
                            }
                            // In --all mode, continue best-effort
                        }
                    }

                    Console.WriteLine($"[DBCTool] Summary for {alias}: {okCount} ok, {failCount} failed");
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

        private static (string dbdDir, string outBase, string locale, List<string> tables, List<(string build, string dir)> inputs, string buildOverride, bool compareArea, bool compareAreaV2, bool exportAll, string srcAlias, string srcBuild, string tgtBuild, string exportRemap, string applyRemap, bool disallowDoNotUse) ParseArgs(string[] args)
        {
            string dbdDir = DefaultDbdDir;
            string outBase = DefaultOutBase;
            string locale = DefaultLocale;
            string buildOverride = string.Empty;
            bool compareArea = false;
            bool compareAreaV2 = false;
            bool exportAll = false;
            string srcAlias = string.Empty;
            string srcBuild = string.Empty;
            string tgtBuild = string.Empty;
            string exportRemap = string.Empty;
            string applyRemap = string.Empty;
            bool disallowDoNotUse = true; // default: do not map to DO NOT USE areas
            var tables = new List<string>();
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
                    case "--src-alias":
                        srcAlias = RequireValue(args, ref i, a);
                        break;
                    case "--src-build":
                        srcBuild = RequireValue(args, ref i, a);
                        break;
                    case "--tgt-build":
                        tgtBuild = RequireValue(args, ref i, a);
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
                    case "--compare-area-v2":
                        compareAreaV2 = true;
                        break;
                    case "--all":
                    case "--export-all":
                        exportAll = true;
                        break;
                    case "--export-remap":
                        exportRemap = RequireValue(args, ref i, a);
                        break;
                    case "--apply-remap":
                        applyRemap = RequireValue(args, ref i, a);
                        break;
                    case "--allow-do-not-use":
                        // If set, allow mapping to target areas named DO NOT USE
                        disallowDoNotUse = false;
                        break;
                    default:
                        break;
                }
            }

            // Only normalize remap paths when provided; keep empty when flags are omitted
            var normExportRemap = string.IsNullOrWhiteSpace(exportRemap) ? string.Empty : NormalizePath(exportRemap);
            var normApplyRemap = string.IsNullOrWhiteSpace(applyRemap) ? string.Empty : NormalizePath(applyRemap);
            return (NormalizePath(dbdDir), NormalizePath(outBase), locale, tables, inputs.Select(t => (t.build, NormalizePath(t.dir))).ToList(), buildOverride, compareArea, compareAreaV2, exportAll, srcAlias, srcBuild, tgtBuild, normExportRemap, normApplyRemap, disallowDoNotUse);
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
                "0.6.0" => "0.6.0.3592",
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
            if (s.StartsWith("0.6.0")) return "0.6.0";
            if (s.StartsWith("3.3.5")) return "3.3.5";
            return string.Empty;
        }

        private static string InferAliasFromPath(string path)
        {
            var p = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar).ToLowerInvariant();
            if (p.Contains("0.5.3")) return "0.5.3";
            if (p.Contains("0.5.5")) return "0.5.5";
            if (p.Contains("0.6.0")) return "0.6.0";
            if (p.Contains("3.3.5")) return "3.3.5";
            return string.Empty;
        }

        private static List<string> DiscoverTables(string dbcDir, string dbdDir, FilesystemDBDProvider dbdProvider, string alias, string canonicalBuild)
        {
            var dirNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            try
            {
                // For classic/early builds (0.5.x, 3.3.5), only *.dbc are valid
                foreach (var p in Directory.EnumerateFiles(dbcDir, "*.dbc", SearchOption.TopDirectoryOnly))
                    dirNames.Add(Path.GetFileNameWithoutExtension(p));
                // Include *.db2 only for non-classic aliases
                if (!(alias == "0.5.3" || alias == "0.5.5" || alias == "3.3.5"))
                {
                    foreach (var p in Directory.EnumerateFiles(dbcDir, "*.db2", SearchOption.TopDirectoryOnly))
                        dirNames.Add(Path.GetFileNameWithoutExtension(p));
                }
            }
            catch { }

            // Intersect with known .dbd definitions
            var dbdNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            try
            {
                foreach (var p in Directory.EnumerateFiles(dbdDir, "*.dbd", SearchOption.TopDirectoryOnly))
                    dbdNames.Add(Path.GetFileNameWithoutExtension(p));
            }
            catch { }

            var candidates = dirNames.Where(name => dbdNames.Contains(name)).ToList();

            // Filter to those listing the build (best-effort)
            var result = new List<string>();
            foreach (var name in candidates)
            {
                try
                {
                    if (dbdProvider.ContainsBuild(name, canonicalBuild))
                        result.Add(name);
                }
                catch
                {
                    // If ContainsBuild throws, include anyway and let loader decide
                    result.Add(name);
                }
            }

            result.Sort(StringComparer.OrdinalIgnoreCase);
            return result;
        }

        private static void PrintHelp()
        {
            Console.WriteLine("DBCTool - Export WoW DBC tables to CSV using DBCD + WoWDBDefs (filesystem only)");
            Console.WriteLine();
            Console.WriteLine("Usage (with build alias):");
            Console.WriteLine("  dotnet run -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS ");
            Console.WriteLine("    --table AreaTable [--table Map]  (or use --all) ");
            Console.WriteLine("    --input 3.3.5=path/to/3.3.5/DBFilesClient");
            Console.WriteLine();
            Console.WriteLine("Usage (bare directory; build inferred from path tokens 0.5.3|0.5.5|3.3.5):");
            Console.WriteLine("  dotnet run -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS ");
            Console.WriteLine("    --table AreaTable  (or use --all) ");
            Console.WriteLine("    --input path/to/0.5.3/DBFilesClient");
            Console.WriteLine();
            Console.WriteLine("Export all tables found in the input folder:");
            Console.WriteLine("  dotnet run -- --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS --all --input path/to/DBFilesClient");
            Console.WriteLine();
            Console.WriteLine("Compare AreaTable 0.5.3 → 3.3.5 mapping and MapId→Name reports:");
            Console.WriteLine("  dotnet run -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS ");
            Console.WriteLine("    --compare-area ");
            Console.WriteLine("    --input 0.5.3=path/to/0.5.3/DBFilesClient ");
            Console.WriteLine("    --input 3.3.5=path/to/3.3.5/DBFilesClient");
            Console.WriteLine();
            Console.WriteLine("Compare AreaTable 0.5.x → 3.3.5 using V2 strict map-locked engine:");
            Console.WriteLine("  dotnet run -- ");
            Console.WriteLine("    --dbd-dir lib/WoWDBDefs/definitions --out out --locale enUS ");
            Console.WriteLine("    --compare-area-v2 ");
            Console.WriteLine("    --input 0.5.3=path/to/0.5.3/DBFilesClient ");
            Console.WriteLine("    --input 3.3.5=path/to/3.3.5/DBFilesClient");
            Console.WriteLine();
        }

        // Compare 0.5.3 → 3.3.5 AreaTable entries using per-build truth (names, parents, map names)
        private static int CompareAreas(string dbdDir, string outBase, DBCD.Locale locale, List<(string build, string dir)> inputs, string buildOverride, string srcAliasFlag, string srcBuildFlag, string tgtBuildFlag, string exportRemap, string applyRemap, bool disallowDoNotUse)
        {
            // Normalize inputs and pick source (0.5.3 or 0.5.5 or 0.6.0) and 3.3.5
            string? dir053 = null, dir055 = null, dir060 = null, dir335 = null;
            foreach (var (build, dir) in inputs)
            {
                var alias = ResolveAliasOrInfer(build, dir, buildOverride);
                if (alias == "0.5.3") dir053 = NormalizePath(dir);
                else if (alias == "0.5.5") dir055 = NormalizePath(dir);
                else if (alias == "0.6.0") dir060 = NormalizePath(dir);
                else if (alias == "3.3.5") dir335 = NormalizePath(dir);
            }
            // Determine source alias if not explicitly provided
            string srcAlias = ResolveAlias(srcAliasFlag);
            if (string.IsNullOrWhiteSpace(srcAlias)) srcAlias = !string.IsNullOrEmpty(dir053) ? "0.5.3" : (!string.IsNullOrEmpty(dir055) ? "0.5.5" : (!string.IsNullOrEmpty(dir060) ? "0.6.0" : "0.5.3"));
            string? dirSrc = srcAlias == "0.5.3" ? dir053 : (srcAlias == "0.5.5" ? dir055 : (srcAlias == "0.6.0" ? dir060 : null));
            if (string.IsNullOrEmpty(dirSrc) || string.IsNullOrEmpty(dir335))
            {
                Console.Error.WriteLine("ERROR: --compare-area requires both a source (0.5.3 or 0.5.5 or 0.6.0) and 3.3.5 inputs.");
                return 2;
            }

            var outDir = Path.Combine(outBase, "compare");
            Directory.CreateDirectory(outDir);

            var dbdProvider = new FilesystemDBDProvider(dbdDir);

            // Load storages with fallback to Locale.None
            string canonicalSrcBuild = !string.IsNullOrWhiteSpace(srcBuildFlag) ? srcBuildFlag : CanonicalizeBuild(srcAlias);
            string canonicalTgtBuild = !string.IsNullOrWhiteSpace(tgtBuildFlag) ? tgtBuildFlag : CanonicalizeBuild("3.3.5");
            var stor053_Area = LoadTable("AreaTable", canonicalSrcBuild, dirSrc!, dbdProvider, locale);
            var stor053_Map  = LoadTable("Map",       canonicalSrcBuild, dirSrc!, dbdProvider, locale);
            var stor335_Area = LoadTable("AreaTable", canonicalTgtBuild, dir335!, dbdProvider, locale);
            var stor335_Map  = LoadTable("Map",       canonicalTgtBuild, dir335!, dbdProvider, locale);

            // Helper: pick best column name for strings
            string DetectColumn(IDBCDStorage storage, params string[] preferred)
            {
                var cols = storage.AvailableColumns ?? Array.Empty<string>();
                foreach (var c in preferred)
                    if (cols.Any(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase))) return cols.First(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase));
                // fallback: first column containing "name"
                var any = cols.FirstOrDefault(x => x.IndexOf("name", StringComparison.OrdinalIgnoreCase) >= 0);
                return any ?? (preferred.Length > 0 ? preferred[0] : string.Empty);
            }

            // Detect Map.dbc ID columns and build MapID → Row dictionaries (real IDs, not storage keys)
            string idCol053 = DetectIdColumn(stor053_Map);
            string idCol335 = DetectIdColumn(stor335_Map);
            var map053ById = new Dictionary<int, DBCDRow>();
            foreach (var mid in stor053_Map.Keys)
            {
                var mrow = stor053_Map[mid];
                int midReal = !string.IsNullOrWhiteSpace(idCol053) ? SafeField<int>(mrow, idCol053) : mid;
                if (!map053ById.ContainsKey(midReal)) map053ById[midReal] = mrow;
            }
            var map335ById = new Dictionary<int, DBCDRow>();
            foreach (var mid in stor335_Map.Keys)
            {
                var mrow = stor335_Map[mid];
                int midReal = !string.IsNullOrWhiteSpace(idCol335) ? SafeField<int>(mrow, idCol335) : mid;
                if (!map335ById.ContainsKey(midReal)) map335ById[midReal] = mrow;
            }

            // Build MapId → Name for each build using their own columns
            // Prefer Directory token extraction (cannot use local functions here)
            string mapNameCol053 = DetectColumn(stor053_Map, "Directory", "InternalName", "MapName_lang", "MapName");
            string mapNameCol335 = DetectColumn(stor335_Map, "Directory", "InternalName", "MapName_lang", "MapName");
            string areaNameCol053 = DetectColumn(stor053_Area, "AreaName_lang", "AreaName", "Name");
            string areaNameCol335 = DetectColumn(stor335_Area, "AreaName_lang", "AreaName", "Name");

            string NormName(string s) => (s ?? string.Empty).Trim().ToLowerInvariant();
            string Slug(string s)
            {
                if (string.IsNullOrWhiteSpace(s)) return string.Empty;
                var sb = new StringBuilder(s.Length);
                foreach (var ch in s)
                {
                    if (char.IsLetterOrDigit(ch)) sb.Append(char.ToLowerInvariant(ch));
                }
                return sb.ToString();
            }

            string ExtractDirToken(string s)
            {
                if (string.IsNullOrWhiteSpace(s)) return string.Empty;
                var parts = s.Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                return parts.Length > 0 ? parts[^1] : s.Trim();
            }

            // Build 3.3.5 map indexes for crosswalk (by Directory token and by normalized name)
            var dir335Index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var name335Index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var mid in map335ById.Keys)
            {
                var mrow = map335ById[mid];
                var dirTok = ExtractDirToken(SafeField<string>(mrow, "Directory"));
                if (!string.IsNullOrWhiteSpace(dirTok) && !dir335Index.ContainsKey(dirTok)) dir335Index[dirTok] = mid;
                var nm = NormName(SafeField<string>(mrow, "MapName_lang") ?? SafeField<string>(mrow, "MapName") ?? SafeField<string>(mrow, "InternalName") ?? dirTok);
                if (!string.IsNullOrWhiteSpace(nm) && !name335Index.ContainsKey(nm)) name335Index[nm] = mid;
            }

            // 0.5.x → 3.3.5 mapping
            var cw053To335 = new Dictionary<int, int>();
            var cwReport = new StringBuilder();
            cwReport.AppendLine("srcMapId,srcDirToken,srcName,tgtMapId,tgtDirToken,tgtName,method");
            foreach (var srcId in map053ById.Keys)
            {
                var srcRow = map053ById[srcId];
                var srcDirTok = ExtractDirToken(SafeField<string>(srcRow, "Directory"));
                var srcName = FirstNonEmpty(
                    SafeField<string>(srcRow, "MapName_lang"),
                    SafeField<string>(srcRow, "MapName"),
                    SafeField<string>(srcRow, "InternalName"),
                    srcDirTok
                ) ?? string.Empty;

                int tgtId = -1; string method = string.Empty;
                var srcDirTokSlug = Slug(srcDirTok);
                var srcNameSlug = Slug(srcName);
                // No special-cases; rely on directory or name matching only
                if (!string.IsNullOrWhiteSpace(srcDirTok) && dir335Index.TryGetValue(srcDirTok, out var byDir))
                { tgtId = byDir; method = "directory"; }
                else
                {
                    var key = NormName(srcName);
                    if (!string.IsNullOrWhiteSpace(key) && name335Index.TryGetValue(key, out var byNameId))
                    { tgtId = byNameId; method = "name"; }
                }

                if (tgtId >= 0 && map335ById.TryGetValue(tgtId, out var tgtRow))
                {
                    cw053To335[srcId] = tgtId;
                    var tgtDirTok = ExtractDirToken(SafeField<string>(tgtRow, "Directory"));
                    var tgtName = FirstNonEmpty(
                        SafeField<string>(tgtRow, "MapName_lang"),
                        SafeField<string>(tgtRow, "MapName"),
                        SafeField<string>(tgtRow, "InternalName"),
                        tgtDirTok
                    );
                    cwReport.AppendLine(string.Join(',', new[]
                    {
                        srcId.ToString(CultureInfo.InvariantCulture),
                        Csv(srcDirTok),
                        Csv(srcName),
                        tgtId.ToString(CultureInfo.InvariantCulture),
                        Csv(tgtDirTok),
                        Csv(tgtName),
                        method
                    }));
                }
                else
                {
                    cwReport.AppendLine(string.Join(',', new[]
                    {
                        srcId.ToString(CultureInfo.InvariantCulture),
                        Csv(srcDirTok),
                        Csv(srcName),
                        "-1","","","unmatched"
                    }));
                }
            }

            // Source map index by Directory token → MapID (to translate ContinentID enum to actual map row)
            var dir053Index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var mid in map053ById.Keys)
            {
                var mrow = map053ById[mid];
                var dirTok = ExtractDirToken(SafeField<string>(mrow, "Directory"));
                if (!string.IsNullOrWhiteSpace(dirTok) && !dir053Index.ContainsKey(dirTok)) dir053Index[dirTok] = mid;
            }

            // 0.5.x AreaNumber → Row index for parent resolution and alpha decode
            var idx053_NumToRow = new Dictionary<int, DBCDRow>();
            foreach (var sid in stor053_Area.Keys)
            {
                var srow = stor053_Area[sid];
                int areaKey = SafeField<int>(srow, srcAlias == "0.5.3" || srcAlias == "0.5.5" ? "AreaNumber" : "ID");
                if (areaKey > 0 && !idx053_NumToRow.ContainsKey(areaKey)) idx053_NumToRow[areaKey] = srow;
            }

            // Helper: pick best column name for strings
            string DetectColumn2(IDBCDStorage storage, params string[] preferred)
            {
                var cols = storage.AvailableColumns ?? Array.Empty<string>();
                foreach (var c in preferred)
                    if (cols.Any(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase))) return cols.First(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase));
                // fallback: first column containing "name"
                var any = cols.FirstOrDefault(x => x.IndexOf("name", StringComparison.OrdinalIgnoreCase) >= 0);
                return any ?? (preferred.Length > 0 ? preferred[0] : string.Empty);
            }

            // Token index of source maps to help infer expected map when ContinentID is unreliable
            var token053Index = new Dictionary<string, List<int>>(StringComparer.OrdinalIgnoreCase);
            void IndexToken053(string token, int id)
            {
                if (string.IsNullOrWhiteSpace(token)) return;
                if (!token053Index.TryGetValue(token, out var list)) { list = new List<int>(); token053Index[token] = list; }
                if (!list.Contains(id)) list.Add(id);
            }
            foreach (var mid in map053ById.Keys)
            {
                var mrow = map053ById[mid];
                var dirTok = ExtractDirToken(SafeField<string>(mrow, "Directory"));
                var tok1 = Slug(dirTok);
                var tok2 = Slug(SafeField<string>(mrow, "InternalName") ?? string.Empty);
                var tok3 = Slug(FirstNonEmpty(
                    SafeField<string>(mrow, "MapName_lang"),
                    SafeField<string>(mrow, "MapName"),
                    SafeField<string>(mrow, "InternalName"),
                    dirTok
                ) ?? string.Empty);
                IndexToken053(tok1, mid);
                IndexToken053(tok2, mid);
                IndexToken053(tok3, mid);
            }

            int Infer053MapIdForArea(string name, string parentName)
            {
                var tokens = new List<string>();
                var tParent = Slug(parentName);
                var tName = Slug(name);
                if (!string.IsNullOrWhiteSpace(tParent)) tokens.Add(tParent);
                if (!string.IsNullOrWhiteSpace(tName) && tName != tParent) tokens.Add(tName);
                foreach (var t in tokens)
                {
                    if (token053Index.TryGetValue(t, out var list) && list.Count > 0)
                    {
                        if (list.Count == 1) return list[0];
                        // Prefer exact Directory token match
                        foreach (var cand in list)
                        {
                            var d = ExtractDirToken(SafeField<string>(map053ById[cand], "Directory"));
                            if (Slug(d) == t) return cand;
                        }
                        return list[0];
                    }
                }
                return -1;
            }

            // 3.3.5 indices and parent resolution via ParentAreaID
            var idx335_IdToRow = new Dictionary<int, DBCDRow>();
            foreach (var id in stor335_Area.Keys) idx335_IdToRow[id] = stor335_Area[id];

            (string name, string parentName, int areaNum, int parentId, int mapId, string mapName, string path) Extract335(int id)
            {
                var row = idx335_IdToRow[id];
                string name = SafeField<string>(row, areaNameCol335);
                int areaNum = SafeField<int>(row, "AreaNumber");
                int parentId = SafeField<int>(row, "ParentAreaID");
                if (parentId <= 0) parentId = id; // parent=self
                int cont = SafeField<int>(row, "ContinentID");
                string mapName = string.Empty;
                if (map335ById.TryGetValue(cont, out var tmp)) mapName = FirstNonEmpty(
                    SafeField<string>(tmp, "MapName_lang"),
                    SafeField<string>(tmp, "MapName"),
                    SafeField<string>(tmp, "InternalName"),
                    ExtractDirToken(SafeField<string>(tmp, "Directory"))
                );
                string parentName = name;
                if (parentId != id && idx335_IdToRow.TryGetValue(parentId, out var prow))
                    parentName = SafeField<string>(prow, areaNameCol335);
                var path = $"{NormName(mapName)}/{NormName(parentName)}/{NormName(name)}";
                return (name, parentName, areaNum, parentId, cont, mapName ?? string.Empty, path);
            }

            (string name, string parentNameForInfer, int areaNum, int parentNum, int mapId, string mapName, int mapIdX, string mapNameX, string path) Extract053(DBCDRow row)
            {
                string name = SafeField<string>(row, areaNameCol053);
                int areaNum = SafeField<int>(row, srcAlias == "0.5.3" || srcAlias == "0.5.5" ? "AreaNumber" : "ID");
                int parentNum = SafeField<int>(row, srcAlias == "0.5.3" || srcAlias == "0.5.5" ? "ParentAreaNum" : "ParentAreaID");
                if (parentNum <= 0) parentNum = areaNum; // parent=self
                // Resolve parent name via AreaNumber index (never index storage by ParentAreaNum)
                string parentNameForInfer = name;
                if (parentNum != areaNum && idx053_NumToRow.TryGetValue(parentNum, out var prowName))
                    parentNameForInfer = SafeField<string>(prowName, areaNameCol053) ?? name;
                int cont = SafeField<int>(row, "ContinentID");
                string mapName = srcAlias == "0.5.3" || srcAlias == "0.5.5" ? MapName053FromCont(cont) : string.Empty;
                if (string.IsNullOrWhiteSpace(mapName) && map053ById.TryGetValue(cont, out var tmp053)) mapName = FirstNonEmpty(
                    SafeField<string>(tmp053, "MapName_lang"),
                    SafeField<string>(tmp053, "MapName"),
                    SafeField<string>(tmp053, "InternalName"),
                    ExtractDirToken(SafeField<string>(tmp053, "Directory"))
                );
                int contX = -1;
                // Step A: For 0.5.x, ContinentID may be enum (0=Azeroth,1=Kalimdor). Map enum→dir token→Src MapID→crosswalk.
                var contDir = srcAlias == "0.5.3" || srcAlias == "0.5.5" ? MapName053FromCont(cont) : string.Empty;
                if (!string.IsNullOrWhiteSpace(contDir))
                {
                    if (dir053Index.TryGetValue(contDir, out var cont053Id) && cw053To335.TryGetValue(cont053Id, out var x1)) contX = x1;
                    else if (dir335Index.TryGetValue(contDir, out var x2)) contX = x2; // fallback: direct to 335 by dir
                }
                // Step B: Some rows may already carry a source MapID in ContinentID; try crosswalk by numeric id.
                if (contX < 0 && cw053To335.TryGetValue(cont, out var x3)) contX = x3;
                // Step C: Infer from names to source MapID and crosswalk
                if (contX < 0)
                {
                    var inferred053 = Infer053MapIdForArea(name, parentNameForInfer);
                    if (inferred053 >= 0 && cw053To335.TryGetValue(inferred053, out var tmpX2)) contX = tmpX2;
                }
                string mapNameX = string.Empty;
                if (contX >= 0 && map335ById.TryGetValue(contX, out var tmpMapRowX))
                    mapNameX = FirstNonEmpty(
                        SafeField<string>(tmpMapRowX, "MapName_lang"),
                        SafeField<string>(tmpMapRowX, "MapName"),
                        SafeField<string>(tmpMapRowX, "InternalName"),
                        ExtractDirToken(SafeField<string>(tmpMapRowX, "Directory"))
                    );
                string path;
                if (!string.IsNullOrWhiteSpace(mapNameX))
                    path = $"{NormName(mapNameX)}/{NormName(parentNameForInfer)}/{NormName(name)}";
                else
                    path = $"{NormName(parentNameForInfer)}/{NormName(name)}";
                return (name, parentNameForInfer, areaNum, parentNum, cont, mapName ?? string.Empty, contX, mapNameX ?? string.Empty, path);
            }

            // Alias table (data-guided) and name variants to catch common renames and articles
            var aliasMap = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
            {
                { "dark portal", new[] { "the dark portal" } },
                { "demonic stronghold", new[] { "dreadmaul hold" } },
                { "shadowfang", new[] { "shadowfang keep" } },
                { "lik'ash tar pits", new[] { "lakkari tar pits" } },
                { "kargathia outpost", new[] { "kargathia keep" } },
                { "the wellspring river", new[] { "wellspring river" } },
                { "wellspring river", new[] { "the wellspring river" } },
            };

            // Apply remap file if provided: override aliases and load explicit maps/options
            RemapDefinition? appliedRemap = null;
            var explicitMap = new Dictionary<int, int>();
            try
            {
                if (!string.IsNullOrWhiteSpace(applyRemap) && File.Exists(applyRemap))
                {
                    appliedRemap = JsonSerializer.Deserialize<RemapDefinition>(File.ReadAllText(applyRemap));
                    if (appliedRemap != null)
                    {
                        if (appliedRemap.aliases != null && appliedRemap.aliases.Count > 0)
                            aliasMap = new Dictionary<string, string[]>(appliedRemap.aliases, StringComparer.OrdinalIgnoreCase);
                        if (appliedRemap.options != null)
                            disallowDoNotUse = appliedRemap.options.disallow_do_not_use_targets;
                        if (appliedRemap.explicit_map != null)
                        {
                            foreach (var e in appliedRemap.explicit_map)
                                explicitMap[e.src_areaNumber] = e.tgt_areaID;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Remap] Failed to load {applyRemap}: {ex.Message}");
            }

            bool DisallowCandidate(string nm)
            {
                if (!disallowDoNotUse) return false;
                if (string.IsNullOrWhiteSpace(nm)) return false;
                return nm.IndexOf("DO NOT USE", StringComparison.OrdinalIgnoreCase) >= 0;
            }

            // Simple Levenshtein distance for fuzzy fallback
            int EditDistance(string a, string b)
            {
                a = a ?? string.Empty; b = b ?? string.Empty;
                int n = a.Length, m = b.Length;
                var d = new int[n + 1, m + 1];
                for (int i = 0; i <= n; i++) d[i, 0] = i;
                for (int j = 0; j <= m; j++) d[0, j] = j;
                for (int i = 1; i <= n; i++)
                {
                    for (int j = 1; j <= m; j++)
                    {
                        int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                        d[i, j] = Math.Min(
                            Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                            d[i - 1, j - 1] + cost);
                    }
                }
                return d[n, m];
            }

            IEnumerable<string> NameVariants(string srcName)
            {
                var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var baseName = (srcName ?? string.Empty).Trim();
                if (string.IsNullOrEmpty(baseName)) yield break;
                // base
                if (set.Add(baseName)) yield return baseName;
                // add/remove leading "The "
                if (baseName.StartsWith("The ", StringComparison.OrdinalIgnoreCase))
                {
                    var noThe = baseName.Substring(4).Trim();
                    if (noThe.Length > 0 && set.Add(noThe)) yield return noThe;
                }
                else
                {
                    var withThe = "The " + baseName;
                    if (set.Add(withThe)) yield return withThe;
                }
                // aliases
                var key = NormName(baseName);
                if (aliasMap.TryGetValue(key, out var al))
                {
                    foreach (var a in al)
                        if (!string.IsNullOrWhiteSpace(a) && set.Add(a)) yield return a;
                }
            }

            IEnumerable<string> NameVariants_Obsolete(string srcName)
            {
                #if false
                var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var baseName = (srcName ?? string.Empty).Trim();
                if (string.IsNullOrEmpty(baseName)) yield break;
                // base
                if (set.Add(baseName)) yield return baseName;
                // add/remove leading "The "
                if (baseName.StartsWith("The ", StringComparison.OrdinalIgnoreCase))
                {
                    var noThe = baseName.Substring(4).Trim();
                    if (noThe.Length > 0 && set.Add(noThe)) yield return noThe;
                }
                else
                {
                    var withThe = "The " + baseName;
                    if (set.Add(withThe)) yield return withThe;
                }
                // aliases
                var key = NormName(baseName);
                if (aliasMap.TryGetValue(key, out var al))
                {
                    foreach (var a in al)
                        if (!string.IsNullOrWhiteSpace(a) && set.Add(a)) yield return a;
                }
                #endif
                yield break;
            }

            // 3.3.5 indices for matching (by name globally, and by name within map)
            var idx335_ByName = new Dictionary<string, List<int>>();
            var idx335_ByMap = new Dictionary<int, List<int>>();
            var idx335_ByMapName = new Dictionary<int, Dictionary<string, List<int>>>();
            foreach (var id in stor335_Area.Keys)
            {
                var rec = Extract335(id);
                var nameN = NormName(rec.name);
                if (!idx335_ByName.TryGetValue(nameN, out var ln)) { ln = new List<int>(); idx335_ByName[nameN] = ln; }
                ln.Add(id);
                if (!idx335_ByMap.TryGetValue(rec.mapId, out var lm)) { lm = new List<int>(); idx335_ByMap[rec.mapId] = lm; }
                lm.Add(id);
                if (!idx335_ByMapName.TryGetValue(rec.mapId, out var dict)) { dict = new Dictionary<string, List<int>>(); idx335_ByMapName[rec.mapId] = dict; }
                if (!dict.TryGetValue(nameN, out var lnm)) { lnm = new List<int>(); dict[nameN] = lnm; }
                lnm.Add(id);
            }

            (int id, string method) ChooseTargetByName(string srcName, int mapIdX, bool requireMap, bool topLevelOnly)
            {
                var variants = NameVariants(srcName).Select(NormName).Distinct().ToList();
                if (variants.Count == 0) variants = new List<string> { NormName(srcName) };

                bool Accept(int tid)
                {
                    var rec = Extract335(tid);
                    if (DisallowCandidate(rec.name)) return false;
                    if (requireMap && rec.mapId != mapIdX) return false;
                    if (topLevelOnly && rec.parentId != tid) return false; // only zones (parent==self)
                    return true;
                }

                // 1) Exact matches
                if (requireMap)
                {
                    if (mapIdX >= 0 && idx335_ByMapName.TryGetValue(mapIdX, out var byNameMap))
                    {
                        foreach (var v in variants)
                            if (byNameMap.TryGetValue(v, out var lm) && lm.Count > 0)
                            {
                                var lmOk = lm.Where(Accept).ToList();
                                if (lmOk.Count > 0) return (lmOk.Min(), "name+map");
                            }
                    }
                }
                else
                {
                    // map-biased exact
                    if (mapIdX >= 0 && idx335_ByMapName.TryGetValue(mapIdX, out var byNameMap))
                    {
                        foreach (var v in variants)
                            if (byNameMap.TryGetValue(v, out var lm) && lm.Count > 0)
                            {
                                var lmOk = lm.Where(Accept).ToList();
                                if (lmOk.Count > 0) return (lmOk.Min(), "name+map");
                            }
                    }
                    // global exact
                    foreach (var v in variants)
                        if (idx335_ByName.TryGetValue(v, out var lg) && lg.Count > 0)
                        {
                            var lgOk = lg.Where(Accept).ToList();
                            if (lgOk.Count > 0) return (lgOk.Min(), "name");
                        }
                }

                // 2) Fuzzy search (map-only or global depending on requireMap)
                IEnumerable<int> domain;
                if (requireMap)
                {
                    if (!(mapIdX >= 0 && idx335_ByMap.TryGetValue(mapIdX, out var lm2))) return (-1, "unmatched");
                    domain = lm2;
                }
                else
                {
                    domain = (mapIdX >= 0 && idx335_ByMap.TryGetValue(mapIdX, out var lm2)) ? lm2 : idx335_IdToRow.Keys;
                }

                int bestId = -1; int bestDist = int.MaxValue;
                foreach (var tid in domain)
                {
                    if (!Accept(tid)) continue;
                    var rec = Extract335(tid);
                    var rn = NormName(rec.name);
                    foreach (var v in variants)
                    {
                        var dn = EditDistance(v, rn);
                        if (dn <= 3 && (dn < bestDist || (dn == bestDist && tid < bestId)))
                        {
                            bestDist = dn; bestId = tid;
                        }
                    }
                }
                if (bestId >= 0) return (bestId, requireMap ? "fuzzy+map" : (mapIdX >= 0 ? "fuzzy+map" : "fuzzy"));

                // 3) Only allow global fuzzy when not requiring map
                if (!requireMap)
                {
                    int gBestId = -1; int gBestDist = int.MaxValue;
                    foreach (var tid in idx335_IdToRow.Keys)
                    {
                        if (!Accept(tid)) continue;
                        var rec = Extract335(tid);
                        var rn = NormName(rec.name);
                        foreach (var v in variants)
                        {
                            var dn = EditDistance(v, rn);
                            if (dn <= 3 && (dn < gBestDist || (dn == gBestDist && tid < gBestId)))
                            {
                                gBestDist = dn; gBestId = tid;
                            }
                        }
                    }
                    if (gBestId >= 0) return (gBestId, "fuzzy-global");
                }

                return (-1, "unmatched");
            }

            // Choose a LK sub-area within a specific LK zone (by ParentAreaID), constrained to same map
            (int id, string method) ChooseSubWithinZone(string subName, int zoneId)
            {
                if (zoneId < 0) return (-1, "unmatched");
                var zoneRec = Extract335(zoneId);
                var baseNorm = NormName(subName);
                var variants = NameVariants(subName).Select(NormName).Distinct().ToList();
                if (variants.Count == 0) variants = new List<string> { baseNorm };

                // Exact name (including alias/variant) within the same zone and same map
                foreach (var v in variants)
                {
                    if (idx335_ByName.TryGetValue(v, out var list) && list.Count > 0)
                    {
                        var kids = list.Where(tid => {
                            var rec = Extract335(tid);
                            return rec.parentId == zoneId && rec.mapId == zoneRec.mapId;
                        }).ToList();
                        if (kids.Count > 0) return (kids.Min(), v == baseNorm ? "name" : "name_alias");
                    }
                }

                // Fuzzy among children of the selected zone (and same map)
                int bestId = -1; int bestDist = int.MaxValue;
                foreach (var tid in idx335_IdToRow.Keys)
                {
                    var rec = Extract335(tid);
                    if (rec.parentId != zoneId) continue;
                    if (rec.mapId != zoneRec.mapId) continue;
                    var dn = EditDistance(baseNorm, NormName(rec.name));
                    if (dn <= 3 && (dn < bestDist || (dn == bestDist && tid < bestId)))
                    {
                        bestDist = dn; bestId = tid;
                    }
                }
                if (bestId >= 0) return (bestId, "fuzzy");
                return (-1, "unmatched");
            }

            // Rename suggestions report for unmatched rows (top-3 by edit distance, data-driven from storages)
            var suggestCsv = new StringBuilder();
            suggestCsv.AppendLine("src_areaNumber,src_name,src_mapId_xwalk,src_mapName_xwalk,cand1_id,cand1_name,cand1_dist,cand2_id,cand2_name,cand2_dist,cand3_id,cand3_name,cand3_dist");

            // Map rows 0.5.x → 3.3.5
            var mapping = new StringBuilder();
            var unmatchedCsv = new StringBuilder();
            var patchCsv = new StringBuilder();
            patchCsv.AppendLine("src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name");
            var header = string.Join(',', new[]
            {
                "src_row_id","src_areaNumber","src_parentNumber","src_name","src_mapId","src_mapName","src_mapId_xwalk","src_mapName_xwalk","src_path",
                "tgt_id_335","tgt_name","tgt_parent_id","tgt_parent_name","tgt_mapId","tgt_mapName","tgt_path","match_method"
            });
            mapping.AppendLine(header);
            unmatchedCsv.AppendLine(header);

            int byName = 0, unmatched = 0, skippedDev = 0;
            var exportExplicit = new List<RemapDefinition.ExplicitMap>();
            var ignoredAreas = new List<int>();
            var mappingByMap = new Dictionary<int, StringBuilder>();
            var unmatchedByMap = new Dictionary<int, StringBuilder>();
            var patchByMap = new Dictionary<int, StringBuilder>();
            foreach (var id in stor053_Area.Keys)
            {
                var src = Extract053(stor053_Area[id]);
                int raw = src.areaNum;
                if (raw == 0)
                {
                    continue;
                }

                var nn = NormName(src.name);
                if (nn.Contains("***on map dungeon***") || nn.Contains("programmer isle") || nn.Contains("plains of snow") || (nn.Contains("jeff") && nn.Contains("quadrant")))
                {
                    skippedDev++; ignoredAreas.Add(src.areaNum); continue;
                }

                // Collision-resistant selection (map-locked + parent-anchored)
                int chosen = -1; string method = string.Empty;
                // Apply explicit mapping if provided
                if (explicitMap.TryGetValue(src.areaNum, out var expId)) { chosen = expId; method = "explicit"; }
                else
                {
                    if (src.parentNum == src.areaNum)
                    {
                        // Top-level zone: map-locked and top-level-only
                        (chosen, method) = ChooseTargetByName(src.parentNameForInfer, src.mapIdX, requireMap: true, topLevelOnly: true);
                    }
                    else
                    {
                        // Sub-area: resolve parent zone first on same map, then sub within that zone
                        var (zoneChosen, zMethod) = ChooseTargetByName(src.parentNameForInfer, src.mapIdX, requireMap: true, topLevelOnly: true);
                        if (zoneChosen >= 0)
                        {
                            var sub = ChooseSubWithinZone(src.name, zoneChosen);
                            chosen = sub.id;
                            method = $"zone:{zMethod}:sub({sub.method})";
                            if (chosen < 0) method = $"zone:{zMethod}:zone_only_no_sub";
                        }
                        else
                        {
                            chosen = -1;
                            method = "unmatched";
                        }
                    }
                }
                // Cross-map guard: demote to unmatched if selected target map differs from expected map
                if (chosen >= 0 && src.mapIdX >= 0)
                {
                    var recSel = Extract335(chosen);
                    if (recSel.mapId != src.mapIdX)
                    {
                        method = string.Equals(method, "explicit", StringComparison.OrdinalIgnoreCase) ? "explicit_cross_map" : "cross_map_violation";
                        chosen = -1;
                    }
                }

                var mappingLine = string.Join(',', new[]
                {
                    id.ToString(CultureInfo.InvariantCulture),
                    src.areaNum.ToString(CultureInfo.InvariantCulture),
                    src.parentNum.ToString(CultureInfo.InvariantCulture),
                    Csv(src.name),
                    src.mapId.ToString(CultureInfo.InvariantCulture),
                    Csv(src.mapName),
                    src.mapIdX.ToString(CultureInfo.InvariantCulture),
                    Csv(src.mapNameX),
                    Csv(src.path),
                    (chosen >= 0 ? chosen.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(chosen >= 0 ? Extract335(chosen).name : string.Empty),
                    (chosen >= 0 ? Extract335(chosen).parentId.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(chosen >= 0 ? Extract335(chosen).parentName : string.Empty),
                    (chosen >= 0 ? Extract335(chosen).mapId.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(chosen >= 0 ? Extract335(chosen).mapName : string.Empty),
                    Csv(chosen >= 0 ? Extract335(chosen).path : string.Empty),
                    method
                });
                mapping.AppendLine(mappingLine);
                if (src.mapIdX >= 0)
                {
                    if (!mappingByMap.TryGetValue(src.mapIdX, out var sbMap))
                    {
                        sbMap = new StringBuilder();
                        sbMap.AppendLine(header);
                        mappingByMap[src.mapIdX] = sbMap;
                    }
                    sbMap.AppendLine(mappingLine);
                }

                if (string.Equals(method, "explicit", StringComparison.OrdinalIgnoreCase))
                    exportExplicit.Add(new RemapDefinition.ExplicitMap { src_areaNumber = src.areaNum, tgt_areaID = chosen, note = method });
                if (chosen >= 0)
                {
                    var trecPatch = Extract335(chosen);
                    // Determine target parent for patch from mapping of source parent
                    int tgtParentForPatch = chosen;
                    if (src.parentNum != src.areaNum)
                    {
                        // Optional: if parent is mapped, use that as parent for patch
                        tgtParentForPatch = chosen; // keep simple
                    }
                    var patchLine = string.Join(',', new[]
                    {
                        src.mapId.ToString(CultureInfo.InvariantCulture),
                        Csv(src.mapName),
                        src.areaNum.ToString(CultureInfo.InvariantCulture),
                        src.parentNum.ToString(CultureInfo.InvariantCulture),
                        Csv(src.name),
                        src.mapIdX.ToString(CultureInfo.InvariantCulture),
                        Csv(src.mapNameX),
                        chosen.ToString(CultureInfo.InvariantCulture),
                        tgtParentForPatch.ToString(CultureInfo.InvariantCulture),
                        Csv(trecPatch.name)
                    });
                    patchCsv.AppendLine(patchLine);
                    if (src.mapIdX >= 0)
                    {
                        if (!patchByMap.TryGetValue(src.mapIdX, out var sbPatch))
                        {
                            sbPatch = new StringBuilder();
                            sbPatch.AppendLine("src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name");
                            patchByMap[src.mapIdX] = sbPatch;
                        }
                        sbPatch.AppendLine(patchLine);
                    }
                }

                var unmatchedLine = string.Join(',', new[]
                {
                    id.ToString(CultureInfo.InvariantCulture),
                    src.areaNum.ToString(CultureInfo.InvariantCulture),
                    src.parentNum.ToString(CultureInfo.InvariantCulture),
                    Csv(src.name),
                    src.mapId.ToString(CultureInfo.InvariantCulture),
                    Csv(src.mapName),
                    src.mapIdX.ToString(CultureInfo.InvariantCulture),
                    Csv(src.mapNameX),
                    Csv(src.path),
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    string.Empty,
                    method // carry method (e.g., unmatched, zone_only_no_sub, cross_map_violation, explicit_cross_map)
                });
                unmatchedCsv.AppendLine(unmatchedLine);
                if (src.mapIdX >= 0)
                {
                    if (!unmatchedByMap.TryGetValue(src.mapIdX, out var sbUn))
                    {
                        sbUn = new StringBuilder();
                        sbUn.AppendLine(header);
                        unmatchedByMap[src.mapIdX] = sbUn;
                    }
                    sbUn.AppendLine(unmatchedLine);
                }

                if (chosen >= 0) byName++;
                else
                {
                    unmatched++;
                    // Record top-3 rename suggestions for unmatched (global domain, by edit distance)
                    var srcNameN = NormName(src.name);
                    var cands = idx335_IdToRow.Keys
                        .Select(tid => { var r = Extract335(tid); return new { tid, nm = r.name, dist = EditDistance(srcNameN, NormName(r.name)) }; })
                        .OrderBy(x => x.dist).ThenBy(x => x.tid).Take(3).ToList();
                    string c1 = (cands.Count > 0 ? cands[0].tid.ToString(CultureInfo.InvariantCulture) : "");
                    string n1 = (cands.Count > 0 ? Csv(cands[0].nm ?? string.Empty) : "");
                    string d1 = (cands.Count > 0 ? cands[0].dist.ToString(CultureInfo.InvariantCulture) : "");
                    string c2 = (cands.Count > 1 ? cands[1].tid.ToString(CultureInfo.InvariantCulture) : "");
                    string n2 = (cands.Count > 1 ? Csv(cands[1].nm ?? string.Empty) : "");
                    string d2 = (cands.Count > 1 ? cands[1].dist.ToString(CultureInfo.InvariantCulture) : "");
                    string c3 = (cands.Count > 2 ? cands[2].tid.ToString(CultureInfo.InvariantCulture) : "");
                    string n3 = (cands.Count > 2 ? Csv(cands[2].nm ?? string.Empty) : "");
                    string d3 = (cands.Count > 2 ? cands[2].dist.ToString(CultureInfo.InvariantCulture) : "");
                    suggestCsv.AppendLine(string.Join(',', new[]
                    {
                        src.areaNum.ToString(CultureInfo.InvariantCulture),
                        Csv(src.name),
                        src.mapIdX.ToString(CultureInfo.InvariantCulture),
                        Csv(src.mapNameX),
                        c1,n1,d1,c2,n2,d2,c3,n3,d3
                    }));
                }
            }

            // Alpha decode: build strict hi16/lo16 mapping with parent validation
            var alphaDecode = new StringBuilder();
            alphaDecode.AppendLine("alpha_raw,alpha_raw_hex,zone_num,zone_name_alpha,sub_num,sub_name_alpha,parent_ok,alpha_continent");
            var seenAlpha = new HashSet<int>();
            foreach (var id in stor053_Area.Keys)
            {
                var src = Extract053(stor053_Area[id]);
                int raw = src.areaNum;
                if (!seenAlpha.Add(raw)) continue;
                int z = (raw >> 16) & 0xFFFF;
                int s = raw & 0xFFFF;
                int zoneBase = (z << 16);
                string zoneNameAlpha = src.name;
                if (idx053_NumToRow.TryGetValue(zoneBase, out var zrow))
                    zoneNameAlpha = SafeField<string>(zrow, areaNameCol053) ?? zoneNameAlpha;
                string subNameAlpha = (s == 0 ? string.Empty : src.name);
                bool parentOk = (s == 0) || (src.parentNum == zoneBase);
                int alphaCont = src.mapIdX >= 0 ? src.mapIdX : src.mapId;
                alphaDecode.AppendLine(string.Join(',', new[]
                {
                    raw.ToString(CultureInfo.InvariantCulture),
                    $"0x{raw:X8}",
                    z.ToString(CultureInfo.InvariantCulture),
                    Csv(zoneNameAlpha ?? string.Empty),
                    s.ToString(CultureInfo.InvariantCulture),
                    Csv(subNameAlpha ?? string.Empty),
                    parentOk ? "1" : "0",
                    alphaCont.ToString(CultureInfo.InvariantCulture)
                }));
            }

            // Alpha → LK suggestions using name + map bias; sub constrained within chosen zone
            string SimplifyMethod(string m)
            {
                if (string.Equals(m, "unmatched", StringComparison.OrdinalIgnoreCase)) return "unmatched";
                return (m?.IndexOf("map", StringComparison.OrdinalIgnoreCase) >= 0) ? "map_biased" : "global";
            }

            var alphaSuggest = new StringBuilder();
            alphaSuggest.AppendLine("alpha_raw,alpha_raw_hex,zone_num,zone_name_alpha,sub_num,sub_name_alpha,alpha_continent,lk_zone_id_suggested,lk_zone_name,lk_sub_id_suggested,lk_sub_name,method");
            var seenAlphaSug = new HashSet<int>();
            foreach (var id in stor053_Area.Keys)
            {
                var src = Extract053(stor053_Area[id]);
                int raw = src.areaNum;
                if (!seenAlphaSug.Add(raw)) continue;
                if (raw == 0)
                {
                    continue;
                }
                int z = (raw >> 16) & 0xFFFF;
                int s = raw & 0xFFFF;
                int zoneBase = (z << 16);
                string zoneNameAlpha = src.name;
                if (idx053_NumToRow.TryGetValue(zoneBase, out var zrow))
                    zoneNameAlpha = SafeField<string>(zrow, areaNameCol053) ?? zoneNameAlpha;
                string subNameAlpha = (s == 0 ? string.Empty : src.name);
                int alphaCont = src.mapIdX >= 0 ? src.mapIdX : src.mapId;

                var (zoneChosen, zMethodRaw) = ChooseTargetByName(zoneNameAlpha, src.mapIdX, requireMap: true, topLevelOnly: true);
                string zMethod = SimplifyMethod(zMethodRaw);

                int subChosen = -1; string subMethod = string.Empty; string outMethod = zMethod;
                if (s != 0)
                {
                    (subChosen, subMethod) = ChooseSubWithinZone(subNameAlpha, zoneChosen);
                    if (subChosen < 0) outMethod = outMethod + ":fallback_to_zone";
                }

                string lkZoneName = zoneChosen >= 0 ? Extract335(zoneChosen).name : string.Empty;
                string lkSubName = subChosen >= 0 ? Extract335(subChosen).name : string.Empty;

                alphaSuggest.AppendLine(string.Join(',', new[]
                {
                    raw.ToString(CultureInfo.InvariantCulture),
                    $"0x{raw:X8}",
                    z.ToString(CultureInfo.InvariantCulture),
                    Csv(zoneNameAlpha ?? string.Empty),
                    s.ToString(CultureInfo.InvariantCulture),
                    Csv(subNameAlpha ?? string.Empty),
                    alphaCont.ToString(CultureInfo.InvariantCulture),
                    zoneChosen.ToString(CultureInfo.InvariantCulture),
                    Csv(lkZoneName ?? string.Empty),
                    subChosen.ToString(CultureInfo.InvariantCulture),
                    Csv(lkSubName ?? string.Empty),
                    outMethod
                }));
            }

            // Write per-map outputs (flat files: map{ID} suffix)
            foreach (var kv in mappingByMap)
            {
                var mapOutFlat = Path.Combine(outDir, $"AreaTable_mapping_map{kv.Key}_{srcAlias}_to_335.csv");
                File.WriteAllText(mapOutFlat, kv.Value.ToString(), new UTF8Encoding(true));
            }
            foreach (var kv in unmatchedByMap)
            {
                var unOutFlat = Path.Combine(outDir, $"AreaTable_unmatched_map{kv.Key}_{srcAlias}_to_335.csv");
                File.WriteAllText(unOutFlat, kv.Value.ToString(), new UTF8Encoding(true));
            }
            foreach (var kv in patchByMap)
            {
                var patchOutFlat = Path.Combine(outDir, $"Area_patch_crosswalk_map{kv.Key}_{srcAlias}_to_335.csv");
                File.WriteAllText(patchOutFlat, kv.Value.ToString(), new UTF8Encoding(true));
            }

            // Write Area outputs
            var mapOut = Path.Combine(outDir, $"AreaTable_mapping_{srcAlias}_to_335.csv");
            File.WriteAllText(mapOut, mapping.ToString(), new UTF8Encoding(true));
            var unOut = Path.Combine(outDir, $"AreaTable_unmatched_{srcAlias}_to_335.csv");
            File.WriteAllText(unOut, unmatchedCsv.ToString(), new UTF8Encoding(true));
            var patchOut = Path.Combine(outDir, $"Area_patch_crosswalk_{srcAlias}_to_335.csv");
            File.WriteAllText(patchOut, patchCsv.ToString(), new UTF8Encoding(true));
            var suggestOut = Path.Combine(outDir, $"AreaTable_rename_suggestions_{srcAlias}_to_335.csv");
            File.WriteAllText(suggestOut, suggestCsv.ToString(), new UTF8Encoding(true));

            // Write Alpha decode/suggestion CSVs
            var alphaDecodeOut = Path.Combine(outDir, "alpha_areaid_decode.csv");
            File.WriteAllText(alphaDecodeOut, alphaDecode.ToString(), new UTF8Encoding(true));
            var alphaTo335Out = Path.Combine(outDir, "alpha_to_335_suggestions.csv");
            File.WriteAllText(alphaTo335Out, alphaSuggest.ToString(), new UTF8Encoding(true));

            // Write Map crosswalk CSV
            var srcTagShort = srcAlias == "0.5.5" ? "055" : (srcAlias == "0.6.0" ? "060" : "053");
            var cwOut = Path.Combine(outDir, $"Map_crosswalk_{srcTagShort}_to_335.csv");
            File.WriteAllText(cwOut, cwReport.ToString(), new UTF8Encoding(true));

            // Build MapId → Name reports and write CSVs using real MapIDs
            var map053 = BuildMapNames(stor053_Map, mapNameCol053);
            var map335 = BuildMapNames(stor335_Map, mapNameCol335);
            WriteSimpleMapReport(Path.Combine(outDir, $"MapId_to_Name_{srcAlias}.csv"), map053);
            WriteSimpleMapReport(Path.Combine(outDir, "MapId_to_Name_3.3.5.csv"), map335);

            // Export remap definition if requested
            if (!string.IsNullOrWhiteSpace(exportRemap))
            {
                try
                {
                    var remapDir = Path.GetDirectoryName(exportRemap);
                    if (!string.IsNullOrWhiteSpace(remapDir) && !Directory.Exists(remapDir)) Directory.CreateDirectory(remapDir);
                    var def = new RemapDefinition
                    {
                        meta = new RemapDefinition.Meta
                        {
                            src_alias = srcAlias,
                            src_build = canonicalSrcBuild,
                            tgt_build = canonicalTgtBuild,
                            generated_at = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture)
                        },
                        aliases = aliasMap,
                        explicit_map = exportExplicit,
                        ignore_area_numbers = ignoredAreas,
                        options = new RemapDefinition.Options { disallow_do_not_use_targets = disallowDoNotUse }
                    };
                    var jsonOpts = new JsonSerializerOptions { WriteIndented = true };
                    File.WriteAllText(exportRemap, JsonSerializer.Serialize(def, jsonOpts), new UTF8Encoding(true));
                    Console.WriteLine($"[Compare] Wrote {exportRemap}");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[Remap] Failed to write {exportRemap}: {ex.Message}");
                }
            }

            Console.WriteLine($"[Compare] Wrote {mapOut}");
            Console.WriteLine($"[Compare] Wrote {unOut}");
            Console.WriteLine($"[Compare] Wrote {patchOut}");
            Console.WriteLine($"[Compare] Wrote {suggestOut}");
            Console.WriteLine($"[Compare] Wrote {alphaDecodeOut}");
            Console.WriteLine($"[Compare] Wrote {alphaTo335Out}");
            Console.WriteLine($"[Compare] Wrote {cwOut}");
            Console.WriteLine($"[Compare] Wrote MapId_to_Name CSVs");

            Console.WriteLine("[Compare] Done.");
            return 0;
        }

        // V2: Strict map-locked, parent-anchored exact matching (scaffold)
        private static int CompareAreasV2(string dbdDir, string outBase, DBCD.Locale locale, List<(string build, string dir)> inputs, string buildOverride, string srcAliasFlag, string srcBuildFlag, string tgtBuildFlag, bool disallowDoNotUse)
        {
            // Normalize inputs and pick source (0.5.3/0.5.5/0.6.0) and 3.3.5
            string? dir053 = null, dir055 = null, dir060 = null, dir335 = null;
            foreach (var (build, dir) in inputs)
            {
                var alias = ResolveAliasOrInfer(build, dir, buildOverride);
                if (alias == "0.5.3") dir053 = NormalizePath(dir);
                else if (alias == "0.5.5") dir055 = NormalizePath(dir);
                else if (alias == "0.6.0") dir060 = NormalizePath(dir);
                else if (alias == "3.3.5") dir335 = NormalizePath(dir);
            }
            string srcAlias = ResolveAlias(srcAliasFlag);
            if (string.IsNullOrWhiteSpace(srcAlias)) srcAlias = !string.IsNullOrEmpty(dir053) ? "0.5.3" : (!string.IsNullOrEmpty(dir055) ? "0.5.5" : (!string.IsNullOrEmpty(dir060) ? "0.6.0" : "0.5.3"));
            string? dirSrc = srcAlias == "0.5.3" ? dir053 : (srcAlias == "0.5.5" ? dir055 : (srcAlias == "0.6.0" ? dir060 : null));
            if (string.IsNullOrEmpty(dirSrc) || string.IsNullOrEmpty(dir335))
            {
                Console.Error.WriteLine("ERROR: --compare-area-v2 requires both a source (0.5.3/0.5.5/0.6.0) and 3.3.5 inputs.");
                return 2;
            }

            var outDir = Path.Combine(outBase, "compare", "v2");
            Directory.CreateDirectory(outDir);

            var dbdProvider = new FilesystemDBDProvider(dbdDir);
            string canonicalSrcBuild = !string.IsNullOrWhiteSpace(srcBuildFlag) ? srcBuildFlag : CanonicalizeBuild(srcAlias);
            string canonicalTgtBuild = !string.IsNullOrWhiteSpace(tgtBuildFlag) ? tgtBuildFlag : CanonicalizeBuild("3.3.5");
            var storSrc_Area = LoadTable("AreaTable", canonicalSrcBuild, dirSrc!, dbdProvider, locale);
            var storSrc_Map  = LoadTable("Map",       canonicalSrcBuild, dirSrc!, dbdProvider, locale);
            var storTgt_Area = LoadTable("AreaTable", canonicalTgtBuild, dir335!, dbdProvider, locale);
            var storTgt_Map  = LoadTable("Map",       canonicalTgtBuild, dir335!, dbdProvider, locale);

            // Helpers (local to V2)
            string Norm(string s) => (s ?? string.Empty).Trim().ToLowerInvariant();
            string DirTok(string s)
            {
                if (string.IsNullOrWhiteSpace(s)) return string.Empty;
                var parts = s.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar).Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                return parts.Length > 0 ? parts[^1] : s.Trim();
            }

            // Detect columns for names/IDs
            string idColMapSrc = DetectIdColumn(storSrc_Map);
            string idColMapTgt = DetectIdColumn(storTgt_Map);
            string idColAreaTgt = DetectIdColumn(storTgt_Area);
            string areaNameColSrc = FirstNonEmpty("AreaName_lang", "AreaName", "Name");
            string areaNameColTgt = FirstNonEmpty("AreaName_lang", "AreaName", "Name");

            // Build MapID→row (real IDs)
            var srcMapById = new Dictionary<int, DBCDRow>();
            foreach (var k in storSrc_Map.Keys)
            {
                var row = storSrc_Map[k];
                int id = !string.IsNullOrWhiteSpace(idColMapSrc) ? SafeField<int>(row, idColMapSrc) : k;
                if (!srcMapById.ContainsKey(id)) srcMapById[id] = row;
            }
            var tgtMapById = new Dictionary<int, DBCDRow>();
            foreach (var k in storTgt_Map.Keys)
            {
                var row = storTgt_Map[k];
                int id = !string.IsNullOrWhiteSpace(idColMapTgt) ? SafeField<int>(row, idColMapTgt) : k;
                if (!tgtMapById.ContainsKey(id)) tgtMapById[id] = row;
            }

            // Crosswalk 0.5.x → 3.3.5 maps (directory/name)
            var dir335Index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var name335Index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var mid in tgtMapById.Keys)
            {
                var mrow = tgtMapById[mid];
                var dirTok = DirTok(SafeField<string>(mrow, "Directory"));
                if (!string.IsNullOrWhiteSpace(dirTok) && !dir335Index.ContainsKey(dirTok)) dir335Index[dirTok] = mid;
                var nm = Norm(FirstNonEmpty(SafeField<string>(mrow, "MapName_lang"), SafeField<string>(mrow, "MapName"), SafeField<string>(mrow, "InternalName"), dirTok));
                if (!string.IsNullOrWhiteSpace(nm) && !name335Index.ContainsKey(nm)) name335Index[nm] = mid;
            }
            var cw053To335 = new Dictionary<int, int>();
            foreach (var mid in srcMapById.Keys)
            {
                var srow = srcMapById[mid];
                var sDir = DirTok(SafeField<string>(srow, "Directory"));
                var sName = FirstNonEmpty(SafeField<string>(srow, "MapName_lang"), SafeField<string>(srow, "MapName"), SafeField<string>(srow, "InternalName"), sDir) ?? string.Empty;
                int tgt = -1;
                if (!string.IsNullOrWhiteSpace(sDir) && dir335Index.TryGetValue(sDir, out var byDir)) tgt = byDir;
                else
                {
                    var key = Norm(sName);
                    if (!string.IsNullOrWhiteSpace(key) && name335Index.TryGetValue(key, out var byName)) tgt = byName;
                }
                if (tgt >= 0) cw053To335[mid] = tgt;
            }

            // Build 3.3.5 Area indices
            var idxTgtTopZonesByMap = new Dictionary<int, Dictionary<string, int>>(); // mapId -> normName -> areaID
            var idxTgtChildrenByZone = new Dictionary<int, Dictionary<string, int>>(); // parentAreaID -> normName -> areaID
            foreach (var key in storTgt_Area.Keys)
            {
                var row = storTgt_Area[key];
                int id = !string.IsNullOrWhiteSpace(idColAreaTgt) ? SafeField<int>(row, idColAreaTgt) : key;
                string name = FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;
                int parentId = SafeField<int>(row, "ParentAreaID");
                if (parentId <= 0) parentId = id;
                int mapId = SafeField<int>(row, "ContinentID");
                if (parentId == id)
                {
                    if (!idxTgtTopZonesByMap.TryGetValue(mapId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idxTgtTopZonesByMap[mapId] = dict; }
                    dict[Norm(name)] = id;
                }
                else
                {
                    if (!idxTgtChildrenByZone.TryGetValue(parentId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idxTgtChildrenByZone[parentId] = dict; }
                    dict[Norm(name)] = id;
                }
            }

            // Source Area index for parent name lookup (AreaNumber) and map inference (ContinentID crosswalk)
            var idxSrcNumToRow = new Dictionary<int, DBCDRow>();
            foreach (var sid in storSrc_Area.Keys)
            {
                var srow = storSrc_Area[sid];
                int areaKey = SafeField<int>(srow, (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID");
                if (areaKey > 0 && !idxSrcNumToRow.ContainsKey(areaKey)) idxSrcNumToRow[areaKey] = srow;
            }

            // Outputs
            var mapping = new StringBuilder();
            var unmatched = new StringBuilder();
            var patch = new StringBuilder();
            var header = string.Join(',', new[]
            {
                "src_row_id","src_areaNumber","src_parentNumber","src_name","src_mapId","src_mapName","src_mapId_xwalk","src_mapName_xwalk","src_path",
                "tgt_id_335","tgt_name","tgt_parent_id","tgt_parent_name","tgt_mapId","tgt_mapName","tgt_path","match_method"
            });
            mapping.AppendLine(header);
            unmatched.AppendLine(header);
            var perMap = new Dictionary<int, (StringBuilder map, StringBuilder un, StringBuilder patch)>();
            patch.AppendLine("src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name");

            // Build map name lookup
            var mapSrcNames = BuildMapNames(storSrc_Map, FirstNonEmpty("MapName_lang", "MapName", "InternalName", "Directory"));
            var mapTgtNames = BuildMapNames(storTgt_Map, FirstNonEmpty("MapName_lang", "MapName", "InternalName", "Directory"));

            foreach (var key in storSrc_Area.Keys)
            {
                var row = storSrc_Area[key];
                string nm = FirstNonEmpty(SafeField<string>(row, areaNameColSrc)) ?? string.Empty;
                int areaNum = SafeField<int>(row, (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID");
                int parentNum = SafeField<int>(row, (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "ParentAreaNum" : "ParentAreaID");
                if (parentNum <= 0) parentNum = areaNum;
                int cont = SafeField<int>(row, "ContinentID");
                string mapName = srcMapById.TryGetValue(cont, out var sm) ? (FirstNonEmpty(SafeField<string>(sm, "MapName_lang"), SafeField<string>(sm, "MapName"), SafeField<string>(sm, "InternalName"), DirTok(SafeField<string>(sm, "Directory"))) ?? string.Empty) : string.Empty;
                var hasMapX = cw053To335.TryGetValue(cont, out var mapIdX);
                string mapNameX = hasMapX && tgtMapById.TryGetValue(mapIdX, out var tm) ? (FirstNonEmpty(SafeField<string>(tm, "MapName_lang"), SafeField<string>(tm, "MapName"), SafeField<string>(tm, "InternalName"), DirTok(SafeField<string>(tm, "Directory"))) ?? string.Empty) : string.Empty;
                string parentName = nm;
                if (parentNum != areaNum && idxSrcNumToRow.TryGetValue(parentNum, out var pRow)) parentName = FirstNonEmpty(SafeField<string>(pRow, areaNameColSrc)) ?? parentName;
                string path = (!string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/" : string.Empty) + $"{Norm(parentName)}/{Norm(nm)}";

                int chosen = -1; string method = string.Empty; int tgtParentId = -1; string tgtName = string.Empty; string tgtParentName = string.Empty; int tgtMap = -1; string tgtMapName = string.Empty; string tgtPath = string.Empty;
                if (hasMapX)
                {
                    // Zone vs Sub
                    bool isZone = (parentNum == areaNum);
                    if (isZone)
                    {
                        if (idxTgtTopZonesByMap.TryGetValue(mapIdX, out var zones) && zones.TryGetValue(Norm(nm), out var zId))
                        {
                            chosen = zId; method = "v2:zone:exact";
                        }
                    }
                    else
                    {
                        int zoneId = -1;
                        if (idxTgtTopZonesByMap.TryGetValue(mapIdX, out var zones) && zones.TryGetValue(Norm(parentName), out var zId)) zoneId = zId;
                        if (zoneId >= 0 && idxTgtChildrenByZone.TryGetValue(zoneId, out var kids) && kids.TryGetValue(Norm(nm), out var sId))
                        {
                            chosen = sId; method = "v2:sub:exact";
                        }
                        else if (zoneId >= 0)
                        {
                            method = "v2:zone_only_no_sub";
                        }
                    }
                }

                if (chosen >= 0)
                {
                    // Extract tgt details
                    var tRow = storTgt_Area[storTgt_Area.Keys.First(k => {
                        try { return (!string.IsNullOrWhiteSpace(idColAreaTgt) ? SafeField<int>(storTgt_Area[k], idColAreaTgt) : k) == chosen; } catch { return false; }
                    })];
                    tgtName = FirstNonEmpty(SafeField<string>(tRow, areaNameColTgt)) ?? string.Empty;
                    tgtParentId = SafeField<int>(tRow, "ParentAreaID"); if (tgtParentId <= 0) tgtParentId = chosen;
                    tgtMap = SafeField<int>(tRow, "ContinentID");
                    tgtMapName = mapTgtNames.TryGetValue(tgtMap, out var mn) ? mn : string.Empty;
                    tgtParentName = tgtParentId == chosen ? tgtName : (storTgt_Area.Keys.Select(k => storTgt_Area[k]).Select(r => (r, id: (!string.IsNullOrWhiteSpace(idColAreaTgt) ? SafeField<int>(r, idColAreaTgt) : 0))).Where(x => x.id == tgtParentId).Select(x => FirstNonEmpty(SafeField<string>(x.r, areaNameColTgt))).FirstOrDefault() ?? string.Empty);
                    tgtPath = $"{Norm(tgtMapName)}/{Norm(tgtParentName)}/{Norm(tgtName)}";
                }

                var line = string.Join(',', new[]
                {
                    key.ToString(CultureInfo.InvariantCulture),
                    areaNum.ToString(CultureInfo.InvariantCulture),
                    parentNum.ToString(CultureInfo.InvariantCulture),
                    Csv(nm),
                    cont.ToString(CultureInfo.InvariantCulture),
                    Csv(mapName),
                    hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                    Csv(mapNameX),
                    Csv(path),
                    (chosen >= 0 ? chosen.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(tgtName),
                    tgtParentId.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtParentName),
                    tgtMap.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtMapName),
                    Csv(tgtPath),
                    (string.IsNullOrWhiteSpace(method) ? "unmatched" : method)
                });
                if (chosen >= 0) mapping.AppendLine(line); else unmatched.AppendLine(line);
                if (hasMapX)
                {
                    if (!perMap.TryGetValue(mapIdX, out var tuple)) { tuple = (new StringBuilder(), new StringBuilder(), new StringBuilder()); tuple.map.AppendLine(header); tuple.un.AppendLine(header); tuple.patch.AppendLine("src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name"); perMap[mapIdX] = tuple; }
                    if (chosen >= 0) tuple.map.AppendLine(line); else tuple.un.AppendLine(line);
                    if (chosen >= 0)
                    {
                        tuple.patch.AppendLine(string.Join(',', new[]
                        {
                            cont.ToString(CultureInfo.InvariantCulture),
                            Csv(mapName),
                            areaNum.ToString(CultureInfo.InvariantCulture),
                            parentNum.ToString(CultureInfo.InvariantCulture),
                            Csv(nm),
                            hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                            Csv(mapNameX),
                            chosen.ToString(CultureInfo.InvariantCulture),
                            (chosen >= 0 ? (tgtParentId.ToString(CultureInfo.InvariantCulture)) : "-1"),
                            Csv(tgtName)
                        }));
                    }
                }
            }

            // Write outputs under v2 folder
            File.WriteAllText(Path.Combine(outDir, $"AreaTable_mapping_{srcAlias}_to_335.csv"), mapping.ToString(), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(outDir, $"AreaTable_unmatched_{srcAlias}_to_335.csv"), unmatched.ToString(), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(outDir, $"Area_patch_crosswalk_{srcAlias}_to_335.csv"), patch.ToString(), new UTF8Encoding(true));
            foreach (var kv in perMap)
            {
                File.WriteAllText(Path.Combine(outDir, $"AreaTable_mapping_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.map.ToString(), new UTF8Encoding(true));
                File.WriteAllText(Path.Combine(outDir, $"AreaTable_unmatched_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.un.ToString(), new UTF8Encoding(true));
                File.WriteAllText(Path.Combine(outDir, $"Area_patch_crosswalk_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.patch.ToString(), new UTF8Encoding(true));
            }

            Console.WriteLine("[V2] Wrote mapping/unmatched/patch CSVs under out/compare/v2/");
            Console.WriteLine("[V2] Done.");
            return 0;
        }

        private static IDBCDStorage LoadTable(string table, string canonicalBuild, string dbcDir, FilesystemDBDProvider dbdProvider, DBCD.Locale locale)
        {
            var provider = new FilesystemDBCProvider(dbcDir, useCache: true);
            var dbcd = new DBCD.DBCD(provider, dbdProvider);
            try { return dbcd.Load(table, canonicalBuild, locale); }
            catch { return dbcd.Load(table, canonicalBuild, DBCD.Locale.None); }
        }

        private static Dictionary<int, string> BuildMapNames(IDBCDStorage mapStorage, string mapNameCol)
        {
            string idCol = DetectIdColumn(mapStorage);
            var dict = new Dictionary<int, string>();
            foreach (var k in mapStorage.Keys)
            {
                var row = mapStorage[k];
                string dirRaw = SafeField<string>(row, "Directory");
                string dirTok = string.Empty;
                if (!string.IsNullOrWhiteSpace(dirRaw))
                {
                    var parts = dirRaw.Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                    dirTok = parts.Length > 0 ? parts[^1] : dirRaw.Trim();
                }
                string name = FirstNonEmpty(
                    SafeField<string>(row, "MapName_lang"),
                    SafeField<string>(row, "MapName"),
                    SafeField<string>(row, "InternalName"),
                    dirTok
                );
                int mapId = !string.IsNullOrWhiteSpace(idCol) ? SafeField<int>(row, idCol) : k;
                dict[mapId] = name;
            }
            return dict;
        }

        private static void WriteSimpleMapReport(string path, Dictionary<int, string> map)
        {
            var sb = new StringBuilder();
            sb.AppendLine("MapID,Name");
            foreach (var kv in map.OrderBy(k => k.Key))
            {
                sb.Append(kv.Key.ToString(CultureInfo.InvariantCulture));
                sb.Append(',');
                sb.AppendLine(Csv(kv.Value ?? string.Empty));
            }
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(path, sb.ToString(), new UTF8Encoding(true));
        }

        private static string Csv(string s)
        {
            if (s.IndexOfAny(new[] { '"', ',', '\n', '\r' }) >= 0) return '"' + s.Replace("\"", "\"\"") + '"';
            return s;
        }

        private static string FirstNonEmpty(params string[] vals)
        {
            foreach (var v in vals) if (!string.IsNullOrWhiteSpace(v)) return v; return string.Empty;
        }

        private static T SafeField<T>(DBCDRow row, string col)
        {
            try { return row.Field<T>(col); } catch { return default!; }
        }

        private static string MapName053FromCont(int cont)
        {
            if (cont == 0) return "Azeroth";
            if (cont == 1) return "Kalimdor";
            return string.Empty;
        }

        private static string DetectIdColumn(IDBCDStorage storage)
        {
            try
            {
                var cols = storage.AvailableColumns ?? Array.Empty<string>();
                string[] prefers = new[] { "ID", "Id", "MapID", "MapId", "m_ID" };
                foreach (var p in prefers)
                {
                    var match = cols.FirstOrDefault(x => string.Equals(x, p, StringComparison.OrdinalIgnoreCase));
                    if (!string.IsNullOrEmpty(match)) return match;
                }
                var anyId = cols.FirstOrDefault(x => x.EndsWith("ID", StringComparison.OrdinalIgnoreCase));
                return anyId ?? string.Empty;
            }
            catch { return string.Empty; }
        }

        // RemapDefinition JSON model for export/apply deterministic mappings
        private sealed class RemapDefinition
        {
            public Meta meta { get; set; } = new Meta();
            public Dictionary<string, string[]> aliases { get; set; } = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            public List<ExplicitMap> explicit_map { get; set; } = new List<ExplicitMap>();
            public List<int> ignore_area_numbers { get; set; } = new List<int>();
            public Options options { get; set; } = new Options();

            public sealed class Meta
            {
                public string src_alias { get; set; } = string.Empty;
                public string src_build { get; set; } = string.Empty;
                public string tgt_build { get; set; } = string.Empty;
                public string generated_at { get; set; } = string.Empty;
            }

            public sealed class ExplicitMap
            {
                public int src_areaNumber { get; set; }
                public int tgt_areaID { get; set; }
                public string? note { get; set; }
            }

            public sealed class Options
            {
                public bool disallow_do_not_use_targets { get; set; } = true;
            }
        }
    }
}
