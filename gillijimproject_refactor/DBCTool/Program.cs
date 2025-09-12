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
                var (dbdDir, outBase, localeStr, tables, inputs, buildOverride, compareArea, exportAll, srcAliasFlag, srcBuildFlag, tgtBuildFlag, exportRemap, applyRemap, disallowDoNotUse) = ParseArgs(args);

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

        private static (string dbdDir, string outBase, string locale, List<string> tables, List<(string build, string dir)> inputs, string buildOverride, bool compareArea, bool exportAll, string srcAlias, string srcBuild, string tgtBuild, string exportRemap, string applyRemap, bool disallowDoNotUse) ParseArgs(string[] args)
        {
            string dbdDir = DefaultDbdDir;
            string outBase = DefaultOutBase;
            string locale = DefaultLocale;
            string buildOverride = string.Empty;
            bool compareArea = false;
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
            return (NormalizePath(dbdDir), NormalizePath(outBase), locale, tables, inputs.Select(t => (t.build, NormalizePath(t.dir))).ToList(), buildOverride, compareArea, exportAll, srcAlias, srcBuild, tgtBuild, normExportRemap, normApplyRemap, disallowDoNotUse);
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
            foreach (var k in stor053_Map.Keys)
            {
                var r = stor053_Map[k];
                int mid = !string.IsNullOrWhiteSpace(idCol053) ? SafeField<int>(r, idCol053) : k;
                if (!map053ById.ContainsKey(mid)) map053ById[mid] = r;
            }
            var map335ById = new Dictionary<int, DBCDRow>();
            foreach (var k in stor335_Map.Keys)
            {
                var r = stor335_Map[k];
                int mid = !string.IsNullOrWhiteSpace(idCol335) ? SafeField<int>(r, idCol335) : k;
                if (!map335ById.ContainsKey(mid)) map335ById[mid] = r;
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

            // Build a token index of source maps to help infer expected map when ContinentID is unreliable
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

            // Source indices and parent resolution via AreaNumber (0.5.x) or ID (0.6.0+)
            var idx053_NumToRow = new Dictionary<int, DBCDRow>();
            foreach (var id in stor053_Area.Keys)
            {
                var row = stor053_Area[id];
                int sid = SafeField<int>(row, srcAlias == "0.5.3" || srcAlias == "0.5.5" ? "AreaNumber" : "ID");
                if (sid > 0 && !idx053_NumToRow.ContainsKey(sid)) idx053_NumToRow[sid] = row;
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

            (string name, string parentName, int areaNum, int parentNum, int mapId, string mapName, int mapIdX, string mapNameX, string path) Extract053(DBCDRow row)
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

            (int id, string method) ChooseTargetByName(string srcName, int mapIdX)
            {
                var variants = NameVariants(srcName).Select(NormName).Distinct().ToList();
                if (variants.Count == 0) variants = new List<string> { NormName(srcName) };

                // 1) Exact matches (map-biased then global) across variants
                if (mapIdX >= 0 && idx335_ByMapName.TryGetValue(mapIdX, out var byNameMap))
                {
                    foreach (var v in variants)
                        if (byNameMap.TryGetValue(v, out var lm) && lm.Count > 0)
                        {
                            var lmOk = lm.Where(tid => !DisallowCandidate(Extract335(tid).name)).ToList();
                            if (lmOk.Count > 0) return (lmOk.Min(), "name+map");
                        }
                }
                foreach (var v in variants)
                    if (idx335_ByName.TryGetValue(v, out var lg) && lg.Count > 0)
                    {
                        var lgOk = lg.Where(tid => !DisallowCandidate(Extract335(tid).name)).ToList();
                        if (lgOk.Count > 0) return (lgOk.Min(), "name");
                    }

                // 2) Fuzzy within map across variants
                IEnumerable<int> domain = (mapIdX >= 0 && idx335_ByMap.TryGetValue(mapIdX, out var lm2)) ? lm2 : idx335_IdToRow.Keys;
                int bestId = -1; int bestDist = int.MaxValue;
                foreach (var tid in domain)
                {
                    var rec = Extract335(tid);
                    if (DisallowCandidate(rec.name)) continue;
                    var rn = NormName(rec.name);
                    foreach (var v in variants)
                    {
                        var dn = EditDistance(v, rn);
                        if (dn <= 2 && (dn < bestDist || (dn == bestDist && tid < bestId)))
                        {
                            bestDist = dn; bestId = tid;
                        }
                    }
                }
                if (bestId >= 0) return (bestId, (mapIdX >= 0 ? "fuzzy+map" : "fuzzy"));

                // 3) Global fuzzy fallback across variants
                int gBestId = -1; int gBestDist = int.MaxValue;
                foreach (var tid in idx335_IdToRow.Keys)
                {
                    var rec = Extract335(tid);
                    if (DisallowCandidate(rec.name)) continue;
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
            foreach (var id in stor053_Area.Keys)
            {
                var src = Extract053(stor053_Area[id]);

                bool ShouldSkipDev(string n)
                {
                    var nn = NormName(n);
                    if (nn.Contains("***on map dungeon***")) return true;
                    if (nn.Contains("programmer isle")) return true;
                    if (nn.Contains("plains of snow")) return true;
                    if (nn.Contains("jeff") && nn.Contains("quadrant")) return true;
                    return false;
                }
                if (ShouldSkipDev(src.name)) { skippedDev++; ignoredAreas.Add(src.areaNum); continue; }

                // Name-only selection (ignore parent mismatches)
                int chosen = -1; string method = string.Empty;
                // Apply explicit mapping if provided
                if (explicitMap.TryGetValue(src.areaNum, out var expId)) { chosen = expId; method = "explicit"; }
                else { (chosen, method) = ChooseTargetByName(src.name, src.mapIdX); }
                if (chosen >= 0) byName++;

                string lineFor(int tgtId, string mth)
                {
                    string tgtName = string.Empty, tgtParentName = string.Empty, tgtMapName = string.Empty; int tgtParentId = -1, tgtMapId = -1; string tgtPath = string.Empty;
                    if (tgtId >= 0)
                    {
                        var rec = Extract335(tgtId);
                        tgtName = rec.name; tgtParentName = rec.parentName; tgtParentId = rec.parentId; tgtMapId = rec.mapId; tgtMapName = rec.mapName; tgtPath = rec.path;
                    }
                    return string.Join(',', new[]
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
                        (tgtId >= 0 ? tgtId.ToString(CultureInfo.InvariantCulture) : "-1"),
                        Csv(tgtName),
                        tgtParentId.ToString(CultureInfo.InvariantCulture),
                        Csv(tgtParentName),
                        tgtMapId.ToString(CultureInfo.InvariantCulture),
                        Csv(tgtMapName),
                        Csv(tgtPath),
                        mth
                    });
                }

                if (chosen >= 0)
                {
                    mapping.AppendLine(lineFor(chosen, method));
                    exportExplicit.Add(new RemapDefinition.ExplicitMap { src_areaNumber = src.areaNum, tgt_areaID = chosen, note = method });
                    var trecPatch = Extract335(chosen);
                    // Determine target parent for patch from mapping of source parent
                    int tgtParentForPatch = chosen;
                    if (src.parentNum != src.areaNum)
                    {
                        // Optional: if parent is mapped, use that as parent for patch
                        tgtParentForPatch = chosen; // keep simple
                    }
                    patchCsv.AppendLine(string.Join(',', new[]
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
                    }));
                }
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
                    unmatchedCsv.AppendLine(lineFor(-1, "unmatched"));
                }
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
            Console.WriteLine($"[Compare] Wrote {cwOut}");
            Console.WriteLine($"[Compare] Wrote MapId_to_Name CSVs");

            Console.WriteLine("[Compare] Done.");
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
