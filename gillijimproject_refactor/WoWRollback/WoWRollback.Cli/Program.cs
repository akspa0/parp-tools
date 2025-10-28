using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using SixLabors.ImageSharp.Formats.Jpeg;
using WoWRollback.AnalysisModule;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services;
using WoWRollback.Core.Services.Config;
using WoWRollback.Core.Services.Viewer;
using WoWRollback.Core.Services.Archive;
using WoWRollback.Core.Services.Minimap;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.Alpha;
using AlphaWdtAnalyzer.Core.Export;
using AlphaWdtAnalyzer.Core.Dbc;
using WoWRollback.Core.Services.AreaMapping;
using DBCTool.V2.Cli;

namespace WoWRollback.Cli;

internal static class Program
{
    private static readonly Dictionary<(string Version, string Map), string?> AlphaWdtCache = new(new TupleComparer());

    private static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintHelp();
            return 0;
        }
        var cmd = args[0].ToLowerInvariant();
        var opts = ParseArgs(args.Skip(1).ToArray());

        try
        {
            switch (cmd)
            {
                case "analyze-alpha-wdt":
                    return RunAnalyzeAlphaWdt(opts);
                case "analyze-lk-adt":
                case "analyze-ranges": // Legacy alias for analyze-lk-adt
                    return RunAnalyzeLkAdt(opts);
                case "analyze-map-adts":
                    return RunAnalyzeMapAdts(opts);
                case "analyze-map-adts-mpq":
                    return RunAnalyzeMapAdtsMpq(opts);
                case "debug-single-adt":
                    return RunDebugSingleAdt(opts);
                case "discover-maps":
                    return RunDiscoverMaps(opts);
                case "probe-archive":
                    return RunProbeArchive(opts);
                case "probe-minimap":
                    return RunProbeMinimap(opts);
                case "prepare-layers":
                    return RunPrepareLayers(opts);
                case "alpha-to-lk":
                    return RunAlphaToLk(opts);
                case "lk-to-alpha":
                    return RunLkToAlpha(opts);
                case "gen-area-remap":
                    return RunGenAreaRemap(opts);
                case "rollback":
                    return RunRollback(opts);
                case "serve-viewer":
                case "serve":
                    return RunServeViewer(opts);
                case "gui":
                    return RunGui(opts);
                case "run-preset":
                    return RunRunPreset(opts);
                case "fix-minimap-webp":
                    return RunFixMinimapWebp(opts);
                case "dry-run":
                    return RunDryRun(opts);
                case "compare-versions":
                    return RunCompareVersions(opts);
                default:
                    Console.Error.WriteLine($"Unknown command: {cmd}");
                    PrintHelp();
                    return 2;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[error] {ex.Message}");
            return 1;
        }
    }

    private static string? EnsureDbcDirFromClient(string alias, string? build, string? clientDir, string outRoot)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(clientDir) || !Directory.Exists(clientDir)) return null;
            var mpqs = ArchiveLocator.LocateMpqs(clientDir);
            if (mpqs.Count == 0) { Console.WriteLine($"[patchmap] no MPQs found under: {clientDir}"); return null; }
            using var src = new MpqArchiveSource(mpqs);

            var dest = Path.Combine(outRoot, "dbc_extract", (alias ?? "unknown").Replace('.', '_'));
            Directory.CreateDirectory(dest);

            bool okMap = TryExtractDbc(src, "DBFilesClient/Map.dbc", Path.Combine(dest, "Map.dbc"));
            bool okArea = TryExtractDbc(src, "DBFilesClient/AreaTable.dbc", Path.Combine(dest, "AreaTable.dbc"));

            if (!okMap || !okArea)
            {
                Console.WriteLine($"[patchmap] DBCs missing from MPQs (Map={okMap}, AreaTable={okArea}) under: {clientDir}");
                return null;
            }
            return dest;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[patchmap] DBC extraction error: {ex.Message}");
            return null;
        }
    }

    private static bool TryExtractDbc(WoWRollback.Core.Services.Archive.IArchiveSource src, string virtualPath, string destPath)
    {
        // Attempt common case and case-variants
        var baseName = Path.GetFileName(virtualPath);
        var candidates = new List<string>
        {
            virtualPath,
            virtualPath.ToLowerInvariant(),
            virtualPath.ToUpperInvariant(),
            virtualPath.Replace("DBFilesClient", "dbfilesclient"),
            virtualPath.Replace("DBFilesClient", "DBFILESCLIENT"),
            baseName
        };

        foreach (var v in candidates)
        {
            try
            {
                if (!src.FileExists(v)) continue;
                using var s = src.OpenFile(v);
                using var fs = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None);
                s.CopyTo(fs);
                return true;
            }
            catch { /* try next variant */ }
        }
        return false;
    }

    private static int RunProbeMinimap(Dictionary<string, string> opts)
    {
        Require(opts, "client-path");
        Require(opts, "map");
        var clientRoot = opts["client-path"]; 
        var mapName = opts["map"]; 
        var limit = TryParseInt(opts, "limit") ?? 12;

        Console.WriteLine($"[probe] Client root: {clientRoot}");
        Console.WriteLine($"[probe] Map: {mapName}");
        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("[error] --client-path does not exist");
            return 1;
        }

        EnsureStormLibOnPath();

        // Build prioritized source
        var mpqs = ArchiveLocator.LocateMpqs(clientRoot);
        using var src = new PrioritizedArchiveSource(clientRoot, mpqs);

        // Load md5translate if present
        WoWRollback.Core.Services.Minimap.Md5TranslateIndex? index = null;
        if (Md5TranslateResolver.TryLoad(src, out var loaded, out var usedPath))
        {
            index = loaded;
            Console.WriteLine($"[probe] md5translate loaded: {usedPath}");
        }
        else
        {
            Console.WriteLine("[probe] md5translate not found; will attempt plain and scan fallbacks");
        }

        var resolver = new MinimapFileResolver(src, index);
        using var fsOnly = new FileSystemArchiveSource(clientRoot);
        using var mpqOnly = new MpqArchiveSource(mpqs);

        // Gather candidate tiles using md5 index when available (preferred for hashed root layouts)
        var candidates = new List<(int X, int Y)>();

        if (index is not null)
        {
            var found = 0;
            foreach (var plain in index.PlainToHash.Keys)
            {
                // restrict to requested map; accept keys within textures/minimap/<map>/... or starting with <map>_
                var containsMapFolder = plain.IndexOf($"/" + mapName + "/", StringComparison.OrdinalIgnoreCase) >= 0;
                var startsWithMap = Path.GetFileNameWithoutExtension(plain).StartsWith(mapName + "_", StringComparison.OrdinalIgnoreCase);
                if (!containsMapFolder && !startsWithMap) continue;

                if (TryParseTileFromPath(plain, mapName, out var x, out var y))
                {
                    var tuple = (x, y);
                    if (!candidates.Contains(tuple))
                    {
                        candidates.Add(tuple);
                        found++;
                        if (found >= limit) break;
                    }
                }
            }
        }

        // Fallback to filename enumeration if md5 index didn’t produce candidates
        if (candidates.Count == 0)
        {
            foreach (var p in src.EnumerateFiles($"textures/Minimap/{mapName}/*.blp"))
            {
                if (TryParseTileFromPath(p, mapName, out var x, out var y))
                {
                    candidates.Add((x, y));
                }
                if (candidates.Count >= limit) break;
            }
            if (candidates.Count < limit)
            {
                foreach (var p in src.EnumerateFiles("textures/Minimap/*.blp"))
                {
                    if (TryParseTileFromPath(p, mapName, out var x, out var y))
                    {
                        if (!candidates.Contains((x, y))) candidates.Add((x, y));
                    }
                    if (candidates.Count >= limit) break;
                }
            }
        }

        if (candidates.Count == 0)
        {
            Console.WriteLine("[probe] No candidate tiles discovered via enumeration; try with different --map or ensure assets exist.");
            return 0;
        }

        Console.WriteLine($"[probe] Resolving up to {candidates.Count} tiles using resolver:");
        int resolved = 0;
        foreach (var (x, y) in candidates)
        {
            if (resolver.TryResolveTile(mapName, x, y, out var path) && path is not null)
            {
                // Determine origin: loose or MPQ
                bool isLoose = fsOnly.FileExists(path) || fsOnly.FileExists(Path.Combine("Data", path).Replace('\\','/'));
                bool isMpq = mpqOnly.FileExists(path);

                long size = 0;
                try
                {
                    using var s = src.OpenFile(path);
                    // obtain size; for FileStream use Length, for MemoryStream use Length, otherwise copy small chunk
                    if (s.CanSeek) 
                    {
                        size = s.Length;
                        if (size == 0)
                        {
                            Console.WriteLine($"  {mapName}_{x}_{y} -> {path} [warning: stream length is 0, stream type: {s.GetType().Name}]");
                        }
                    }
                    else 
                    { 
                        using var ms = new MemoryStream(); 
                        s.CopyTo(ms); 
                        size = ms.Length; 
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  {mapName}_{x}_{y} -> {path} [open failed: {ex.Message}]");
                    continue;
                }

                var origin = isLoose ? "loose" : (isMpq ? "mpq" : "unknown");
                Console.WriteLine($"  {mapName}_{x}_{y} -> {path}  [origin: {origin}, size: {size}]");
                resolved++;
            }
            else
            {
                Console.WriteLine($"  {mapName}_{x}_{y} -> (not found)");
            }
        }

        Console.WriteLine($"[ok] Resolved {resolved}/{candidates.Count} tiles (no viewer changes).");
        return 0;
    }

    private static string BuildLayersUiArgs(string proj, string wdtPath, string outDir, int gap, string? dbdDirOpt, string? dbcDirOpt, string? buildOpt, string? lkDbcDirOpt)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append($"run --project \"{proj}\" -- layers-ui --wdt \"{wdtPath}\" --output-dir \"{outDir}\" --gap-threshold {gap}");
        if (!string.IsNullOrWhiteSpace(dbdDirOpt)) sb.Append($" --dbd-dir \"{dbdDirOpt}\"");
        // Prefer explicit --dbc-dir, else fall back to --lk-dbc-dir when provided
        var dbcEffective = !string.IsNullOrWhiteSpace(dbcDirOpt) ? dbcDirOpt : lkDbcDirOpt;
        if (!string.IsNullOrWhiteSpace(dbcEffective)) sb.Append($" --dbc-dir \"{dbcEffective}\"");
        if (!string.IsNullOrWhiteSpace(buildOpt)) sb.Append($" --build \"{buildOpt}\"");
        return sb.ToString();
    }

    private static int RunGui(Dictionary<string, string> opts)
    {
        var cache = GetOption(opts, "cache") ?? Path.Combine(Directory.GetCurrentDirectory(), "work", "cache");
        var presets = GetOption(opts, "presets") ?? Path.Combine(Directory.GetCurrentDirectory(), "work", "presets");
        Directory.CreateDirectory(cache);
        Directory.CreateDirectory(presets);

        try
        {
            var proj = ResolveProjectCsproj("WoWRollback", "WoWRollback.Gui");
            var args = $"run --project \"{proj}\" -- --cache \"{cache}\" --presets \"{presets}\"";
            var psi = new ProcessStartInfo
            {
                FileName = "dotnet",
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = false
            };
            using var proc = new Process { StartInfo = psi };
            proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine(e.Data); };
            proc.ErrorDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine(e.Data); };
            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            proc.WaitForExit();
            return proc.ExitCode;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[error] GUI launch failed: {ex.Message}");
            return 1;
        }
    }

    private static int RunRunPreset(Dictionary<string, string> opts)
    {
        var presetPath = GetOption(opts, "preset");
        if (string.IsNullOrWhiteSpace(presetPath) || !File.Exists(presetPath))
        {
            Console.Error.WriteLine("[error] --preset file not found");
            return 2;
        }
        var mapsOpt = GetOption(opts, "maps") ?? "all";
        var outRoot = GetOption(opts, "out-root") ?? Path.Combine(Directory.GetCurrentDirectory(), "work", "patches");
        var lkOut = GetOption(opts, "lk-out") ?? Path.Combine(Directory.GetCurrentDirectory(), "work", "lk_adts");
        var dryRun = opts.ContainsKey("dry-run");

        Console.WriteLine("══════════════════════════════════════════");
        Console.WriteLine("          🎮 WoWRollback - RUN PRESET (dry)");
        Console.WriteLine("══════════════════════════════════════════");
        Console.WriteLine($"Preset:      {presetPath}");
        Console.WriteLine($"Maps:        {mapsOpt}");
        Console.WriteLine($"Out Root:    {outRoot}");
        Console.WriteLine($"LK Out:      {lkOut}");
        Console.WriteLine($"Mode:        {(dryRun ? "dry-run" : "execute")}");

        try
        {
            var json = File.ReadAllText(presetPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            int mapCount = 0;
            if (root.TryGetProperty("maps", out var mapsEl) && mapsEl.ValueKind == JsonValueKind.Object)
            {
                mapCount = mapsEl.EnumerateObject().Count();
            }
            Console.WriteLine($"Preset contains {mapCount} map entries.");
        }
        catch { Console.WriteLine("[warn] Could not parse preset; proceeding"); }

        if (dryRun)
        {
            Console.WriteLine("[dry-run] No actions taken");
            return 0;
        }

        Console.WriteLine("[todo] Execution not implemented in this build. Use generated CLI commands from UI for now.");
        return 0;
    }

    private static int RunPrepareLayers(Dictionary<string, string> opts)
    {
        var outRoot = GetOption(opts, "out") ?? Path.Combine(Directory.GetCurrentDirectory(), "work", "cache");
        Directory.CreateDirectory(outRoot);

        var gap = TryParseInt(opts, "gap-threshold") ?? 50;
        var dbdDirOpt = GetOption(opts, "dbd-dir");
        var dbcDirOpt = GetOption(opts, "dbc-dir");
        var buildOpt = GetOption(opts, "build");
        var lkDbcDirOpt = GetOption(opts, "lk-dbc-dir");
        var lkClientOpt = GetOption(opts, "lk-client-path");

        // Mode A: explicit single WDT
        var wdtPath = GetOption(opts, "wdt");
        if (!string.IsNullOrWhiteSpace(wdtPath))
        {
            if (!File.Exists(wdtPath)) { Console.Error.WriteLine($"[error] --wdt not found: {wdtPath}"); return 1; }
            var mapName = Path.GetFileNameWithoutExtension(wdtPath);
            var outDir = Path.Combine(outRoot, mapName);
            Directory.CreateDirectory(outDir);
            return RunLayersUiGenerator(wdtPath, outDir, gap, dbdDirOpt, dbcDirOpt, buildOpt, lkDbcDirOpt, lkClientOpt);
        }

        // Mode B: scan client-root for loose Alpha WDTs under World/Maps
        var clientRoot = GetOption(opts, "client-root");
        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("[error] Provide either --wdt <path> or --client-root <dir>");
            return 2;
        }

        // Enumerate WDTs: <clientRoot>/World/Maps/<Map>/<Map>.wdt
        var mapsDir = Path.Combine(clientRoot, "World", "Maps");
        if (!Directory.Exists(mapsDir))
        {
            Console.Error.WriteLine($"[error] Not a valid client root (missing World/Maps): {clientRoot}");
            return 2;
        }

        var requested = GetOption(opts, "maps"); // null|"all"|csv
        var allowAll = string.IsNullOrWhiteSpace(requested) || string.Equals(requested, "all", StringComparison.OrdinalIgnoreCase);
        var allowSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (!allowAll)
        {
            foreach (var m in requested!.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                allowSet.Add(m);
        }

        var wdtCandidates = new List<(string Map, string Path)>();
        try
        {
            foreach (var dir in Directory.EnumerateDirectories(mapsDir))
            {
                var map = Path.GetFileName(dir);
                if (!allowAll && !allowSet.Contains(map)) continue;
                var wdt = Path.Combine(dir, map + ".wdt");
                if (File.Exists(wdt)) wdtCandidates.Add((map, wdt));
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[error] Scan failed: {ex.Message}");
            return 2;
        }

        if (wdtCandidates.Count == 0)
        {
            Console.WriteLine("[info] No maps found to prepare.");
            return 0;
        }

        Console.WriteLine($"[prepare] Building layer caches for {wdtCandidates.Count} map(s) → {outRoot}");
        int ok = 0, fail = 0;
        foreach (var (map, path) in wdtCandidates.OrderBy(t => t.Map, StringComparer.OrdinalIgnoreCase))
        {
            Console.WriteLine($"—— {map} ——");
            var outDir = Path.Combine(outRoot, map);
            Directory.CreateDirectory(outDir);
            var code = RunLayersUiGenerator(path, outDir, gap, dbdDirOpt, dbcDirOpt, buildOpt, lkDbcDirOpt, lkClientOpt);
            if (code == 0) { ok++; Console.WriteLine($"[ok] {map}"); }
            else { fail++; Console.WriteLine($"[fail] {map} (exit={code})"); }
        }

        Console.WriteLine($"[summary] success={ok}, failed={fail}");
        return fail == 0 ? 0 : 1;
    }

    private static int RunLayersUiGenerator(string wdtPath, string outDir, int gap, string? dbdDirOpt, string? dbcDirOpt, string? buildOpt, string? lkDbcDirOpt, string? lkClientOpt)
    {
        try
        {
            var proj = ResolveProjectCsproj("WoWRollback", "WoWDataPlot");
            // Pre-step: generate LK ADTs from Alpha WDT (mock run, keep everything)
            try
            {
                var mapName = Path.GetFileNameWithoutExtension(wdtPath);
                var rollOpts = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["input"] = wdtPath,
                    ["out"] = outDir,
                    ["max-uniqueid"] = int.MaxValue.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    ["export-lk-adts"] = "true",
                    ["force"] = "true"
                };
                if (!string.IsNullOrWhiteSpace(lkClientOpt)) rollOpts["lk-client-path"] = lkClientOpt!;
                if (!string.IsNullOrWhiteSpace(lkDbcDirOpt)) rollOpts["lk-dbc-dir"] = lkDbcDirOpt!;
                if (!string.IsNullOrWhiteSpace(dbdDirOpt)) rollOpts["dbd-dir"] = dbdDirOpt!;
                Console.WriteLine($"[info] Preparing LK ADTs for AreaIDs: {mapName}");
                var code = RunRollback(rollOpts);
                if (code != 0)
                {
                    Console.WriteLine($"[warn] Mock rollback (LK export) failed for {mapName} (exit={code}); areas.csv enrichment may be empty.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[warn] Mock rollback (LK export) error: {ex.Message}");
            }

            var psi = new ProcessStartInfo
            {
                FileName = "dotnet",
                Arguments = BuildLayersUiArgs(proj, wdtPath, outDir, gap, dbdDirOpt, dbcDirOpt, buildOpt, lkDbcDirOpt),
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var proc = new Process { StartInfo = psi };
            proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine(e.Data); };
            proc.ErrorDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine(e.Data); };
            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            proc.WaitForExit();
            return proc.ExitCode;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[error] layers-ui run failed: {ex.Message}");
            return 1;
        }
    }

    private static bool TryParseTileFromPath(string virtualPath, string mapName, out int x, out int y)
    {
        x = 0; y = 0;
        var file = Path.GetFileNameWithoutExtension(virtualPath);
        if (file.StartsWith(mapName + "_", StringComparison.OrdinalIgnoreCase))
        {
            var tail = file.Substring(mapName.Length + 1);
            var parts = tail.Split('_');
            if (parts.Length >= 2 && int.TryParse(parts[0], out x) && int.TryParse(parts[1], out y))
                return true;
        }
        if (file.StartsWith("map", StringComparison.OrdinalIgnoreCase))
        {
            var tail = file.Substring(3);
            var parts = tail.Split('_');
            if (parts.Length >= 2 && int.TryParse(parts[0], out x) && int.TryParse(parts[1], out y))
                return true;
        }
        return false;
    }

    private static int RunAnalyzeAlphaWdt(Dictionary<string, string> opts)
    {
        Require(opts, "wdt-file");
        var wdtFile = opts["wdt-file"];
        var outRoot = opts.GetValueOrDefault("out", "");
        var mapName = Path.GetFileNameWithoutExtension(wdtFile);

        var buildTag = BuildTagResolver.ResolveForPath(Path.GetDirectoryName(Path.GetFullPath(wdtFile)) ?? wdtFile);
        var sessionDir = OutputSession.Create(outRoot, mapName, buildTag);
        Console.WriteLine($"[info] Archaeological analysis session: {sessionDir}");
        Console.WriteLine($"[info] Excavating Alpha WDT: {wdtFile}");
        Console.WriteLine($"[info] Using raw Alpha coordinates (no transforms)");

        var analysis = WoWRollback.Core.Services.AlphaWdtAnalyzer.AnalyzeAlphaWdt(wdtFile);
        var csvResult = RangeCsvWriter.WritePerMapCsv(sessionDir, $"alpha_{mapName}", analysis.Ranges, analysis.Assets);

        Console.WriteLine($"[ok] Extracted {analysis.Ranges.Count} archaeological placement layers");
        Console.WriteLine($"[ok] Alpha UniqueID ranges written to: {csvResult.PerMapPath}");
        if (!string.IsNullOrEmpty(csvResult.TimelinePath))
        {
            Console.WriteLine($"[ok] Timeline CSV: {csvResult.TimelinePath}");
        }
        if (!string.IsNullOrEmpty(csvResult.AssetLedgerPath))
        {
            Console.WriteLine($"[ok] Asset ledger CSV: {csvResult.AssetLedgerPath}");
        }
        if (!string.IsNullOrEmpty(csvResult.TimelineAssetsPath))
        {
            Console.WriteLine($"[ok] Timeline asset summary CSV: {csvResult.TimelineAssetsPath}");
        }

        return 0;
    }

    private static bool ShouldGenerateViewer(Dictionary<string, string> opts)
    {
        if (!opts.TryGetValue("viewer-report", out var value))
            return false;

        return !string.Equals(value, "false", StringComparison.OrdinalIgnoreCase);
    }

    private static ViewerOptions BuildViewerOptions(Dictionary<string, string> opts)
    {
        var defaults = ViewerOptions.CreateDefault();

        var defaultVersion = opts.TryGetValue("default-version", out var requestedDefault)
            ? requestedDefault
            : defaults.DefaultVersion;

        var minimapWidth = TryParseInt(opts, "minimap-width") ?? defaults.MinimapWidth;
        var minimapHeight = TryParseInt(opts, "minimap-height") ?? defaults.MinimapHeight;
        var distanceThreshold = TryParseDouble(opts, "distance-threshold") ?? defaults.DiffDistanceThreshold;
        var moveEpsilon = TryParseDouble(opts, "move-epsilon") ?? defaults.MoveEpsilonRatio;

        return new ViewerOptions(
            defaultVersion,
            DiffPair: null,
            MinimapWidth: minimapWidth,
            MinimapHeight: minimapHeight,
            DiffDistanceThreshold: distanceThreshold,
            MoveEpsilonRatio: moveEpsilon);
    }

    private static (string Baseline, string Comparison)? ParseDiffPair(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        var parts = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length != 2)
            return null;

        return (parts[0], parts[1]);
    }

    private static int? TryParseInt(Dictionary<string, string> opts, string key)
    {
        if (!opts.TryGetValue(key, out var value))
            return null;

        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : null;
    }

    private static double? TryParseDouble(Dictionary<string, string> opts, string key)
    {
        if (!opts.TryGetValue(key, out var value))
            return null;

        return double.TryParse(value, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : null;
    }

    private static string ResolveProjectCsproj(string folder, string projectName)
    {
        try
        {
            var start = new DirectoryInfo(AppContext.BaseDirectory);
            for (var dir = start; dir != null; dir = dir.Parent)
            {
                var csproj = Path.Combine(dir.FullName, folder, projectName, projectName + ".csproj");
                if (File.Exists(csproj)) return csproj;
            }
        }
        catch { }
        return Path.Combine(folder, projectName);
    }

    private static int RunCompareVersions(Dictionary<string, string> opts)
    {
        Require(opts, "versions");
        var rootCandidate = GetOption(opts, "root");
        var outCandidate = GetOption(opts, "out");
        var root = string.IsNullOrWhiteSpace(rootCandidate) ? (outCandidate ?? string.Empty) : rootCandidate!;
        var outputRoot = string.IsNullOrWhiteSpace(root)
            ? Path.Combine(Directory.GetCurrentDirectory(), "rollback_outputs")
            : Path.GetFullPath(root);

        var versions = opts["versions"].Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim())
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();

        IReadOnlyList<string>? maps = null;
        if (opts.TryGetValue("maps", out var mapsSpec) && !string.IsNullOrWhiteSpace(mapsSpec))
        {
            maps = mapsSpec.Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .ToList();
        }

        EnsureComparisonPrerequisites(outputRoot, versions, maps, opts);

        Console.WriteLine($"[info] Comparing versions: {string.Join(", ", versions)}");
        if (!string.IsNullOrWhiteSpace(root)) Console.WriteLine($"[info] Root directory: {root}");
        if (maps is not null && maps.Count > 0) Console.WriteLine($"[info] Map filter: {string.Join(", ", maps)}");

        var result = VersionComparisonService.CompareVersions(root, versions, maps);
        var paths = VersionComparisonWriter.WriteOutputs(root, result);
        var wantYaml = opts.TryGetValue("yaml-report", out var yamlOpt) && !string.Equals(yamlOpt, "false", StringComparison.OrdinalIgnoreCase);
        if (wantYaml)
        {
            var yamlRoot = VersionComparisonWriter.WriteYamlReports(root, result);
            Console.WriteLine($"[ok] YAML reports written to: {yamlRoot}");
        }

        if (ShouldGenerateViewer(opts))
        {
            var options = BuildViewerOptions(opts);
            var diffPair = ParseDiffPair(GetOption(opts, "diff"));
            var viewerWriter = new ViewerReportWriter();
            var viewerRoot = viewerWriter.Generate(paths.ComparisonDirectory, result, options, diffPair);
            if (string.IsNullOrEmpty(viewerRoot))
            {
                Console.WriteLine("[info] Viewer assets skipped (no placement data).");
            }
            else
            {
                Console.WriteLine($"[ok] Viewer assets written to: {viewerRoot}");
            }
        }

        Console.WriteLine($"[ok] Comparison key: {result.ComparisonKey}");
        Console.WriteLine($"[ok] Outputs written to: {paths.ComparisonDirectory}");
        if (result.Warnings.Count > 0)
        {
            Console.WriteLine($"[warn] {result.Warnings.Count} warnings emitted. See warnings_{result.ComparisonKey}.txt");
        }
        return 0;
    }

    private static int RunAnalyzeLkAdt(Dictionary<string, string> opts)
    {
        Require(opts, "input-dir");
        var map = opts["map"]; 
        var inputDir = opts["input-dir"]; 
        var outRoot = opts.GetValueOrDefault("out", "");
        
        var buildTag = BuildTagResolver.ResolveForPath(inputDir);
        var sessionDir = OutputSession.Create(outRoot, map, buildTag);
        Console.WriteLine($"[info] LK ADT analysis session: {sessionDir}");
        Console.WriteLine($"[info] Analyzing converted LK ADT files for map: {map}");

        var ranges = RangeScanner.AnalyzeRangesForMap(inputDir, map);
        var csvResult = RangeCsvWriter.WritePerMapCsv(sessionDir, $"lk_{map}", ranges);

        Console.WriteLine($"[ok] Extracted {ranges.Count} preserved placement ranges from LK ADTs");
        Console.WriteLine($"[ok] LK UniqueID ranges written to: {csvResult.PerMapPath}");
        if (!string.IsNullOrEmpty(csvResult.TimelinePath))
        {
            Console.WriteLine($"[ok] LK timeline CSV: {csvResult.TimelinePath}");
        }
        if (!string.IsNullOrEmpty(csvResult.AssetLedgerPath))
        {
            Console.WriteLine($"[ok] LK asset ledger CSV: {csvResult.AssetLedgerPath}");
        }
        if (!string.IsNullOrEmpty(csvResult.TimelineAssetsPath))
        {
            Console.WriteLine($"[ok] LK timeline asset summary CSV: {csvResult.TimelineAssetsPath}");
        }

        return 0;
    }

    private static int RunLkToAlpha(Dictionary<string, string> opts)
    {
        // Symmetric patcher for LK ADTs: bury by UniqueID and optionally clear holes / zero MCSH
        var inputDir = opts.GetValueOrDefault("lk-adts-dir", opts.GetValueOrDefault("input-dir", ""));
        if (string.IsNullOrWhiteSpace(inputDir)) throw new ArgumentException("Missing --lk-adts-dir (or --input-dir)");
        Require(opts, "max-uniqueid");

        var mapName = opts.GetValueOrDefault("map", Path.GetFileName(Path.GetFullPath(inputDir).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)));
        var maxUniqueId = (uint)(TryParseInt(opts, "max-uniqueid") ?? throw new ArgumentException("Missing --max-uniqueid"));
        var userOut = GetOption(opts, "out");
        uint rangeMinLabel, rangeMaxLabel;
        string outRoot;
        if (string.IsNullOrWhiteSpace(userOut))
        {
            outRoot = ResolveSessionRoot(opts, mapName, maxUniqueId, out rangeMinLabel, out rangeMaxLabel);
        }
        else
        {
            outRoot = userOut!;
            if (!TryComputeRangeFromPresetOption(opts, out rangeMinLabel, out var _presetMaxTmp))
            {
                rangeMinLabel = 0;
            }
            rangeMaxLabel = maxUniqueId;
        }
        Directory.CreateDirectory(outRoot);
        var lkAdtRoot = Path.Combine(outRoot, "lk_adts", "World", "Maps", mapName);

        var buryDepth = opts.TryGetValue("bury-depth", out var buryStr) && float.TryParse(buryStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var bd)
            ? bd : -5000.0f;
        var fixHoles = opts.ContainsKey("fix-holes");
        var disableMcsh = opts.ContainsKey("disable-mcsh");
        var holesScope = opts.TryGetValue("holes-scope", out var holesScopeStr) ? holesScopeStr.ToLowerInvariant() : "self";
        var holesNeighbors = string.Equals(holesScope, "neighbors", StringComparison.OrdinalIgnoreCase);
        var holesPreserveWmo = !(opts.TryGetValue("holes-wmo-preserve", out var preserveStr) && string.Equals(preserveStr, "false", StringComparison.OrdinalIgnoreCase));

        Console.WriteLine("══════════════════════════════════════════");
        Console.WriteLine("          🎮 WoWRollback - LK PATCHER");
        Console.WriteLine("══════════════════════════════════════════");
        Console.WriteLine($"Map:            {mapName}");
        Console.WriteLine($"LK ADT Dir:     {inputDir}");
        Console.WriteLine($"Session Dir:    {outRoot}");
        Console.WriteLine($"LK Output Dir:  {lkAdtRoot}");
        Console.WriteLine($"Max UniqueID:   {maxUniqueId:N0}");
        Console.WriteLine($"Bury Depth:     {buryDepth:F1}");
        // Display label range (no preset for LK patcher unless provided via opts)
        uint presetMinTmp2, presetMaxTmp2;
        if (TryComputeRangeFromPresetOption(opts, out presetMinTmp2, out presetMaxTmp2))
        {
            Console.WriteLine($"Preset Range:   {presetMinTmp2}-{presetMaxTmp2}");
        }
        Console.WriteLine($"Session Label:  {rangeMinLabel}-{rangeMaxLabel}");
        Console.WriteLine($"Bury Threshold: UniqueID > {maxUniqueId:N0}");
        if (fixHoles)
        {
            Console.WriteLine($"Option:         --fix-holes (scope={holesScope}, preserve-wmo={holesPreserveWmo.ToString().ToLowerInvariant()})");
        }
        if (disableMcsh) Console.WriteLine("Option:         --disable-mcsh (zero baked shadows)");
        Console.WriteLine();

        int filesProcessed = 0, placementsProcessed = 0, placementsBuried = 0;
        int holesCleared = 0, mcshZeroed = 0, mcnkScanned = 0;

        var allAdts = Directory.EnumerateFiles(inputDir, "*.adt", SearchOption.AllDirectories)
            .Where(p => Path.GetFileNameWithoutExtension(p).StartsWith(mapName + "_", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p)
            .ToList();

        foreach (var inPath in allAdts)
        {
            var bytes = File.ReadAllBytes(inPath);

            // Track buried arrays for MCRF gating per file
            bool[] mddfBuried = Array.Empty<bool>();
            bool[] modfBuried = Array.Empty<bool>();

            void BuryMddf()
            {
                int start = FindChunk(bytes, "FDDM");
                if (start < 0) return;
                int size = BitConverter.ToInt32(bytes, start + 4);
                int data = start + 8;
                const int entry = 36;
                int count = size / entry;
                mddfBuried = new bool[count];
                for (int off = 0; off + entry <= size; off += entry)
                {
                    int entryStart = data + off;
                    uint uid = BitConverter.ToUInt32(bytes, entryStart + 4);
                    placementsProcessed++;
                    if (uid > maxUniqueId)
                    {
                        var newY = BitConverter.GetBytes(buryDepth); // height at +12
                        Array.Copy(newY, 0, bytes, entryStart + 12, 4);
                        placementsBuried++;
                        int idx = off / entry; if (idx >= 0 && idx < mddfBuried.Length) mddfBuried[idx] = true;
                    }
                }
            }

            void BuryModf()
            {
                int start = FindChunk(bytes, "FDOM");
                if (start < 0) return;
                int size = BitConverter.ToInt32(bytes, start + 4);
                int data = start + 8;
                const int entry = 64;
                int count = size / entry;
                modfBuried = new bool[count];
                for (int off = 0; off + entry <= size; off += entry)
                {
                    int entryStart = data + off;
                    uint uid = BitConverter.ToUInt32(bytes, entryStart + 4);
                    placementsProcessed++;
                    if (uid > maxUniqueId)
                    {
                        var newY = BitConverter.GetBytes(buryDepth); // height at +12
                        Array.Copy(newY, 0, bytes, entryStart + 12, 4);
                        placementsBuried++;
                        int idx = off / entry; if (idx >= 0 && idx < modfBuried.Length) modfBuried[idx] = true;
                    }
                }
            }

            BuryMddf();
            BuryModf();

            // Optional MCNK passes (holes + mcsh)
            if (fixHoles || disableMcsh)
            {
                long fileLen = bytes.LongLength;
                using var ms = new MemoryStream(bytes);
                using var br = new BinaryReader(ms);
                using var bw = new BinaryWriter(ms);

                // Find MCIN
                long mcinDataPos = -1; int mcinSize = 0;
                ms.Position = 0;
                while (ms.Position + 8 <= fileLen)
                {
                    var rev = br.ReadBytes(4);
                    if (rev.Length < 4) break;
                    var fourcc = ReverseFourCC(System.Text.Encoding.ASCII.GetString(rev));
                    int sz = br.ReadInt32();
                    long dataPos = ms.Position;
                    if (fourcc == "MCIN") { mcinDataPos = dataPos; mcinSize = sz; break; }
                    ms.Position = dataPos + sz + ((sz & 1) == 1 ? 1 : 0);
                }

                if (mcinDataPos >= 0 && mcinSize >= 16)
                {
                    ms.Position = mcinDataPos;
                    int need = Math.Min(mcinSize, 256 * 16);
                    var mcinBytes = br.ReadBytes(need);

                    var chunkHasHoles = new bool[256];
                    var holesOffsetByIdx = new int[256];
                    var chunkHasBuriedRef = new bool[256];
                    var chunkHasKeepWmo = new bool[256];
                    Array.Fill(holesOffsetByIdx, -1);

                    // Pre-scan holes and buried references
                    for (int i = 0; i < 256; i++)
                    {
                        int mcnkOffset = (mcinBytes.Length >= (i + 1) * 16) ? BitConverter.ToInt32(mcinBytes, i * 16) : 0;
                        if (mcnkOffset <= 0) continue;
                        mcnkScanned++;

                        int headerStart = mcnkOffset + 8;
                        if (headerStart + 128 > bytes.Length) continue;

                        int holesOffset = headerStart + 0x40;
                        if (holesOffset + 4 <= bytes.Length)
                        {
                            holesOffsetByIdx[i] = holesOffset;
                            int prev = BitConverter.ToInt32(bytes, holesOffset);
                            chunkHasHoles[i] = prev != 0;
                        }

                        try
                        {
                            int m2Number = BitConverter.ToInt32(bytes, headerStart + 0x14);
                            int wmoNumber = BitConverter.ToInt32(bytes, headerStart + 0x3C);
                            int mcrfRel = BitConverter.ToInt32(bytes, headerStart + 0x24);
                            int mcrfChunkOffset = headerStart + 128 + mcrfRel;
                            if (mcrfChunkOffset + 8 <= bytes.Length)
                            {
                                var mcrf = new Mcrf(bytes, mcrfChunkOffset);
                                var m2Idx = mcrf.GetDoodadsIndices(Math.Max(0, m2Number));
                                var wmoIdx = mcrf.GetWmosIndices(Math.Max(0, wmoNumber));
                                foreach (var idx in m2Idx)
                                {
                                    if (idx >= 0 && idx < mddfBuried.Length && mddfBuried[idx]) { chunkHasBuriedRef[i] = true; break; }
                                }
                                if (!chunkHasBuriedRef[i])
                                {
                                    foreach (var idx in wmoIdx)
                                    {
                                        if (idx >= 0 && idx < modfBuried.Length && modfBuried[idx]) { chunkHasBuriedRef[i] = true; break; }
                                    }
                                }
                                if (holesPreserveWmo && !chunkHasKeepWmo[i])
                                {
                                    foreach (var idx in wmoIdx)
                                    {
                                        if (idx >= 0 && idx < modfBuried.Length && !modfBuried[idx]) { chunkHasKeepWmo[i] = true; break; }
                                    }
                                }
                            }
                        }
                        catch { /* best-effort */ }
                    }

                    // Clear holes with scope and WMO-preserve guard
                    if (fixHoles)
                    {
                        var toClear = new bool[256];
                        for (int i = 0; i < 256; i++)
                        {
                            if (!chunkHasBuriedRef[i]) continue;
                            int cx = i % 16, cy = i / 16;
                            for (int dy = -1; dy <= 1; dy++)
                            {
                                for (int dx = -1; dx <= 1; dx++)
                                {
                                    if (!holesNeighbors && (dx != 0 || dy != 0)) continue;
                                    int nx = cx + dx, ny = cy + dy;
                                    if (nx < 0 || ny < 0 || nx >= 16 || ny >= 16) continue;
                                    int j = ny * 16 + nx;
                                    if (chunkHasHoles[j])
                                    {
                                        if (!(holesPreserveWmo && chunkHasKeepWmo[j]))
                                            toClear[j] = true;
                                    }
                                }
                            }
                        }
                        for (int j = 0; j < 256; j++)
                        {
                            if (!toClear[j]) continue;
                            int off = holesOffsetByIdx[j];
                            if (off >= 0 && off + 4 <= bytes.Length)
                            {
                                if (bytes[off + 0] != 0 || bytes[off + 1] != 0 || bytes[off + 2] != 0 || bytes[off + 3] != 0)
                                {
                                    bytes[off + 0] = 0; bytes[off + 1] = 0; bytes[off + 2] = 0; bytes[off + 3] = 0;
                                    holesCleared++;
                                }
                            }
                        }
                    }

                    // MCSH zeroing pass
                    if (disableMcsh)
                    {
                        for (int i = 0; i < 256; i++)
                        {
                            int mcnkOffset = (mcinBytes.Length >= (i + 1) * 16) ? BitConverter.ToInt32(mcinBytes, i * 16) : 0;
                            if (mcnkOffset <= 0) continue;
                            int headerStart = mcnkOffset + 8;
                            if (headerStart + 128 > bytes.Length) continue;
                            int mcshOffset = BitConverter.ToInt32(bytes, headerStart + 0x30);
                            int mcshSize = BitConverter.ToInt32(bytes, headerStart + 0x34);
                            if (mcshSize > 0)
                            {
                                long payloadStart = (long)headerStart + 128 + mcshOffset;
                                long payloadEnd = payloadStart + mcshSize;
                                if (payloadStart >= 0 && payloadEnd <= bytes.Length)
                                {
                                    Array.Clear(bytes, (int)payloadStart, mcshSize);
                                    mcshZeroed++;
                                }
                            }
                        }
                    }
                }
            }

            // Write output preserving relative structure
            var rel = Path.GetRelativePath(inputDir, inPath);
            var outPath = Path.Combine(lkAdtRoot, rel);
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
            File.WriteAllBytes(outPath, bytes);
            filesProcessed++;
            if (filesProcessed % 50 == 0) Console.WriteLine($"  [lk] Patched {filesProcessed}/{allAdts.Count} ADTs...");
        }

        Console.WriteLine($"[ok] LK patch complete: files={filesProcessed}, placementsProcessed={placementsProcessed}, buried={placementsBuried}");
        if (fixHoles || disableMcsh)
        {
            Console.WriteLine($"[ok] MCNK pass: holesCleared={holesCleared}, mcshZeroed={mcshZeroed}, mcnkScanned={mcnkScanned}");
        }
        return 0;
    }

    private static int FindChunk(byte[] bytes, string reversedFourCC)
    {
        var pattern = System.Text.Encoding.ASCII.GetBytes(reversedFourCC);
        for (int i = 0; i < bytes.Length - 4; i++)
        {
            if (bytes[i] == pattern[0] && bytes[i + 1] == pattern[1] && bytes[i + 2] == pattern[2] && bytes[i + 3] == pattern[3])
                return i;
        }
        return -1;
    }

    private static string? GenerateCrosswalksIfNeeded(string alias, string inputPath, string? dbdDirOpt, string? srcDbcDirOpt, string lkDbcDir, string? pivot060DirOpt, bool chainVia060, string outRoot, string? srcClientDirOpt, string? lkClientDirOpt, string? pivot060ClientDirOpt)
    {
        try
        {
            // Ensure LK DBC dir (prefer provided, else extract from MPQs)
            string? lkDbcResolved = null;
            if (!string.IsNullOrWhiteSpace(lkDbcDir) && Directory.Exists(lkDbcDir)) lkDbcResolved = lkDbcDir;
            else lkDbcResolved = EnsureDbcDirFromClient("3.3.5", null, lkClientDirOpt, outRoot);
            if (string.IsNullOrWhiteSpace(lkDbcResolved)) { Console.WriteLine("[patchmap] generation requires LK DBCs via --lk-dbc-dir or --lk-client-path"); return null; }

            string? dbdDir = null;
            if (!string.IsNullOrWhiteSpace(dbdDirOpt) && Directory.Exists(dbdDirOpt)) dbdDir = dbdDirOpt;
            else
            {
                var probes = new[]
                {
                    Path.Combine(Directory.GetCurrentDirectory(), "lib", "WoWDBDefs", "definitions"),
                    Path.Combine(AppContext.BaseDirectory, "..","..","lib", "WoWDBDefs", "definitions"),
                    Path.Combine(AppContext.BaseDirectory, "..","..","..", "lib", "WoWDBDefs", "definitions"),
                };
                foreach (var p in probes)
                {
                    var full = Path.GetFullPath(p);
                    if (Directory.Exists(full)) { dbdDir = full; break; }
                }
            }
            if (string.IsNullOrWhiteSpace(dbdDir))
            {
                Console.WriteLine("[patchmap] generation requires --dbd-dir (WoWDBDefs/definitions). Not found in common locations.");
                return null;
            }

            // Ensure source DBC dir (prefer provided, else infer from input folder, else extract from MPQs)
            string? srcDbcDir = null;
            if (!string.IsNullOrWhiteSpace(srcDbcDirOpt) && Directory.Exists(srcDbcDirOpt)) srcDbcDir = srcDbcDirOpt;
            if (string.IsNullOrWhiteSpace(srcDbcDir))
            {
                var wdtDir = Path.GetDirectoryName(Path.GetFullPath(inputPath)) ?? string.Empty;
                var cur = new DirectoryInfo(wdtDir);
                for (int i = 0; i < 8 && cur != null; i++)
                {
                    var cand = Path.Combine(cur.FullName, "DBFilesClient");
                    if (Directory.Exists(cand)) { srcDbcDir = cand; break; }
                    cur = cur.Parent;
                }
            }
            if (string.IsNullOrWhiteSpace(srcDbcDir))
            {
                srcDbcDir = EnsureDbcDirFromClient(alias, null, srcClientDirOpt, outRoot);
            }
            if (string.IsNullOrWhiteSpace(srcDbcDir)) { Console.WriteLine("[patchmap] generation requires source DBCs via --src-dbc-dir or --src-client-path"); return null; }

            var inputs = new List<(string build, string dir)> { (alias, srcDbcDir), ("3.3.5", lkDbcResolved!) };
            if (!string.IsNullOrWhiteSpace(pivot060DirOpt) && Directory.Exists(pivot060DirOpt))
            {
                inputs.Add(("0.6.0", pivot060DirOpt));
            }
            else if (!string.IsNullOrWhiteSpace(pivot060ClientDirOpt))
            {
                var pivot060 = EnsureDbcDirFromClient("0.6.0", null, pivot060ClientDirOpt, outRoot);
                if (!string.IsNullOrWhiteSpace(pivot060)) inputs.Add(("0.6.0", pivot060));
            }

            var outBase = Path.Combine(outRoot, "dbctool_outputs");
            Directory.CreateDirectory(outBase);
            Console.WriteLine($"[patchmap] generating crosswalks -> {outBase} (alias={alias}, via060={(chainVia060 ? "on" : "off")})");
            var cmd = new CompareAreaV2Command();
            var rc = cmd.Run(dbdDir, outBase, "enUS", inputs, chainVia060);
            if (rc != 0)
            {
                Console.WriteLine($"[patchmap] generation failed with exit code {rc}");
                return null;
            }
            var compareV2 = Path.Combine(outBase, alias, "compare", "v2");
            return compareV2;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[patchmap] generation error: {ex.Message}");
            return null;
        }
    }

    private static int RunAnalyzeMapAdts(Dictionary<string, string> opts)
    {
        Require(opts, "map-dir");
        Require(opts, "map");
        
        var mapDir = opts["map-dir"];
        var mapName = opts["map"];
        var outDir = opts.GetValueOrDefault("out", Path.Combine("analysis_output", mapName));

        Console.WriteLine($"[info] Analyzing ADT files for map: {mapName}");
        Console.WriteLine($"[info] Map directory: {mapDir}");
        Console.WriteLine($"[info] Output directory: {outDir}");

        // Step 1: Extract placements from ADT files
        Console.WriteLine("\n=== Step 1: Extracting placements from ADT files ===");
        var extractor = new AdtPlacementsExtractor();
        var placementsCsvPath = Path.Combine(outDir, $"{mapName}_placements.csv");
        var extractResult = extractor.Extract(mapDir, mapName, placementsCsvPath);

        if (!extractResult.Success)
        {
            Console.Error.WriteLine($"[error] Placement extraction failed: {extractResult.ErrorMessage}");
            return 1;
        }

        Console.WriteLine($"[ok] Extracted {extractResult.M2Count} M2 placements");
        Console.WriteLine($"[ok] Extracted {extractResult.WmoCount} WMO placements");
        Console.WriteLine($"[ok] Processed {extractResult.TilesProcessed} tiles");
        Console.WriteLine($"[ok] Placements CSV: {placementsCsvPath}");

        // Step 1.5: Process minimap tiles
        Console.WriteLine("\n=== Processing minimap tiles ===");
        var minimapHandler = new MinimapHandler();
        var minimapResult = minimapHandler.ProcessMinimaps(mapDir, mapName, outDir);

        if (minimapResult.Success && minimapResult.TilesCopied > 0)
        {
            Console.WriteLine($"[ok] Copied {minimapResult.TilesCopied} minimap tiles");
            Console.WriteLine($"[ok] Minimap directory: {minimapResult.MinimapDir}");
        }
        else if (minimapResult.Success)
        {
            Console.WriteLine($"[info] No minimap PNG files found");
            Console.WriteLine($"[info] Expected location: World\\Textures\\Minimap\\ or World\\Textures\\Minimap\\{mapName}\\");
            Console.WriteLine($"[info] Place PNGs named {mapName}_X_Y.png or mapX_Y.png for minimap support");
        }
        else
        {
            Console.WriteLine($"[warn] Minimap processing failed: {minimapResult.ErrorMessage}");
        }

        // Step 1.75: Extract terrain data (MCNK chunks) for terrain overlays
        Console.WriteLine("\n=== Extracting terrain data (MCNK chunks) ===");
        var terrainExtractor = new AdtTerrainExtractor();
        var terrainResult = terrainExtractor.ExtractTerrainForMap(mapDir, mapName, outDir);
        
        if (!terrainResult.Success)
        {
            Console.WriteLine($"[warn] Terrain extraction failed - terrain overlays will not be available");
        }
        else
        {
            Console.WriteLine($"[ok] Extracted {terrainResult.ChunksExtracted} MCNK chunks from {terrainResult.TilesProcessed} tiles");
            Console.WriteLine($"[ok] Terrain CSV: {terrainResult.CsvPath}");
        }

        // Step 2: Analyze UniqueIDs and detect layers
        Console.WriteLine("\n=== Step 2: Analyzing UniqueIDs and detecting layers ===");
        var analyzer = new UniqueIdAnalyzer(gapThreshold: 100);
        var analysisResult = analyzer.AnalyzeFromPlacementsCsv(placementsCsvPath, mapName, outDir);

        if (!analysisResult.Success)
        {
            Console.Error.WriteLine($"[error] UniqueID analysis failed: {analysisResult.ErrorMessage}");
            return 1;
        }

        Console.WriteLine($"[ok] Analyzed {analysisResult.TileCount} tiles");
        Console.WriteLine($"[ok] UniqueID analysis CSV: {analysisResult.CsvPath}");
        Console.WriteLine($"[ok] Layers JSON: {analysisResult.LayersJsonPath}");

        // Step 3: Detect spatial clusters and patterns (prefabs/brushes)
        Console.WriteLine("\n=== Step 3: Detecting spatial clusters and patterns ===");
        var clusterAnalyzer = new ClusterAnalyzer(proximityThreshold: 50.0f, minClusterSize: 3);
        var clusterResult = clusterAnalyzer.Analyze(placementsCsvPath, mapName, outDir);

        if (!clusterResult.Success)
        {
            Console.WriteLine($"[warn] Cluster analysis failed: {clusterResult.ErrorMessage}");
        }
        else
        {
            Console.WriteLine($"[ok] Detected {clusterResult.TotalClusters} spatial clusters");
            Console.WriteLine($"[ok] Identified {clusterResult.TotalPatterns} recurring patterns (potential prefabs)");
            Console.WriteLine($"[ok] Clusters JSON: {clusterResult.ClustersJsonPath}");
            Console.WriteLine($"[ok] Patterns JSON: {clusterResult.PatternsJsonPath}");
            Console.WriteLine($"[ok] Summary CSV: {clusterResult.SummaryCsvPath}");
        }

        // Step 4: Generate viewer using existing infrastructure
        Console.WriteLine("\n=== Step 4: Generating viewer ===");
        var viewerAdapter = new AnalysisViewerAdapter();
        var viewerRoot = viewerAdapter.GenerateViewer(placementsCsvPath, mapName, outDir, minimapResult.MinimapDir);

        if (!string.IsNullOrEmpty(viewerRoot))
        {
            Console.WriteLine($"[ok] Viewer generated: {viewerRoot}");
            Console.WriteLine($"[info] Open: {Path.Combine(viewerRoot, "index.html")}");
        }
        else
        {
            Console.WriteLine($"[warn] Viewer generation skipped (no placements)");
        }

        Console.WriteLine("\n=== Analysis Complete ===");
        Console.WriteLine($"All outputs written to: {outDir}");
        Console.WriteLine("\nℹ️  Spatial clusters reveal object groups placed together - likely prefabs or brushes");
        Console.WriteLine("ℹ️  Recurring patterns show reused object compositions across the map");
        Console.WriteLine("ℹ️  Open the viewer in a web browser to explore your map interactively");
        
        // Auto-serve if --serve flag provided
        if (opts.ContainsKey("serve"))
        {
            Console.WriteLine("\n=== Starting built-in HTTP server ===");
            var port = 8080;
            if (opts.TryGetValue("port", out var portStr) && int.TryParse(portStr, out var parsedPort))
            {
                port = parsedPort;
            }
            
            ViewerServer.Serve(viewerRoot, port, openBrowser: true);
        }
        
        return 0;
    }

    private static int RunServeViewer(Dictionary<string, string> opts)
    {
        // Get viewer directory
        var viewerDir = opts.GetValueOrDefault("viewer-dir", "");
        
        // If not specified, look for common locations
        if (string.IsNullOrWhiteSpace(viewerDir))
        {
            var candidates = new[]
            {
                "analysis_output/viewer",
                "rollback_outputs/viewer",
                "viewer"
            };
            
            foreach (var candidate in candidates)
            {
                if (Directory.Exists(candidate))
                {
                    viewerDir = candidate;
                    break;
                }
            }
        }
        
        if (string.IsNullOrWhiteSpace(viewerDir) || !Directory.Exists(viewerDir))
        {
            Console.Error.WriteLine("[error] Viewer directory not found.");
            Console.Error.WriteLine("[info] Usage: dotnet run -- serve-viewer [--viewer-dir <path>] [--port <port>] [--no-browser]");
            Console.Error.WriteLine("[info] Common locations checked: analysis_output/viewer, rollback_outputs/viewer, viewer");
            return 1;
        }
        
        // Get port (default 8080)
        var port = 8080;
        if (opts.TryGetValue("port", out var portStr) && int.TryParse(portStr, out var parsedPort))
        {
            port = parsedPort;
        }
        
        // Check if should open browser
        var openBrowser = !opts.ContainsKey("no-browser");
        
        Console.WriteLine($"[info] Starting viewer server...");
        ViewerServer.Serve(viewerDir, port, openBrowser);
        
        return 0;
    }

    private static int RunGenAreaRemap(Dictionary<string, string> opts)
    {
        Console.WriteLine("[info] gen-area-remap is not yet implemented in this build.");
        Console.WriteLine("[info] Provide --area-remap-json to supply explicit mapping, or use --lk-client-path once available.");
        return 0;
    }

    private static int RunDryRun(Dictionary<string, string> opts)
    {
        Require(opts, "input-dir");
        var map = opts["map"]; var inputDir = opts["input-dir"]; var outRoot = opts.GetValueOrDefault("out", "");

        RangeConfig config = new RangeConfig { Map = map, Mode = opts.GetValueOrDefault("mode", "keep") };
        if (opts.TryGetValue("config", out var cfgPath) && File.Exists(cfgPath))
        {
            config = RangeConfigLoader.LoadFromJson(cfgPath);
        }

        var keepRanges = opts.TryGetValue("keep-range", out var keepSpec) ? new[] { keepSpec } : Array.Empty<string>();
        var dropRanges = opts.TryGetValue("drop-range", out var dropSpec) ? new[] { dropSpec } : Array.Empty<string>();
        RangeConfigLoader.ApplyCliOverrides(config, keepRanges, dropRanges);

        var adts = Directory.EnumerateFiles(inputDir, "*.adt", SearchOption.AllDirectories)
            .Where(p => Path.GetFileNameWithoutExtension(p).StartsWith(map + "_", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p)
            .ToList();

        Console.WriteLine($"[info] Dry-run for map={map}, mode={config.Mode}, include={config.IncludeRanges.Count}, exclude={config.ExcludeRanges.Count}");
        int total = 0, removed = 0;
        foreach (var adt in adts)
        {
            int localTotal = 0, localRemoved = 0;
            foreach (var entry in AdtPlacementAnalyzer.EnumeratePlacements(adt))
            {
                localTotal++;
                if (RangeSelector.ShouldRemove(config, entry.UniqueId)) localRemoved++;
            }
            total += localTotal; removed += localRemoved;
            Console.WriteLine($"{Path.GetFileName(adt)}, total={localTotal}, would_remove={localRemoved}");
        }
        Console.WriteLine($"[summary] total={total}, would_remove={removed}");
        return 0;
    }

    private static void Require(Dictionary<string, string> opts, string key)
    {
        if (!opts.ContainsKey(key) || string.IsNullOrWhiteSpace(opts[key]))
            throw new ArgumentException($"Missing required --{key}");
    }
    
    private static string? ExtractMinimapsFromMpq(IArchiveSource src, string mapName, string outputDir)
    {
        try
        {
            // Load md5translate if present (checks loose files first, then MPQ)
            WoWRollback.Core.Services.Minimap.Md5TranslateIndex? index = null;
            string? md5Path = null;
            if (Md5TranslateResolver.TryLoad(src, out var loaded, out md5Path))
            {
                index = loaded;
                Console.WriteLine($"[info] Loaded md5translate from: {md5Path}");
                Console.WriteLine($"[info] md5translate contains {(loaded?.PlainToHash.Count ?? 0)} minimap mappings");
            }
            else
            {
                Console.WriteLine($"[info] No md5translate file found, using direct BLP paths");
            }

            var resolver = new MinimapFileResolver(src, index);
            var minimapOutDir = Path.Combine(outputDir, "minimaps");
            Directory.CreateDirectory(minimapOutDir);

            // Scan for all tiles (0-63 grid)
            int extracted = 0;
            int failed = 0;
            for (int x = 0; x < 64; x++)
            {
                for (int y = 0; y < 64; y++)
                {
                    if (resolver.TryResolveTile(mapName, x, y, out var virtualPath) && !string.IsNullOrEmpty(virtualPath))
                    {
                        try
                        {
                            // Read BLP (IArchiveSource checks loose files FIRST, then MPQ)
                            using var blpStream = src.OpenFile(virtualPath);
                            using var ms = new MemoryStream();
                            blpStream.CopyTo(ms);
                            var blpData = ms.ToArray();
                            
                            // Convert BLP to JPG using Warcraft.NET
                            var blp = new Warcraft.NET.Files.BLP.BLP(blpData);
                            var image = blp.GetMipMap(0); // Get highest resolution mipmap
                            
                            var outputPath = Path.Combine(minimapOutDir, $"{mapName}_{x}_{y}.jpg");
                            using var outStream = File.Create(outputPath);
                            image.Save(outStream, new JpegEncoder { Quality = 85 });
                            extracted++;
                            
                            if (extracted == 1)
                            {
                                Console.WriteLine($"[info] First minimap found: {virtualPath}");
                            }
                        }
                        catch (Exception ex)
                        {
                            failed++;
                            if (failed <= 3) // Only log first few failures
                            {
                                Console.WriteLine($"[debug] Failed to extract tile [{x},{y}] from {virtualPath}: {ex.Message}");
                            }
                        }
                    }
                }
            }

            if (extracted > 0)
            {
                Console.WriteLine($"[info] Extracted and converted {extracted} minimap tiles (BLP→JPG)");
                if (failed > 0)
                {
                    Console.WriteLine($"[info] Failed to extract {failed} tiles (missing or corrupt)");
                }
                return minimapOutDir;
            }

            Console.WriteLine($"[warn] No minimap tiles found for {mapName}");
            Console.WriteLine($"[info] Checked for: md5translate.txt/trs, loose BLPs in Data/, and MPQ archives");
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[warn] Minimap extraction failed: {ex.Message}");
            return null;
        }
    }
    
    private static string? ExtractVersionFromPath(string clientPath)
    {
        // Try to extract version from path patterns:
        // - E:\Archive\0.6.0.3592\World of Warcraft\Data
        // - E:\Archive\0.X_Pre-Release_Windows_enUS_0.6.0.3592\World of Warcraft
        // - E:\WoW_Clients\1.12.1\Data
        // - E:\WoW\3.3.5a\World of Warcraft
        
        var currentPath = clientPath;
        
        // Walk up the directory tree looking for version pattern
        for (int i = 0; i < 5; i++) // Check up to 5 levels up
        {
            if (string.IsNullOrEmpty(currentPath))
                break;
                
            // Try to find version pattern in current path segment
            var dirName = Path.GetFileName(currentPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            
            // Pattern 1: Full version (0.6.0.3592, 1.12.1.5875, 3.3.5.12340)
            var match = System.Text.RegularExpressions.Regex.Match(dirName, @"^(\d+\.\d+\.\d+\.\d+)");
            if (match.Success)
            {
                Console.WriteLine($"[info] Detected build version from path: {match.Groups[1].Value}");
                return match.Groups[1].Value;
            }
            
            // Pattern 2: Version with suffix (3.3.5a, 1.12.1b)
            match = System.Text.RegularExpressions.Regex.Match(dirName, @"^(\d+\.\d+\.\d+)[a-z]?$");
            if (match.Success)
            {
                // For versions without build number, add a default
                var version = match.Groups[1].Value;
                var withBuild = version + ".0"; // Default build 0
                Console.WriteLine($"[info] Detected version from path: {version} (using {withBuild})");
                return withBuild;
            }
            
            // Pattern 3: Version embedded in longer string (0.X_Pre-Release_Windows_enUS_0.6.0.3592)
            match = System.Text.RegularExpressions.Regex.Match(dirName, @"(\d+\.\d+\.\d+\.\d+)");
            if (match.Success)
            {
                Console.WriteLine($"[info] Detected build version from path: {match.Groups[1].Value}");
                return match.Groups[1].Value;
            }
            
            // Move up one directory
            currentPath = Path.GetDirectoryName(currentPath);
        }
        
        Console.WriteLine("[warn] Could not detect build version from path, using default");
        return null;
    }

    private static Dictionary<string, string> ParseArgs(string[] args)
    {
        var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a.StartsWith("--"))
            {
                var key = a.Substring(2);
                string val = "true";
                if (i + 1 < args.Length && !args[i + 1].StartsWith("--")) { val = args[++i]; }
                dict[key] = val;
            }
        }
        return dict;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("WoWRollback CLI - Digital Archaeology of World of Warcraft Development");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  gui  [--cache <dir>] [--presets <dir>]");
        Console.WriteLine("    Launch native GUI (Avalonia) to manage presets across ALL maps");
        Console.WriteLine();
        Console.WriteLine("  run-preset  --preset <file> [--maps all|m1,m2] [--out-root <dir>] [--lk-out <dir>] [--dry-run]");
        Console.WriteLine("    Apply a preset to selected maps. Dry-run prints summary only in this build");
        Console.WriteLine();
        Console.WriteLine("  prepare-layers  [--wdt <WDT>] | [--client-root <dir> [--maps all|m1,m2]] [--out <dir>] [--gap-threshold <N>]");
        Console.WriteLine("    Build per-map layer caches (placements, tile_layers.csv, layers.json) without patching");
        Console.WriteLine("    Outputs under <out>/<map>/; usable by GUI and Layers UI");
        Console.WriteLine();
        Console.WriteLine("  discover-maps  --client-path <dir> [--version <ver>] [--dbd-dir <path>] [--out <csv>]");
        Console.WriteLine("    Discover all maps from Map.dbc and analyze their WDT files");
        Console.WriteLine("    Shows terrain vs WMO-only maps, tile counts, and hybrid maps");
        Console.WriteLine("    Version auto-detected from path or use --version (e.g., 0.6.0.3592)");
        Console.WriteLine();
        Console.WriteLine("  analyze-map-adts-mpq  --client-path <dir> (--map <name> | --all-maps) [--out <dir>] [--version <ver>]");
        Console.WriteLine("                        [--serve] [--port <port>]");
        Console.WriteLine("    Analyze ADT files directly from MPQs (patched view), detect layers/clusters, and generate viewer");
        Console.WriteLine("    Use --all-maps to discover and analyze all maps from Map.dbc");
        Console.WriteLine("    Version auto-detected from path or use --version (e.g., 0.6.0.3592)");
        Console.WriteLine();
        Console.WriteLine("  analyze-map-adts  --map <name> --map-dir <dir> [--out <dir>] [--serve] [--port <port>]");
        Console.WriteLine("    Analyze ADT files (pre-Cata or Cata+ split) from loose files");
        Console.WriteLine("    Supports 0.6.0 through 4.0.0+ ADT formats");
        Console.WriteLine();
        Console.WriteLine("  analyze-alpha-wdt --wdt-file <path> [--out <dir>]");
        Console.WriteLine("    Extract UniqueID ranges from Alpha WDT files (archaeological excavation)");
        Console.WriteLine();
        Console.WriteLine("  rollback  --input <WDT> --max-uniqueid <N> [--bury-depth <float>] [--out <dir>] [--fix-holes] [--disable-mcsh] [--export-lk-adts] [--lk-out <dir>] [--lk-client-path <dir>] [--area-remap-json <path>] [--default-unmapped <id>] [--force]");
        Console.WriteLine("    Modify Alpha WDT by burying placements with UniqueID > N, then write output + MD5");
        Console.WriteLine("    --fix-holes        Clear MCNK Holes flags across all chunks (terrain hole masks)");
        Console.WriteLine("    --disable-mcsh     Zero MCSH subchunk payloads (remove baked shadows)");
        Console.WriteLine("    --export-lk-adts   After writing modified WDT, convert present tiles to LK ADT files");
        Console.WriteLine("    --lk-out <dir>     Output directory root for LK ADTs (default: <out>/lk_adts/World/Maps/<map>)");
        Console.WriteLine("    --force            Rebuild even if LK ADTs appear complete (disables preflight skip)");
        Console.WriteLine("    --lk-client-path   LK client folder with MPQs (used for AreaTable auto-mapping)");
        Console.WriteLine("    --area-remap-json  JSON file mapping AlphaAreaId->LKAreaId to set MCNK.AreaId");
        Console.WriteLine("    --default-unmapped AreaId to use when no mapping/ID exists in LK (default 0)");
        Console.WriteLine("    --crosswalk-dir    Preferred: directory of crosswalk CSVs (Area_patch_crosswalk_*.csv)");
        Console.WriteLine("    --crosswalk-file   Preferred: specific crosswalk CSV to load (resolved against dir or out root)");
        Console.WriteLine("    --dbctool-out-root Root of DBCTool.V2 output (expects <alias>/compare/v2|v3 for crosswalk CSVs)");
        Console.WriteLine("    --dbctool-patch-dir Legacy alias for --crosswalk-dir");
        Console.WriteLine("    --dbctool-patch-file Legacy alias for --crosswalk-file");
        Console.WriteLine("    --lk-dbc-dir Directory with extracted LK DBCs (Map.dbc/AreaTable.dbc), else read from --lk-client-path");
        Console.WriteLine("    Default bury-depth = -5000.0, default out dir = <input_basename>_out next to input");
        Console.WriteLine();
        Console.WriteLine("  alpha-to-lk  --input <WDT> --max-uniqueid <N> [--bury-depth <float>] [--out <dir>] [--fix-holes] [--holes-scope self|neighbors] [--holes-wmo-preserve true|false] [--disable-mcsh] [--lk-out <dir>] [--lk-client-path <dir>] [--area-remap-json <path>] [--default-unmapped <id>] [--force]");
        Console.WriteLine("    One-shot: rollback + (optional) fix-holes/MCSH + LK export with AreaTable mapping");
        Console.WriteLine("    Crosswalk mapping (strict, map-locked):");
        Console.WriteLine("      --crosswalk-dir <dir>         Preferred per-run CSV directory (Area_patch_crosswalk_*.csv)");
        Console.WriteLine("      --crosswalk-file <file>       Preferred specific CSV (resolved vs dir/out roots)");
        Console.WriteLine("      --dbctool-out-root <root>     Root of crosswalk outputs (<alias>/compare/v2|v3)");
        Console.WriteLine("      --dbctool-patch-dir <dir>     Legacy alias for --crosswalk-dir");
        Console.WriteLine("      --dbctool-patch-file <file>   Legacy alias for --crosswalk-file");
        Console.WriteLine("      --lk-dbc-dir <dir>            LK DBFilesClient (required for guard and auto-gen)");
        Console.WriteLine("      --strict-areaid [true|false]  Strict map-locked patching (default true)");
        Console.WriteLine("      --report-areaid               Write per-ADT and summary CSVs");
        Console.WriteLine("      --copy-crosswalks             Copy used CSVs into <session>/reports/crosswalk");
        Console.WriteLine("    Preflight (skip-if-exists):");
        Console.WriteLine("      LK export is skipped when <map>.wdt and all ADTs already exist in --lk-out; pass --force to rebuild");
        Console.WriteLine("    Auto-generate crosswalks (default on when none found):");
        Console.WriteLine("      --auto-crosswalks [true|false]  Enable CSV generation via DBCTool.V2 (default true)");
        Console.WriteLine("      --dbd-dir <dir>                WoWDBDefs/definitions path (required if not probed)");
        Console.WriteLine("      --src-dbc-dir <dir>            Source (Alpha) DBFilesClient directory");
        Console.WriteLine("      --src-client-path <dir>        Source client root (MPQs); auto-extract DBCs when no --src-dbc-dir");
        Console.WriteLine("      --lk-client-path <dir>         LK client root (MPQs); auto-extract DBCs when no --lk-dbc-dir");
        Console.WriteLine("      --pivot-060-dbc-dir <dir>      Optional 0.6.0 DBFilesClient for pivot");
        Console.WriteLine("      --pivot-060-client-path <dir>  Optional 0.6.0 client root (MPQs) for pivot extraction");
        Console.WriteLine("      --chain-via-060                Force pivot chain resolution via 0.6.0");
        Console.WriteLine();
        Console.WriteLine("  lk-to-alpha  --lk-adts-dir <dir> --map <name> --max-uniqueid <N> [--bury-depth <float>] [--out <dir>] [--fix-holes] [--holes-scope self|neighbors] [--holes-wmo-preserve true|false] [--disable-mcsh]");
        Console.WriteLine("    Patch existing LK ADTs: bury placements with UniqueID > N, optionally clear holes (MCRF-gated) and zero MCSH");
        Console.WriteLine();
        Console.WriteLine("  probe-archive    --client-path <dir> [--map <name>] [--limit <n>]");
        Console.WriteLine("    Probe mixed inputs (loose Data + MPQs)");
        Console.WriteLine();
        Console.WriteLine("  probe-minimap    --client-path <dir> --map <name> [--limit <n>]");
        Console.WriteLine("    Resolve sample minimap tiles using md5translate");
        Console.WriteLine();
        Console.WriteLine("  serve-viewer  [--viewer-dir <path>] [--port <port>] [--no-browser]");
        Console.WriteLine("    Start built-in HTTP server to host the viewer");
        Console.WriteLine();
        Console.WriteLine("Archaeological Perspective:");
        Console.WriteLine("  Each UniqueID range represents a 'volume of work' by ancient developers.");
        Console.WriteLine("  We're uncovering sedimentary layers of 20+ years of WoW development history.");
    }

    private static int RunDiscoverMaps(Dictionary<string, string> opts)
    {
        Require(opts, "client-path");
        var clientRoot = opts["client-path"];
        
        // Try auto-detection first, then check both --version and --build parameters
        var detectedVersion = ExtractVersionFromPath(clientRoot);
        var buildVersion = opts.GetValueOrDefault("version", 
                          opts.GetValueOrDefault("build", 
                          detectedVersion ?? "0.5.3"));
        
        var dbdDir = opts.GetValueOrDefault("dbd-dir", Path.Combine(Directory.GetCurrentDirectory(), "..", "lib", "WoWDBDefs", "definitions"));
        var outCsv = opts.GetValueOrDefault("out", "discovered_maps.csv");

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"[error] Client path not found: {clientRoot}");
            return 1;
        }

        if (!Directory.Exists(dbdDir))
        {
            Console.Error.WriteLine($"[error] DBD definitions not found: {dbdDir}");
            Console.Error.WriteLine($"[info] Clone WoWDBDefs: git clone https://github.com/wowdev/WoWDBDefs.git");
            return 1;
        }

        Console.WriteLine($"[info] Discovering maps from: {clientRoot}");
        Console.WriteLine($"[info] Build version: {buildVersion}");
        Console.WriteLine($"[info] DBD definitions: {dbdDir}");

        EnsureStormLibOnPath();
        var mpqs = ArchiveLocator.LocateMpqs(clientRoot);
        using var src = new PrioritizedArchiveSource(clientRoot, mpqs);

        // Extract Map.dbc to temp directory
        var tempDir = Path.Combine(Path.GetTempPath(), "wowrollback_dbc_" + Guid.NewGuid().ToString("N"));
        var discoveryService = new MapDiscoveryService(dbdDir);
        
        Console.WriteLine($"[info] Extracting Map.dbc from MPQ...");
        var dbcDir = discoveryService.ExtractMapDbc(src, tempDir);
        
        if (string.IsNullOrEmpty(dbcDir))
        {
            Console.Error.WriteLine($"[error] Failed to extract Map.dbc from MPQ");
            return 1;
        }

        Console.WriteLine($"[info] Analyzing maps and WDT files...");
        var result = discoveryService.DiscoverMaps(src, buildVersion, Path.GetDirectoryName(dbcDir)!);

        if (!result.Success)
        {
            Console.Error.WriteLine($"[error] {result.ErrorMessage}");
            return 1;
        }

        // Write CSV
        var csv = new System.Text.StringBuilder();
        csv.AppendLine("id,name,folder,wdt_exists,map_type,tile_count,has_wmo,wmo_path");

        foreach (var map in result.Maps.OrderBy(m => m.Id))
        {
            var wmoPath = map.WmoPlacement?.WmoPath ?? "";
            csv.AppendLine($"{map.Id},{map.Name},{map.Folder},{map.WdtExists},{map.MapType},{map.TileCount},{map.WmoPlacement != null},{wmoPath}");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outCsv))!);
        File.WriteAllText(outCsv, csv.ToString());

        // Print summary
        Console.WriteLine($"\n[ok] Discovered {result.Maps.Length} maps");
        Console.WriteLine($"[ok] Maps CSV: {outCsv}");
        
        var terrainMaps = result.Maps.Count(m => m.HasTerrain && !m.IsWmoOnly);
        var wmoOnlyMaps = result.Maps.Count(m => m.IsWmoOnly && !m.HasTerrain);
        var hybridMaps = result.Maps.Count(m => m.IsWmoOnly && m.HasTerrain);
        var noWdtMaps = result.Maps.Count(m => !m.WdtExists);

        Console.WriteLine($"\n=== Map Type Summary ===");
        Console.WriteLine($"  Terrain maps: {terrainMaps}");
        Console.WriteLine($"  WMO-only maps: {wmoOnlyMaps}");
        Console.WriteLine($"  Hybrid (WMO + Terrain): {hybridMaps}");
        Console.WriteLine($"  No WDT: {noWdtMaps}");

        // Cleanup temp directory
        try { Directory.Delete(tempDir, true); } catch { }

        return 0;
    }

    private static int RunAnalyzeMapAdtsMpq(Dictionary<string, string> opts)
    {
        Require(opts, "client-path");
        var clientRoot = opts["client-path"];
        var allMaps = opts.ContainsKey("all-maps");
        var mapName = opts.GetValueOrDefault("map", "");
        
        if (!allMaps && string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("[error] Either --map <name> or --all-maps is required");
            return 1;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"[error] client path not found: {clientRoot}");
            return 1;
        }

        EnsureStormLibOnPath();
        var mpqs = ArchiveLocator.LocateMpqs(clientRoot);
        using var src = new PrioritizedArchiveSource(clientRoot, mpqs);

        // Batch mode: discover all maps and analyze each
        if (allMaps)
        {
            return RunBatchAnalysis(src, clientRoot, opts);
        }

        // Single map mode
        var outDir = opts.GetValueOrDefault("out", Path.Combine("analysis_output", mapName));
        return AnalyzeSingleMap(src, clientRoot, mapName, outDir, opts);
    }

    private static int RunBatchAnalysis(IArchiveSource src, string clientRoot, Dictionary<string, string> opts)
    {
        // Try auto-detection first, then check both --version and --build parameters
        var detectedVersion = ExtractVersionFromPath(clientRoot);
        var buildVersion = opts.GetValueOrDefault("version", 
                          opts.GetValueOrDefault("build", 
                          detectedVersion ?? "0.6.0"));
        
        var dbdDir = opts.GetValueOrDefault("dbd-dir", Path.Combine(Directory.GetCurrentDirectory(), "..", "lib", "WoWDBDefs", "definitions"));
        var baseOutDir = opts.GetValueOrDefault("out", "analysis_output");
        var versionLabel = buildVersion; // Use the resolved version

        Console.WriteLine($"[info] === Batch Analysis Mode ===");
        Console.WriteLine($"[info] Discovering maps from Map.dbc...");

        // Extract Map.dbc and discover maps
        var tempDir = Path.Combine(Path.GetTempPath(), "wowrollback_dbc_" + Guid.NewGuid().ToString("N"));
        var discoveryService = new MapDiscoveryService(dbdDir);
        
        var dbcDir = discoveryService.ExtractMapDbc(src, tempDir);
        if (string.IsNullOrEmpty(dbcDir))
        {
            Console.Error.WriteLine($"[error] Failed to extract Map.dbc from MPQ");
            return 1;
        }

        var result = discoveryService.DiscoverMaps(src, buildVersion, Path.GetDirectoryName(dbcDir)!);
        if (!result.Success)
        {
            Console.Error.WriteLine($"[error] {result.ErrorMessage}");
            return 1;
        }

        // Filter to terrain maps only (skip WMO-only for now)
        var terrainMaps = result.Maps.Where(m => m.HasTerrain && m.TileCount > 0).ToList();
        
        Console.WriteLine($"[ok] Discovered {result.Maps.Length} total maps");
        Console.WriteLine($"[info] Analyzing {terrainMaps.Count} terrain maps (skipping WMO-only maps)");

        int successCount = 0;
        int failCount = 0;
        var failedMaps = new List<string>();
        var processedMaps = new List<(string MapName, string PlacementsCsv, string? MinimapDir)>();

        // Phase 1: Extract and analyze each map
        foreach (var map in terrainMaps)
        {
            Console.WriteLine($"\n=== Analyzing map: {map.Folder} ({map.Name}) ===");
            
            try
            {
                var mapOutDir = Path.Combine(baseOutDir, map.Folder);
                var exitCode = AnalyzeSingleMapNoViewer(src, clientRoot, map.Folder, mapOutDir, out var placementsCsv, out var minimapDir);
                
                if (exitCode == 0 && !string.IsNullOrEmpty(placementsCsv))
                {
                    successCount++;
                    processedMaps.Add((map.Folder, placementsCsv, minimapDir));
                    Console.WriteLine($"[ok] {map.Folder} completed successfully");
                }
                else if (exitCode == 2)
                {
                    // Non-fatal: map skipped (no data, no WDT, etc.)
                    Console.WriteLine($"[info] {map.Folder} skipped (no data)");
                }
                else
                {
                    failCount++;
                    failedMaps.Add(map.Folder);
                    Console.WriteLine($"[warn] {map.Folder} failed with exit code {exitCode}");
                }
            }
            catch (Exception ex)
            {
                failCount++;
                failedMaps.Add(map.Folder);
                Console.WriteLine($"[error] {map.Folder} failed: {ex.Message}");
            }
        }

        // Phase 2: Generate unified viewer for all processed maps
        if (processedMaps.Count > 0)
        {
            Console.WriteLine($"\n=== Generating Unified Viewer ===");
            Console.WriteLine($"[info] Creating viewer for {processedMaps.Count} maps...");
            
            try
            {
                var viewerAdapter = new AnalysisViewerAdapter();
                var viewerRoot = viewerAdapter.GenerateUnifiedViewer(processedMaps, baseOutDir, versionLabel);
                
                if (!string.IsNullOrEmpty(viewerRoot))
                {
                    Console.WriteLine($"[ok] Unified viewer generated: {viewerRoot}");
                    Console.WriteLine($"[info] Open: {Path.Combine(viewerRoot, "index.html")}");
                    
                    // Auto-serve if requested
                    if (opts.ContainsKey("serve"))
                    {
                        var port = 8080;
                        if (opts.TryGetValue("port", out var portStr) && int.TryParse(portStr, out var parsedPort))
                        {
                            port = parsedPort;
                        }
                        ViewerServer.Serve(viewerRoot, port, openBrowser: !opts.ContainsKey("no-browser"));
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[error] Unified viewer generation failed: {ex.Message}");
            }
        }

        // Cleanup temp directory
        try { Directory.Delete(tempDir, true); } catch { }

        // Summary
        Console.WriteLine($"\n=== Batch Analysis Complete ===");
        Console.WriteLine($"[ok] Successfully analyzed: {successCount}/{terrainMaps.Count} maps");
        if (failCount > 0)
        {
            Console.WriteLine($"[warn] Failed maps ({failCount}): {string.Join(", ", failedMaps)}");
        }

        return failCount > 0 ? 1 : 0;
    }

    private static int AnalyzeSingleMapNoViewer(IArchiveSource src, string clientRoot, string mapName, string outDir, out string? placementsCsv, out string? minimapDir)
    {
        placementsCsv = null;
        minimapDir = null;
        
        Console.WriteLine($"[info] Analyzing ADTs from MPQs for map: {mapName}");
        Console.WriteLine($"[info] Output directory: {outDir}");

        // Check if map folder exists in MPQ
        var wdtPath = $"world/maps/{mapName}/{mapName}.wdt";
        if (!src.FileExists(wdtPath))
        {
            Console.WriteLine($"[warn] Map folder not found in MPQ: {mapName}");
            Console.WriteLine($"[info] Skipping map (no WDT file)");
            return 2; // Non-fatal error code
        }
        
        // Step 1: Extract placements
        Console.WriteLine("  Step 1: Extracting placements...");
        var extractor = new AdtMpqChunkPlacementsExtractor();
        placementsCsv = Path.Combine(outDir, $"{mapName}_placements.csv");
        var extractResult = extractor.ExtractFromArchive(src, mapName, placementsCsv);

        if (!extractResult.Success)
        {
            Console.WriteLine($"[warn] {extractResult.ErrorMessage}");
            placementsCsv = null;
            return 2; // Non-fatal error code
        }

        var totalPlacements = extractResult.M2Count + extractResult.WmoCount;
        if (totalPlacements == 0)
        {
            Console.WriteLine($"[info] No placements found for map: {mapName}");
            placementsCsv = null;
            return 0; // Success but no data
        }

        Console.WriteLine($"  [ok] Extracted {extractResult.M2Count} M2 + {extractResult.WmoCount} WMO placements");

        // Step 2: Extract minimaps
        Console.WriteLine("  Step 2: Extracting minimaps...");
        minimapDir = ExtractMinimapsFromMpq(src, mapName, outDir);
        if (!string.IsNullOrEmpty(minimapDir))
        {
            var count = Directory.GetFiles(minimapDir, "*.jpg").Length;
            Console.WriteLine($"  [ok] Extracted {count} minimap tiles");
        }

        // Step 3: Analyze UniqueIDs
        Console.WriteLine("  Step 3: Analyzing UniqueIDs...");
        var analyzer = new UniqueIdAnalyzer(gapThreshold: 100);
        var analysisResult = analyzer.AnalyzeFromPlacementsCsv(placementsCsv, mapName, outDir);

        if (!analysisResult.Success)
        {
            Console.WriteLine($"  [warn] UniqueID analysis failed: {analysisResult.ErrorMessage}");
        }
        else
        {
            Console.WriteLine($"  [ok] Analyzed {analysisResult.TileCount} tiles");
        }

        // Step 4: Detect clusters
        Console.WriteLine("  Step 4: Detecting spatial clusters...");
        var clusterAnalyzer = new ClusterAnalyzer(proximityThreshold: 50.0f, minClusterSize: 3);
        var clusterResult = clusterAnalyzer.Analyze(placementsCsv, mapName, outDir);

        if (!clusterResult.Success)
        {
            Console.WriteLine($"  [warn] Cluster analysis failed: {clusterResult.ErrorMessage}");
        }
        else
        {
            Console.WriteLine($"  [ok] Detected {clusterResult.TotalClusters} clusters, {clusterResult.TotalPatterns} patterns");
        }

        // Step 5: Extract terrain data from ADTs in MPQ
        Console.WriteLine("  Step 5: Extracting terrain data (MCNK chunks)...");
        try
        {
            var terrainExtractor = new AdtMpqTerrainExtractor();
            var terrainCsvPath = Path.Combine(outDir, $"{mapName}_terrain.csv");
            var terrainResult = terrainExtractor.ExtractFromArchive(src, mapName, terrainCsvPath);
            
            if (terrainResult.Success && terrainResult.ChunksExtracted > 0)
            {
                Console.WriteLine($"  [ok] Extracted {terrainResult.ChunksExtracted} MCNK chunks from {terrainResult.TilesProcessed} tiles");
            }
            else if (terrainResult.Success && terrainResult.ChunksExtracted == 0)
            {
                Console.WriteLine($"  [info] No terrain data found (map may be WMO-only)");
            }
            else
            {
                Console.WriteLine($"  [warn] Terrain extraction failed");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [warn] Terrain extraction error: {ex.Message}");
        }

        // Step 6: Extract terrain meshes (GLB + OBJ) for 3D visualization
        Console.WriteLine("  Step 6: Extracting terrain meshes (GLB + OBJ)...");
        try
        {
            var meshExtractor = new AdtMeshExtractor();
            var meshResult = meshExtractor.ExtractFromArchive(src, mapName, outDir, exportGlb: true, exportObj: true, maxTiles: 0);
            
            if (meshResult.Success && meshResult.TilesProcessed > 0)
            {
                Console.WriteLine($"  [ok] Extracted {meshResult.TilesProcessed} tile meshes to {meshResult.MeshDirectory}");
                if (!string.IsNullOrEmpty(meshResult.ManifestPath))
                {
                    Console.WriteLine($"  [ok] Mesh manifest: {Path.GetFileName(meshResult.ManifestPath)}");
                }
            }
            else if (meshResult.Success && meshResult.TilesProcessed == 0)
            {
                Console.WriteLine($"  [info] No mesh data extracted (map may be WMO-only)");
            }
            else
            {
                Console.WriteLine($"  [warn] Mesh extraction failed");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [warn] Mesh extraction error: {ex.Message}");
        }

        return 0;
    }

    private static int AnalyzeSingleMap(IArchiveSource src, string clientRoot, string mapName, string outDir, Dictionary<string, string> opts)
    {
        Console.WriteLine($"[info] Analyzing ADTs from MPQs for map: {mapName}");
        Console.WriteLine($"[info] Client: {clientRoot}");
        Console.WriteLine($"[info] Output directory: {outDir}");

        // Step 1: Extract placements from MPQ archives
        Console.WriteLine("\n=== Step 1: Extracting placements from MPQ archives ===");
        
        // Check if map folder exists in MPQ
        var wdtPath = $"world/maps/{mapName}/{mapName}.wdt";
        if (!src.FileExists(wdtPath))
        {
            Console.WriteLine($"[warn] Map folder not found in MPQ: {mapName}");
            Console.WriteLine($"[info] Skipping map (no WDT file)");
            return 2; // Non-fatal error code
        }
        
        var extractor = new AdtMpqChunkPlacementsExtractor();
        var placementsCsvPath = Path.Combine(outDir, $"{mapName}_placements.csv");
        var extractResult = extractor.ExtractFromArchive(src, mapName, placementsCsvPath);

        if (!extractResult.Success)
        {
            Console.WriteLine($"[warn] {extractResult.ErrorMessage}");
            Console.WriteLine($"[info] Skipping map (extraction failed)");
            return 2; // Non-fatal error code
        }

        // Check if any placements were found
        var totalPlacements = extractResult.M2Count + extractResult.WmoCount;
        if (totalPlacements == 0)
        {
            Console.WriteLine($"[info] No placements found for map: {mapName}");
            Console.WriteLine($"[info] Skipping remaining analysis steps");
            return 0; // Success but no data
        }

        Console.WriteLine($"[ok] {extractResult.ErrorMessage}");
        Console.WriteLine($"[ok] Placements CSV: {placementsCsvPath}");

        // Step 1.5: Extract minimaps from MPQ
        Console.WriteLine("\n=== Step 1.5: Extracting minimaps from MPQ ===");
        var minimapDir = ExtractMinimapsFromMpq(src, mapName, outDir);
        if (!string.IsNullOrEmpty(minimapDir))
        {
            Console.WriteLine($"[ok] Extracted minimaps to: {minimapDir}");
        }
        else
        {
            Console.WriteLine($"[info] No minimaps found in MPQ");
        }

        // Step 1.6: Extract AreaTable from DBC for terrain overlay enrichment
        // NOTE: Only for 0.6.0+. Alpha (0.5.x) uses specialized hand-crafted mapping.
        var versionStr = opts.GetValueOrDefault("version", "unknown");
        var isAlpha = versionStr.StartsWith("0.5.");
        
        if (!isAlpha)
        {
            Console.WriteLine("\n=== Step 1.6: Extracting AreaTable from DBC ===");
            try
            {
                ExtractAreaTableFromMpq(src, outDir, versionStr);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[warn] AreaTable extraction failed: {ex.Message}");
                Console.WriteLine($"[info] Terrain overlays will work without area names");
            }
        }
        else
        {
            Console.WriteLine("\n=== Step 1.6: Skipping AreaTable extraction (Alpha version uses specialized mapping) ===");
        }

        // Step 2: Analyze UniqueIDs and detect layers
        Console.WriteLine("\n=== Step 2: Analyzing UniqueIDs and detecting layers ===");
        var analyzer = new UniqueIdAnalyzer(gapThreshold: 100);
        var analysisResult = analyzer.AnalyzeFromPlacementsCsv(placementsCsvPath, mapName, outDir);

        if (!analysisResult.Success)
        {
            Console.Error.WriteLine($"[error] UniqueID analysis failed: {analysisResult.ErrorMessage}");
            return 1;
        }

        Console.WriteLine($"[ok] Analyzed {analysisResult.TileCount} tiles");
        Console.WriteLine($"[ok] UniqueID analysis CSV: {analysisResult.CsvPath}");
        Console.WriteLine($"[ok] Layers JSON: {analysisResult.LayersJsonPath}");

        // Step 3: Detect spatial clusters and patterns
        Console.WriteLine("\n=== Step 3: Detecting spatial clusters and patterns ===");
        var clusterAnalyzer = new ClusterAnalyzer(proximityThreshold: 50.0f, minClusterSize: 3);
        var clusterResult = clusterAnalyzer.Analyze(placementsCsvPath, mapName, outDir);

        if (!clusterResult.Success)
        {
            Console.WriteLine($"[warn] Cluster analysis failed: {clusterResult.ErrorMessage}");
        }
        else
        {
            Console.WriteLine($"[ok] Detected {clusterResult.TotalClusters} spatial clusters");
            Console.WriteLine($"[ok] Identified {clusterResult.TotalPatterns} recurring patterns (potential prefabs)");
            Console.WriteLine($"[ok] Clusters JSON: {clusterResult.ClustersJsonPath}");
            Console.WriteLine($"[ok] Patterns JSON: {clusterResult.PatternsJsonPath}");
            Console.WriteLine($"[ok] Summary CSV: {clusterResult.SummaryCsvPath}");
        }

        // Step 4: Extract terrain data from ADTs in MPQ
        Console.WriteLine("\n=== Step 4: Extracting terrain data (MCNK chunks) ===");
        try
        {
            var terrainExtractor = new AdtMpqTerrainExtractor();
            var terrainCsvPath = Path.Combine(outDir, $"{mapName}_terrain.csv");
            var terrainResult = terrainExtractor.ExtractFromArchive(src, mapName, terrainCsvPath);
            
            if (terrainResult.Success && terrainResult.ChunksExtracted > 0)
            {
                Console.WriteLine($"[ok] Extracted {terrainResult.ChunksExtracted} MCNK chunks from {terrainResult.TilesProcessed} tiles");
            }
            else if (terrainResult.Success && terrainResult.ChunksExtracted == 0)
            {
                Console.WriteLine($"[info] No terrain data found (map may be WMO-only)");
            }
            else
            {
                Console.WriteLine($"[warn] Terrain extraction failed");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[warn] Terrain extraction error: {ex.Message}");
        }

        // Step 5: Extract terrain meshes (GLB + OBJ) for 3D visualization
        Console.WriteLine("\n=== Step 5: Extracting terrain meshes (GLB + OBJ) ===");
        try
        {
            var meshExtractor = new AdtMeshExtractor();
            var meshResult = meshExtractor.ExtractFromArchive(src, mapName, outDir, exportGlb: true, exportObj: true, maxTiles: 0);
            
            if (meshResult.Success && meshResult.TilesProcessed > 0)
            {
                Console.WriteLine($"[ok] Extracted {meshResult.TilesProcessed} tile meshes to {meshResult.MeshDirectory}");
                if (!string.IsNullOrEmpty(meshResult.ManifestPath))
                {
                    Console.WriteLine($"[ok] Mesh manifest: {Path.GetFileName(meshResult.ManifestPath)}");
                }
            }
            else if (meshResult.Success && meshResult.TilesProcessed == 0)
            {
                Console.WriteLine($"[info] No mesh data extracted (map may be WMO-only)");
            }
            else
            {
                Console.WriteLine($"[warn] Mesh extraction failed");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[warn] Mesh extraction error: {ex.Message}");
        }

        // Step 6: Generate viewer
        Console.WriteLine("\n=== Step 6: Generating viewer ===");
        var viewerAdapter = new AnalysisViewerAdapter();
        var versionLabel = ExtractVersionFromPath(clientRoot) ?? "mpq-analysis";
        var viewerRoot = viewerAdapter.GenerateViewer(placementsCsvPath, mapName, outDir, minimapDir: minimapDir, versionLabel: versionLabel);

        if (!string.IsNullOrEmpty(viewerRoot))
        {
            Console.WriteLine($"[ok] Viewer generated: {viewerRoot}");
            Console.WriteLine($"[info] Open: {Path.Combine(viewerRoot, "index.html")}");
        }
        else
        {
            Console.WriteLine($"[warn] Viewer generation skipped (no placements)");
        }

        Console.WriteLine("\n=== Analysis Complete ===");
        Console.WriteLine($"All outputs written to: {outDir}");
        
        // Check if --serve flag is present
        if (opts.ContainsKey("serve"))
        {
            Console.WriteLine("\n=== Starting viewer server ===");
            var port = TryParseInt(opts, "port") ?? 8080;
            var openBrowser = !opts.ContainsKey("no-browser");
            ViewerServer.Serve(viewerRoot, port, openBrowser);
        }
        else
        {
            Console.WriteLine("\nℹ️  Use --serve to auto-start web server after analysis");
        }
        
        return 0;
    }

    private static int RunProbeArchive(Dictionary<string, string> opts)
    {
        Require(opts, "client-path");
        var clientRoot = opts["client-path"]; 
        var mapName = opts.GetValueOrDefault("map", "");
        var limit = TryParseInt(opts, "limit") ?? 10;

        Console.WriteLine($"[probe] Client root: {clientRoot}");
        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("[error] --client-path does not exist");
            return 1;
        }

        EnsureStormLibOnPath();

        // Locate MPQs with base then ascending patches
        var mpqs = ArchiveLocator.LocateMpqs(clientRoot);
        Console.WriteLine($"[probe] MPQs found: {mpqs.Count}");
        if (mpqs.Count == 0) Console.WriteLine("[probe] No MPQs detected; loose files only.");

        using var src = new PrioritizedArchiveSource(clientRoot, mpqs);

        // md5translate: prefer loose Data/textures/Minimap
        var md5Candidates = new[]
        {
            "textures/Minimap/md5translate.txt",
            "textures/Minimap/md5translate.trs"
        };

        string? md5Used = null;
        foreach (var cand in md5Candidates)
        {
            if (src.FileExists(cand)) { md5Used = cand; break; }
        }

        if (md5Used is null)
        {
            Console.WriteLine("[probe] md5translate not found in Data or MPQs");
        }
        else
        {
            Console.WriteLine($"[probe] md5translate detected at virtual path: {md5Used}");
            try
            {
                using var s = src.OpenFile(md5Used);
                using var r = new StreamReader(s);
                Console.WriteLine("[probe] md5translate preview (first 5 non-empty lines):");
                int shown = 0;
                while (!r.EndOfStream && shown < 5)
                {
                    var line = r.ReadLine();
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    Console.WriteLine("  " + line);
                    shown++;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[warn] Failed reading md5translate: {ex.Message}");
            }
        }

        // Enumerate a few minimap BLPs from Textures/Minimap (loose-first, then MPQ union)
        var pattern = string.IsNullOrWhiteSpace(mapName)
            ? "textures/Minimap/*.blp"
            : $"textures/Minimap/{mapName}/*.blp";

        Console.WriteLine($"[probe] Enumerating BLPs with pattern: {pattern}");
        var all = src.EnumerateFiles(pattern).Take(limit).ToList();
        if (all.Count == 0)
        {
            Console.WriteLine("[probe] No BLPs found for minimap pattern.");
        }
        else
        {
            Console.WriteLine($"[probe] Showing up to {limit} BLPs:");
            foreach (var f in all)
            {
                Console.WriteLine("  " + f);
            }
        }

        Console.WriteLine("[ok] Probe complete (no viewer changes made).");
        return 0;
    }

    private static void EnsureStormLibOnPath()
    {
        // Try to ensure native StormLib.dll can be resolved by the process loader.
        var baseDir = AppContext.BaseDirectory;
        var local = Path.Combine(baseDir, "StormLib.dll");
        if (File.Exists(local)) return;

        // Common repo-relative locations during dev
        var candidates = new List<string>();
        var cwd = Directory.GetCurrentDirectory();
        candidates.Add(Path.Combine(cwd, "WoWRollback.Mpq", "runtimes", "win-x64", "native"));
        candidates.Add(Path.Combine(cwd, "..", "WoWRollback.Mpq", "runtimes", "win-x64", "native"));
        candidates.Add(Path.Combine(cwd, "..", "..", "WoWRollback.Mpq", "runtimes", "win-x64", "native"));

        foreach (var dir in candidates)
        {
            if (!Directory.Exists(dir)) continue;
            var dll = Path.Combine(dir, "StormLib.dll");
            if (!File.Exists(dll)) continue;

            var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            if (!path.Split(Path.PathSeparator).Any(p => string.Equals(Path.GetFullPath(p), Path.GetFullPath(dir), StringComparison.OrdinalIgnoreCase)))
            {
                Environment.SetEnvironmentVariable("PATH", dir + Path.PathSeparator + path);
                Console.WriteLine($"[probe] Added to PATH: {dir}");
            }
            return;
        }
    }

    private static void EnsureComparisonPrerequisites(
        string outputRoot,
        IReadOnlyList<string> versions,
        IReadOnlyList<string>? maps,
        Dictionary<string, string> opts)
    {
        if (!Directory.Exists(outputRoot))
        {
            Directory.CreateDirectory(outputRoot);
        }

        if (maps is null || maps.Count == 0)
        {
            return;
        }

        var alphaRoot = GetOption(opts, "alpha-root");
        var convertedAdtRoot = GetOption(opts, "converted-adt-root");
        var convertedAdtCacheRoot = GetOption(opts, "converted-adt-cache");

        string? normalizedAlphaRoot = null;
        if (!string.IsNullOrWhiteSpace(alphaRoot))
        {
            normalizedAlphaRoot = Path.GetFullPath(alphaRoot);
            if (!Directory.Exists(normalizedAlphaRoot))
            {
                throw new DirectoryNotFoundException($"Alpha test data root not found: {normalizedAlphaRoot}");
            }
        }

        string? normalizedConvertedRoot = null;
        if (!string.IsNullOrWhiteSpace(convertedAdtRoot))
        {
            var candidate = Path.GetFullPath(convertedAdtRoot);
            if (Directory.Exists(candidate))
            {
                normalizedConvertedRoot = candidate;
            }
            else
            {
                Console.WriteLine($"[warn] Converted ADT root not found: {candidate}. Coordinates will fall back to raw Alpha data.");
            }
        }

        string? normalizedCacheRoot = null;
        if (!string.IsNullOrWhiteSpace(convertedAdtCacheRoot))
        {
            var candidate = Path.GetFullPath(convertedAdtCacheRoot);
            if (Directory.Exists(candidate))
            {
                normalizedCacheRoot = candidate;
            }
            else
            {
                Console.WriteLine($"[warn] Converted ADT cache not found: {candidate}. Rebuilding may be necessary.");
            }
        }

        foreach (var version in versions)
        {
            foreach (var map in maps)
            {
                if (HasAlphaOutputs(outputRoot, version, map))
                {
                    continue;
                }

                if (normalizedAlphaRoot is null)
                {
                    throw new InvalidOperationException(
                        $"Required comparison data for version '{version}', map '{map}' is missing under '{outputRoot}'. " +
                        "Supply --alpha-root so the CLI can auto-generate the placement ranges.");
                }

                var wdtPath = FindAlphaWdt(normalizedAlphaRoot, version, map);
                if (wdtPath is null)
                {
                    throw new InvalidOperationException(
                        $"Could not locate Alpha WDT for version '{version}', map '{map}' beneath '{normalizedAlphaRoot}'.");
                }

                var buildTag = BuildTagResolver.ResolveForPath(Path.GetDirectoryName(Path.GetFullPath(wdtPath)) ?? wdtPath);
                var sessionDir = OutputSession.Create(outputRoot, map, buildTag);
                var convertedDir = ResolveConvertedAdtDirectory(normalizedConvertedRoot, version, map);
                if (string.IsNullOrWhiteSpace(convertedDir) && normalizedCacheRoot is not null)
                {
                    foreach (var candidate in EnumerateCacheCandidates(normalizedCacheRoot, version, map))
                    {
                        if (Directory.Exists(candidate))
                        {
                            convertedDir = candidate;
                            break;
                        }
                    }
                }

                Console.WriteLine($"[auto] Generating placement ranges for {version} / {map}");
                Console.WriteLine($"[auto]  WDT: {wdtPath}");
                Console.WriteLine($"[auto]  Using raw Alpha coordinates (no transforms)");

                var analysis = WoWRollback.Core.Services.AlphaWdtAnalyzer.AnalyzeAlphaWdt(wdtPath);
                RangeCsvWriter.WritePerMapCsv(sessionDir, $"alpha_{map}", analysis.Ranges, analysis.Assets);
            }
        }
    }

    private static bool HasAlphaOutputs(string outputRoot, string version, string map)
    {
        var versionDirectory = Path.Combine(outputRoot, version);
        if (!Directory.Exists(versionDirectory))
        {
            return false;
        }

        var mapDirectory = Path.Combine(versionDirectory, map);
        if (!Directory.Exists(mapDirectory))
        {
            return false;
        }

        var idRanges = Path.Combine(mapDirectory, $"id_ranges_by_map_alpha_{map}.csv");
        return File.Exists(idRanges);
    }

    private static string? FindAlphaWdt(string alphaRoot, string version, string map)
    {
        var key = (version, map);
        if (AlphaWdtCache.TryGetValue(key, out var cached))
        {
            return cached;
        }

        try
        {
            var matches = Directory.EnumerateFiles(alphaRoot, map + ".wdt", SearchOption.AllDirectories)
                .Where(path => path.EndsWith(Path.DirectorySeparatorChar + map + ".wdt", StringComparison.OrdinalIgnoreCase) ||
                               path.EndsWith(Path.AltDirectorySeparatorChar + map + ".wdt", StringComparison.OrdinalIgnoreCase))
                .Where(path => path.IndexOf($"{Path.DirectorySeparatorChar}World{Path.DirectorySeparatorChar}Maps{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) >= 0 ||
                               path.IndexOf($"{Path.AltDirectorySeparatorChar}World{Path.AltDirectorySeparatorChar}Maps{Path.AltDirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) >= 0)
                .OrderBy(path => ScoreVersionMatch(path, version))
                .ThenBy(path => path.Length)
                .ToList();

            var resolved = matches.FirstOrDefault();
            AlphaWdtCache[key] = resolved;
            return resolved;
        }
        catch
        {
            AlphaWdtCache[key] = null;
            return null;
        }
    }

    private static int ScoreVersionMatch(string path, string version)
    {
        if (path.IndexOf(version, StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return 0;
        }

        var prefixLength = Math.Min(5, version.Length);
        var prefix = version[..prefixLength];
        if (path.IndexOf(prefix, StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return 1;
        }

        var majorMinor = version.Split('.', StringSplitOptions.RemoveEmptyEntries);
        if (majorMinor.Length >= 2)
        {
            var partial = string.Join('.', majorMinor.Take(2));
            if (path.IndexOf(partial, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return 2;
            }
        }

        return 3;
    }

    private static IEnumerable<string> EnumerateCacheCandidates(string root, string version, string map)
    {
        yield return Path.Combine(root, "World", "Maps", map, version);
        yield return Path.Combine(root, "World", "Maps", map);
        yield return Path.Combine(root, version, "World", "Maps", map);
        yield return Path.Combine(root, version, map);
        yield return Path.Combine(root, map);
    }

    private static string? ResolveConvertedAdtDirectory(string? convertedRoot, string version, string map)
    {
        if (string.IsNullOrWhiteSpace(convertedRoot))
        {
            return null;
        }

        foreach (var candidate in EnumerateCacheCandidates(convertedRoot, version, map))
        {
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private sealed class TupleComparer : IEqualityComparer<(string Version, string Map)>
    {
        public bool Equals((string Version, string Map) x, (string Version, string Map) y) =>
            string.Equals(x.Version, y.Version, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(x.Map, y.Map, StringComparison.OrdinalIgnoreCase);

        public int GetHashCode((string Version, string Map) obj) =>
            HashCode.Combine(obj.Version.ToUpperInvariant(), obj.Map.ToUpperInvariant());
    }

    private static string? GetOption(Dictionary<string, string> opts, string key, string? fallback = null) =>
        opts.TryGetValue(key, out var value) ? value : fallback;

    // Build an auto-named session root like: outputs/<prefix?>{Map}-{min}-{max}_timestamp
    private static string ResolveSessionRoot(Dictionary<string, string> opts, string mapName, uint maxUniqueId, out uint rangeMin, out uint rangeMax)
    {
        var outputsRoot = opts.GetValueOrDefault("outputs-root", "outputs");
        var prefix = opts.GetValueOrDefault("prefix", "");
        var noTs = opts.ContainsKey("no-timestamp");
        var tsFmt = opts.GetValueOrDefault("timestamp-format", "yyyyMMdd-HHmmss");

        // Range label resolution precedence: --id-range > preset-json scan > 0-maxUniqueId
        var idRangeSpec = GetOption(opts, "id-range");
        if (!TryParseRangeOverride(idRangeSpec, out rangeMin, out rangeMax))
        {
            if (!TryComputeRangeFromPresetOption(opts, out rangeMin, out var presetMax))
            {
                rangeMin = 0; rangeMax = maxUniqueId;
            }
            else
            {
                // Label min from preset; label max must reflect the actual run ceiling
                rangeMax = maxUniqueId;
            }
        }

        var ts = noTs ? string.Empty : DateTime.UtcNow.ToString(tsFmt, CultureInfo.InvariantCulture);
        var name = string.IsNullOrEmpty(ts)
            ? $"{prefix}{mapName}-{rangeMin}-{rangeMax}"
            : $"{prefix}{mapName}-{rangeMin}-{rangeMax}_{ts}";
        return Path.Combine(outputsRoot, name);
    }

    private static bool TryParseRangeOverride(string? spec, out uint min, out uint max)
    {
        min = 0; max = 0;
        if (string.IsNullOrWhiteSpace(spec)) return false;
        var parts = spec.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length != 2) return false;
        if (!uint.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out min)) return false;
        if (!uint.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out max)) return false;
        if (max < min) { var t = min; min = max; max = t; }
        return true;
    }

    private static bool TryComputeRangeFromPresetOption(Dictionary<string, string> opts, out uint min, out uint max)
    {
        min = 0; max = 0;
        if (!opts.TryGetValue("preset-json", out var path) || string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return false;
        return TryComputeRangeFromPreset(path, out min, out max);
    }

    // Minimal scan of preset JSON to find enabled ranges with min/max fields
    private static bool TryComputeRangeFromPreset(string presetPath, out uint min, out uint max)
    {
        uint accMin = uint.MaxValue; uint accMax = 0;
        try
        {
            using var fs = File.OpenRead(presetPath);
            using var doc = JsonDocument.Parse(fs);

            void Scan(JsonElement el)
            {
                switch (el.ValueKind)
                {
                    case JsonValueKind.Object:
                        bool enabled = true;
                        if (el.TryGetProperty("enabled", out var enProp) && enProp.ValueKind == JsonValueKind.False)
                            enabled = false;
                        if (enabled && el.TryGetProperty("min", out var minProp) && el.TryGetProperty("max", out var maxProp))
                        {
                            if (minProp.TryGetInt64(out var mn) && maxProp.TryGetInt64(out var mx))
                            {
                                if (mn < 0) mn = 0;
                                var umin = (uint)Math.Min(uint.MaxValue, (ulong)mn);
                                var umax = (uint)Math.Min(uint.MaxValue, (ulong)mx);
                                if (umin < accMin) accMin = umin;
                                if (umax > accMax) accMax = umax;
                            }
                        }
                        foreach (var prop in el.EnumerateObject()) Scan(prop.Value);
                        break;
                    case JsonValueKind.Array:
                        foreach (var item in el.EnumerateArray()) Scan(item);
                        break;
                }
            }

            Scan(doc.RootElement);
            if (accMax == 0 && accMin == uint.MaxValue) { min = 0; max = 0; return false; }
            if (accMin == uint.MaxValue) accMin = 0;
            min = accMin; max = accMax; return true;
        }
        catch { min = 0; max = 0; return false; }
    }

    private static int RunFixMinimapWebp(Dictionary<string, string> opts)
    {
        var outputDir = GetOption(opts, "out");
        if (string.IsNullOrEmpty(outputDir) || !Directory.Exists(outputDir))
        {
            Console.Error.WriteLine("[error] --out directory not found or not specified");
            Console.Error.WriteLine("Usage: fix-minimap-webp --out <output_directory>");
            return 1;
        }

        Console.WriteLine($"[info] === Minimap WebP Fix-up Tool ===");
        Console.WriteLine($"[info] Scanning: {outputDir}");

        // Find all PNG files in {version}/World/Textures/Minimap/{map}/ folders
        var versionDirs = Directory.GetDirectories(outputDir)
            .Where(d => !Path.GetFileName(d).Equals("viewer", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (versionDirs.Count == 0)
        {
            Console.WriteLine("[warn] No version directories found");
            return 0;
        }

        int totalConverted = 0;
        int totalMaps = 0;

        foreach (var versionDir in versionDirs)
        {
            var version = Path.GetFileName(versionDir);
            var minimapBaseDir = Path.Combine(versionDir, "World", "Textures", "Minimap");
            
            if (!Directory.Exists(minimapBaseDir))
            {
                Console.WriteLine($"[info] No minimap directory for version: {version}");
                continue;
            }

            // Find all map subdirectories
            var mapDirs = Directory.GetDirectories(minimapBaseDir);
            
            foreach (var mapDir in mapDirs)
            {
                var mapName = Path.GetFileName(mapDir);
                var pngFiles = Directory.GetFiles(mapDir, "*.png", SearchOption.TopDirectoryOnly);
                
                if (pngFiles.Length == 0) continue;

                totalMaps++;
                Console.WriteLine($"[info] Processing {mapName} ({version}): {pngFiles.Length} PNG files");

                // Create viewer minimap directory
                var viewerMinimapDir = Path.Combine(outputDir, "viewer", "minimap", version, mapName);
                Directory.CreateDirectory(viewerMinimapDir);

                int converted = 0;
                foreach (var pngFile in pngFiles)
                {
                    try
                    {
                        // Load PNG
                        using var image = SixLabors.ImageSharp.Image.Load(pngFile);
                        
                        // Save as WebP
                        var fileName = Path.GetFileNameWithoutExtension(pngFile);
                        var webpPath = Path.Combine(viewerMinimapDir, $"{fileName}.webp");
                        
                        using var outStream = File.Create(webpPath);
                        image.Save(outStream, new SixLabors.ImageSharp.Formats.Webp.WebpEncoder { Quality = 90 });
                        converted++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[warn] Failed to convert {Path.GetFileName(pngFile)}: {ex.Message}");
                    }
                }

                totalConverted += converted;
                Console.WriteLine($"[ok] Converted {converted}/{pngFiles.Length} tiles for {mapName}");
            }
        }

        Console.WriteLine($"\n[ok] === Fix-up Complete ===");
        Console.WriteLine($"[ok] Processed {totalMaps} maps, converted {totalConverted} tiles to WebP");
        Console.WriteLine($"[ok] WebP files are now in: {Path.Combine(outputDir, "viewer", "minimap")}");

        return 0;
    }

    private static void ExtractAreaTableFromMpq(IArchiveSource src, string outputDir, string version)
    {
        // Check if AreaTable.dbc exists in MPQ
        const string dbcPath = "DBFilesClient/AreaTable.dbc";
        if (!src.FileExists(dbcPath))
        {
            Console.WriteLine($"[warn] AreaTable.dbc not found in MPQ");
            return;
        }

        try
        {
            // Get WoWDBDefs path
            var dbdDir = GetWoWDBDefsPath();
            if (string.IsNullOrEmpty(dbdDir))
            {
                Console.WriteLine($"[warn] WoWDBDefs not found - cannot parse AreaTable");
                return;
            }

            // Extract DBC to temp location
            var tempDbcDir = Path.Combine(Path.GetTempPath(), $"dbc_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDbcDir);
            var tempDbcPath = Path.Combine(tempDbcDir, "AreaTable.dbc");
            
            using (var dbcStream = src.OpenFile(dbcPath))
            using (var fileStream = File.Create(tempDbcPath))
            {
                dbcStream.CopyTo(fileStream);
            }

            Console.WriteLine($"[info] Extracted AreaTable.dbc from MPQ");

            // Use DBCD to parse and export to CSV in AreaTableReader-compatible format
            var dbdProvider = new DBCD.Providers.FilesystemDBDProvider(dbdDir);
            var dbcProvider = new DBCD.Providers.FilesystemDBCProvider(tempDbcDir, useCache: false);
            var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);
            var areaTable = dbcd.Load("AreaTable", version);

            // Export to CSV in format: row_key,id,parent,continentId,name
            var csvPath = Path.Combine(outputDir, "AreaTable_Alpha.csv");
            using (var writer = new StreamWriter(csvPath))
            {
                // Write header matching AreaTableReader expectations
                writer.WriteLine("row_key,id,parent,continentId,name");

                // Write rows
                int rowKey = 0;
                foreach (var kvp in (IEnumerable<KeyValuePair<int, DBCD.DBCDRow>>)areaTable)
                {
                    try
                    {
                        var row = kvp.Value;
                        var id = GetField<int>(row, "ID");
                        var parent = GetField<int?>(row, "ParentAreaID") ?? GetField<int?>(row, "AreaTableParentID") ?? 0;
                        var continentId = GetField<int?>(row, "ContinentID") ?? GetField<int?>(row, "MapID") ?? 0;
                        var name = GetField<string>(row, "AreaName_Lang") ?? GetField<string>(row, "Name_Lang") ?? "";
                        
                        // Escape name if it contains commas
                        if (name.Contains(',') || name.Contains('"'))
                        {
                            name = $"\"{name.Replace("\"", "\"\"")}\"";
                        }
                        
                        writer.WriteLine($"{rowKey},{id},{parent},{continentId},{name}");
                        rowKey++;
                    }
                    catch
                    {
                        // Skip rows that don't have required fields
                        continue;
                    }
                }
            }

            Console.WriteLine($"[ok] AreaTable CSV: {csvPath} ({areaTable.Count} rows)");

            // Cleanup temp files
            try
            {
                Directory.Delete(tempDbcDir, recursive: true);
            }
            catch { }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[warn] AreaTable extraction error: {ex.Message}");
        }
    }

    private static string? GetWoWDBDefsPath()
    {
        // Try common locations for WoWDBDefs
        var candidates = new[]
        {
            Path.Combine(Directory.GetCurrentDirectory(), "..", "lib", "WoWDBDefs", "definitions"),
            Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "lib", "WoWDBDefs", "definitions"),
            Path.Combine(Directory.GetCurrentDirectory(), "lib", "WoWDBDefs", "definitions"),
            "C:\\WoWDBDefs\\definitions"
        };

        foreach (var path in candidates)
        {
            if (Directory.Exists(path))
            {
                return Path.GetFullPath(path);
            }
        }

        return null;
    }

    private static T? GetField<T>(DBCD.DBCDRow row, string fieldName)
    {
        try
        {
            var value = row[fieldName];
            if (value is T typedValue)
                return typedValue;
            
            // Try conversion for common types
            if (typeof(T) == typeof(string))
                return (T)(object)(value?.ToString() ?? "");
            
            if (value != null)
                return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            // Field doesn't exist or conversion failed
        }
        
        return default;
    }

    private static int RunDebugSingleAdt(Dictionary<string, string> opts)
    {
        if (!opts.TryGetValue("tile-x", out var tileXStr) ||
            !opts.TryGetValue("tile-y", out var tileYStr) ||
            !opts.TryGetValue("client-path", out var clientPath) ||
            !opts.TryGetValue("out", out var outDir) ||
            !opts.TryGetValue("map", out var mapName))
        {
            Console.WriteLine("[ERROR] Required: --tile-x, --tile-y, --client-path, --out, --map");
            return 1;
        }

        if (!int.TryParse(tileXStr, out var tileX) || !int.TryParse(tileYStr, out var tileY))
        {
            Console.WriteLine("[ERROR] tile-x and tile-y must be integers");
            return 1;
        }

        Commands.DebugSingleAdtCommand.ExecuteAsync(tileX, tileY, clientPath, outDir, mapName).Wait();
        return 0;
    }

    private static int RunAlphaToLk(Dictionary<string, string> opts)
    {
        // Compose rollback with LK export enabled
        Require(opts, "input");
        Require(opts, "max-uniqueid");

        var merged = new Dictionary<string, string>(opts, StringComparer.OrdinalIgnoreCase)
        {
            ["export-lk-adts"] = "true"
        };

        return RunRollback(merged);
    }

    private static int RunRollback(Dictionary<string, string> opts)
    {
        Require(opts, "input");
        Require(opts, "max-uniqueid");

        var inputPath = opts["input"];
        var mapName = Path.GetFileNameWithoutExtension(inputPath);
        var userOut = GetOption(opts, "out");

        var buryDepth = opts.TryGetValue("bury-depth", out var buryStr) && float.TryParse(buryStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var bd)
            ? bd
            : -5000.0f;
        var maxUniqueId = (uint)(TryParseInt(opts, "max-uniqueid") ?? throw new ArgumentException("Missing --max-uniqueid"));
        var fixHoles = opts.ContainsKey("fix-holes");
        var disableMcsh = opts.ContainsKey("disable-mcsh");
        var holesScope = opts.TryGetValue("holes-scope", out var holesScopeStr) ? holesScopeStr.ToLowerInvariant() : "self";
        var holesNeighbors = string.Equals(holesScope, "neighbors", StringComparison.OrdinalIgnoreCase);
        var holesPreserveWmo = !(opts.TryGetValue("holes-wmo-preserve", out var preserveStr) && string.Equals(preserveStr, "false", StringComparison.OrdinalIgnoreCase));
        var exportLkAdts = opts.ContainsKey("export-lk-adts");
        var force = opts.ContainsKey("force");

        // Resolve session and primary output paths
        uint rangeMinLabel, rangeMaxLabel;
        string outRoot;
        if (string.IsNullOrWhiteSpace(userOut))
        {
            outRoot = ResolveSessionRoot(opts, mapName, maxUniqueId, out rangeMinLabel, out rangeMaxLabel);
        }
        else
        {
            outRoot = userOut!;
            if (!TryComputeRangeFromPresetOption(opts, out rangeMinLabel, out var _presetMaxTmp))
            {
                rangeMinLabel = 0;
            }
            rangeMaxLabel = maxUniqueId;
        }
        Directory.CreateDirectory(outRoot);
        var alphaWdtDir = Path.Combine(outRoot, "alpha_wdt");
        Directory.CreateDirectory(alphaWdtDir);
        var outputPath = Path.Combine(alphaWdtDir, Path.GetFileName(inputPath));

        var lkOutDefault = Path.Combine(outRoot, "lk_adts", "World", "Maps", mapName);
        var lkOutDir = opts.GetValueOrDefault("lk-out", lkOutDefault);
        var lkClientPath = opts.GetValueOrDefault("lk-client-path", "");
        var srcClientPath = opts.GetValueOrDefault("src-client-path", "");
        var areaRemapJsonPath = opts.GetValueOrDefault("area-remap-json", "");
        var defaultUnmapped = TryParseInt(opts, "default-unmapped") ?? 0;
        var dbctoolOutRoot = opts.GetValueOrDefault("dbctool-out-root", "");
        var strictAreaId = !opts.TryGetValue("strict-areaid", out var strictStr) || !string.Equals(strictStr, "false", StringComparison.OrdinalIgnoreCase);
        var reportAreaId = opts.ContainsKey("report-areaid");
        var copyCrosswalks = opts.ContainsKey("copy-crosswalks");
        var autoCrosswalks = !opts.TryGetValue("auto-crosswalks", out var autoX) || !string.Equals(autoX, "false", StringComparison.OrdinalIgnoreCase);
        var chainVia060 = opts.ContainsKey("chain-via-060");
        // support preferred aliases --crosswalk-dir/--crosswalk-file while keeping legacy dbctool-* flags
        var dbctoolPatchDir = opts.ContainsKey("dbctool-patch-dir") ? opts["dbctool-patch-dir"] : opts.GetValueOrDefault("crosswalk-dir", "");
        var dbctoolPatchFile = opts.ContainsKey("dbctool-patch-file") ? opts["dbctool-patch-file"] : opts.GetValueOrDefault("crosswalk-file", "");
        var lkDbcDir = opts.GetValueOrDefault("lk-dbc-dir", "");
        var dbdDirOpt = GetOption(opts, "dbd-dir");
        var srcDbcDirOpt = GetOption(opts, "src-dbc-dir");
        var pivot060DirOpt = GetOption(opts, "pivot-060-dbc-dir");

        Console.WriteLine("══════════════════════════════════════════");
        Console.WriteLine("          🎮 WoWRollback - ROLLBACK");
        Console.WriteLine("══════════════════════════════════════════");
        Console.WriteLine($"Input WDT:      {inputPath}");
        Console.WriteLine($"Session Dir:    {outRoot}");
        Console.WriteLine($"Alpha Out Dir:  {alphaWdtDir}");
        Console.WriteLine($"Max UniqueID:   {maxUniqueId:N0}");
        Console.WriteLine($"Bury Depth:     {buryDepth:F1}");
        // Display label range and preset range for clarity
        uint presetMinTmp, presetMaxTmp;
        if (TryComputeRangeFromPresetOption(opts, out presetMinTmp, out presetMaxTmp))
        {
            Console.WriteLine($"Preset Range:   {presetMinTmp}-{presetMaxTmp}");
        }
        Console.WriteLine($"Session Label:  {rangeMinLabel}-{rangeMaxLabel}");
        Console.WriteLine($"Bury Threshold: UniqueID > {maxUniqueId:N0}");
        if (fixHoles)
        {
            Console.WriteLine($"Option:         --fix-holes (scope={holesScope}, preserve-wmo={holesPreserveWmo.ToString().ToLowerInvariant()})");
        }
        if (disableMcsh) Console.WriteLine("Option:         --disable-mcsh (zero baked shadows)");
        if (exportLkAdts)
        {
            Console.WriteLine("Option:         --export-lk-adts (write LK ADTs)");
            Console.WriteLine($"LK ADT Out:     {lkOutDir}");
            if (!string.IsNullOrWhiteSpace(lkClientPath)) Console.WriteLine($"LK Client:      {lkClientPath}");
            if (!string.IsNullOrWhiteSpace(areaRemapJsonPath)) Console.WriteLine($"Area Map JSON:  {areaRemapJsonPath}");
            Console.WriteLine($"Strict AreaIDs: {(strictAreaId ? "true" : "false")}");
            if (reportAreaId) Console.WriteLine("Option:         --report-areaid (write summary CSV)");
        }
        Console.WriteLine();

        try
        {
            var wdt = new WdtAlpha(inputPath);
            var existingAdts = wdt.GetExistingAdtsNumbers();
            var adtOffsets = wdt.GetAdtOffsetsInMain();
            Console.WriteLine($"[info] Tiles detected: {existingAdts.Count}");
            var expectedAdtCount = existingAdts.Count;

            var wdtBytes = File.ReadAllBytes(inputPath);
            int totalPlacements = 0, removed = 0, tilesProcessed = 0;
            int holesCleared = 0, mcshZeroed = 0;
            var processedMcnk = new HashSet<int>();

            foreach (var adtNum in existingAdts)
            {
                int adtOffset = adtOffsets[adtNum];
                if (adtOffset == 0) continue;

                var adt = new AdtAlpha(inputPath, adtOffset, adtNum);
                var mddf = adt.GetMddf();
                var modf = adt.GetModf();

                const int mddfEntrySize = 36;
                int mddfCount = mddf.Data.Length / mddfEntrySize;
                var mddfBuried = new bool[mddfCount];
                for (int offset = 0; offset + mddfEntrySize <= mddf.Data.Length; offset += mddfEntrySize)
                {
                    uint uid = BitConverter.ToUInt32(mddf.Data, offset + 4);
                    totalPlacements++;
                    if (uid > maxUniqueId)
                    {
                        var newZ = BitConverter.GetBytes(buryDepth);
                        Array.Copy(newZ, 0, mddf.Data, offset + 12, 4);
                        removed++;
                        int idx = offset / mddfEntrySize;
                        if (idx >= 0 && idx < mddfBuried.Length) mddfBuried[idx] = true;
                    }
                }

                const int modfEntrySize = 64;
                int modfCount = modf.Data.Length / modfEntrySize;
                var modfBuried = new bool[modfCount];
                for (int offset = 0; offset + modfEntrySize <= modf.Data.Length; offset += modfEntrySize)
                {
                    uint uid = BitConverter.ToUInt32(modf.Data, offset + 4);
                    totalPlacements++;
                    if (uid > maxUniqueId)
                    {
                        var newZ = BitConverter.GetBytes(buryDepth);
                        Array.Copy(newZ, 0, modf.Data, offset + 12, 4);
                        removed++;
                        int idx = offset / modfEntrySize;
                        if (idx >= 0 && idx < modfBuried.Length) modfBuried[idx] = true;
                    }
                }

                // Per-ADT MCNK passes using MCIN offsets (neighbor-aware holes clearing)
                if (fixHoles || disableMcsh)
                {
                    // Parse MHDR -> MCIN for this ADT
                    var mhdr = new Chunk(wdtBytes, adtOffset);
                    int mhdrStart = adtOffset + 8;
                    int mcinRel = mhdr.GetOffset(0x0);
                    int mcinChunkOffset = mhdrStart + mcinRel;
                    var mcin = new Mcin(wdtBytes, mcinChunkOffset);
                    var mcnkOffsets = mcin.GetMcnkOffsets();
                    var localHolesNeighbors = holesNeighbors;
                    var localHolesPreserveWmo = holesPreserveWmo;

                    // Pre-scan: which chunks currently have holes and which reference to-be-buried placements
                    var chunkHasHoles = new bool[256];
                    var holesOffsetByIdx = new int[256];
                    var chunkHasBuriedRef = new bool[256];
                    var chunkHasKeepWmo = new bool[256];
                    Array.Fill(holesOffsetByIdx, -1);

                    for (int i = 0; i < mcnkOffsets.Count && i < 256; i++)
                    {
                        int pos = mcnkOffsets[i];
                        if (pos <= 0) continue;
                        if (!processedMcnk.Add(pos)) { /* de-dup across tiles for MCSH pass */ }

                        int headerStart = pos + 8; // skip 'MCNK' header
                        if (headerStart + 128 > wdtBytes.Length) continue;

                        // Record holes flag state and offset
                        int holesOffset = headerStart + 0x40; // McnkAlphaHeader.Holes
                        if (holesOffset + 4 <= wdtBytes.Length)
                        {
                            holesOffsetByIdx[i] = holesOffset;
                            int prev = BitConverter.ToInt32(wdtBytes, holesOffset);
                            chunkHasHoles[i] = prev != 0;
                        }

                        // Determine if this chunk references any soon-to-be-buried MDDF/MODF entries
                        try
                        {
                            int m2Number = BitConverter.ToInt32(wdtBytes, headerStart + 0x14);
                            int wmoNumber = BitConverter.ToInt32(wdtBytes, headerStart + 0x3C);
                            int mcrfRel = BitConverter.ToInt32(wdtBytes, headerStart + 0x24);
                            int mcrfChunkOffset = headerStart + 128 + mcrfRel;
                            if (mcrfChunkOffset + 8 <= wdtBytes.Length)
                            {
                                var mcrf = new Mcrf(wdtBytes, mcrfChunkOffset);
                                var m2Idx = mcrf.GetDoodadsIndices(Math.Max(0, m2Number));
                                var wmoIdx = mcrf.GetWmosIndices(Math.Max(0, wmoNumber));

                                foreach (var idx in m2Idx)
                                {
                                    if (idx >= 0 && idx < mddfBuried.Length && mddfBuried[idx]) { chunkHasBuriedRef[i] = true; break; }
                                }
                                if (!chunkHasBuriedRef[i])
                                {
                                    foreach (var idx in wmoIdx)
                                    {
                                        if (idx >= 0 && idx < modfBuried.Length && modfBuried[idx]) { chunkHasBuriedRef[i] = true; break; }
                                    }
                                }
                                if (localHolesPreserveWmo && !chunkHasKeepWmo[i])
                                {
                                    // If any unburied WMO is referenced by this chunk, preserve holes
                                    foreach (var idx in wmoIdx)
                                    {
                                        if (idx >= 0 && idx < modfBuried.Length && !modfBuried[idx]) { chunkHasKeepWmo[i] = true; break; }
                                    }
                                }
                            }
                        }
                        catch { /* best-effort */ }
                    }

                    // Holes clearing with scope and WMO-preserve guard
                    if (fixHoles)
                    {
                        var toClear = new bool[256];
                        for (int i = 0; i < 256; i++)
                        {
                            if (!chunkHasBuriedRef[i]) continue;
                            int cx = i % 16, cy = i / 16;
                            for (int dy = -1; dy <= 1; dy++)
                            {
                                for (int dx = -1; dx <= 1; dx++)
                                {
                                    if (!localHolesNeighbors && (dx != 0 || dy != 0)) continue; // self-only
                                    int nx = cx + dx, ny = cy + dy;
                                    if (nx < 0 || ny < 0 || nx >= 16 || ny >= 16) continue;
                                    int j = ny * 16 + nx;
                                    if (chunkHasHoles[j])
                                    {
                                        if (!(localHolesPreserveWmo && chunkHasKeepWmo[j]))
                                            toClear[j] = true;
                                    }
                                }
                            }
                        }
                        for (int j = 0; j < 256; j++)
                        {
                            if (!toClear[j]) continue;
                            int off = holesOffsetByIdx[j];
                            if (off >= 0 && off + 4 <= wdtBytes.Length)
                            {
                                if (wdtBytes[off + 0] != 0 || wdtBytes[off + 1] != 0 || wdtBytes[off + 2] != 0 || wdtBytes[off + 3] != 0)
                                {
                                    wdtBytes[off + 0] = 0; wdtBytes[off + 1] = 0; wdtBytes[off + 2] = 0; wdtBytes[off + 3] = 0;
                                    holesCleared++;
                                }
                            }
                        }
                    }

                    // MCSH zeroing pass
                    if (disableMcsh)
                    {
                        for (int i = 0; i < mcnkOffsets.Count && i < 256; i++)
                        {
                            int pos = mcnkOffsets[i];
                            if (pos <= 0) continue;
                            int headerStart = pos + 8;
                            if (headerStart + 128 > wdtBytes.Length) continue;
                            int mcshOffset = BitConverter.ToInt32(wdtBytes, headerStart + 0x30);
                            int mcshSize = BitConverter.ToInt32(wdtBytes, headerStart + 0x34);
                            if (mcshSize > 0)
                            {
                                long payloadStart = (long)headerStart + 128 + mcshOffset;
                                long payloadEnd = payloadStart + mcshSize;
                                if (payloadStart >= 0 && payloadEnd <= wdtBytes.Length)
                                {
                                    Array.Clear(wdtBytes, (int)payloadStart, mcshSize);
                                    mcshZeroed++;
                                }
                            }
                        }
                    }
                }

                // Commit MDDF/MODF changes after hole/MCSH passes
                if (mddf.Data.Length > 0)
                {
                    int mddfFileOffset = adt.GetMddfDataOffset();
                    Array.Copy(mddf.Data, 0, wdtBytes, mddfFileOffset, mddf.Data.Length);
                }
                if (modf.Data.Length > 0)
                {
                    int modfFileOffset = adt.GetModfDataOffset();
                    Array.Copy(modf.Data, 0, wdtBytes, modfFileOffset, modf.Data.Length);
                }

                tilesProcessed++;
                if (tilesProcessed % 50 == 0) Console.WriteLine($"  Processed {tilesProcessed}/{existingAdts.Count} tiles...");
            }

            // Log MCNK stats if passes were requested
            if (fixHoles || disableMcsh)
            {
                Console.WriteLine($"[ok] MCNK pass: holesCleared={holesCleared}, mcshZeroed={mcshZeroed}, mcnkScanned={processedMcnk.Count}");
            }

            File.WriteAllBytes(outputPath, wdtBytes);
            Console.WriteLine($"[ok] Saved: {outputPath}");

            using (var md5 = System.Security.Cryptography.MD5.Create())
            {
                var hash = md5.ComputeHash(wdtBytes);
                var hashString = BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
                var md5FilePath = Path.Combine(alphaWdtDir, Path.GetFileNameWithoutExtension(outputPath) + ".md5");
                File.WriteAllText(md5FilePath, hashString);
                Console.WriteLine($"[ok] MD5: {hashString}");
                Console.WriteLine($"[ok] Saved: {md5FilePath}");
            }

            Console.WriteLine($"Total Placements:  {totalPlacements:N0}");
            Console.WriteLine($"  Removed:          {removed:N0}");

            // Optional: Export LK ADTs from modified Alpha WDT (with preflight skip)
            if (exportLkAdts && !force && PreflightChecks.HasCompleteLkAdts(mapName, lkOutDir, expectedAdtCount))
            {
                Console.WriteLine($"[preflight] SKIP LK export (already complete): {lkOutDir}");
            }
            else if (exportLkAdts)
            {
                try
                {
                    // Load optional area remap JSON (AlphaAreaId -> LkAreaId)
                    Dictionary<int,int>? areaRemap = null;
                    if (!string.IsNullOrWhiteSpace(areaRemapJsonPath) && File.Exists(areaRemapJsonPath))
                    {
                        try
                        {
                            var json = File.ReadAllText(areaRemapJsonPath);
                            var tmp = JsonSerializer.Deserialize<Dictionary<string, int>>(json);
                            if (tmp != null)
                            {
                                areaRemap = new Dictionary<int,int>();
                                foreach (var kv in tmp) { if (int.TryParse(kv.Key, out var k)) areaRemap[k] = kv.Value; }
                                Console.WriteLine($"[lk] areaRemap entries={areaRemap.Count}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[warn] Failed to parse area remap JSON: {ex.Message}");
                        }
                    }

                    var wdtOut = new WdtAlpha(outputPath);
                    var existingAfter = wdtOut.GetExistingAdtsNumbers();
                    var adtOffsetsAfter = wdtOut.GetAdtOffsetsInMain();
                    var mdnmNames = wdtOut.GetMdnmFileNames();
                    var monmNames = wdtOut.GetMonmFileNames();

                    // Auto-map fallback when no JSON provided: use LK AreaTable.dbc to keep IDs that exist; others -> defaultUnmapped
                    if (areaRemap == null)
                    {
                        var alphaAreas = new HashSet<int>();
                        foreach (var adtNum2 in existingAfter)
                        {
                            int adtOff2 = adtOffsetsAfter[adtNum2];
                            if (adtOff2 == 0) continue;
                            var aScan = new AdtAlpha(outputPath, adtOff2, adtNum2);
                            foreach (var aid in aScan.GetAlphaMcnkAreaIds())
                            {
                                if (aid >= 0) alphaAreas.Add(aid);
                            }
                        }

                        var lkIds = string.IsNullOrWhiteSpace(lkClientPath)
                            ? new HashSet<int>()
                            : LkAreaTableDbcReader.LoadLkAreaIdsFromClient(lkClientPath);

                        areaRemap = new Dictionary<int, int>(capacity: alphaAreas.Count);
                        foreach (var aid in alphaAreas)
                        {
                            areaRemap[aid] = lkIds.Contains(aid) ? aid : defaultUnmapped;
                        }
                        Console.WriteLine($"[lk] auto-mapped {areaRemap.Count} alpha area IDs (default-unmapped={defaultUnmapped})");
                    }

                    // Load crosswalk patch mapping (CSV) and resolve current map id for guard
                    var aliasUsed = ResolveSrcAlias(GetOption(opts, "version"), inputPath);
                    var (patchMap, loadedFiles) = LoadPatchMapping(aliasUsed, dbctoolOutRoot, dbctoolPatchDir, dbctoolPatchFile);
                    int currentMapId = ResolveMapIdFromOptions(dbctoolOutRoot, lkDbcDir, mapName);
                    if (patchMap != null)
                    {
                        Console.WriteLine($"[patchmap] loaded crosswalks: per-map={patchMap.PerMapCount} global={patchMap.GlobalCount} files={loadedFiles.Count}");
                        if (copyCrosswalks && loadedFiles.Count > 0)
                        {
                            var cwOutDir = Path.Combine(outRoot, "reports", "crosswalk");
                            Directory.CreateDirectory(cwOutDir);
                            int copied = 0;
                            foreach (var f in loadedFiles)
                            {
                                try
                                {
                                    var dest = Path.Combine(cwOutDir, Path.GetFileName(f));
                                    File.Copy(f, dest, true);
                                    copied++;
                                }
                                catch { }
                            }
                            Console.WriteLine($"[patchmap] copied {copied}/{loadedFiles.Count} CSVs to: {cwOutDir}");
                        }
                    }
                    else
                    {
                        Console.WriteLine("[patchmap] no crosswalk CSVs resolved. Will attempt auto-generation if enabled.");
                        if (autoCrosswalks)
                        {
                            var genDir = GenerateCrosswalksIfNeeded(aliasUsed, inputPath, dbdDirOpt, srcDbcDirOpt, lkDbcDir, pivot060DirOpt, chainVia060, outRoot, srcClientPath, lkClientPath, opts.GetValueOrDefault("pivot-060-client-path", ""));
                            if (!string.IsNullOrWhiteSpace(genDir) && Directory.Exists(genDir!))
                            {
                                var genBase = Path.Combine(outRoot, "dbctool_outputs");
                                var reload = LoadPatchMapping(aliasUsed, genBase, genDir!, "");
                                patchMap = reload.Map; loadedFiles = reload.Files;
                                if (patchMap != null)
                                {
                                    Console.WriteLine($"[patchmap] generated crosswalks at: {genDir}");
                                    Console.WriteLine($"[patchmap] loaded crosswalks: per-map={patchMap.PerMapCount} global={patchMap.GlobalCount} files={loadedFiles.Count}");
                                    if (copyCrosswalks && loadedFiles.Count > 0)
                                    {
                                        var cwOutDir = Path.Combine(outRoot, "reports", "crosswalk");
                                        Directory.CreateDirectory(cwOutDir);
                                        int copied = 0;
                                        foreach (var f in loadedFiles)
                                        {
                                            try { File.Copy(f, Path.Combine(cwOutDir, Path.GetFileName(f)), true); copied++; } catch { }
                                        }
                                        Console.WriteLine($"[patchmap] copied {copied}/{loadedFiles.Count} CSVs to: {cwOutDir}");
                                    }
                                }
                                else
                                {
                                    Console.WriteLine("[patchmap] generation completed but mapping reload failed; AreaIDs will remain 0 in strict mode.");
                                }
                            }
                            else
                            {
                                Console.WriteLine("[patchmap] auto-generation failed or prerequisites missing. Provide --dbctool-patch-dir or fix inputs.");
                            }
                        }
                        else
                        {
                            Console.WriteLine("[patchmap] auto-generation disabled. Pass --auto-crosswalks (default) or provide --dbctool-patch-dir.");
                        }
                    }

                    Directory.CreateDirectory(lkOutDir);
                    int written = 0;
                    long totalAreaPresent = 0, totalAreaPatched = 0, totalAreaMapped = 0;
                    var perAdtRows = new List<string>();
                    foreach (var adtNum2 in existingAfter)
                    {
                        int adtOff2 = adtOffsetsAfter[adtNum2];
                        if (adtOff2 == 0) continue;
                        var a = new AdtAlpha(outputPath, adtOff2, adtNum2);
                        var lk = a.ToAdtLk(mdnmNames, monmNames, areaRemap);
                        lk.ToFile(lkOutDir); // Treats directory as output root

                        // Crosswalk-based AreaID patching (in-place)
                        if (patchMap != null)
                        {
                            var alphaAreaIds = a.GetAlphaMcnkAreaIds();
                            var outFile = Path.Combine(lkOutDir, $"{mapName}_{adtNum2 % 64}_{adtNum2 / 64}.adt");
                            try
                            {
                                var (present, patched, mapped) = PatchMcnkAreaIdsOnDiskV2(outFile, mapName, alphaAreaIds, patchMap, currentMapId, strictAreaId, chainVia060);
                                totalAreaPresent += present; totalAreaPatched += patched; totalAreaMapped += mapped;
                                if (reportAreaId)
                                {
                                    var fname = Path.GetFileName(outFile);
                                    var unmatched = Math.Max(0, present - mapped);
                                    perAdtRows.Add($"{fname},{present},{mapped},{patched},{unmatched}");
                                }
                                if (present > 0 && written < 4)
                                {
                                    var unmatched = Math.Max(0, present - mapped);
                                    Console.WriteLine($"  [AreaIds] {Path.GetFileName(outFile)} present={present} mapped={mapped} patched={patched} unmatched={unmatched}");
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[warn] AreaID patch failed for {outFile}: {ex.Message}");
                            }
                        }
                        written++;
                        if (written % 50 == 0) Console.WriteLine($"  [lk] Wrote {written}/{existingAfter.Count} ADTs...");
                    }
                    Console.WriteLine($"[ok] Exported {written}/{existingAfter.Count} LK ADTs to: {lkOutDir}");
                    // Also emit LK WDT alongside LK ADTs: convert modified Alpha WDT -> LK WDT and place as <mapName>.wdt
                    try
                    {
                        var wdtAlphaForLk = new WdtAlpha(outputPath);
                        var wdtLk = wdtAlphaForLk.ToWdt();
                        // Write using library's default naming (Azeroth.wdt_new), then rename to Azeroth.wdt
                        wdtLk.ToFile(lkOutDir);
                        var emitted = Path.Combine(lkOutDir, Path.GetFileName(outputPath) + "_new");
                        var desired = Path.Combine(lkOutDir, mapName + ".wdt");
                        try { if (File.Exists(desired)) File.Delete(desired); } catch { /* best-effort */ }
                        if (File.Exists(emitted)) File.Move(emitted, desired, overwrite: true);
                        Console.WriteLine($"[lk] Wrote LK WDT: {desired}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[warn] Failed to write LK WDT: {ex.Message}");
                    }
                    if (patchMap != null)
                    {
                        var totalUnmatched = Math.Max(0, totalAreaPresent - totalAreaMapped);
                        Console.WriteLine($"[AreaIds] summary: present={totalAreaPresent} mapped={totalAreaMapped} patched={totalAreaPatched} unmatched={totalUnmatched}");
                        if (reportAreaId)
                        {
                            var reportDir = Path.Combine(outRoot, "reports");
                            Directory.CreateDirectory(reportDir);
                            var reportPath = Path.Combine(reportDir, $"areaid_patch_summary_{mapName}.csv");
                            File.WriteAllLines(reportPath, new[]{"file,present,mapped,patched,unmatched"}.Concat(perAdtRows));
                            Console.WriteLine($"[AreaIds] report: {reportPath}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[warn] LK ADT export failed: {ex.Message}");
                }
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[error] Rollback failed: {ex.Message}");
            return 1;
        }
    }

    // === Crosswalk integration helpers ===
    private static (DbcPatchMapping? Map, List<string> Files) LoadPatchMapping(string alias, string dbctoolOutRoot, string patchDir, string patchFile)
    {
        try
        {
            bool any = false;
            var map = new DbcPatchMapping();
            var loaded = new List<string>();

            IEnumerable<string> EnumerateCrosswalkCsvs(string dir)
            {
                if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir)) yield break;
                var patterns = new[] { "Area_patch_crosswalk_*.csv", "Area_crosswalk_v*.csv" };
                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var pat in patterns)
                {
                    foreach (var f in Directory.EnumerateFiles(dir, pat, SearchOption.AllDirectories))
                    {
                        if (seen.Add(f)) yield return f;
                    }
                }
            }

            if (!string.IsNullOrWhiteSpace(patchFile))
            {
                // Resolve relative against patchDir or outRoot/<alias>/compare/v[3|2]
                if (Path.IsPathFullyQualified(patchFile) && File.Exists(patchFile))
                {
                    map.LoadFile(patchFile); any = true; loaded.Add(patchFile);
                }
                else
                {
                    if (!string.IsNullOrWhiteSpace(patchDir))
                    {
                        var cand = Path.Combine(patchDir, patchFile);
                        if (File.Exists(cand)) { map.LoadFile(cand); any = true; loaded.Add(cand); }
                    }
                    if (!any && !string.IsNullOrWhiteSpace(dbctoolOutRoot) && !string.IsNullOrWhiteSpace(alias))
                    {
                        var v3 = Path.Combine(dbctoolOutRoot, alias, "compare", "v3", patchFile);
                        var v2 = Path.Combine(dbctoolOutRoot, alias, "compare", "v2", patchFile);
                        if (File.Exists(v3)) { map.LoadFile(v3); any = true; loaded.Add(v3); }
                        else if (File.Exists(v2)) { map.LoadFile(v2); any = true; loaded.Add(v2); }
                    }
                }
            }

            if (!any && !string.IsNullOrWhiteSpace(patchDir) && Directory.Exists(patchDir))
            {
                foreach (var f in EnumerateCrosswalkCsvs(patchDir)) { map.LoadFile(f); any = true; loaded.Add(f); }
            }

            if (!any && !string.IsNullOrWhiteSpace(dbctoolOutRoot) && !string.IsNullOrWhiteSpace(alias))
            {
                var v3Dir = Path.Combine(dbctoolOutRoot, alias, "compare", "v3");
                var v2Dir = Path.Combine(dbctoolOutRoot, alias, "compare", "v2");
                if (Directory.Exists(v3Dir)) { foreach (var f in EnumerateCrosswalkCsvs(v3Dir)) { map.LoadFile(f); any = true; loaded.Add(f); } }
                if (Directory.Exists(v2Dir)) { foreach (var f in EnumerateCrosswalkCsvs(v2Dir)) { map.LoadFile(f); any = true; loaded.Add(f); } }
            }
            return any ? (map, loaded) : (null, new List<string>());
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[warn] Failed to load crosswalk mapping: {ex.Message}");
            return (null, new List<string>());
        }
    }

    private static int ResolveMapIdFromOptions(string dbctoolOutRoot, string lkDbcDir, string mapName)
    {
        try
        {
            // Prefer MapIdResolver from DBCTool outputs if available
            if (!string.IsNullOrWhiteSpace(dbctoolOutRoot))
            {
                // Try common aliases in priority order
                foreach (var alias in new[] { "0.6.0", "0.5.5", "0.5.3" })
                {
                    var res = MapIdResolver.LoadFromDbcToolOutput(dbctoolOutRoot, alias);
                    if (res != null)
                    {
                        var id = res.GetMapIdByDirectory(mapName);
                        if (id.HasValue) return id.Value;
                    }
                }
            }

            // Fallback to reading Map.dbc directly
            var dbc = string.Empty;
            if (!string.IsNullOrWhiteSpace(lkDbcDir))
            {
                dbc = Path.Combine(lkDbcDir, "Map.dbc");
            }
            if (!string.IsNullOrWhiteSpace(dbc) && File.Exists(dbc))
            {
                return ReadMapDbcIdByDirectory(dbc, mapName);
            }
        }
        catch { }
        return -1;
    }

    private static int ReadMapDbcIdByDirectory(string dbcPath, string targetName)
    {
        using var fs = new FileStream(dbcPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs, System.Text.Encoding.UTF8, leaveOpen: false);
        var magic = br.ReadBytes(4);
        if (magic.Length != 4) return -1;
        int recordCount = br.ReadInt32();
        int fieldCount = br.ReadInt32();
        int recordSize = br.ReadInt32();
        int stringBlockSize = br.ReadInt32();
        var records = br.ReadBytes(recordCount * recordSize);
        var stringBlock = br.ReadBytes(stringBlockSize);
        for (int i = 0; i < recordCount; i++)
        {
            int baseOff = i * recordSize;
            var ints = new int[fieldCount];
            for (int f = 0; f < fieldCount; f++)
            {
                int off = baseOff + (f * 4);
                if (off + 4 <= records.Length) ints[f] = BitConverter.ToInt32(records, off);
            }
            for (int f = 0; f < fieldCount; f++)
            {
                int sOff = ints[f];
                if (sOff > 0 && sOff < stringBlock.Length)
                {
                    int end = sOff;
                    while (end < stringBlock.Length && stringBlock[end] != 0) end++;
                    if (end > sOff)
                    {
                        var s = System.Text.Encoding.UTF8.GetString(stringBlock, sOff, end - sOff);
                        if (!string.IsNullOrWhiteSpace(s) && s.Equals(targetName, StringComparison.OrdinalIgnoreCase))
                            return ints[0];
                    }
                }
            }
        }
        return -1;
    }

    private static string ResolveSrcAlias(string? explicitAlias, string inputPath)
    {
        static string Normalize(string s)
        {
            var t = (s ?? string.Empty).Trim().ToLowerInvariant();
            if (t.StartsWith("0.6.0")) return "0.6.0";
            if (t.StartsWith("0.5.5")) return "0.5.5";
            if (t.StartsWith("0.5.3")) return "0.5.3";
            return s ?? string.Empty;
        }
        if (!string.IsNullOrWhiteSpace(explicitAlias)) return Normalize(explicitAlias!);
        var corpus = inputPath?.ToLowerInvariant() ?? string.Empty;
        if (corpus.Contains("0.6.0") || corpus.Contains("\\060\\") || corpus.Contains("/060/") || corpus.Contains("0_6_0")) return "0.6.0";
        if (corpus.Contains("0.5.5") || corpus.Contains("\\055\\") || corpus.Contains("/055/") || corpus.Contains("0_5_5")) return "0.5.5";
        if (corpus.Contains("0.5.3") || corpus.Contains("\\053\\") || corpus.Contains("/053/") || corpus.Contains("0_5_3")) return "0.5.3";
        return "0.5.3";
    }

    private static string ReverseFourCC(string s) => new string(s.Reverse().ToArray());

    private static string? ResolveTargetMapNameFromId(int? mapId)
    {
        if (!mapId.HasValue || mapId.Value < 0) return null;
        return mapId.Value switch
        {
            0 => "Eastern Kingdoms",
            1 => "Kalimdor",
            530 => "Outland",
            571 => "Northrend",
            _ => null,
        };
    }

    private static (int present, int patched, int mapped) PatchMcnkAreaIdsOnDiskV2(string filePath, string mapName, IReadOnlyList<int> alphaAreaIds, DbcPatchMapping patchMap, int currentMapId, bool strictMapLocked, bool chainVia060)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read, bufferSize: 65536, options: FileOptions.RandomAccess);
        using var br = new BinaryReader(fs);
        using var bw = new BinaryWriter(fs);

        long fileLen = fs.Length;
        long mcinDataPos = -1; int mcinSize = 0;
        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(System.Text.Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32();
            long dpos = fs.Position;
            if (fourcc == "MCIN") { mcinDataPos = dpos; mcinSize = sz; break; }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }

        int present = 0, patched = 0, mappedCount = 0;
        if (mcinDataPos >= 0 && mcinSize >= 16)
        {
            // Pre-read MCIN
            fs.Position = mcinDataPos;
            int need = Math.Min(mcinSize, 256 * 16);
            var mcinBytes = br.ReadBytes(need);
            for (int i = 0; i < 256; i++)
            {
                int mcnkOffset = (mcinBytes.Length >= (i + 1) * 16) ? BitConverter.ToInt32(mcinBytes, i * 16) : 0;
                if (mcnkOffset <= 0) continue;
                present++;

                int lkAreaId = 0; bool mapped = false;
                // Derive alpha numeric area (zone<<16|sub)
                int aIdNum = -1;
                if (alphaAreaIds != null && alphaAreaIds.Count == 256)
                {
                    int alt = alphaAreaIds[i];
                    if (alt > 0) aIdNum = ((alt >> 16) == 0) ? (alt << 16) : alt;
                }
                int zoneBase = (aIdNum > 0) ? (aIdNum & unchecked((int)0xFFFF0000)) : 0;
                int subLo = (aIdNum > 0) ? (aIdNum & 0xFFFF) : 0;

                bool Accept(int cand)
                {
                    if (cand <= 0) return false;
                    lkAreaId = cand; mapped = true; return true;
                }

                if (aIdNum > 0)
                {
                    if (currentMapId >= 0 && patchMap.TryMapByTarget(currentMapId, aIdNum, out var numMap)) { Accept(numMap); }
                    if (!mapped)
                    {
                        var tgtName = ResolveTargetMapNameFromId(currentMapId);
                        if (!string.IsNullOrWhiteSpace(tgtName) && patchMap.TryMapByTargetName(tgtName!, aIdNum, out var numMapName)) { Accept(numMapName); }
                    }
                    if (!mapped && patchMap.TryMapBySrcAreaSimple(mapName, aIdNum, out var byName)) { Accept(byName); }
                    if (!mapped && !strictMapLocked)
                    {
                        if (patchMap.TryMapBySrcAreaNumber(aIdNum, out var exactId, out _)) { Accept(exactId); }
                    }
                    if (!mapped && chainVia060)
                    {
                        if (patchMap.TryMapViaMid(currentMapId, aIdNum, out var midId, out _, out _)) { Accept(midId); }
                    }
                }

                // Write AreaId (0 if unmapped)
                long areaFieldPos = (long)mcnkOffset + 8 + 0x34;
                if (areaFieldPos + 4 <= fileLen)
                {
                    long save = fs.Position;
                    fs.Position = areaFieldPos;
                    uint existing = br.ReadUInt32();
                    int effective = mapped && lkAreaId > 0 ? lkAreaId : 0;
                    if (mapped && lkAreaId > 0) mappedCount++;
                    if (existing != (uint)effective)
                    {
                        fs.Position = areaFieldPos;
                        bw.Write((uint)effective);
                        patched++;
                    }
                    fs.Position = save;
                }
            }
        }
        return (present, patched, mappedCount);
    }

    private static List<int> FindAllChunks(byte[] data, string chunkName)
    {
        var positions = new List<int>();
        var pattern = System.Text.Encoding.ASCII.GetBytes(chunkName);
        for (int i = 0; i < data.Length - pattern.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < pattern.Length; j++)
            {
                if (data[i + j] != pattern[j]) { match = false; break; }
            }
            if (match)
            {
                positions.Add(i);
                i += 8; // skip chunk header to avoid immediate re-match
            }
        }
        return positions;
    }
}
