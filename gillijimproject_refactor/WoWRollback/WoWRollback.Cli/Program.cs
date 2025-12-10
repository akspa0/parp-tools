using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
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
using GillijimProject.WowFiles.LichKing;
using GillijimProject.Utilities;
using AlphaWdtAnalyzer.Core.Export;
using AlphaWdtAnalyzer.Core.Dbc;
using WoWRollback.Core.Services.AreaMapping;
using DBCTool.V2.Cli;
using WoWRollback.LkToAlphaModule;
using WoWRollback.LkToAlphaModule.Writers;
using MPQToTACT.MPQ;
using WoWRollback.Core.Services.Assets;
using WoWRollback.Core.Services.PM4;
using WoWRollback.Cli.Commands;

namespace WoWRollback.Cli;

internal static class Program
{
    private static readonly Dictionary<(string Version, string Map), string?> AlphaWdtCache = new(new TupleComparer());
    
    private static bool TryIsAlphaV18Wdt(byte[] buf, out string reason)
    {
        reason = string.Empty;
        try
        {
            if (buf == null || buf.Length < 24) { reason = "buffer too small"; return false; }
            static int Next(int start, int size) => start + 8 + size + ((size & 1) == 1 ? 1 : 0);

            // Chunk 0: MVER (on-disk letters reversed: REVM)
            var c0 = Encoding.ASCII.GetString(buf, 0, 4);
            if (!string.Equals(c0, "REVM", StringComparison.Ordinal)) { reason = $"first FourCC not MVER (got {c0})"; return false; }
            int sz0 = BitConverter.ToInt32(buf, 4);
            if (8 + sz0 > buf.Length) { reason = "MVER size out of range"; return false; }
            int ver = BitConverter.ToInt32(buf, 8);
            if (ver != 18) { reason = $"MVER version {ver} (expected 18)"; return false; }
            int off1 = Next(0, sz0);
            if (off1 + 8 > buf.Length) { reason = "missing MPHD"; return false; }

            // Chunk 1: MPHD (on-disk DHPM), size 128 in Alpha
            var c1 = Encoding.ASCII.GetString(buf, off1, 4);
            if (!string.Equals(c1, "DHPM", StringComparison.Ordinal)) { reason = $"second FourCC not MPHD (got {c1})"; return false; }
            int sz1 = BitConverter.ToInt32(buf, off1 + 4);
            if (sz1 != 128) { reason = $"MPHD size {sz1} (expected 128)"; return false; }
            int off2 = Next(off1, sz1);
            if (off2 + 8 > buf.Length) { reason = "missing MAIN"; return false; }

            // Chunk 2: MAIN (on-disk NIAM)
            var c2 = Encoding.ASCII.GetString(buf, off2, 4);
            if (!string.Equals(c2, "NIAM", StringComparison.Ordinal)) { reason = $"third FourCC not MAIN (got {c2})"; return false; }
            return true;
        }
        catch (Exception ex)
        {
            reason = ex.Message;
            return false;
        }
    }

    private static void SetupTeeLogging(string cmd, Dictionary<string, string> opts)
    {
        try
        {
            var logDir = GetOption(opts, "log-dir");
            var logFile = GetOption(opts, "log-file");
            if (string.IsNullOrWhiteSpace(logDir) && string.IsNullOrWhiteSpace(logFile)) return;

            string dir = string.IsNullOrWhiteSpace(logDir) ? Directory.GetCurrentDirectory() : logDir!;
            Directory.CreateDirectory(dir);

            string fileName = string.IsNullOrWhiteSpace(logFile)
                ? $"wowrollback_{cmd}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.log"
                : logFile!;
            if (!Path.IsPathRooted(fileName)) fileName = Path.Combine(dir, fileName);

            var fs = new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.Read);
            var writer = new StreamWriter(fs) { AutoFlush = true };
            Console.SetOut(new SplitWriter(Console.Out, writer));
            Console.SetError(new SplitWriter(Console.Error, writer));
            Console.WriteLine($"[log] tee: {fileName}");
        }
        catch { }
    }

    private sealed class SplitWriter : System.IO.TextWriter
    {
        private readonly System.IO.TextWriter _a;
        private readonly System.IO.TextWriter _b;
        public SplitWriter(System.IO.TextWriter a, System.IO.TextWriter b) { _a = a; _b = b; }
        public override Encoding Encoding => _a.Encoding;
        public override void Write(char value) { _a.Write(value); _b.Write(value); }
        public override void Write(string? value) { _a.Write(value); _b.Write(value); }
        public override void WriteLine(string? value) { _a.WriteLine(value); _b.WriteLine(value); }
        public override void Flush() { _a.Flush(); _b.Flush(); }
    }

    private static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintHelp();
            return 0;
        }

        var cmd = args[0].ToLowerInvariant();
        var opts = ParseArgs(args.Skip(1).ToArray());
        SetupTeeLogging(cmd, opts);

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
                case "analyze-map-adts-casc":
                    return RunAnalyzeMapAdtsCasc(opts);
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
                case "snapshot-listfile":
                    return RunSnapshotListfile(opts);
                case "diff-listfiles":
                    return RunDiffListfiles(opts);
                case "pack-monolithic-alpha-wdt":
                    return RunPackMonolithicAlphaWdt(opts);
                case "pack-wdl-from-lk":
                    return RunPackWdlFromLk(opts);
                case "build-test-adt":
                    return RunBuildTestAdt(opts);
                case "gui":
                    return RunGui(opts);
                case "regen-layers":
                case "analyze-layers-from-placements":
                    return RunRegenLayers(opts);
                case "run-preset":
                    return RunRunPreset(opts);
                case "fix-minimap-webp":
                    return RunFixMinimapWebp(opts);
                case "dry-run":
                    return RunDryRun(opts);
                case "compare-versions":
                    return RunCompareVersions(opts);
                case "compute-heatmap-stats":
                case "heatmap-stats":
                    return RunComputeHeatmapStats(opts);
                case "alpha-roundtrip-verify":
                    return RunAlphaRoundtripVerify(opts);
                case "dump-alpha-mcnk":
                    return RunDumpAlphaMcnk(opts);
                case "pm4-adt-correlate":
                    return RunPm4AdtCorrelate(opts);
                case "wmo-walkable-extract":
                    return RunWmoWalkableExtract(opts);
                case "pm4-export-obj":
                    return RunPm4ExportObj(opts);
                case "pm4-wmo-match":
                    return RunPm4WmoMatch(opts);
                case "pm4-reconstruct-modf":
                case "modf-reconstruct":
                    return RunPm4ReconstructModf(opts);
                case "wmo-batch-extract":
                    return RunWmoBatchExtract(opts);
                case "pm4-export-modf":
                    return RunPm4ExportModf(opts);
                case "pm4-create-adt":
                    return RunPm4CreateAdt(opts);
                case "development-repair":
                    return DevelopmentRepairCommand.Execute(opts);
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

    private static int RunPackMonolithicAlphaWdt(Dictionary<string, string> opts)
    {
        // Common options
        var targetListfile = GetOption(opts, "target-listfile");
        var modernListfile = GetOption(opts, "modern-listfile");
        var strict = !opts.TryGetValue("strict-target-assets", out var strictStr) || !string.Equals(strictStr, "false", StringComparison.OrdinalIgnoreCase);
        var exportMccv = GetOption(opts, "export-mccv");
        var outWdt = GetOption(opts, "out");
        if (string.IsNullOrWhiteSpace(outWdt)) { Console.Error.WriteLine("[error] --out <wdt> is required"); return 2; }

        var options = new LkToAlphaOptions
        {
            TargetListfilePath = targetListfile,
            ModernListfilePath = modernListfile,
            StrictTargetAssets = strict
        };
        if (opts.ContainsKey("extract-assets"))
        {
            options = options with { ExtractAssets = true, AssetScope = "textures+models" };
        }
        var assetScopeOpt = GetOption(opts, "asset-scope");
        if (!string.IsNullOrWhiteSpace(assetScopeOpt))
        {
            options = options with { AssetScope = assetScopeOpt! };
        }
        var assetsOutOpt = GetOption(opts, "assets-out");
        if (!string.IsNullOrWhiteSpace(assetsOutOpt))
        {
            options = options with { AssetsOut = assetsOutOpt };
        }
        // New flags: skip toggles and conversion controls
        if (opts.ContainsKey("skip-wmos")) options = options with { SkipWmos = true };
        if (opts.ContainsKey("skip-m2")) options = options with { SkipM2 = true };
        var assetsSourceRootOpt = GetOption(opts, "assets-source-root");
        if (!string.IsNullOrWhiteSpace(assetsSourceRootOpt)) options = options with { AssetsSourceRoot = assetsSourceRootOpt };
        if (opts.ContainsKey("convert-models-to-legacy")) options = options with { ConvertModelsToLegacy = true };
        if (opts.ContainsKey("convert-wmos-to-legacy")) options = options with { ConvertWmosToLegacy = true };
        var wantVerbose = opts.ContainsKey("verbose-logging") || opts.ContainsKey("alpha-debug") || opts.ContainsKey("verbose");
        if (wantVerbose) options = options with { VerboseLogging = true, Verbose = true };
        if (opts.ContainsKey("raw-copy-layers")) options = options with { RawCopyLkLayers = true };
        if (opts.ContainsKey("prefer-tex") || opts.ContainsKey("prefer-tex-layers")) options = options with { PreferTexLayers = true };
        if (!string.IsNullOrWhiteSpace(exportMccv)) options = options with { ExportMccvDir = exportMccv };

        // MPQ client mode
        var clientPath = GetOption(opts, "client-path");
        var mapName = GetOption(opts, "map");
        if (!string.IsNullOrWhiteSpace(clientPath) && !string.IsNullOrWhiteSpace(mapName))
        {
            Console.WriteLine("[pack] Building monolithic Alpha WDT from MPQ client...");
            Console.WriteLine($"[pack] Client (MPQ): {clientPath}");
            Console.WriteLine($"[pack] Map: {mapName}");
            if (!string.IsNullOrWhiteSpace(targetListfile)) Console.WriteLine($"[pack] Target listfile: {targetListfile}");
            Console.WriteLine($"[pack] Strict target assets: {options.StrictTargetAssets.ToString().ToLowerInvariant()}");
            if (options.ExtractAssets) Console.WriteLine("[pack] Extract assets: true");

            if (!Directory.Exists(clientPath)) { Console.Error.WriteLine("[error] --client-path does not exist"); return 2; }
            EnsureStormLibOnPath();
            var mpqs = ArchiveLocator.LocateMpqs(clientPath);
            if (wantVerbose)
            {
                Console.WriteLine($"[mpq][order] total={mpqs.Count}");
                int n = Math.Min(5, mpqs.Count);
                for (int i = 0; i < n; i++) Console.WriteLine($"[mpq][first] {mpqs[i]}");
                for (int i = Math.Max(0, mpqs.Count - n); i < mpqs.Count; i++) Console.WriteLine($"[mpq][last]  {mpqs[i]}");

                static bool IsLocalePathLocal(string path)
                {
                    var p = path.Replace('\\', '/');
                    var idx = p.IndexOf("/Data/", StringComparison.OrdinalIgnoreCase);
                    if (idx < 0) return false;
                    var rest = p.Substring(idx + 6);
                    int slash = rest.IndexOf('/');
                    if (slash <= 0) return false;
                    var seg = rest.Substring(0, slash);
                    if (seg.Length != 4) return false;
                    return char.IsLetter(seg[0]) && char.IsLetter(seg[1]) && char.IsLetter(seg[2]) && char.IsLetter(seg[3]);
                }

                var reNum = new Regex(@"patch(?:[-_][a-z]{2}[A-Z]{2})?[-_]?([0-9]+)\.mpq", RegexOptions.IgnoreCase);
                var reLet = new Regex(@"patch(?:[-_][a-z]{2}[A-Z]{2})?[-_]?([A-Za-z])\.mpq", RegexOptions.IgnoreCase);
                var rootNums = new List<string>();
                var localeNums = new List<string>();
                var rootLets = new List<string>();
                var localeLets = new List<string>();
                foreach (var mpq in mpqs)
                {
                    var file = Path.GetFileName(mpq);
                    if (reNum.IsMatch(file))
                    {
                        if (IsLocalePathLocal(mpq)) localeNums.Add(mpq); else rootNums.Add(mpq);
                    }
                    if (reLet.IsMatch(file))
                    {
                        if (IsLocalePathLocal(mpq)) localeLets.Add(mpq); else rootLets.Add(mpq);
                    }
                }
                Console.WriteLine($"[mpq][numeric] root={rootNums.Count}, locale={localeNums.Count}");
                foreach (var p in rootNums.OrderBy(x => x, StringComparer.OrdinalIgnoreCase)) Console.WriteLine($"[mpq][numeric][root] {p}");
                foreach (var p in localeNums.OrderBy(x => x, StringComparer.OrdinalIgnoreCase)) Console.WriteLine($"[mpq][numeric][locale] {p}");
                Console.WriteLine($"[mpq][letter] root={rootLets.Count}, locale={localeLets.Count}");
                foreach (var p in rootLets.OrderBy(x => x, StringComparer.OrdinalIgnoreCase)) Console.WriteLine($"[mpq][letter][root] {p}");
                foreach (var p in localeLets.OrderBy(x => x, StringComparer.OrdinalIgnoreCase)) Console.WriteLine($"[mpq][letter][locale] {p}");
                Console.WriteLine("[mpq][overlay] FS > root-letter > locale-letter > root-numeric > locale-numeric > base");
            }
            using var src = new PrioritizedArchiveSource(clientPath, mpqs);
            AlphaWdtMonolithicWriter.WriteMonolithicFromArchive(src, mapName, outWdt!, options);
            // Optional: export LK ADTs for diagnostics
            if (opts.ContainsKey("export-lk-adts-after-pack"))
            {
                try
                {
                    Console.WriteLine("[diag] Exporting LK ADTs after pack (diagnostic phase)...");
                    var roll = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                    {
                        ["input"] = outWdt!,
                        ["out"] = Path.GetDirectoryName(Path.GetFullPath(outWdt!)) ?? ".",
                        ["max-uniqueid"] = int.MaxValue.ToString(System.Globalization.CultureInfo.InvariantCulture),
                        ["export-lk-adts"] = "true",
                        ["force"] = "true"
                    };
                    var lkOut = GetOption(opts, "lk-out");
                    if (!string.IsNullOrWhiteSpace(lkOut)) roll["lk-out"] = lkOut!;
                    // Pre-create diagnostics CSV from driver as early as possible
                    try
                    {
                        var outRootForDiag = roll["out"];
                        string[] diagCandidates = new[]
                        {
                            Path.Combine(outRootForDiag, "lk_export_diag.csv"),
                            Path.Combine(!string.IsNullOrWhiteSpace(lkOut) ? lkOut! : outRootForDiag, "lk_export_diag.csv"),
                            Path.Combine(Directory.GetCurrentDirectory(), "lk_export_diag.csv"),
                        };
                        var header = "tile_yy,tile_xx,chunk_idx,nDoodadRefs,nMapObjRefs,offsRefs,offsLayer,offsAlpha,offsShadow,offsSnd,offsLiquid,mddf_count,modf_count,mcrf_expected,mcrf_payload,d_samples,w_samples,exception\n";
                        var diagDriverWritten = new List<string>();
                        foreach (var cand in diagCandidates)
                        {
                            try
                            {
                                var dir = Path.GetDirectoryName(cand);
                                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                                if (!File.Exists(cand)) File.WriteAllText(cand, header);
                                try { File.AppendAllText(cand, "-1,-1,-1,,,,,,,,,,,,,,,export_begin_driver\n"); } catch { }
                                diagDriverWritten.Add(cand);
                            }
                            catch { }
                        }
                        Console.WriteLine($"[diag] lk_export_diag.csv (driver) => {string.Join(" | ", diagDriverWritten)}");
                    }
                    catch { }
                    var lkClient = GetOption(opts, "lk-client-path");
                    if (!string.IsNullOrWhiteSpace(lkClient)) roll["lk-client-path"] = lkClient!;
                    var lkDbcDir = GetOption(opts, "lk-dbc-dir");
                    if (!string.IsNullOrWhiteSpace(lkDbcDir)) roll["lk-dbc-dir"] = lkDbcDir!;
                    var areaRemapJson = GetOption(opts, "area-remap-json");
                    if (!string.IsNullOrWhiteSpace(areaRemapJson)) roll["area-remap-json"] = areaRemapJson!;
                    var code = RunRollback(roll);
                    Console.WriteLine($"[diag] LK export finished (exit={code})");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[warn] Post-pack LK export failed: {ex.Message}");
                }
            }
            Console.WriteLine($"[ok] WDT written: {outWdt}");
            return 0;
        }

        // Loose file mode
        if (!opts.ContainsKey("lk-wdt")) { Console.Error.WriteLine("[error] Provide either --client-path & --map, or --lk-wdt (<file>)"); return 2; }
        var lkWdt = opts["lk-wdt"];
        var lkMapDir = opts.GetValueOrDefault("lk-map-dir", Path.GetDirectoryName(lkWdt) ?? ".");

        Console.WriteLine("[pack] Building monolithic Alpha WDT from LK inputs...");
        Console.WriteLine($"[pack] LK WDT: {lkWdt}");
        Console.WriteLine($"[pack] LK Map Dir: {lkMapDir}");
        if (!string.IsNullOrWhiteSpace(targetListfile)) Console.WriteLine($"[pack] Target listfile: {targetListfile}");
        Console.WriteLine($"[pack] Strict target assets: {options.StrictTargetAssets.ToString().ToLowerInvariant()}");

        AlphaWdtMonolithicWriter.WriteMonolithic(lkWdt, lkMapDir, outWdt!, options);
        // Emit a sibling WDL next to the WDT for distant terrain rendering
        try
        {
            var outWdl = Path.ChangeExtension(outWdt, ".wdl");
            WdlWriterV18.WriteFromLk(lkWdt, lkMapDir, outWdl);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[warn] WDL generation skipped: {ex.Message}");
        }
        // Optional: export LK ADTs for diagnostics
        if (opts.ContainsKey("export-lk-adts-after-pack"))
        {
            try
            {
                Console.WriteLine("[diag] Exporting LK ADTs after pack (diagnostic phase)...");
                var roll = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["input"] = outWdt!,
                    ["out"] = Path.GetDirectoryName(Path.GetFullPath(outWdt!)) ?? ".",
                    ["max-uniqueid"] = int.MaxValue.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    ["export-lk-adts"] = "true",
                    ["force"] = "true"
                };
                var lkOut = GetOption(opts, "lk-out");
                if (!string.IsNullOrWhiteSpace(lkOut)) roll["lk-out"] = lkOut!;
                // Pre-create diagnostics CSV from driver as early as possible
                try
                {
                    var outRootForDiag = roll["out"];
                    string[] diagCandidates = new[]
                    {
                        Path.Combine(outRootForDiag, "lk_export_diag.csv"),
                        Path.Combine(!string.IsNullOrWhiteSpace(lkOut) ? lkOut! : outRootForDiag, "lk_export_diag.csv"),
                        Path.Combine(Directory.GetCurrentDirectory(), "lk_export_diag.csv"),
                    };
                    var header = "tile_yy,tile_xx,chunk_idx,nDoodadRefs,nMapObjRefs,offsRefs,offsLayer,offsAlpha,offsShadow,offsSnd,offsLiquid,mddf_count,modf_count,mcrf_expected,mcrf_payload,d_samples,w_samples,exception\n";
                    var diagDriverWritten = new List<string>();
                    foreach (var cand in diagCandidates)
                    {
                        try
                        {
                            var dir = Path.GetDirectoryName(cand);
                            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                            if (!File.Exists(cand)) File.WriteAllText(cand, header);
                            try { File.AppendAllText(cand, "-1,-1,-1,,,,,,,,,,,,,,,export_begin_driver\n"); } catch { }
                            diagDriverWritten.Add(cand);
                        }
                        catch { }
                    }
                    Console.WriteLine($"[diag] lk_export_diag.csv (driver) => {string.Join(" | ", diagDriverWritten)}");
                }
                catch { }
                var lkClient = GetOption(opts, "lk-client-path");
                if (!string.IsNullOrWhiteSpace(lkClient)) roll["lk-client-path"] = lkClient!;
                var lkDbcDir = GetOption(opts, "lk-dbc-dir");
                if (!string.IsNullOrWhiteSpace(lkDbcDir)) roll["lk-dbc-dir"] = lkDbcDir!;
                var areaRemapJson = GetOption(opts, "area-remap-json");
                if (!string.IsNullOrWhiteSpace(areaRemapJson)) roll["area-remap-json"] = areaRemapJson!;
                var code = RunRollback(roll);
                Console.WriteLine($"[diag] LK export finished (exit={code})");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[warn] Post-pack LK export failed: {ex.Message}");
            }
        }
        Console.WriteLine($"[ok] WDT written: {outWdt}");
        return 0;
    }

    private static int RunBuildTestAdt(Dictionary<string, string> opts)
    {
        if (!opts.ContainsKey("lk-adt")) { Console.Error.WriteLine("[error] --lk-adt <file> is required"); return 2; }
        var inAdtPath = opts["lk-adt"]; var outAdtPath = GetOption(opts, "out");
        if (string.IsNullOrWhiteSpace(outAdtPath)) { Console.Error.WriteLine("[error] --out <file> is required"); return 2; }
        var pattern = opts.GetValueOrDefault("pattern", "solid").ToLowerInvariant();

        byte[] adtFile = File.ReadAllBytes(inAdtPath);
        int offsetInFile = 0;
        int currentChunkSize;
        var mver = new Chunk(adtFile, offsetInFile);
        offsetInFile += 4; currentChunkSize = BitConverter.ToInt32(adtFile, offsetInFile); offsetInFile = 4 + offsetInFile + currentChunkSize;
        int mhdrStartOffset = offsetInFile + 8;
        var mhdr = new Mhdr(adtFile, offsetInFile);

        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.McinOffset);
        var mcin = new Mcin(adtFile, offsetInFile);

        Mh2o mh2o = new Mh2o();
        if (mhdr.GetOffset(Mhdr.Mh2oOffset) != 0)
        {
            var off = mhdrStartOffset + mhdr.GetOffset(Mhdr.Mh2oOffset);
            mh2o = new Mh2o(adtFile, off);
        }

        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MtexOffset);
        var mtex = new Chunk(adtFile, offsetInFile);
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MmdxOffset);
        var mmdx = new Mmdx(adtFile, offsetInFile);
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MmidOffset);
        var mmid = new Mmid(adtFile, offsetInFile);
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MwmoOffset);
        var mwmo = new Mwmo(adtFile, offsetInFile);
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MwidOffset);
        var mwid = new Mwid(adtFile, offsetInFile);
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MddfOffset);
        var mddf = new Mddf(adtFile, offsetInFile);
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.ModfOffset);
        var modf = new Modf(adtFile, offsetInFile);

        Chunk mfbo = new Chunk("MFBO", 0, Array.Empty<byte>());
        if (mhdr.GetOffset(Mhdr.MfboOffset) != 0)
        {
            var off = mhdrStartOffset + mhdr.GetOffset(Mhdr.MfboOffset);
            mfbo = new Chunk(adtFile, off);
        }
        Chunk mtxf = new Chunk("MTXF", 0, Array.Empty<byte>());
        if (mhdr.GetOffset(Mhdr.MtxfOffset) != 0)
        {
            var off = mhdrStartOffset + mhdr.GetOffset(Mhdr.MtxfOffset);
            mtxf = new Chunk(adtFile, off);
        }

        var texNames = ParseMtexNames(mtex.Data);
        int texCount = texNames.Count == 0 ? 1 : texNames.Count;

        var mcnkOffsets = mcin.GetMcnkOffsets();
        var newMcnks = new List<McnkLk>(256);

        for (int idx = 0; idx < 256; idx++)
        {
            int mOff = mcnkOffsets[idx];
            if (mOff <= 0 || mOff + 8 > adtFile.Length)
            {
                newMcnks.Add(McnkLk.CreatePlaceholder());
                continue;
            }

            int headerStart = mOff;
            byte[] hdr = new byte[128];
            Buffer.BlockCopy(adtFile, headerStart + 8, hdr, 0, 128);
            var h = Utilities.ByteArrayToStruct<McnkHeader>(hdr);

            var mcvt = new Chunk(adtFile, headerStart + h.McvtOffset);
            Chunk? mccv = null;
            if (h.MccvOffset != 0) mccv = new Chunk(adtFile, headerStart + h.MccvOffset);
            var mcnr = new McnrLk(adtFile, headerStart + h.McnrOffset);
            var mcrf = new Mcrf(adtFile, headerStart + h.McrfOffset);
            Chunk? mcsh = null;
            if (h.McshOffset != 0 && h.McshOffset != h.McalOffset) mcsh = new Chunk(adtFile, headerStart + h.McshOffset);
            Chunk? mclq = null;
            if (h.MclqOffset != 0) mclq = new Chunk(adtFile, headerStart + h.MclqOffset);
            Chunk? mcse = null;
            if (h.McseOffset != 0) mcse = new Chunk(adtFile, headerStart + h.McseOffset);

            int chosenTex = idx % texCount;
            int baseTex = (chosenTex + 1) % texCount; // ensure visible contrast
            int nLayers = 2;
            var mclyBytes = new byte[nLayers * 16];
            void WriteLayer(int i, uint texId, uint flags, uint offs, uint effect)
            {
                int b = i * 16; Buffer.BlockCopy(BitConverter.GetBytes(texId), 0, mclyBytes, b + 0, 4); Buffer.BlockCopy(BitConverter.GetBytes(flags), 0, mclyBytes, b + 4, 4); Buffer.BlockCopy(BitConverter.GetBytes(offs), 0, mclyBytes, b + 8, 4); Buffer.BlockCopy(BitConverter.GetBytes(effect), 0, mclyBytes, b + 12, 4);
            }
            uint effNone = 0xFFFFFFFF;
            WriteLayer(0, (uint)(Math.Min(baseTex, int.MaxValue)), 0u, 0u, effNone); // base
            WriteLayer(1, (uint)(Math.Min(chosenTex, int.MaxValue)), 0x100u, 0u, effNone); // blend uses alpha
            var mcly = new Chunk("MCLY", mclyBytes.Length, mclyBytes);

            var alpha = new byte[64 * 64];
            switch (pattern)
            {
                case "solid":
                    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) alpha[y * 64 + x] = (byte)255; break;
                case "half-top":
                    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) alpha[y * 64 + x] = (byte)((y < 32) ? 255 : 0); break;
                case "half-left":
                    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) alpha[y * 64 + x] = (byte)((x < 32) ? 255 : 0); break;
                case "checker":
                    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) alpha[y * 64 + x] = (byte)((((x >> 3) ^ (y >> 3)) & 1) == 0 ? 255 : 0); break;
                case "diagonal":
                    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) alpha[y * 64 + x] = (byte)((x > y) ? 255 : 0); break;
                case "center-dot":
                    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) { int dx = x - 32, dy = y - 32; alpha[y * 64 + x] = (byte)((dx * dx + dy * dy <= 8 * 8) ? 255 : 0); } break;
                default:
                    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) alpha[y * 64 + x] = (byte)255; break;
            }
            var mcal = new Mcal("MCAL", alpha.Length, alpha);

            var outHdr = h;
            outHdr.NLayers = nLayers;
            var m = new McnkLk(outHdr, mcvt, mccv, mcnr, mcly, mcrf, mcsh, mcal, mclq, mcse);
            newMcnks.Add(m);
        }

        var adtName = Path.GetFileName(outAdtPath);
        var test = new AdtLk(adtName, mver, 0, mh2o, mtex, mmdx, mmid, mwmo, mwid, mddf, modf, newMcnks, mfbo, mtxf);
        test.ToFile(outAdtPath!);
        Console.WriteLine($"[ok] Test ADT written: {outAdtPath}");
        return 0;

        static List<string> ParseMtexNames(byte[] data)
        {
            var list = new List<string>();
            if (data == null || data.Length == 0) return list;
            int i = 0;
            while (i < data.Length)
            {
                int j = i; while (j < data.Length && data[j] != 0) j++;
                if (j > i) list.Add(Encoding.ASCII.GetString(data, i, j - i));
                i = j + 1;
            }
            return list;
        }
    }

    private static int RunPackWdlFromLk(Dictionary<string, string> opts)
    {
        if (!opts.ContainsKey("lk-wdt")) { Console.Error.WriteLine("[error] --lk-wdt <file> is required"); return 2; }
        var lkWdt = opts["lk-wdt"];
        var lkMapDir = opts.GetValueOrDefault("lk-map-dir", Path.GetDirectoryName(lkWdt) ?? ".");
        var outWdl = GetOption(opts, "out");
        if (string.IsNullOrWhiteSpace(outWdl)) { Console.Error.WriteLine("[error] --out <map.wdl> is required"); return 2; }

        Console.WriteLine("[pack] Building WDL v18 from LK inputs...");
        Console.WriteLine($"[pack] LK WDT: {lkWdt}");
        Console.WriteLine($"[pack] LK Map Dir: {lkMapDir}");
        WdlWriterV18.WriteFromLk(lkWdt, lkMapDir, outWdl!);
        return 0;
    }

    private static int RunSnapshotListfile(Dictionary<string, string> opts)
    {
        Require(opts, "client-path");
        Require(opts, "alias");
        Require(opts, "out");

        var clientPath = opts["client-path"]; // game root
        var alias = opts["alias"];            // full build string
        var outPath = opts["out"];            // json file
        var csvOut = GetOption(opts, "csv-out");
        var csvMissingToken = GetOption(opts, "csv-missing-fdid");
        if (string.IsNullOrWhiteSpace(csvMissingToken)) csvMissingToken = "0"; // default: zeros
        var communityPath = GetOption(opts, "community-listfile");

        if (!Directory.Exists(clientPath)) throw new DirectoryNotFoundException(clientPath);

        var snapshot = new ListfileSnapshot
        {
            Source = "mpq-scan",
            ClientRoot = Path.GetFullPath(clientPath),
            Version = alias,
            GeneratedAt = DateTime.UtcNow,
            Entries = new List<ListfileSnapshot.Entry>()
        };

        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1) Pull from MPQ embedded (listfile) where available (0.6.0+)
        var mpqs = ArchiveLocator.LocateMpqs(clientPath);
        foreach (var mpqPath in mpqs)
        {
            try
            {
                using var mpq = new MpqArchive(mpqPath, FileAccess.Read);
                const string LISTFILE = "(listfile)";
                if (mpq.HasFile(LISTFILE))
                {
                    using var fs = mpq.OpenFile(LISTFILE);
                    if (fs != null && fs.CanRead && fs.Length > 0)
                    {
                        using var sr = new StreamReader(fs);
                        while (!sr.EndOfStream)
                        {
                            var line = sr.ReadLine();
                            if (string.IsNullOrWhiteSpace(line)) continue;
                            var norm = NormalizeAssetPath(line);
                            paths.Add(norm);
                        }
                    }
                }
            }
            catch { /* best-effort */ }
        }

        // 2) Scan loose files under client root (prefer relative under Data/ if present)
        foreach (var file in Directory.EnumerateFiles(clientPath, "*", SearchOption.AllDirectories))
        {
            try
            {
                var rel = GetRelativeUnderDataOrRoot(clientPath, file);
                var norm = NormalizeAssetPath(rel);
                if (!string.IsNullOrEmpty(norm)) paths.Add(norm);
                if (!string.IsNullOrEmpty(norm) && norm.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
                {
                    var noMpq = norm.Substring(0, norm.Length - 4);
                    var ext = Path.GetExtension(noMpq);
                    if (!string.IsNullOrEmpty(ext))
                    {
                        switch (ext.ToLowerInvariant())
                        {
                            case ".wdt":
                            case ".wdl":
                            case ".wmo":
                            case ".mdx":
                            case ".m2":
                            case ".blp":
                            case ".adt":
                                paths.Add(noMpq);
                                if (ext.Equals(".m2", StringComparison.OrdinalIgnoreCase))
                                {
                                    var asMdx = noMpq.Substring(0, noMpq.Length - 3) + "mdx";
                                    paths.Add(asMdx);
                                }
                                break;
                        }
                    }
                }
            }
            catch { }
        }

        // Optional FDID enrichment from a community listfile
        Dictionary<string, uint?>? comm = null;
        Dictionary<string, List<(string Path, uint? Id)>>? byFile = null;
        if (!string.IsNullOrWhiteSpace(communityPath) && File.Exists(communityPath))
        {
            comm = LoadGenericListfile(communityPath!);
            byFile = new Dictionary<string, List<(string, uint?)>>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in comm)
            {
                var fname = Path.GetFileName(kv.Key);
                if (!byFile.TryGetValue(fname, out var list)) { list = new List<(string, uint?)>(); byFile[fname] = list; }
                list.Add((kv.Key, kv.Value));
                // m2<->mdx alias on filename
                if (fname.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                {
                    var aliasName = fname.Substring(0, fname.Length - 3) + "mdx";
                    if (!byFile.TryGetValue(aliasName, out var list2)) { list2 = new List<(string, uint?)>(); byFile[aliasName] = list2; }
                    list2.Add((kv.Key, kv.Value));
                }
                else if (fname.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
                {
                    var aliasName = fname.Substring(0, fname.Length - 4) + "m2";
                    if (!byFile.TryGetValue(aliasName, out var list3)) { list3 = new List<(string, uint?)>(); byFile[aliasName] = list3; }
                    list3.Add((kv.Key, kv.Value));
                }
            }
        }

        var missing = new List<string>();
        var ambiguous = new List<(string Path, List<string> Candidates)>();

        foreach (var p in paths.OrderBy(p => p, StringComparer.OrdinalIgnoreCase))
        {
            uint? fdid = null;
            if (comm != null)
            {
                // exact path
                if (!comm.TryGetValue(p, out fdid))
                {
                    // mdx<->m2 alias on path
                    string pathAlias = p;
                    if (p.EndsWith(".m2", StringComparison.OrdinalIgnoreCase)) pathAlias = p.Substring(0, p.Length - 3) + "mdx";
                    else if (p.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)) pathAlias = p.Substring(0, p.Length - 4) + "m2";
                    if (!string.Equals(pathAlias, p, StringComparison.OrdinalIgnoreCase)) comm.TryGetValue(pathAlias, out fdid);

                    // fallback: filename-only heuristic
                    if (!fdid.HasValue && byFile != null)
                    {
                        var fname = Path.GetFileName(p);
                        if (byFile.TryGetValue(fname, out var list) && list.Count == 1)
                        {
                            fdid = list[0].Id;
                        }
                        else if (byFile.TryGetValue(fname, out var many) && many.Count > 1)
                        {
                            ambiguous.Add((p, many.Select(t => t.Path).OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToList()));
                        }
                    }
                }
            }

            snapshot.Entries.Add(new ListfileSnapshot.Entry { Path = p, FileDataId = fdid });
            if (comm != null && !fdid.HasValue)
            {
                // log truly unresolved ones (exclude those recorded as ambiguous)
                if (!ambiguous.Any(a => a.Path.Equals(p, StringComparison.OrdinalIgnoreCase))) missing.Add(p);
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? ".");
        // Write diagnostics when community listfile provided
        if (comm != null)
        {
            var outDir = Path.GetDirectoryName(outPath) ?? ".";
            if (missing.Count > 0)
            {
                File.WriteAllText(Path.Combine(outDir, "snapshot_missing_fdid.csv"), "path\n" + string.Join("\n", missing));
                Console.WriteLine($"[info] missing fdid: {missing.Count}");
            }
            if (ambiguous.Count > 0)
            {
                using var sw = new StreamWriter(Path.Combine(outDir, "snapshot_ambiguous_fdid.csv"));
                sw.WriteLine("path,candidates");
                foreach (var a in ambiguous)
                {
                    var cand = string.Join('|', a.Candidates);
                    sw.WriteLine($"{a.Path},{EscapeCsvLocal(cand)}");
                }
                Console.WriteLine($"[info] ambiguous fdid: {ambiguous.Count}");
            }
        }
        var json = JsonSerializer.Serialize(snapshot, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(outPath, json);
        Console.WriteLine($"[ok] Snapshot written: {outPath} (paths={paths.Count})");
        if (!string.IsNullOrWhiteSpace(csvOut))
        {
            var sb = new StringBuilder();
            foreach (var e in snapshot.Entries.OrderBy(e => e.Path, StringComparer.OrdinalIgnoreCase))
            {
                if (e.FileDataId.HasValue)
                    sb.AppendLine($"{e.FileDataId.Value};{e.Path}");
                else
                {
                    if (csvMissingToken.Equals("none", StringComparison.OrdinalIgnoreCase))
                        sb.AppendLine(e.Path);
                    else
                        sb.AppendLine($"{csvMissingToken};{e.Path}");
                }
            }
            File.WriteAllText(csvOut!, sb.ToString());
            Console.WriteLine($"[ok] CSV written: {csvOut} (rows={snapshot.Entries.Count})");
        }
        return 0;

        static string GetRelativeUnderDataOrRoot(string root, string fullPath)
        {
            var rel = Path.GetRelativePath(root, fullPath);
            var parts = rel.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            for (int i = 0; i < parts.Length; i++)
            {
                if (string.Equals(parts[i], "data", StringComparison.OrdinalIgnoreCase))
                {
                    var sub = string.Join(Path.DirectorySeparatorChar, parts.Skip(i + 1));
                    return sub;
                }
            }
            return rel;
        }
        static string EscapeCsvLocal(string s)
        {
            return (s.Contains(',') || s.Contains('"')) ? '"' + s.Replace("\"", "\"\"") + '"' : s;
        }
    }

    // Parse listfile in plain/CSV/JSON (our snapshot format) into path->fdid map
    private static Dictionary<string, uint?> LoadGenericListfile(string path)
    {
        if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            var snap = JsonSerializer.Deserialize<ListfileSnapshot>(File.ReadAllText(path));
            var dict = new Dictionary<string, uint?>(StringComparer.OrdinalIgnoreCase);
            if (snap?.Entries != null)
            {
                foreach (var e in snap.Entries)
                {
                    if (string.IsNullOrWhiteSpace(e.Path)) continue;
                    var norm = NormalizeAssetPath(e.Path);
                    uint? id = e.FileDataId.HasValue ? e.FileDataId.Value : (uint?)null;
                    dict[norm] = id;
                }
            }
            return dict;
        }
        else
        {
            var dict = new Dictionary<string, uint?>(StringComparer.OrdinalIgnoreCase);
            foreach (var raw in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(raw)) continue;
                var line = raw.Trim();
                if (line.StartsWith("#") || line.StartsWith("//")) continue;
                string[] parts = line.Split(';');
                if (parts.Length < 2) parts = line.Split(',');
                if (parts.Length < 2) parts = line.Split('\t');
                if (parts.Length >= 2 && uint.TryParse(parts[0].Trim(), out var fdid))
                {
                    var pathPart = NormalizeAssetPath(parts[1]);
                    if (pathPart.Length == 0) continue;
                    if (!dict.ContainsKey(pathPart)) dict[pathPart] = fdid;
                }
                else
                {
                    var onlyPath = NormalizeAssetPath(line);
                    if (onlyPath.Length == 0) continue;
                    if (!dict.ContainsKey(onlyPath)) dict[onlyPath] = null;
                }
            }
            return dict;
        }
    }

    private static int RunDiffListfiles(Dictionary<string, string> opts)
    {
        Require(opts, "a");
        Require(opts, "b");
        Require(opts, "out");

        var fileA = opts["a"]; var fileB = opts["b"]; var outDir = opts["out"]; Directory.CreateDirectory(outDir);

        var a = LoadGeneric(fileA);
        var b = LoadGeneric(fileB);

        var pathsA = new HashSet<string>(a.Keys, StringComparer.OrdinalIgnoreCase);
        var pathsB = new HashSet<string>(b.Keys, StringComparer.OrdinalIgnoreCase);

        // added (in B not in A)
        var added = pathsB.Except(pathsA, StringComparer.OrdinalIgnoreCase).OrderBy(p => p).ToList();
        // removed (in A not in B)
        var removed = pathsA.Except(pathsB, StringComparer.OrdinalIgnoreCase).OrderBy(p => p).ToList();
        // changed FDIDs (same path present in both and both have fdid known but differ)
        var changed = new List<(string Path, uint? OldId, uint? NewId)>();
        foreach (var path in pathsA.Intersect(pathsB, StringComparer.OrdinalIgnoreCase))
        {
            var oldId = a[path]; var newId = b[path];
            if (oldId.HasValue && newId.HasValue && oldId.Value != newId.Value)
            {
                changed.Add((path, oldId, newId));
            }
        }

        // write CSVs
        File.WriteAllText(Path.Combine(outDir, "added_paths.csv"), "path\n" + string.Join("\n", added));
        File.WriteAllText(Path.Combine(outDir, "removed_paths.csv"), "path\n" + string.Join("\n", removed));
        using (var sw = new StreamWriter(Path.Combine(outDir, "changed_fdid.csv")))
        {
            sw.WriteLine("path,old_fdid,new_fdid");
            foreach (var c in changed)
            {
                sw.WriteLine($"{EscapeCsv(c.Path)},{c.OldId},{c.NewId}");
            }
        }

        Console.WriteLine($"[ok] Diff written to: {outDir} (added={added.Count}, removed={removed.Count}, changed={changed.Count})");
        return 0;

        static Dictionary<string, uint?> LoadGeneric(string path)
        {
            if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                var snap = JsonSerializer.Deserialize<ListfileSnapshot>(File.ReadAllText(path));
                var dict = new Dictionary<string, uint?>(StringComparer.OrdinalIgnoreCase);
                if (snap?.Entries != null)
                {
                    foreach (var e in snap.Entries)
                    {
                        if (string.IsNullOrWhiteSpace(e.Path)) continue;
                        var norm = NormalizeAssetPath(e.Path);
                        uint? id = e.FileDataId.HasValue ? e.FileDataId.Value : (uint?)null;
                        dict[norm] = id;
                    }
                }
                return dict;
            }
            else
            {
                // CSV or plain listfile format
                var dict = new Dictionary<string, uint?>(StringComparer.OrdinalIgnoreCase);
                foreach (var raw in File.ReadLines(path))
                {
                    if (string.IsNullOrWhiteSpace(raw)) continue;
                    var line = raw.Trim();
                    if (line.StartsWith("#") || line.StartsWith("//")) continue;
                    string[] parts = line.Split(';');
                    if (parts.Length < 2) parts = line.Split(',');
                    if (parts.Length < 2) parts = line.Split('\t');
                    if (parts.Length >= 2 && uint.TryParse(parts[0].Trim(), out var fdid))
                    {
                        var pathPart = NormalizeAssetPath(parts[1]);
                        if (pathPart.Length == 0) continue;
                        dict[pathPart] = fdid;
                    }
                    else
                    {
                        var onlyPath = NormalizeAssetPath(line);
                        if (onlyPath.Length == 0) continue;
                        if (!dict.ContainsKey(onlyPath)) dict[onlyPath] = null;
                    }
                }
                return dict;
            }
        }

        static string EscapeCsv(string s)
        {
            return (s.Contains(',') || s.Contains('"')) ? '"' + s.Replace("\"", "\"\"") + '"' : s;
        }
    }

    private static int RunRegenLayers(Dictionary<string, string> opts)
    {
        var inDir = opts.GetValueOrDefault("in", opts.GetValueOrDefault("dir", opts.GetValueOrDefault("out", "analysis_output")));
        if (string.IsNullOrWhiteSpace(inDir) || !Directory.Exists(inDir))
        {
            Console.Error.WriteLine($"[error] input directory not found: {inDir}");
            return 1;
        }

        var mapsCsv = opts.GetValueOrDefault("maps");
        HashSet<string>? only = null;
        if (!string.IsNullOrWhiteSpace(mapsCsv))
        {
            only = new HashSet<string>(mapsCsv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries), StringComparer.OrdinalIgnoreCase);
        }

        var force = opts.ContainsKey("force");
        var placements = Directory.EnumerateFiles(inDir, "*_placements.csv", SearchOption.AllDirectories).ToList();
        if (placements.Count == 0)
        {
            Console.WriteLine("[info] no placements found to regenerate layers");
            return 0;
        }

        int ok = 0, skipped = 0, failed = 0;
        foreach (var file in placements)
        {
            try
            {
                var name = Path.GetFileName(file);
                var map = name.EndsWith("_placements.csv", StringComparison.OrdinalIgnoreCase)
                    ? name.Substring(0, name.Length - "_placements.csv".Length)
                    : Path.GetFileName(Path.GetDirectoryName(file) ?? string.Empty);
                if (string.IsNullOrWhiteSpace(map)) { skipped++; continue; }
                if (only != null && !only.Contains(map)) { skipped++; continue; }

                var parent = Path.GetDirectoryName(file)!;
                var mapDir = Path.GetFileName(parent).Equals(map, StringComparison.OrdinalIgnoreCase) ? parent : Path.Combine(parent, map);
                Directory.CreateDirectory(mapDir);
                var layersCsv = Path.Combine(mapDir, "tile_layers.csv");
                if (!force && File.Exists(layersCsv) && new FileInfo(layersCsv).Length > 0)
                {
                    Console.WriteLine($"[skip] {map}: tile_layers.csv already exists (use --force to overwrite)");
                    skipped++;
                    continue;
                }

                Console.WriteLine($"[regen] {map}: analyzing {Path.GetFileName(file)} -> {mapDir}");
                var analyzer = new UniqueIdAnalyzer(gapThreshold: 100);
                var res = analyzer.AnalyzeFromPlacementsCsv(file, map, mapDir);
                if (!res.Success)
                {
                    Console.WriteLine($"[warn] {map}: analysis failed: {res.ErrorMessage}");
                    failed++;
                }
                else
                {
                    Console.WriteLine($"[ok] {map}: layers written");
                    ok++;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[error] {file}: {ex.Message}");
                failed++;
            }
        }

        Console.WriteLine($"[done] regen-layers ok={ok} skipped={skipped} failed={failed}");
        return failed > 0 ? 1 : 0;
    }

    private static int RunComputeHeatmapStats(Dictionary<string, string> opts)
    {
        var buildRoot = GetOption(opts, "build-root");
        if (string.IsNullOrWhiteSpace(buildRoot))
        {
            var cache = GetOption(opts, "cache");
            var build = GetOption(opts, "build");
            if (!string.IsNullOrWhiteSpace(cache) && !string.IsNullOrWhiteSpace(build))
            {
                buildRoot = Path.Combine(cache!, build!);
            }
        }
        if (string.IsNullOrWhiteSpace(buildRoot) || !Directory.Exists(buildRoot!))
        {
            Console.Error.WriteLine("[error] Provide --build-root <dir> or both --cache <dir> and --build <label>");
            return 2;
        }
        var force = opts.ContainsKey("force");
        Console.WriteLine($"[stats] Scanning build root: {buildRoot}");
        var res = StatsService.GenerateHeatmapStats(buildRoot!, force);
        Console.WriteLine(res.Skipped
            ? $"[skip] Up-to-date: {res.OutputPath} (min={res.MinUnique}, max={res.MaxUnique}, maps={res.PerMapCount})"
            : $"[ok] heatmap_stats.json written: {res.OutputPath} (min={res.MinUnique}, max={res.MaxUnique}, maps={res.PerMapCount})");
        return 0;
    }

    private static string NormalizeAssetPath(string p)
    {
        if (string.IsNullOrWhiteSpace(p)) return string.Empty;
        var t = p.Trim().Replace('\\', '/');
        return t;
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

        // Step 1.5: Process minimap tiles (disabled by default)
        string? minimapDir = null;
        if (opts.ContainsKey("export-minimaps"))
        {
            Console.WriteLine("\n=== Processing minimap tiles ===");
            var minimapHandler = new MinimapHandler();
            var minimapResult = minimapHandler.ProcessMinimaps(mapDir, mapName, outDir);
            if (minimapResult.Success && minimapResult.TilesCopied > 0)
            {
                Console.WriteLine($"[ok] Copied {minimapResult.TilesCopied} minimap tiles");
                Console.WriteLine($"[ok] Minimap directory: {minimapResult.MinimapDir}");
                minimapDir = minimapResult.MinimapDir;
            }
            else if (minimapResult.Success)
            {
                Console.WriteLine($"[info] No minimap PNG files found");
                minimapDir = minimapResult.MinimapDir;
            }
            else
            {
                Console.WriteLine($"[warn] Minimap processing failed: {minimapResult.ErrorMessage}");
            }
        }
        else
        {
            Console.WriteLine("\n=== Skipping minimap tiles (enable with --export-minimaps) ===");
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

        // Placements-only: short-circuit after UniqueID analysis (layers written)
        if (opts.ContainsKey("placements-only"))
        {
            Console.WriteLine("[info] placements-only: skipping clusters, terrain, mesh, and viewer");
            return 0;
        }

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
        var viewerRoot = viewerAdapter.GenerateViewer(placementsCsvPath, mapName, outDir, minimapDir);

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
                // On CASC (retail) builds, minimaps are typically FDID-only; skip to avoid long scans/decode
                if (src is WoWRollback.Core.Services.Archive.CascArchiveSource)
                {
                    Console.WriteLine("[info] CASC build without md5translate: skipping minimap extraction (FDID-only). Continuing.");
                    return null;
                }
            }

            var resolver = new MinimapFileResolver(src, index);
            var minimapOutDir = Path.Combine(outputDir, "minimaps");
            Directory.CreateDirectory(minimapOutDir);

            // Scan for all tiles (0-63 grid)
            int extracted = 0;
            int failed = 0;
            int attempts = 0;
            for (int x = 0; x < 64; x++)
            {
                for (int y = 0; y < 64; y++)
                {
                    attempts++;
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

    private static string? ExtractVersionFromBuildInfo(string clientRoot)
    {
        try
        {
            string? dir = clientRoot;
            for (int i = 0; i < 4 && !string.IsNullOrEmpty(dir); i++, dir = Path.GetDirectoryName(dir))
            {
                var path = Path.Combine(dir!, ".build.info");
                if (!File.Exists(path)) continue;
                var lines = File.ReadAllLines(path);
                if (lines.Length < 2) return null;
                var header = lines[0];
                var delim = header.Contains('|') ? '|' : (header.Contains(';') ? ';' : ' ');
                var cols = header.Split(delim, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                int idxVersion = Array.FindIndex(cols, c => c.StartsWith("Version", StringComparison.OrdinalIgnoreCase));
                if (idxVersion < 0) return null;
                for (int j = 1; j < lines.Length; j++)
                {
                    var parts = lines[j].Split(delim, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                    if (idxVersion >= parts.Length) continue;
                    var ver = parts[idxVersion];
                    if (!string.IsNullOrWhiteSpace(ver)) return ver;
                }
            }
        }
        catch { }
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
        Console.WriteLine("  compute-heatmap-stats  --build-root <dir> [--force]");
        Console.WriteLine("    Compute global heatmap stats (min/max per build, per-map ranges) from tile_layers.csv under build root.");
        Console.WriteLine("    Alternatively: provide --cache <dir> and --build <label> (e.g., 0.5.3) to compose the build root.");
        Console.WriteLine();
        Console.WriteLine("  discover-maps  --client-path <dir> [--version <ver>] [--dbd-dir <path>] [--out <csv>]");
        Console.WriteLine("    Discover all maps from Map.dbc and analyze their WDT files");
        Console.WriteLine("    Shows terrain vs WMO-only maps, tile counts, and hybrid maps");
        Console.WriteLine("    Version auto-detected from path or use --version (e.g., 0.6.0.3592)");
        Console.WriteLine();
        Console.WriteLine("Common analyze flags:");
        Console.WriteLine("  --maps <m1,m2>        Limit to specific maps by name/folder");
        Console.WriteLine("  --map-ids <ids>       Limit to specific Map.dbc IDs");
        Console.WriteLine("  --max-maps <N>        Process at most N maps (batch mode)");
        Console.WriteLine("  --max-tiles <N>       Limit mesh export tiles (when --export-mesh)");
        Console.WriteLine("  --export-minimaps     Enable minimap extraction (disabled by default)");
        Console.WriteLine("  --export-mesh         Enable GLB/OBJ mesh export (disabled by default)");
        Console.WriteLine();
        Console.WriteLine("  analyze-map-adts-mpq  --client-path <dir> (--map <name> | --all-maps) [--out <dir>] [--version <ver>]");
        Console.WriteLine("                        [--serve] [--port <port>]");
        Console.WriteLine("    Analyze ADT files directly from MPQs (patched view), detect layers/clusters, and generate viewer");
        Console.WriteLine("    Use --all-maps to discover and analyze all maps from Map.dbc");
        Console.WriteLine("    Version auto-detected from path or use --version (e.g., 0.6.0.3592)");
        Console.WriteLine();
        Console.WriteLine("  analyze-map-adts-casc  --client-path <dir> (--map <name> | --all-maps) [--product wow|wowt] [--locale enUS]");
        Console.WriteLine("                         [--listfile <path>] [--out <dir>] [--version <ver>] [--serve] [--port <port>]");
        Console.WriteLine("    Analyze ADTs from CASC depots. Requires a listfile to enumerate named paths reliably.");
        Console.WriteLine();
        Console.WriteLine("  pack-monolithic-alpha-wdt  --lk-wdt <file> [--lk-map-dir <dir>] --out <wdt> [--target-listfile <335.csv>] [--strict-target-assets true|false]");
        Console.WriteLine("                             [--extract-assets] [--asset-scope textures|models|textures+models] [--assets-out <dir>]");
        Console.WriteLine("                             [--export-lk-adts-after-pack] [--lk-out <dir>] [--lk-client-path <dir>] [--lk-dbc-dir <dir>] [--area-remap-json <path>]");
        Console.WriteLine("    Pack alpha WDT (monolithic) from LK inputs. Optionally gate MDNM/MONM assets to 3.3.5 listfile.");
        Console.WriteLine("    When --export-lk-adts-after-pack is set, runs a post-pack diagnostic phase to write LK ADTs from the new WDT.");
        Console.WriteLine();
        Console.WriteLine("  snapshot-listfile  --client-path <dir> --alias <name> --out <json>");
        Console.WriteLine("    Build a JSON listfile snapshot from MPQs (reads (listfile) entries) and loose files under Data.");
        Console.WriteLine();
        Console.WriteLine("  diff-listfiles  --a <listfileA> --b <listfileB> --out <dir>");
        Console.WriteLine("    Produce CSVs of added/removed paths and FDID changes between two listfiles.");
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
        Console.WriteLine("  alpha-roundtrip-verify  --input <alpha.wdt> --rt-out <dir> [--lk-out <dir>] [--map <name>] [--threads <N>] [--accept-padding] [--accept-name-normalization]");
        Console.WriteLine("    Run AlphaWDT → LK ADTs → AlphaWDT and write chunk-aware diff CSVs (skeleton)");
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

    private static int RunAlphaRoundtripVerify(Dictionary<string, string> opts)
    {
        var input = GetOption(opts, "input");
        var rtOut = GetOption(opts, "rt-out");
        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(rtOut))
        {
            Console.Error.WriteLine("[error] --input <alpha.wdt> and --rt-out <dir> are required");
            return 2;
        }
        var inputFull = Path.GetFullPath(input!);
        Directory.CreateDirectory(rtOut!);
        var session = Path.Combine(rtOut!, $"session_{DateTime.UtcNow:yyyyMMdd_HHmmss}");
        Directory.CreateDirectory(session);

        var map = GetOption(opts, "map");
        if (string.IsNullOrWhiteSpace(map))
        {
            var fnNoExt = Path.GetFileNameWithoutExtension(inputFull);
            map = string.IsNullOrWhiteSpace(fnNoExt) ? "unknown" : fnNoExt;
        }

        bool acceptPadding = opts.ContainsKey("accept-padding");
        bool acceptNameNorm = opts.ContainsKey("accept-name-normalization");
        int threads = TryParseInt(opts, "threads") ?? (Environment.ProcessorCount / 2);
        if (threads < 1) threads = 1;
        if (threads > Environment.ProcessorCount) threads = Environment.ProcessorCount;

        Console.WriteLine("[rt] alpha-roundtrip-verify");
        Console.WriteLine($"[rt] input: {inputFull}");
        Console.WriteLine($"[rt] session: {session}");
        Console.WriteLine($"[rt] map: {map}");
        Console.WriteLine($"[rt] threads={threads}");
        Console.WriteLine($"[rt] accept: padding={acceptPadding.ToString().ToLowerInvariant()}, name_norm={acceptNameNorm.ToString().ToLowerInvariant()}");

        var lkOutRoot = GetOption(opts, "lk-out");
        if (string.IsNullOrWhiteSpace(lkOutRoot)) lkOutRoot = Path.Combine(session, "lk_out");
        Directory.CreateDirectory(lkOutRoot!);

        var roll = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["input"] = inputFull,
            ["out"] = session,
            ["max-uniqueid"] = int.MaxValue.ToString(CultureInfo.InvariantCulture),
            ["export-lk-adts"] = "true",
            ["force"] = "true",
            ["lk-out"] = lkOutRoot!,
            ["threads"] = threads.ToString(CultureInfo.InvariantCulture)
        };
        var lkClient = GetOption(opts, "lk-client-path");
        if (!string.IsNullOrWhiteSpace(lkClient)) roll["lk-client-path"] = lkClient!;
        var lkDbcDir = GetOption(opts, "lk-dbc-dir");
        if (!string.IsNullOrWhiteSpace(lkDbcDir)) roll["lk-dbc-dir"] = lkDbcDir!;
        var areaRemapJson = GetOption(opts, "area-remap-json");
        if (!string.IsNullOrWhiteSpace(areaRemapJson)) roll["area-remap-json"] = areaRemapJson!;

        Console.WriteLine("[rt] Phase A: Alpha → LK ADTs");
        var codeA = RunRollback(roll);
        if (codeA != 0)
        {
            Console.Error.WriteLine($"[rt][error] Phase A failed (exit={codeA})");
            return codeA;
        }

        string lkMapDir;
        string lkWdtPath;
        var nestedDir = Path.Combine(lkOutRoot!, "World", "Maps", map!);
        var nestedWdt = Path.Combine(nestedDir, map! + ".wdt");
        var flatWdt = Path.Combine(lkOutRoot!, map! + ".wdt");
        if (File.Exists(nestedWdt))
        {
            lkMapDir = nestedDir;
            lkWdtPath = nestedWdt;
        }
        else if (File.Exists(flatWdt))
        {
            lkMapDir = lkOutRoot!;
            lkWdtPath = flatWdt;
        }
        else
        {
            var found = Directory.EnumerateFiles(lkOutRoot!, "*.wdt", SearchOption.AllDirectories)
                                 .FirstOrDefault(f => string.Equals(Path.GetFileNameWithoutExtension(f), map!, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrEmpty(found))
            {
                lkWdtPath = found!;
                lkMapDir = Path.GetDirectoryName(found!) ?? lkOutRoot!;
            }
            else
            {
                Console.Error.WriteLine($"[rt][error] Expected LK WDT not found under: {Path.Combine(lkOutRoot!, "(flat or World/Maps)")}");
                return 1;
            }
        }

        Console.WriteLine("[rt] Phase B: LK → Alpha (pack monolithic)");
        var repackedAlpha = Path.Combine(session, map! + ".roundtrip.wdt");
        var pack = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["lk-wdt"] = lkWdtPath,
            ["lk-map-dir"] = lkMapDir,
            ["out"] = repackedAlpha,
            ["threads"] = threads.ToString(CultureInfo.InvariantCulture)
        };
        var swPack = Stopwatch.StartNew();
        var codeB = RunPackMonolithicAlphaWdt(pack);
        swPack.Stop();
        if (codeB != 0)
        {
            Console.Error.WriteLine($"[rt][error] Phase B failed (exit={codeB})");
            return codeB;
        }
        try { File.WriteAllText(Path.Combine(session, "pack_timing.csv"), "metric,ms\npack_total," + swPack.Elapsed.TotalMilliseconds.ToString(CultureInfo.InvariantCulture) + "\n"); } catch { }

        var repackedBytes = File.ReadAllBytes(repackedAlpha);
        if (!TryIsAlphaV18Wdt(repackedBytes, out var reason))
        {
            Console.Error.WriteLine($"[rt][error] Validation failed: {reason}");
            return 1;
        }

        static int Next(int start, int size) => start + 8 + size + ((size & 1) == 1 ? 1 : 0);
        static int[] ReadMainOffsets(byte[] buf, out int mainSize)
        {
            mainSize = 0;
            if (buf.Length < 24) return Array.Empty<int>();
            var c0 = Encoding.ASCII.GetString(buf, 0, 4);
            if (!string.Equals(c0, "REVM", StringComparison.Ordinal)) return Array.Empty<int>();
            int sz0 = BitConverter.ToInt32(buf, 4);
            int off1 = Next(0, sz0);
            if (off1 + 8 > buf.Length) return Array.Empty<int>();
            var c1 = Encoding.ASCII.GetString(buf, off1, 4);
            if (!string.Equals(c1, "DHPM", StringComparison.Ordinal)) return Array.Empty<int>();
            int sz1 = BitConverter.ToInt32(buf, off1 + 4);
            int off2 = Next(off1, sz1);
            if (off2 + 8 > buf.Length) return Array.Empty<int>();
            var c2 = Encoding.ASCII.GetString(buf, off2, 4);
            if (!string.Equals(c2, "NIAM", StringComparison.Ordinal)) return Array.Empty<int>();
            int sz2 = BitConverter.ToInt32(buf, off2 + 4);
            mainSize = sz2;
            int dataStart = off2 + 8;
            int n = Math.Min(sz2 / 4, 64 * 64);
            var arr = new int[64 * 64];
            for (int i = 0; i < n; i++)
            {
                int pos = dataStart + i * 4;
                if (pos + 4 <= buf.Length) arr[i] = BitConverter.ToInt32(buf, pos);
            }
            return arr;
        }

        var origBytes = File.ReadAllBytes(inputFull);
        int mainSzA, mainSzB;
        var mainA = ReadMainOffsets(origBytes, out mainSzA);
        var mainB = ReadMainOffsets(repackedBytes, out mainSzB);

        var wdtCsv = new StringBuilder();
        wdtCsv.AppendLine("field,a,b,equal");
        long sizeA = new FileInfo(inputFull).Length;
        long sizeB = new FileInfo(repackedAlpha).Length;
        wdtCsv.AppendLine($"file_size,{sizeA},{sizeB},{(sizeA==sizeB).ToString().ToLowerInvariant()}");
        wdtCsv.AppendLine($"main_size,{mainSzA},{mainSzB},{(mainSzA==mainSzB).ToString().ToLowerInvariant()}");
        var wdtCsvPath = Path.Combine(session, "wdt_diff.csv");
        File.WriteAllText(wdtCsvPath, wdtCsv.ToString());

        var adtCsv = new StringBuilder();
        adtCsv.AppendLine("tile_index,off_a,off_b,equal");
        int tiles = Math.Max(mainA.Length, mainB.Length);
        for (int i = 0; i < tiles; i++)
        {
            int a = i < mainA.Length ? mainA[i] : 0;
            int b = i < mainB.Length ? mainB[i] : 0;
            bool eq = a == b;
            adtCsv.AppendLine($"{i},{a},{b},{eq.ToString().ToLowerInvariant()}");
        }
        var adtCsvPath = Path.Combine(session, "adt_diff.csv");
        File.WriteAllText(adtCsvPath, adtCsv.ToString());

        var tilesCsv = new StringBuilder();
        tilesCsv.AppendLine("tile_x,tile_y,index,offset");
        for (int i = 0; i < mainB.Length; i++)
        {
            int off = mainB[i];
            if (off > 0)
            {
                int ty = i / 64;
                int tx = i % 64;
                tilesCsv.AppendLine($"{tx},{ty},{i},{off}");
            }
        }
        File.WriteAllText(Path.Combine(session, "tiles_written.csv"), tilesCsv.ToString());

        var summaryPath = Path.Combine(session, "rt_summary.txt");
        var sb = new StringBuilder();
        sb.AppendLine($"input_alpha_wdt={inputFull}");
        sb.AppendLine($"lk_out_root={lkOutRoot}");
        sb.AppendLine($"lk_map_dir={lkMapDir}");
        sb.AppendLine($"lk_wdt={lkWdtPath}");
        sb.AppendLine($"repacked_alpha_wdt={repackedAlpha}");
        File.WriteAllText(summaryPath, sb.ToString());

        Console.WriteLine("[rt] roundtrip complete (validation passed)");
        Console.WriteLine($"[rt] summary: {summaryPath}");
        return 0;
    }

    private static int RunDumpAlphaMcnk(Dictionary<string, string> opts)
    {
        Require(opts, "wdt-file");
        var wdtFile = opts["wdt-file"];

        var tx = TryParseInt(opts, "tile-x") ?? -1;
        var ty = TryParseInt(opts, "tile-y") ?? -1;
        var ci = TryParseInt(opts, "chunk") ?? 0;
        if (tx < 0 || tx >= 64 || ty < 0 || ty >= 64)
        {
            Console.Error.WriteLine("[dump] tile-x and tile-y must be in range [0,63]");
            return 2;
        }
        if (ci < 0 || ci >= 256)
        {
            Console.Error.WriteLine("[dump] chunk must be in range [0,255]");
            return 2;
        }

        var outPathOpt = GetOption(opts, "out");
        string outCsv;
        if (!string.IsNullOrWhiteSpace(outPathOpt))
        {
            outCsv = outPathOpt!;
        }
        else
        {
            var baseDir = Path.GetDirectoryName(Path.GetFullPath(wdtFile)) ?? Directory.GetCurrentDirectory();
            var baseName = Path.GetFileNameWithoutExtension(wdtFile);
            outCsv = Path.Combine(baseDir, $"{baseName}_t{ty:D2}_{tx:D2}_c{ci:D3}_mcnk.csv");
        }

        Console.WriteLine("[dump] Alpha WDT: " + wdtFile);
        Console.WriteLine($"[dump] tile=({tx},{ty}) chunkIndex={ci}");

        var buf = File.ReadAllBytes(wdtFile);

        // Local helper: read MAIN-style offsets table (64x64) from an Alpha WDT buffer
        static int[] ReadMainOffsetsForDump(byte[] b, out int mainSize)
        {
            mainSize = 0;
            if (b.Length < 24) return Array.Empty<int>();
            string c0 = Encoding.ASCII.GetString(b, 0, 4);
            if (!string.Equals(c0, "REVM", StringComparison.Ordinal)) return Array.Empty<int>();
            int sz0 = BitConverter.ToInt32(b, 4);
            static int Next(int start, int size) => start + 8 + size + ((size & 1) == 1 ? 1 : 0);
            int off1 = Next(0, sz0);
            if (off1 + 8 > b.Length) return Array.Empty<int>();
            string c1 = Encoding.ASCII.GetString(b, off1, 4);
            if (!string.Equals(c1, "DHPM", StringComparison.Ordinal)) return Array.Empty<int>();
            int sz1 = BitConverter.ToInt32(b, off1 + 4);
            int off2 = Next(off1, sz1);
            if (off2 + 8 > b.Length) return Array.Empty<int>();
            string c2 = Encoding.ASCII.GetString(b, off2, 4);
            if (!string.Equals(c2, "NIAM", StringComparison.Ordinal)) return Array.Empty<int>();
            int sz2 = BitConverter.ToInt32(b, off2 + 4);
            mainSize = sz2;
            int dataStart = off2 + 8;
            int n = Math.Min(sz2 / 4, 64 * 64);
            var arr = new int[64 * 64];
            for (int i = 0; i < n; i++)
            {
                int pos = dataStart + i * 4;
                if (pos + 4 <= b.Length) arr[i] = BitConverter.ToInt32(b, pos);
            }
            return arr;
        }

        int mainSz;
        var main = ReadMainOffsetsForDump(buf, out mainSz);
        if (main.Length == 0)
        {
            Console.Error.WriteLine("[dump][error] MAIN offsets not found or invalid.");
            return 1;
        }

        int tileIndex = ty * 64 + tx;
        if (tileIndex < 0 || tileIndex >= main.Length)
        {
            Console.Error.WriteLine($"[dump][error] tile index {tileIndex} out of MAIN bounds.");
            return 1;
        }

        int adtOff = main[tileIndex];
        if (adtOff <= 0 || adtOff + 8 > buf.Length)
        {
            Console.Error.WriteLine($"[dump][error] tile ({tx},{ty}) has no ADT (offset={adtOff}).");
            return 1;
        }

        int chunkX = ci % 16;
        int chunkY = ci / 16;

        Console.WriteLine($"[dump] MAIN[{tileIndex}] = 0x{adtOff:X} (tile ADT offset)");

        int foundAt = -1;
        int idxX = -1;
        int idxY = -1;

        // Scan chunks within the embedded ADT to find the requested MCNK
        for (int p = adtOff; p + 8 <= buf.Length;)
        {
            string fcc = Encoding.ASCII.GetString(buf, p, 4);
            int size = BitConverter.ToInt32(buf, p + 4);
            int dataStart = p + 8;
            if (size < 0 || dataStart + size > buf.Length) break;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= p) break;

            if (fcc == "KNCM")
            {
                int headerStart = dataStart;
                if (headerStart + 0x80 > buf.Length) break;

                idxX = BitConverter.ToInt32(buf, headerStart + 0x04);
                idxY = BitConverter.ToInt32(buf, headerStart + 0x08);

                if (idxX == chunkX && idxY == chunkY)
                {
                    foundAt = p;
                    break;
                }
            }

            p = next;
        }

        if (foundAt < 0)
        {
            Console.Error.WriteLine($"[dump][error] MCNK for chunk ({chunkX},{chunkY}) not found in tile ({tx},{ty}).");
            return 1;
        }

        Console.WriteLine($"[dump] Found MCNK at 0x{foundAt:X} with indexX={idxX} indexY={idxY}");

        int mcnkSize = BitConverter.ToInt32(buf, foundAt + 4);
        int mcnkHeaderStart = foundAt + 8;
        if (mcnkHeaderStart + 0x80 > buf.Length)
        {
            Console.Error.WriteLine("[dump][error] MCNK header truncated.");
            return 1;
        }

        int flags = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x00);
        int nLayers = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x10);
        int nDoodadRefs = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x14);
        int offsHeight = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x18);
        int offsNormal = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x1C);
        int offsLayer = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x20);
        int offsRefs = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x24);
        int offsAlpha = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x28);
        int sizeAlpha = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x2C);
        int offsShadow = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x30);
        int sizeShadow = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x34);
        int areaId = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x38);
        int nMapObjRefs = BitConverter.ToInt32(buf, mcnkHeaderStart + 0x3C);

        Console.WriteLine($"[dump] MCNK size={mcnkSize} flags=0x{flags:X} nLayers={nLayers} nDoodadRefs={nDoodadRefs} nMapObjRefs={nMapObjRefs}");

        // Extract MCLY raw table
        var mclyEntries = new List<(int Layer, uint TextureId, uint Flags, uint AlphaOffset)>();
        if (offsLayer > 0)
        {
            int mclyPos = foundAt + offsLayer;
            if (mclyPos + 8 <= buf.Length)
            {
                string mclyFcc = Encoding.ASCII.GetString(buf, mclyPos, 4);
                int mclySize = BitConverter.ToInt32(buf, mclyPos + 4);
                int mclyDataStart = mclyPos + 8;
                int mclyDataEnd = Math.Min(mclyDataStart + mclySize, buf.Length);
                int mclyLen = Math.Max(0, mclyDataEnd - mclyDataStart);

                if (mclyFcc == "YLCM" && mclyLen >= 16)
                {
                    int layersFromTable = mclyLen / 16;
                    for (int i = 0; i < layersFromTable; i++)
                    {
                        int baseOff = mclyDataStart + i * 16;
                        if (baseOff + 16 > buf.Length) break;
                        uint texId = BitConverter.ToUInt32(buf, baseOff + 0);
                        uint fl = BitConverter.ToUInt32(buf, baseOff + 4);
                        uint alphaOff = BitConverter.ToUInt32(buf, baseOff + 8);
                        mclyEntries.Add((i, texId, fl, alphaOff));
                    }
                }
                else
                {
                    Console.Error.WriteLine($"[dump][warn] MCLY not found at offsLayer=0x{offsLayer:X} or invalid FourCC '{mclyFcc}'.");
                }
            }
        }

        // MCAL length is derived from header sizeAlpha; payload has no header in Alpha v18
        int mcalLen = sizeAlpha;

        try
        {
            var outDir = Path.GetDirectoryName(outCsv);
            if (!string.IsNullOrWhiteSpace(outDir)) Directory.CreateDirectory(outDir!);
        }
        catch { }

        using (var sw = new StreamWriter(outCsv, false, Encoding.UTF8))
        {
            sw.WriteLine("kind,tileX,tileY,chunkIndex,indexX,indexY,flags,nLayers,nDoodadRefs,nMapObjRefs,offsHeight,offsNormal,offsLayer,offsRefs,offsAlpha,sizeAlpha,offsShadow,sizeShadow,areaId,mcalLen");
            sw.WriteLine(string.Join(',',
                "meta",
                tx.ToString(CultureInfo.InvariantCulture),
                ty.ToString(CultureInfo.InvariantCulture),
                ci.ToString(CultureInfo.InvariantCulture),
                idxX.ToString(CultureInfo.InvariantCulture),
                idxY.ToString(CultureInfo.InvariantCulture),
                $"0x{flags:X}",
                nLayers.ToString(CultureInfo.InvariantCulture),
                nDoodadRefs.ToString(CultureInfo.InvariantCulture),
                nMapObjRefs.ToString(CultureInfo.InvariantCulture),
                offsHeight.ToString(CultureInfo.InvariantCulture),
                offsNormal.ToString(CultureInfo.InvariantCulture),
                offsLayer.ToString(CultureInfo.InvariantCulture),
                offsRefs.ToString(CultureInfo.InvariantCulture),
                offsAlpha.ToString(CultureInfo.InvariantCulture),
                sizeAlpha.ToString(CultureInfo.InvariantCulture),
                offsShadow.ToString(CultureInfo.InvariantCulture),
                sizeShadow.ToString(CultureInfo.InvariantCulture),
                areaId.ToString(CultureInfo.InvariantCulture),
                mcalLen.ToString(CultureInfo.InvariantCulture)));

            sw.WriteLine("kind,layer,textureId,flags,alphaOffset");
            foreach (var e in mclyEntries)
            {
                sw.WriteLine(string.Join(',',
                    "layer",
                    e.Layer.ToString(CultureInfo.InvariantCulture),
                    e.TextureId.ToString(CultureInfo.InvariantCulture),
                    $"0x{e.Flags:X}",
                    e.AlphaOffset.ToString(CultureInfo.InvariantCulture)));
            }
        }

        Console.WriteLine("[dump] CSV written: " + outCsv);
        return 0;
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

    private static string? GuessProductFromPath(string clientRoot)
    {
        try
        {
            var lower = clientRoot.ToLowerInvariant();
            if (lower.Contains("beta")) return "wow_beta";
            if (lower.Contains("ptr") || lower.Contains("wowt")) return "wowt";
            return "wow";
        }
        catch { return null; }
    }

    private static string? ExtractProductFromBuildInfo(string clientRoot)
    {
        try
        {
            string? dir = clientRoot;
            for (int i = 0; i < 4 && !string.IsNullOrEmpty(dir); i++, dir = Path.GetDirectoryName(dir))
            {
                var path = Path.Combine(dir!, ".build.info");
                if (!File.Exists(path)) continue;
                var lines = File.ReadAllLines(path);
                if (lines.Length < 2) return null;
                var header = lines[0];
                var delim = header.Contains('|') ? '|' : (header.Contains(';') ? ';' : ' ');
                var cols = header.Split(delim, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                int idxProduct = Array.FindIndex(cols, c => c.StartsWith("Product", StringComparison.OrdinalIgnoreCase));
                if (idxProduct < 0) return null;
                for (int j = 1; j < lines.Length; j++)
                {
                    var parts = lines[j].Split(delim, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                    if (idxProduct >= parts.Length) continue;
                    var prod = parts[idxProduct];
                    if (!string.IsNullOrWhiteSpace(prod)) return prod;
                }
            }
        }
        catch { }
        return null;
    }

    private static int RunAnalyzeMapAdtsCasc(Dictionary<string, string> opts)
    {
        Require(opts, "client-path");
        var clientRoot = opts["client-path"];
        var allMaps = opts.ContainsKey("all-maps");
        var mapName = opts.GetValueOrDefault("map", "");
        var product = opts.GetValueOrDefault("product", ExtractProductFromBuildInfo(clientRoot) ?? GuessProductFromPath(clientRoot) ?? "wow");
        var listfile = opts.GetValueOrDefault("listfile", "");

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

        // Initialize CASC-backed archive source (local storage via .build.info)
        using var src = new WoWRollback.Core.Services.Archive.CascArchiveSource(clientRoot, product, string.IsNullOrWhiteSpace(listfile) ? null : listfile);

        // Batch: discover from Map.dbc and analyze
        if (allMaps)
        {
            return RunBatchAnalysis(src, clientRoot, opts);
        }

        // Single map
        var outDir = opts.GetValueOrDefault("out", Path.Combine("analysis_output", mapName));
        return AnalyzeSingleMap(src, clientRoot, mapName, outDir, opts);
    }

    private static int RunBatchAnalysis(IArchiveSource src, string clientRoot, Dictionary<string, string> opts)
    {
        // Try auto-detection first, then check both --version and --build parameters
        var detectedVersion = ExtractVersionFromBuildInfo(clientRoot) ?? ExtractVersionFromPath(clientRoot);
        var buildVersion = opts.GetValueOrDefault("version", 
                          opts.GetValueOrDefault("build", 
                          detectedVersion ?? "0.6.0"));
        
        var dbdDir = opts.GetValueOrDefault("dbd-dir", Path.Combine(Directory.GetCurrentDirectory(), "..", "lib", "WoWDBDefs", "definitions"));
        var baseOutDir = opts.GetValueOrDefault("out", "analysis_output");
        var versionLabel = buildVersion; // Use the resolved version

        Console.WriteLine($"[info] === Batch Analysis Mode ===");
        Console.WriteLine($"[info] Discovering maps from Map.dbc...");

        // Extract Map.dbc and discover maps (preferred for MPQ/LK)
        var tempDir = Path.Combine(Path.GetTempPath(), "wowrollback_dbc_" + Guid.NewGuid().ToString("N"));
        var discoveryService = new MapDiscoveryService(dbdDir);
        
        MapDiscoveryResult? result = null;
        var dbcDir = discoveryService.ExtractMapDbc(src, tempDir);
        if (!string.IsNullOrEmpty(dbcDir))
        {
            var res = discoveryService.DiscoverMaps(src, buildVersion, Path.GetDirectoryName(dbcDir)!);
            if (res.Success) result = res; else Console.WriteLine($"[warn] Map.dbc discovery failed: {res.ErrorMessage}");
        }
        
        // If Map.dbc not available or failed, attempt CASC DB2 discovery
        if (result is null)
        {
            var resDb2 = discoveryService.DiscoverMapsFromCasc(src, buildVersion);
            if (resDb2.Success) result = resDb2; else Console.WriteLine($"[warn] Map.db2 discovery failed: {resDb2.ErrorMessage}");
        }
        
        // Fallback for CASC or missing DBC: scan world/maps/*/*.wdt
        if (result is null)
        {
            Console.WriteLine("[info] Falling back to WDT scan (CASC/DB2 builds)");
            var wdtPaths = src.EnumerateFiles("world/maps/*/*.wdt").ToList();
            if (wdtPaths.Count == 0)
            {
                Console.Error.WriteLine("[error] No WDT files found via archive enumeration");
                return 1;
            }
            var byMap = new SortedDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var vp in wdtPaths)
            {
                var norm = vp.Replace('\\','/');
                var parts = norm.Split('/', StringSplitOptions.RemoveEmptyEntries);
                // Expect world/maps/<map>/<map>.wdt
                if (parts.Length >= 4 && parts[0].Equals("world", StringComparison.OrdinalIgnoreCase) && parts[1].Equals("maps", StringComparison.OrdinalIgnoreCase))
                {
                    var map = parts[2];
                    if (!byMap.ContainsKey(map)) byMap[map] = norm;
                }
            }
            var wdtAnalyzer = new WdtAnalyzer();
            var discovered = new List<DiscoveredMap>();
            foreach (var map in byMap.Keys)
            {
                var w = wdtAnalyzer.Analyze(src, map);
                discovered.Add(new DiscoveredMap(
                    Id: -1,
                    Name: map,
                    Folder: map,
                    WdtExists: w.Success,
                    HasTerrain: w.HasTerrain,
                    IsWmoOnly: w.IsWmoOnly,
                    TileCount: w.TileCount,
                    WmoPlacement: w.WmoPlacement
                ));
            }
            result = new MapDiscoveryResult(true, null, discovered.ToArray());
        }

        // Filter to terrain maps only (skip WMO-only for now)
        var terrainMaps = result.Maps.Where(m => m.HasTerrain && m.TileCount > 0).ToList();

        // Apply selection filters: --maps (csv of names/folders) and --map-ids (csv)
        var mapsCsv = opts.GetValueOrDefault("maps");
        var idsCsv = opts.GetValueOrDefault("map-ids");
        if (!string.IsNullOrWhiteSpace(mapsCsv))
        {
            var set = new HashSet<string>(mapsCsv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries), StringComparer.OrdinalIgnoreCase);
            terrainMaps = terrainMaps.Where(m => set.Contains(m.Name) || set.Contains(m.Folder)).ToList();
        }
        if (!string.IsNullOrWhiteSpace(idsCsv))
        {
            var idSet = new HashSet<int>(idsCsv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => int.TryParse(s, out var v) ? v : int.MinValue));
            terrainMaps = terrainMaps.Where(m => idSet.Contains(m.Id)).ToList();
        }

        // Limit number of maps processed
        var maxMaps = TryParseInt(opts, "max-maps");
        if (maxMaps.HasValue && maxMaps.Value > 0)
        {
            terrainMaps = terrainMaps.Take(maxMaps.Value).ToList();
        }
        
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
                var exitCode = AnalyzeSingleMapNoViewer(src, clientRoot, map.Folder, mapOutDir, opts, out var placementsCsv, out var minimapDir);
                
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

        // Phase 2: Generate unified viewer for all processed maps (skip if placements-only)
        if (processedMaps.Count > 0)
        {
            if (opts.ContainsKey("placements-only"))
            {
                Console.WriteLine("[info] placements-only: skipping unified viewer generation");
            }
            else
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

    private static int AnalyzeSingleMapNoViewer(IArchiveSource src, string clientRoot, string mapName, string outDir, Dictionary<string, string> opts, out string? placementsCsv, out string? minimapDir)
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
        
        // Step 0: Reuse cached placements if available (unless --force)
        var cachedCsv = Path.Combine(outDir, $"{mapName}_placements.csv");
        if (File.Exists(cachedCsv) && !opts.ContainsKey("force"))
        {
            Console.WriteLine("  [info] Reusing cached placements CSV (use --force to reprocess)");
            placementsCsv = cachedCsv;
            // Placements-only mode can return now
            if (opts.ContainsKey("placements-only"))
            {
                Console.WriteLine("  [info] placements-only: skipping remaining steps");
                return 0;
            }
        }

        // Step 1: Extract placements
        Console.WriteLine("  Step 1: Extracting placements...");
        var extractor = new AdtMpqChunkPlacementsExtractor();
        placementsCsv = cachedCsv;
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

        // Step 2: Extract minimaps (disabled by default)
        if (opts.ContainsKey("export-minimaps"))
        {
            Console.WriteLine("  Step 2: Extracting minimaps...");
            minimapDir = ExtractMinimapsFromMpq(src, mapName, outDir);
            if (!string.IsNullOrEmpty(minimapDir))
            {
                var count = Directory.GetFiles(minimapDir, "*.jpg").Length;
                Console.WriteLine($"  [ok] Extracted {count} minimap tiles");
            }
        }
        else
        {
            minimapDir = null;
            Console.WriteLine("  Step 2: Skipped minimaps (enable with --export-minimaps)");
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

        // Placements-only: short-circuit after UniqueID analysis (layers written)
        if (opts.ContainsKey("placements-only"))
        {
            Console.WriteLine("  [info] placements-only: skipping clusters, terrain, mesh");
            return 0;
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

        // Step 6: Extract terrain meshes (GLB + OBJ) - disabled by default
        if (opts.ContainsKey("export-mesh"))
        {
            Console.WriteLine("  Step 6: Extracting terrain meshes (GLB + OBJ)...");
            try
            {
                var meshExtractor = new AdtMeshExtractor();
                var maxTiles = TryParseInt(opts, "max-tiles") ?? 0;
                var meshResult = meshExtractor.ExtractFromArchive(src, mapName, outDir, exportGlb: true, exportObj: true, maxTiles: maxTiles);
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
        }
        else
        {
            Console.WriteLine("  Step 6: Skipping mesh export (enable with --export-mesh)");
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
        
        var placementsCsvPath = Path.Combine(outDir, $"{mapName}_placements.csv");
        bool reused = false;
        PlacementsExtractionResult? extractResult = null;
        if (File.Exists(placementsCsvPath) && !opts.ContainsKey("force"))
        {
            Console.WriteLine("[info] Reusing cached placements CSV (use --force to reprocess)");
            reused = true;
        }
        else
        {
            var extractor = new AdtMpqChunkPlacementsExtractor();
            extractResult = extractor.ExtractFromArchive(src, mapName, placementsCsvPath);
            if (!extractResult.Success)
            {
                Console.WriteLine($"[warn] {extractResult.ErrorMessage}");
                Console.WriteLine($"[info] Skipping map (extraction failed)");
                return 2; // Non-fatal error code
            }
        }

        // Check if any placements were found (when freshly extracted). If reused, assume non-empty.
        var totalPlacements = reused ? 1 : (extractResult!.M2Count + extractResult.WmoCount);
        if (totalPlacements == 0)
        {
            Console.WriteLine($"[info] No placements found for map: {mapName}");
            Console.WriteLine($"[info] Skipping remaining analysis steps");
            return 0; // Success but no data
        }

        if (!reused) Console.WriteLine($"[ok] {extractResult!.ErrorMessage}");
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
        var versionLabel = ExtractVersionFromBuildInfo(clientRoot) ?? ExtractVersionFromPath(clientRoot) ?? "analysis";
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
        if (!opts.ContainsKey("input"))
        {
            if (!(opts.ContainsKey("client-path") && opts.ContainsKey("map")))
            {
                Console.Error.WriteLine("[error] Provide --input <WDT> or for CASC: --client-path <dir> --map <name> [--listfile <file>] [--product wow|wowt]");
                return 2;
            }
        }
        Require(opts, "max-uniqueid");

        var merged = new Dictionary<string, string>(opts, StringComparer.OrdinalIgnoreCase)
        {
            ["export-lk-adts"] = "true"
        };

        return RunRollback(merged);
    }

    private static int RunRollback(Dictionary<string, string> opts)
    {
        var hasInputPath = opts.ContainsKey("input");
        var cascMode = !hasInputPath && opts.ContainsKey("client-path") && opts.ContainsKey("map");
        if (!hasInputPath && !cascMode)
        {
            Console.Error.WriteLine("[error] Provide --input <WDT> or for CASC: --client-path <dir> --map <name>");
            return 2;
        }
        Require(opts, "max-uniqueid");

        var inputPath = hasInputPath ? opts["input"] : string.Empty;
        var mapName = cascMode ? opts["map"] : Path.GetFileNameWithoutExtension(inputPath);
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
        int threads = Math.Max(1, (TryParseInt(opts, "threads") ?? 1)); 

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
        var outputPath = Path.Combine(alphaWdtDir, cascMode ? (mapName + ".wdt") : Path.GetFileName(inputPath));

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
        Console.WriteLine(cascMode
            ? $"Input WDT:      cas://world/maps/{mapName}/{mapName}.wdt"
            : $"Input WDT:      {inputPath}");
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

            // Very-early diagnostics CSV creation (before any export try/catch)
            try
            {
                string[] diagCandidatesEarly = new[]
                {
                    Path.Combine(outRoot, "lk_export_diag.csv"),
                    Path.Combine(lkOutDir, "lk_export_diag.csv"),
                    Path.Combine(Directory.GetCurrentDirectory(), "lk_export_diag.csv"),
                };
                var header = "tile_yy,tile_xx,chunk_idx,nDoodadRefs,nMapObjRefs,offsRefs,offsLayer,offsAlpha,offsShadow,offsSnd,offsLiquid,mddf_count,modf_count,mcrf_expected,mcrf_payload,d_samples,w_samples,exception\n";
                var diagEarlyWritten = new List<string>();
                foreach (var cand in diagCandidatesEarly)
                {
                    try
                    {
                        var dir = Path.GetDirectoryName(cand);
                        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                        if (!File.Exists(cand)) File.WriteAllText(cand, header);
                        try { File.AppendAllText(cand, "-1,-1,-1,,,,,,,,,,,,,,,export_begin_pre\n"); } catch { }
                        diagEarlyWritten.Add(cand);
                    }
                    catch { }
                }
                if (diagEarlyWritten.Count > 0)
                {
                    Console.WriteLine($"[diag] lk_export_diag.csv => {string.Join(" | ", diagEarlyWritten)}");
                }
            }
            catch { }
            Console.WriteLine($"Strict AreaIDs: {(strictAreaId ? "true" : "false")}");
            if (reportAreaId) Console.WriteLine("Option:         --report-areaid (write summary CSV)");
        }
        Console.WriteLine();

        try
        {
            byte[] wdtBytes;
            List<int> existingAdts;
            List<int> adtOffsets;
            if (cascMode)
            {
                var clientRoot = opts["client-path"];
                var product = GetOption(opts, "product") ?? (ExtractProductFromBuildInfo(clientRoot) ?? GuessProductFromPath(clientRoot) ?? "wow");
                var listfile = GetOption(opts, "listfile");
                using var src = new WoWRollback.Core.Services.Archive.CascArchiveSource(clientRoot, product, listfile);
                var vpath = $"world/maps/{mapName}/{mapName}.wdt";
                if (!src.FileExists(vpath)) { Console.Error.WriteLine($"[error] CASC WDT not found: {vpath}"); return 1; }
                using (var s = src.OpenFile(vpath))
                {
                    using var ms = new MemoryStream();
                    s.CopyTo(ms);
                    wdtBytes = ms.ToArray();
                }
                if (!TryIsAlphaV18Wdt(wdtBytes, out var reason))
                {
                    Console.Error.WriteLine($"[error] Provided WDT does not appear to be Alpha v18: {reason}");
                    Console.Error.WriteLine("[hint] Use 'pack-monolithic-alpha-wdt' to generate an Alpha WDT from LK/modern inputs, then pass it via --input.");
                    return 2;
                }
                // Library constructor takes a file path. Write bytes to the planned output path and parse from there.
                File.WriteAllBytes(outputPath, wdtBytes);
                var wdt = new WdtAlpha(outputPath);
                existingAdts = wdt.GetExistingAdtsNumbers();
                adtOffsets = wdt.GetAdtOffsetsInMain();
            }
            else
            {
                var wdt = new WdtAlpha(inputPath);
                existingAdts = wdt.GetExistingAdtsNumbers();
                adtOffsets = wdt.GetAdtOffsetsInMain();
                wdtBytes = File.ReadAllBytes(inputPath);
            }
            Console.WriteLine($"[info] Tiles detected: {existingAdts.Count}");
            var expectedAdtCount = existingAdts.Count;
            int totalPlacements = 0, removed = 0, tilesProcessed = 0;
            int holesCleared = 0, mcshZeroed = 0;
            var processedMcnk = new HashSet<int>();

            foreach (var adtNum in existingAdts)
            {
                int adtOffset = adtOffsets[adtNum];
                if (adtOffset == 0) continue;

                // Build AdtAlpha using a file path (library does not expose a byte[] ctor)
                var adtPath = cascMode ? outputPath : inputPath;
                var adt = new AdtAlpha(adtPath, adtOffset, adtNum);

                // Compute MDDF/MODF chunk positions from MHDR relative offsets
                int mhdrDataStart = adtOffset + 8; // skip 'MHDR' header
                int mddfRel = BitConverter.ToInt32(wdtBytes, mhdrDataStart + 0x0C);
                int modfRel = BitConverter.ToInt32(wdtBytes, mhdrDataStart + 0x14);
                int mddfChunkOffset = (mddfRel > 0) ? mhdrDataStart + mddfRel : -1;
                int modfChunkOffset = (modfRel > 0) ? mhdrDataStart + modfRel : -1;

                var mddf = (mddfChunkOffset >= 0 && (mddfChunkOffset + 8) <= wdtBytes.Length)
                    ? new Mddf(wdtBytes, mddfChunkOffset)
                    : new Mddf("MDDF", 0, Array.Empty<byte>());
                var modf = (modfChunkOffset >= 0 && (modfChunkOffset + 8) <= wdtBytes.Length)
                    ? new Modf(wdtBytes, modfChunkOffset)
                    : new Modf("MODF", 0, Array.Empty<byte>());

                // Data payload starts after each subchunk's 8-byte header
                int mddfFileOffsetLocal = (mddfChunkOffset >= 0) ? (mddfChunkOffset + 8) : -1;
                int modfFileOffsetLocal = (modfChunkOffset >= 0) ? (modfChunkOffset + 8) : -1;

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
                if (mddf.Data.Length > 0 && mddfFileOffsetLocal >= 0 && (mddfFileOffsetLocal + mddf.Data.Length) <= wdtBytes.Length)
                {
                    Array.Copy(mddf.Data, 0, wdtBytes, mddfFileOffsetLocal, mddf.Data.Length);
                }
                if (modf.Data.Length > 0 && modfFileOffsetLocal >= 0 && (modfFileOffsetLocal + modf.Data.Length) <= wdtBytes.Length)
                {
                    Array.Copy(modf.Data, 0, wdtBytes, modfFileOffsetLocal, modf.Data.Length);
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
                    var perAdtRows = new ConcurrentBag<string>();
                    var timingRowsPar = new ConcurrentBag<string>();
                    // Prepare diagnostics CSV path and ensure header exists with fallbacks
                    string[] diagCandidates = new[]
                    {
                        Path.Combine(outRoot, "lk_export_diag.csv"),
                        Path.Combine(lkOutDir, "lk_export_diag.csv"),
                        Path.Combine(Directory.GetCurrentDirectory(), "lk_export_diag.csv"),
                    };
                    var header = "tile_yy,tile_xx,chunk_idx,nDoodadRefs,nMapObjRefs,offsRefs,offsLayer,offsAlpha,offsShadow,offsSnd,offsLiquid,mddf_count,modf_count,mcrf_expected,mcrf_payload,d_samples,w_samples,exception\n";
                    var diagWritten = new List<string>();
                    string diagPath = diagCandidates[0];
                    foreach (var cand in diagCandidates)
                    {
                        try
                        {
                            var dir = Path.GetDirectoryName(cand);
                            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                            if (!File.Exists(cand)) File.WriteAllText(cand, header);
                            try { File.AppendAllText(cand, "-1,-1,-1,,,,,,,,,,,,,,,export_begin\n"); } catch { }
                            diagWritten.Add(cand);
                            diagPath = cand; // keep last successful for subsequent appends
                        }
                        catch { /* try next candidate */ }
                    }
                    Console.WriteLine($"[diag] lk_export_diag.csv => {string.Join(" | ", diagWritten)}");
                    string[] mcrfDiagCandidates = new[]
                    {
                        Path.Combine(outRoot, "lk_export_diag_mcrf.csv"),
                        Path.Combine(lkOutDir, "lk_export_diag_mcrf.csv"),
                        Path.Combine(Directory.GetCurrentDirectory(), "lk_export_diag_mcrf.csv"),
                    };
                    var mcrfHeader = "tile_yy,tile_xx,chunk_idx,src,nDoodadRefs,nMapObjRefs,mcrf_payload,d_samples,w_samples\n";
                    var mcrfTargets = new List<string>();
                    foreach (var cand in mcrfDiagCandidates)
                    {
                        try { var dir = Path.GetDirectoryName(cand); if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir); if (!File.Exists(cand)) File.WriteAllText(cand, mcrfHeader); mcrfTargets.Add(cand); } catch { }
                    }
                    static int FindFourCCLocal(byte[] buf, string fwd)
                    {
                        if (buf == null || buf.Length < 8) return -1; if (string.IsNullOrEmpty(fwd) || fwd.Length != 4) return -1;
                        var r = new[] { (byte)fwd[3], (byte)fwd[2], (byte)fwd[1], (byte)fwd[0] };
                        for (int i = 0; i + 8 <= buf.Length;)
                        {
                            if (i < 0 || i + 8 > buf.Length) break;
                            if (buf[i + 0] == r[0] && buf[i + 1] == r[1] && buf[i + 2] == r[2] && buf[i + 3] == r[3]) return i;
                            int size = BitConverter.ToInt32(buf, i + 4); if (size < 0 || size > buf.Length) break; int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0); if (next <= i) break; i = next;
                        }
                        return -1;
                    }
                    var diagTargets = diagWritten.Count > 0 ? diagWritten.ToArray() : new[] { diagPath };
                    // Pre-loop marker with counts so we know we reached the iteration boundary
                    try
                    {
                        var pre = $"-1,-1,-1,,,,,,,,,,,,,,,pre_loop existing={existingAfter.Count} offsets={adtOffsetsAfter.Count}\\n";
                        foreach (var pth in diagTargets) { try { File.AppendAllText(pth, pre); } catch { } }
                        Console.WriteLine($"[diag] pre_loop: existing={existingAfter.Count} offsets={adtOffsetsAfter.Count}");
                    }
                    catch { }

                    if (threads > 1)
                    {
                        var parOpts = new ParallelOptions { MaxDegreeOfParallelism = threads };
                        var diagLock = new object();
                        var mcrfLock = new object();
                        Parallel.ForEach(existingAfter, parOpts, adtNum2 =>
                        {
                            var swTile = Stopwatch.StartNew();
                            int adtOff2 = adtOffsetsAfter[adtNum2];
                            if (adtOff2 == 0) return;
                            try
                            {
                                int tileX0 = adtNum2 % 64, tileY0 = adtNum2 / 64;
                                var line = $"{tileY0},{tileX0},-1,,,,,,,,,,,,,,,tile_begin\n";
                                lock (diagLock) { foreach (var p in diagTargets) { try { File.AppendAllText(p, line); } catch { } } }
                            }
                            catch { }

                            try
                            {
                                byte[] alphaWdtBytes = File.ReadAllBytes(outputPath);
                                int mhdrStart = adtOff2 + 8;
                                int mcinRel2 = BitConverter.ToInt32(alphaWdtBytes, mhdrStart + 0x0);
                                int mddfRel2 = BitConverter.ToInt32(alphaWdtBytes, mhdrStart + 0x0C);
                                int modfRel2 = BitConverter.ToInt32(alphaWdtBytes, mhdrStart + 0x14);
                                int mcinChunkOffset2 = mhdrStart + mcinRel2;
                                int mddfChunkOffset2 = (mddfRel2 > 0) ? mhdrStart + mddfRel2 : -1;
                                int modfChunkOffset2 = (modfRel2 > 0) ? mhdrStart + modfRel2 : -1;
                                int mddfCount2 = 0, modfCount2 = 0;
                                if (mddfChunkOffset2 >= 0 && mddfChunkOffset2 + 8 <= alphaWdtBytes.Length)
                                {
                                    int sz = BitConverter.ToInt32(alphaWdtBytes, mddfChunkOffset2 + 4);
                                    mddfCount2 = Math.Max(0, sz / 36);
                                }
                                if (modfChunkOffset2 >= 0 && modfChunkOffset2 + 8 <= alphaWdtBytes.Length)
                                {
                                    int sz = BitConverter.ToInt32(alphaWdtBytes, modfChunkOffset2 + 4);
                                    modfCount2 = Math.Max(0, sz / 64);
                                }
                                var mcin2 = new Mcin(alphaWdtBytes, mcinChunkOffset2);
                                var mcnkOffsets2 = mcin2.GetMcnkOffsets();
                                int tileX = adtNum2 % 64, tileY = adtNum2 / 64;

                                for (int ci = 0; ci < mcnkOffsets2.Count && ci < 256; ci++)
                                {
                                    int pos = mcnkOffsets2[ci];
                                    if (pos <= 0 || pos + 8 > alphaWdtBytes.Length) continue;
                                    int headerStart = pos + 8;
                                    if (headerStart + 128 > alphaWdtBytes.Length) continue;
                                    int nDoodadRefs = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x14);
                                    int nMapObjRefs = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x3C);
                                    int offsRefs = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x24);
                                    int offsLayer = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x20);
                                    int offsAlpha = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x28);
                                    int offsShadow = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x30);
                                    int offsSnd = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x5C);
                                    int offsLiquid = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x64);

                                    int mcrfPayload = 0;
                                    string dSamples = string.Empty, wSamples = string.Empty;
                                    try
                                    {
                                        int mcrfChunkOffset2 = pos + offsRefs;
                                        if (mcrfChunkOffset2 + 8 <= alphaWdtBytes.Length && offsRefs > 0)
                                        {
                                            var mcrf2 = new Mcrf(alphaWdtBytes, mcrfChunkOffset2);
                                            mcrfPayload = mcrf2.Data.Length;
                                            var dIdx = (System.Collections.Generic.IEnumerable<int>)mcrf2.GetDoodadsIndices(Math.Max(0, nDoodadRefs)) ?? System.Linq.Enumerable.Empty<int>();
                                            var wIdx = (System.Collections.Generic.IEnumerable<int>)mcrf2.GetWmosIndices(Math.Max(0, nMapObjRefs)) ?? System.Linq.Enumerable.Empty<int>();
                                            var dTake = System.Linq.Enumerable.Take(dIdx, 4);
                                            var wTake = System.Linq.Enumerable.Take(wIdx, 4);
                                            dSamples = string.Join('|', System.Linq.Enumerable.Select(dTake, v => v.ToString(CultureInfo.InvariantCulture)));
                                            wSamples = string.Join('|', System.Linq.Enumerable.Select(wTake, v => v.ToString(CultureInfo.InvariantCulture)));
                                            if (mcrfTargets.Count > 0)
                                            {
                                                var row2 = string.Join(',', new[] { tileY.ToString(CultureInfo.InvariantCulture), tileX.ToString(CultureInfo.InvariantCulture), ci.ToString(CultureInfo.InvariantCulture), "alpha", nDoodadRefs.ToString(CultureInfo.InvariantCulture), nMapObjRefs.ToString(CultureInfo.InvariantCulture), mcrfPayload.ToString(CultureInfo.InvariantCulture), dSamples, wSamples }) + "\n";
                                                lock (mcrfLock) { foreach (var p in mcrfTargets) { try { File.AppendAllText(p, row2); } catch { } } }
                                            }
                                        }
                                    }
                                    catch { }

                                    long expected = (long)(nDoodadRefs + nMapObjRefs) * 4L;
                                    var row = string.Join(',', new[] { tileY.ToString(CultureInfo.InvariantCulture), tileX.ToString(CultureInfo.InvariantCulture), ci.ToString(CultureInfo.InvariantCulture), nDoodadRefs.ToString(CultureInfo.InvariantCulture), nMapObjRefs.ToString(CultureInfo.InvariantCulture), offsRefs.ToString(CultureInfo.InvariantCulture), offsLayer.ToString(CultureInfo.InvariantCulture), offsAlpha.ToString(CultureInfo.InvariantCulture), offsShadow.ToString(CultureInfo.InvariantCulture), offsSnd.ToString(CultureInfo.InvariantCulture), offsLiquid.ToString(CultureInfo.InvariantCulture), mddfCount2.ToString(CultureInfo.InvariantCulture), modfCount2.ToString(CultureInfo.InvariantCulture), expected.ToString(CultureInfo.InvariantCulture), mcrfPayload.ToString(CultureInfo.InvariantCulture), dSamples, wSamples, string.Empty }) + "\n";
                                    lock (diagLock) { foreach (var p in diagTargets) { try { File.AppendAllText(p, row); } catch { } } }
                                }
                            }
                            catch (Exception exDiag)
                            {
                                try
                                {
                                    int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                    var line = $"{tileY},{tileX},-1,,,,,,,,,,,,,,,{exDiag.Message}\n";
                                    lock (diagLock) { foreach (var p in diagTargets) { try { File.AppendAllText(p, line); } catch { } } }
                                }
                                catch { }
                            }

                            AdtAlpha? a = null;
                            try { a = new AdtAlpha(outputPath, adtOff2, adtNum2); }
                            catch (Exception exCtor)
                            {
                                try
                                {
                                    int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                    var line = $"{tileY},{tileX},-1,,,,,,,,,,,,,,,{exCtor.Message}\n";
                                    lock (diagLock) { foreach (var p in diagTargets) { try { File.AppendAllText(p, line); } catch { } } }
                                }
                                catch { }
                                return;
                            }

                            try
                            {
                                var lk = a!.ToAdtLk(mdnmNames, monmNames);
                                lk.ToFile(lkOutDir);
                                try
                                {
                                    int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                    var outFile = Path.Combine(lkOutDir, $"{mapName}_{tileX}_{tileY}.adt");
                                    if (File.Exists(outFile))
                                    {
                                        var lkBytes = File.ReadAllBytes(outFile);
                                        int lkMh = FindFourCCLocal(lkBytes, "MHDR");
                                        if (lkMh >= 0)
                                        {
                                            int lkMhStart = lkMh + 8;
                                            int lkMcinRel = BitConverter.ToInt32(lkBytes, lkMhStart + 0x0);
                                            int lkMcinChunk = lkMhStart + lkMcinRel;
                                            if (lkMcinChunk >= 0 && lkMcinChunk + 8 <= lkBytes.Length)
                                            {
                                                int entries = 256;
                                                for (int ci = 0; ci < entries; ci++)
                                                {
                                                    int mcPos = BitConverter.ToInt32(lkBytes, lkMcinChunk + 8 + ci * 16 + 0);
                                                    if (mcPos <= 0 || mcPos + 8 > lkBytes.Length) continue;
                                                    int hStart = mcPos + 8;
                                                    if (hStart + 128 > lkBytes.Length) continue;
                                                    int dN = BitConverter.ToInt32(lkBytes, hStart + 0x14);
                                                    int wN = BitConverter.ToInt32(lkBytes, hStart + 0x3C);
                                                    int oRefs = BitConverter.ToInt32(lkBytes, hStart + 0x20);
                                                    int mcrfPayload = 0;
                                                    string dSamples = string.Empty, wSamples = string.Empty;
                                                    if (oRefs > 0)
                                                    {
                                                        int mcrfOff = mcPos + oRefs;
                                                        if (mcrfOff + 8 <= lkBytes.Length)
                                                        {
                                                            var mcrf3 = new Mcrf(lkBytes, mcrfOff);
                                                            mcrfPayload = mcrf3.Data.Length;
                                                            var dIdx = (System.Collections.Generic.IEnumerable<int>)mcrf3.GetDoodadsIndices(Math.Max(0, dN)) ?? System.Linq.Enumerable.Empty<int>();
                                                            var wIdx = (System.Collections.Generic.IEnumerable<int>)mcrf3.GetWmosIndices(Math.Max(0, wN)) ?? System.Linq.Enumerable.Empty<int>();
                                                            var dTake = System.Linq.Enumerable.Take(dIdx, 4);
                                                            var wTake = System.Linq.Enumerable.Take(wIdx, 4);
                                                            dSamples = string.Join('|', System.Linq.Enumerable.Select(dTake, v => v.ToString(CultureInfo.InvariantCulture)));
                                                            wSamples = string.Join('|', System.Linq.Enumerable.Select(wTake, v => v.ToString(CultureInfo.InvariantCulture)));
                                                        }
                                                    }
                                                    if (mcrfTargets.Count > 0)
                                                    {
                                                        var row3 = string.Join(',', new[] { (adtNum2 / 64).ToString(CultureInfo.InvariantCulture), (adtNum2 % 64).ToString(CultureInfo.InvariantCulture), ci.ToString(CultureInfo.InvariantCulture), "lk", dN.ToString(CultureInfo.InvariantCulture), wN.ToString(CultureInfo.InvariantCulture), mcrfPayload.ToString(CultureInfo.InvariantCulture), dSamples, wSamples }) + "\n";
                                                        lock (mcrfLock) { foreach (var p in mcrfTargets) { try { File.AppendAllText(p, row3); } catch { } } }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                catch { }
                            }
                            catch (Exception exConv)
                            {
                                try
                                {
                                    int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                    var line = $"{tileY},{tileX},-1,,,,,,,,,,,,,,,{exConv.Message}\n";
                                    lock (diagLock) { foreach (var p in diagTargets) { try { File.AppendAllText(p, line); } catch { } } }
                                }
                                catch { }
                            }

                            if (patchMap != null)
                            {
                                var alphaAreaIds = a!.GetAlphaMcnkAreaIds();
                                var outFile = Path.Combine(lkOutDir, $"{mapName}_{adtNum2 % 64}_{adtNum2 / 64}.adt");
                                try
                                {
                                    var (present, patched, mapped) = PatchMcnkAreaIdsOnDiskV2(outFile, mapName, alphaAreaIds, patchMap, currentMapId, strictAreaId, chainVia060);
                                    Interlocked.Add(ref totalAreaPresent, present);
                                    Interlocked.Add(ref totalAreaPatched, patched);
                                    Interlocked.Add(ref totalAreaMapped, mapped);
                                    if (reportAreaId)
                                    {
                                        var fname = Path.GetFileName(outFile);
                                        var unmatched = Math.Max(0, present - mapped);
                                        perAdtRows.Add($"{fname},{present},{mapped},{patched},{unmatched}");
                                    }
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine($"[warn] AreaID patch failed for {outFile}: {ex.Message}");
                                }
                            }
                            Interlocked.Increment(ref written);
                            swTile.Stop();
                            try
                            {
                                int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                timingRowsPar.Add($"{tileX},{tileY},{adtNum2},{swTile.Elapsed.TotalMilliseconds.ToString(CultureInfo.InvariantCulture)}");
                            }
                            catch { }
                        });
                        Console.WriteLine($"[ok] Exported {written}/{existingAfter.Count} LK ADTs to: {lkOutDir}");
                        try
                        {
                            var timingPath = Path.Combine(outRoot, "lk_export_timing.csv");
                            using var tw = new StreamWriter(timingPath, append: false, Encoding.UTF8);
                            tw.WriteLine("tile_x,tile_y,index,ms_total");
                            foreach (var row in timingRowsPar) tw.WriteLine(row);
                        }
                        catch { }
                    }
                    else
                    {
                        var timingRows = new List<string>();
                        foreach (var adtNum2 in existingAfter)
                        {
                        var swTile = Stopwatch.StartNew();
                        int adtOff2 = adtOffsetsAfter[adtNum2];
                        if (adtOff2 == 0) continue;
                        // Per-tile begin marker before any parsing or object construction
                        try
                        {
                            int tileX0 = adtNum2 % 64, tileY0 = adtNum2 / 64;
                            var line = $"{tileY0},{tileX0},-1,,,,,,,,,,,,,,,tile_begin\n";
                            foreach (var p in diagTargets) { try { File.AppendAllText(p, line); } catch { } }
                            Console.WriteLine($"[diag] tile_begin {tileY0},{tileX0}");
                        }
                        catch { }

                        // Gather diagnostics from Alpha ADT bytes prior to conversion
                        try
                        {
                            byte[] alphaWdtBytes = File.ReadAllBytes(outputPath);
                            int mhdrStart = adtOff2 + 8;
                            int mcinRel2 = BitConverter.ToInt32(alphaWdtBytes, mhdrStart + 0x0);
                            int mddfRel2 = BitConverter.ToInt32(alphaWdtBytes, mhdrStart + 0x0C);
                            int modfRel2 = BitConverter.ToInt32(alphaWdtBytes, mhdrStart + 0x14);
                            int mcinChunkOffset2 = mhdrStart + mcinRel2;
                            int mddfChunkOffset2 = (mddfRel2 > 0) ? mhdrStart + mddfRel2 : -1;
                            int modfChunkOffset2 = (modfRel2 > 0) ? mhdrStart + modfRel2 : -1;
                            int mddfCount2 = 0, modfCount2 = 0;
                            if (mddfChunkOffset2 >= 0 && mddfChunkOffset2 + 8 <= alphaWdtBytes.Length)
                            {
                                int sz = BitConverter.ToInt32(alphaWdtBytes, mddfChunkOffset2 + 4);
                                mddfCount2 = Math.Max(0, sz / 36);
                            }
                            if (modfChunkOffset2 >= 0 && modfChunkOffset2 + 8 <= alphaWdtBytes.Length)
                            {
                                int sz = BitConverter.ToInt32(alphaWdtBytes, modfChunkOffset2 + 4);
                                modfCount2 = Math.Max(0, sz / 64);
                            }
                            var mcin2 = new Mcin(alphaWdtBytes, mcinChunkOffset2);
                            var mcnkOffsets2 = mcin2.GetMcnkOffsets();
                            int tileX = adtNum2 % 64, tileY = adtNum2 / 64;

                            for (int ci = 0; ci < mcnkOffsets2.Count && ci < 256; ci++)
                            {
                                int pos = mcnkOffsets2[ci];
                                if (pos <= 0 || pos + 8 > alphaWdtBytes.Length) continue;
                                int headerStart = pos + 8;
                                if (headerStart + 128 > alphaWdtBytes.Length) continue;
                                int nDoodadRefs = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x14);
                                int nMapObjRefs = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x3C);
                                int offsRefs = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x24);
                                int offsLayer = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x20);
                                int offsAlpha = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x28);
                                int offsShadow = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x30);
                                int offsSnd = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x5C);
                                int offsLiquid = BitConverter.ToInt32(alphaWdtBytes, headerStart + 0x64);

                                int mcrfPayload = 0;
                                string dSamples = string.Empty, wSamples = string.Empty;
                                try
                                {
                                    // Offsets in Alpha MCNK header are relative to the beginning of the MCNK chunk (letters position).
                                    int mcrfChunkOffset2 = pos + offsRefs;
                                    if (mcrfChunkOffset2 + 8 <= alphaWdtBytes.Length && offsRefs > 0)
                                    {
                                        var mcrf2 = new Mcrf(alphaWdtBytes, mcrfChunkOffset2);
                                        mcrfPayload = mcrf2.Data.Length;
                                        var dIdx = (System.Collections.Generic.IEnumerable<int>)mcrf2.GetDoodadsIndices(Math.Max(0, nDoodadRefs)) ?? System.Linq.Enumerable.Empty<int>();
                                        var wIdx = (System.Collections.Generic.IEnumerable<int>)mcrf2.GetWmosIndices(Math.Max(0, nMapObjRefs)) ?? System.Linq.Enumerable.Empty<int>();
                                        var dTake = System.Linq.Enumerable.Take(dIdx, 4);
                                        var wTake = System.Linq.Enumerable.Take(wIdx, 4);
                                        dSamples = string.Join('|', System.Linq.Enumerable.Select(dTake, v => v.ToString(CultureInfo.InvariantCulture)));
                                        wSamples = string.Join('|', System.Linq.Enumerable.Select(wTake, v => v.ToString(CultureInfo.InvariantCulture)));
                                        if (mcrfTargets.Count > 0)
                                        {
                                            var row2 = string.Join(',', new[]
                                            {
                                                tileY.ToString(CultureInfo.InvariantCulture),
                                                tileX.ToString(CultureInfo.InvariantCulture),
                                                ci.ToString(CultureInfo.InvariantCulture),
                                                "alpha",
                                                nDoodadRefs.ToString(CultureInfo.InvariantCulture),
                                                nMapObjRefs.ToString(CultureInfo.InvariantCulture),
                                                mcrfPayload.ToString(CultureInfo.InvariantCulture),
                                                dSamples,
                                                wSamples
                                            }) + "\n";
                                            foreach (var p in mcrfTargets) { try { File.AppendAllText(p, row2); } catch { } }
                                        }
                                    }
                                }
                                catch { }

                                long expected = (long)(nDoodadRefs + nMapObjRefs) * 4L;
                                var row = string.Join(',', new[]
                                {
                                    tileY.ToString(CultureInfo.InvariantCulture),
                                    tileX.ToString(CultureInfo.InvariantCulture),
                                    ci.ToString(CultureInfo.InvariantCulture),
                                    nDoodadRefs.ToString(CultureInfo.InvariantCulture),
                                    nMapObjRefs.ToString(CultureInfo.InvariantCulture),
                                    offsRefs.ToString(CultureInfo.InvariantCulture),
                                    offsLayer.ToString(CultureInfo.InvariantCulture),
                                    offsAlpha.ToString(CultureInfo.InvariantCulture),
                                    offsShadow.ToString(CultureInfo.InvariantCulture),
                                    offsSnd.ToString(CultureInfo.InvariantCulture),
                                    offsLiquid.ToString(CultureInfo.InvariantCulture),
                                    mddfCount2.ToString(CultureInfo.InvariantCulture),
                                    modfCount2.ToString(CultureInfo.InvariantCulture),
                                    expected.ToString(CultureInfo.InvariantCulture),
                                    mcrfPayload.ToString(CultureInfo.InvariantCulture),
                                    dSamples,
                                    wSamples,
                                    string.Empty
                                }) + "\n";
                                foreach (var p in diagTargets) { try { File.AppendAllText(p, row); } catch { } }
                            }
                        }
                        catch (Exception exDiag)
                        {
                            try
                            {
                                int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                var line = $"{tileY},{tileX},-1,,,,,,,,,,,,,,,{exDiag.Message}\n";
                                foreach (var p in diagTargets) { try { File.AppendAllText(p, line); } catch { } }
                            }
                            catch { }
                        }

                        // Construct ADT object after diagnostics; capture ctor exceptions
                        AdtAlpha? a = null;
                        try
                        {
                            a = new AdtAlpha(outputPath, adtOff2, adtNum2);
                        }
                        catch (Exception exCtor)
                        {
                            try
                            {
                                int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                var line = $"{tileY},{tileX},-1,,,,,,,,,,,,,,,{exCtor.Message}\n";
                                foreach (var p in diagTargets) { try { File.AppendAllText(p, line); } catch { } }
                            }
                            catch { }
                            continue; // skip this tile
                        }

                        // Perform conversion with exception capture
                        try
                        {
                            var lk = a!.ToAdtLk(mdnmNames, monmNames);
                            lk.ToFile(lkOutDir); // Treats directory as output root
                            try
                            {
                                int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                var outFile = Path.Combine(lkOutDir, $"{mapName}_{tileX}_{tileY}.adt");
                                if (File.Exists(outFile))
                                {
                                    var lkBytes = File.ReadAllBytes(outFile);
                                    int lkMh = FindFourCCLocal(lkBytes, "MHDR");
                                    if (lkMh >= 0)
                                    {
                                        int lkMhStart = lkMh + 8;
                                        int lkMcinRel = BitConverter.ToInt32(lkBytes, lkMhStart + 0x0);
                                        int lkMcinChunk = lkMhStart + lkMcinRel;
                                        if (lkMcinChunk >= 0 && lkMcinChunk + 8 <= lkBytes.Length)
                                        {
                                            int entries = 256;
                                            for (int ci = 0; ci < entries; ci++)
                                            {
                                                int mcPos = BitConverter.ToInt32(lkBytes, lkMcinChunk + 8 + ci * 16 + 0);
                                                if (mcPos <= 0 || mcPos + 8 > lkBytes.Length) continue;
                                                int hStart = mcPos + 8;
                                                if (hStart + 128 > lkBytes.Length) continue;
                                                int dN = BitConverter.ToInt32(lkBytes, hStart + 0x14);
                                                int wN = BitConverter.ToInt32(lkBytes, hStart + 0x3C);
                                                int oRefs = BitConverter.ToInt32(lkBytes, hStart + 0x20);
                                                int mcrfPayload = 0;
                                                string dSamples = string.Empty, wSamples = string.Empty;
                                                if (oRefs > 0)
                                                {
                                                    // Offsets in LK MCNK header are relative to the beginning of the MCNK chunk (letters position).
                                                    int mcrfOff = mcPos + oRefs;
                                                    if (mcrfOff + 8 <= lkBytes.Length)
                                                    {
                                                        var mcrf3 = new Mcrf(lkBytes, mcrfOff);
                                                        mcrfPayload = mcrf3.Data.Length;
                                                        var dIdx = (System.Collections.Generic.IEnumerable<int>)mcrf3.GetDoodadsIndices(Math.Max(0, dN)) ?? System.Linq.Enumerable.Empty<int>();
                                                        var wIdx = (System.Collections.Generic.IEnumerable<int>)mcrf3.GetWmosIndices(Math.Max(0, wN)) ?? System.Linq.Enumerable.Empty<int>();
                                                        var dTake = System.Linq.Enumerable.Take(dIdx, 4);
                                                        var wTake = System.Linq.Enumerable.Take(wIdx, 4);
                                                        dSamples = string.Join('|', System.Linq.Enumerable.Select(dTake, v => v.ToString(CultureInfo.InvariantCulture)));
                                                        wSamples = string.Join('|', System.Linq.Enumerable.Select(wTake, v => v.ToString(CultureInfo.InvariantCulture)));
                                                    }
                                                }
                                                if (mcrfTargets.Count > 0)
                                                {
                                                    var row3 = string.Join(',', new[]
                                                    {
                                                        tileY.ToString(CultureInfo.InvariantCulture),
                                                        tileX.ToString(CultureInfo.InvariantCulture),
                                                        ci.ToString(CultureInfo.InvariantCulture),
                                                        "lk",
                                                        dN.ToString(CultureInfo.InvariantCulture),
                                                        wN.ToString(CultureInfo.InvariantCulture),
                                                        mcrfPayload.ToString(CultureInfo.InvariantCulture),
                                                        dSamples,
                                                        wSamples
                                                    }) + "\n";
                                                    foreach (var p in mcrfTargets) { try { File.AppendAllText(p, row3); } catch { } }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            catch { }
                        }
                        catch (Exception exConv)
                        {
                            try
                            {
                                int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                                var line = $"{tileY},{tileX},-1,,,,,,,,,,,,,,,{exConv.Message}\n";
                                foreach (var p in diagTargets) { try { File.AppendAllText(p, line); } catch { } }
                            }
                            catch { }
                            // Do not rethrow here; continue to next tile to ensure diagnostics are fully emitted
                        }

                        // Crosswalk-based AreaID patching (in-place)
                        if (patchMap != null)
                        {
                            var alphaAreaIds = a!.GetAlphaMcnkAreaIds();
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
                        swTile.Stop();
                        try
                        {
                            int tileX = adtNum2 % 64, tileY = adtNum2 / 64;
                            timingRows.Add($"{tileX},{tileY},{adtNum2},{swTile.Elapsed.TotalMilliseconds.ToString(CultureInfo.InvariantCulture)}");
                        }
                        catch { }
                        }
                        Console.WriteLine($"[ok] Exported {written}/{existingAfter.Count} LK ADTs to: {lkOutDir}");
                        try
                        {
                            var timingPath = Path.Combine(outRoot, "lk_export_timing.csv");
                            using var tw = new StreamWriter(timingPath, append: false, Encoding.UTF8);
                            tw.WriteLine("tile_x,tile_y,index,ms_total");
                            foreach (var row in timingRows) tw.WriteLine(row);
                        }
                        catch { }
                    }
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
                            File.WriteAllLines(reportPath, new[]{"file,present,mapped,patched,unmatched"}.Concat(perAdtRows.ToArray()));
                            Console.WriteLine($"[AreaIds] report: {reportPath}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[warn] LK ADT export failed: {ex.Message}");
                    try
                    {
                        string[] diagCandidatesErr = new[]
                        {
                            Path.Combine(outRoot, "lk_export_diag.csv"),
                            Path.Combine(lkOutDir, "lk_export_diag.csv"),
                            Path.Combine(Directory.GetCurrentDirectory(), "lk_export_diag.csv"),
                        };
                        foreach (var cand in diagCandidatesErr)
                        {
                            try { File.AppendAllText(cand, $"-1,-1,-1,,,,,,,,,,,,,,,{ex.Message}\n"); } catch { }
                        }
                    }
                    catch { }
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

    /// <summary>
    /// Correlate PM4 pathfinding objects with ADT MODF placements.
    /// Uses development_29_39 as the Rosetta Stone for understanding PM4↔WMO relationships.
    /// </summary>
    private static int RunPm4AdtCorrelate(Dictionary<string, string> opts)
    {
        // Required: --pm4 <path> --obj0 <path>
        // Optional: --out <dir>
        var pm4Path = opts.GetValueOrDefault("pm4", "");
        var obj0Path = opts.GetValueOrDefault("obj0", "");
        var outDir = opts.GetValueOrDefault("out", "");

        // If no paths provided, use development_29_39 test data
        if (string.IsNullOrEmpty(pm4Path) && string.IsNullOrEmpty(obj0Path))
        {
            // Try to find test data
            var testDataRoot = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "test_data");
            if (!Directory.Exists(testDataRoot))
            {
                testDataRoot = @"j:\wowDev\parp-tools\gillijimproject_refactor\test_data";
            }

            var devPath = Path.Combine(testDataRoot, "development", "World", "Maps", "development");
            if (Directory.Exists(devPath))
            {
                pm4Path = Path.Combine(devPath, "development_29_39.pm4");
                obj0Path = Path.Combine(devPath, "development_29_39_obj0.adt");
                Console.WriteLine("[INFO] Using default test data: development_29_39");
            }
            else
            {
                // Try 3.3.5 test data
                devPath = Path.Combine(testDataRoot, "3.3.5", "tree", "World", "Maps", "development");
                if (Directory.Exists(devPath))
                {
                    pm4Path = Path.Combine(devPath, "development_29_39.pm4");
                    obj0Path = Path.Combine(devPath, "development_29_39_obj0.adt");
                    Console.WriteLine("[INFO] Using 3.3.5 test data: development_29_39");
                }
            }
        }

        if (string.IsNullOrEmpty(pm4Path))
        {
            Console.Error.WriteLine("[ERROR] Missing required argument: --pm4 <path>");
            Console.Error.WriteLine("  Optional: --obj0 <path> (for full correlation)");
            Console.Error.WriteLine("  Or run without arguments to use development_29_39 test data");
            return 1;
        }

        if (!File.Exists(pm4Path))
        {
            Console.Error.WriteLine($"[ERROR] PM4 file not found: {pm4Path}");
            return 1;
        }

        // Setup output
        if (string.IsNullOrEmpty(outDir))
        {
            outDir = Path.Combine(Path.GetDirectoryName(pm4Path) ?? ".", "pm4_analysis_output");
        }
        Directory.CreateDirectory(outDir);

        var correlator = new Pm4AdtCorrelator();

        // If obj0 provided, do full correlation; otherwise PM4-only analysis
        if (!string.IsNullOrEmpty(obj0Path) && File.Exists(obj0Path))
        {
            var baseName = Path.GetFileNameWithoutExtension(pm4Path);
            var outputCsv = Path.Combine(outDir, $"{baseName}_correlation.csv");

            Console.WriteLine("=== PM4-ADT Correlator ===");
            Console.WriteLine($"PM4: {pm4Path}");
            Console.WriteLine($"_obj0.adt: {obj0Path}");
            Console.WriteLine($"Output: {outDir}");
            Console.WriteLine();

            correlator.AnalyzeTile(pm4Path, obj0Path, outputCsv);
        }
        else
        {
            Console.WriteLine("=== PM4 Analysis (no _obj0.adt) ===");
            Console.WriteLine($"PM4: {pm4Path}");
            Console.WriteLine($"Output: {outDir}");
            Console.WriteLine();

            correlator.AnalyzePm4Only(pm4Path, outDir);
        }

        return 0;
    }

    /// <summary>
    /// Extract walkable surfaces from a WMO file.
    /// </summary>
    private static int RunWmoWalkableExtract(Dictionary<string, string> opts)
    {
        var wmoPath = opts.GetValueOrDefault("wmo", "");
        var outPath = opts.GetValueOrDefault("out", "");
        var clientPath = opts.GetValueOrDefault("client-path", "");
        var v14 = opts.ContainsKey("v14");
        var exportAll = opts.ContainsKey("all");
        var exportFloors = opts.ContainsKey("floors");

        if (string.IsNullOrEmpty(wmoPath))
        {
            Console.WriteLine("Usage: wmo-walkable-extract --wmo <virtual-path> --client-path <path> [--out <obj>] [--all] [--floors]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --wmo          Virtual path to WMO (e.g., World\\wmo\\Azeroth\\Buildings\\...)");
            Console.WriteLine("  --client-path  Path to WoW client installation with MPQ files");
            Console.WriteLine("  --out          Output OBJ file (default: <wmo-name>_collision.obj)");
            Console.WriteLine("  --v14          Parse as v14 monolithic format (for alpha clients)");
            Console.WriteLine("  --all          Export ALL triangles (full render geometry)");
            Console.WriteLine("  --floors       Export upward-facing render faces (walkable floors)");
            return 1;
        }

        Console.WriteLine("=== WMO Walkable Surface Extractor ===");
        Console.WriteLine($"WMO: {wmoPath}");
        Console.WriteLine($"Client: {clientPath}");
        Console.WriteLine();

        var extractor = new WmoWalkableSurfaceExtractor();
        WmoWalkableSurfaceExtractor.WmoWalkableData data;

        // If client-path provided, load from MPQ
        if (!string.IsNullOrEmpty(clientPath))
        {
            if (!Directory.Exists(clientPath))
            {
                Console.Error.WriteLine($"[ERROR] Client path not found: {clientPath}");
                return 1;
            }

            EnsureStormLibOnPath();

            // Locate MPQs
            var mpqs = ArchiveLocator.LocateMpqs(clientPath);
            Console.WriteLine($"[INFO] Found {mpqs.Count} MPQ archives");

            if (mpqs.Count == 0)
            {
                Console.Error.WriteLine("[ERROR] No MPQ archives found");
                return 1;
            }

            using var src = new PrioritizedArchiveSource(clientPath, mpqs);

            // Check if WMO exists
            if (!src.FileExists(wmoPath))
            {
                Console.Error.WriteLine($"[ERROR] WMO not found in archives: {wmoPath}");
                return 1;
            }

            // Load root WMO
            byte[] rootData;
            using (var stream = src.OpenFile(wmoPath))
            using (var ms = new MemoryStream())
            {
                stream.CopyTo(ms);
                rootData = ms.ToArray();
            }
            Console.WriteLine($"[INFO] Loaded root WMO: {rootData.Length} bytes");

            // Group loader function
            byte[]? LoadGroup(string groupPath)
            {
                if (!src.FileExists(groupPath))
                    return null;
                using var stream = src.OpenFile(groupPath);
                using var ms = new MemoryStream();
                stream.CopyTo(ms);
                return ms.ToArray();
            }

            data = extractor.ExtractFromBytes(rootData, wmoPath, LoadGroup);
        }
        else if (File.Exists(wmoPath))
        {
            // Load from local file
            if (v14)
            {
                data = extractor.ExtractFromWmoV14(wmoPath);
            }
            else
            {
                data = extractor.ExtractFromWmoV17(wmoPath);
            }
        }
        else
        {
            Console.Error.WriteLine($"[ERROR] WMO not found: {wmoPath}");
            Console.Error.WriteLine("Provide --client-path to load from MPQ archives");
            return 1;
        }

        // Set output path
        var wmoName = Path.GetFileNameWithoutExtension(wmoPath);
        if (string.IsNullOrEmpty(outPath))
        {
            outPath = exportAll ? $"{wmoName}_full.obj" : 
                      exportFloors ? $"{wmoName}_floors.obj" : 
                      $"{wmoName}_collision.obj";
        }

        Console.WriteLine();
        Console.WriteLine($"[RESULT] Extraction complete:");
        Console.WriteLine($"  Total triangles: {data.AllTriangles.Count}");
        Console.WriteLine($"  Collision triangles: {data.WalkableTriangles.Count}");
        
        if (data.WalkableVertices.Count > 0)
        {
            Console.WriteLine($"  Bounds: ({data.BoundsMin.X:F1}, {data.BoundsMin.Y:F1}, {data.BoundsMin.Z:F1}) to ({data.BoundsMax.X:F1}, {data.BoundsMax.Y:F1}, {data.BoundsMax.Z:F1})");
        }

        // Export based on mode
        if (exportAll)
        {
            extractor.ExportAllToObj(data, outPath);
        }
        else if (exportFloors)
        {
            extractor.ExportWalkableFloorsToObj(data, outPath);
        }
        else if (data.WalkableTriangles.Count > 0)
        {
            extractor.ExportToObj(data, outPath);
        }
        else
        {
            Console.WriteLine("[WARN] No collision surfaces found to export");
        }

        return 0;
    }

    /// <summary>
    /// Export PM4 pathfinding geometry to OBJ.
    /// </summary>
    private static int RunPm4ExportObj(Dictionary<string, string> opts)
    {
        var pm4Path = opts.GetValueOrDefault("pm4", "");
        var outPath = opts.GetValueOrDefault("out", "");

        if (string.IsNullOrEmpty(pm4Path))
        {
            Console.WriteLine("Usage: pm4-export-obj --pm4 <path> [--out <obj>]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --pm4    Path to PM4 file");
            Console.WriteLine("  --out    Output OBJ file (default: <pm4-name>.obj)");
            return 1;
        }

        if (!File.Exists(pm4Path))
        {
            Console.Error.WriteLine($"[ERROR] PM4 file not found: {pm4Path}");
            return 1;
        }

        if (string.IsNullOrEmpty(outPath))
        {
            outPath = Path.ChangeExtension(pm4Path, ".obj");
        }

        Console.WriteLine("=== PM4 Geometry Exporter ===");
        Console.WriteLine($"PM4: {pm4Path}");
        Console.WriteLine($"Output: {outPath}");
        Console.WriteLine();

        var exporter = new Pm4GeometryExporter();
        exporter.ExportToObj(pm4Path, outPath);

        return 0;
    }

    /// <summary>
    /// Match PM4 geometry to WMO geometry to derive placement transform.
    /// </summary>
    private static int RunPm4WmoMatch(Dictionary<string, string> opts)
    {
        var pm4Obj = opts.GetValueOrDefault("pm4-obj", "");
        var wmoObj = opts.GetValueOrDefault("wmo-obj", "");
        var outPath = opts.GetValueOrDefault("out", "");

        if (string.IsNullOrEmpty(pm4Obj) || string.IsNullOrEmpty(wmoObj))
        {
            Console.WriteLine("Usage: pm4-wmo-match --pm4-obj <path> --wmo-obj <path> [--out <transformed.obj>]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --pm4-obj    Path to PM4 geometry OBJ (in world space)");
            Console.WriteLine("  --wmo-obj    Path to WMO geometry OBJ (in local space)");
            Console.WriteLine("  --out        Output transformed WMO OBJ (optional)");
            Console.WriteLine();
            Console.WriteLine("This tool analyzes both geometries to derive the placement transform");
            Console.WriteLine("(position, rotation, scale) that was applied to place the WMO in the world.");
            return 1;
        }

        if (!File.Exists(pm4Obj))
        {
            Console.Error.WriteLine($"[ERROR] PM4 OBJ not found: {pm4Obj}");
            return 1;
        }

        if (!File.Exists(wmoObj))
        {
            Console.Error.WriteLine($"[ERROR] WMO OBJ not found: {wmoObj}");
            return 1;
        }

        Console.WriteLine("=== PM4-WMO Geometry Matcher ===");
        Console.WriteLine($"PM4: {pm4Obj}");
        Console.WriteLine($"WMO: {wmoObj}");
        Console.WriteLine();

        var matcher = new Pm4WmoGeometryMatcher();
        var transform = matcher.AnalyzeAndAlign(pm4Obj, wmoObj, string.IsNullOrEmpty(outPath) ? null : outPath);

        Console.WriteLine("\n=== Derived MODF Placement Data ===");
        Console.WriteLine($"Position: ({transform.Position.X:F2}, {transform.Position.Y:F2}, {transform.Position.Z:F2})");
        Console.WriteLine($"Rotation: ({transform.Rotation.X:F2}°, {transform.Rotation.Y:F2}°, {transform.Rotation.Z:F2}°)");
        Console.WriteLine($"Scale: {transform.Scale:F4}");
        Console.WriteLine($"Confidence: {transform.MatchConfidence:P1}");

        return 0;
    }

    /// <summary>
    /// Reconstruct MODF placement data from PM4 objects by matching against WMO library.
    /// </summary>
    private static int RunPm4ReconstructModf(Dictionary<string, string> opts)
    {
        var pm4Dir = opts.GetValueOrDefault("pm4-dir", "");
        var wmoDir = opts.GetValueOrDefault("wmo-dir", "");
        var outDir = opts.GetValueOrDefault("out", "");
        var minConfStr = opts.GetValueOrDefault("min-confidence", "0.7");

        if (string.IsNullOrEmpty(pm4Dir) || string.IsNullOrEmpty(wmoDir))
        {
            Console.WriteLine("Usage: pm4-reconstruct-modf --pm4-dir <path> --wmo-dir <path> [--out <dir>] [--min-confidence <0.7>]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --pm4-dir         Path to PM4FacesTool output directory");
            Console.WriteLine("  --wmo-dir         Path to WMO collision geometry directory");
            Console.WriteLine("  --out             Output directory for MODF reconstruction");
            Console.WriteLine("  --min-confidence  Minimum match confidence (0.0-1.0, default 0.7)");
            Console.WriteLine();
            Console.WriteLine("This tool matches PM4 pathfinding objects against known WMOs to");
            Console.WriteLine("reconstruct MODF placement data for ADT generation.");
            return 1;
        }

        if (!Directory.Exists(pm4Dir))
        {
            Console.Error.WriteLine($"[ERROR] PM4 directory not found: {pm4Dir}");
            return 1;
        }

        if (!Directory.Exists(wmoDir))
        {
            Console.Error.WriteLine($"[ERROR] WMO directory not found: {wmoDir}");
            return 1;
        }

        if (string.IsNullOrEmpty(outDir))
        {
            outDir = Path.Combine(pm4Dir, "modf_reconstruction");
        }
        Directory.CreateDirectory(outDir);

        float minConfidence = 0.7f;
        float.TryParse(minConfStr, out minConfidence);

        Console.WriteLine("=== PM4 MODF Reconstructor ===");
        Console.WriteLine($"PM4 Dir: {pm4Dir}");
        Console.WriteLine($"WMO Dir: {wmoDir}");
        Console.WriteLine($"Output: {outDir}");
        Console.WriteLine($"Min Confidence: {minConfidence:P0}");
        Console.WriteLine();

        var reconstructor = new Pm4ModfReconstructor();

        // Build WMO library
        var wmoLibrary = reconstructor.BuildWmoLibrary(wmoDir);
        if (wmoLibrary.Count == 0)
        {
            Console.Error.WriteLine("[ERROR] No WMOs found in library");
            return 1;
        }

        // Reconstruct MODF
        var result = reconstructor.ReconstructModf(pm4Dir, wmoLibrary, minConfidence);

        // Apply Coordinate Transform (PM4 -> ADT World)
        Console.WriteLine("[INFO] Applying PM4->ADT coordinate transform (ServerToAdtPosition)...");
        var transformedEntries = result.ModfEntries.Select(e => e with 
        { 
            Position = AdtModfInjector.ServerToAdtPosition(e.Position) 
        }).ToList();
        
        result = result with { ModfEntries = transformedEntries };

        // Export results
        reconstructor.ExportToCsv(result, Path.Combine(outDir, "modf_entries.csv"));
        reconstructor.ExportMwmo(result, Path.Combine(outDir, "mwmo_names.csv"));

        // Export unmatched objects
        if (result.UnmatchedPm4Objects.Count > 0)
        {
            File.WriteAllLines(
                Path.Combine(outDir, "unmatched_objects.txt"),
                result.UnmatchedPm4Objects);
            Console.WriteLine($"[INFO] {result.UnmatchedPm4Objects.Count} unmatched objects written to unmatched_objects.txt");
        }

        Console.WriteLine();
        Console.WriteLine("=== Summary ===");
        Console.WriteLine($"Total MODF entries: {result.ModfEntries.Count}");
        Console.WriteLine($"Unique WMOs: {result.WmoNames.Count}");
        Console.WriteLine($"Unmatched objects: {result.UnmatchedPm4Objects.Count}");

        if (result.MatchCounts.Count > 0)
        {
            Console.WriteLine("\nTop WMO matches:");
            foreach (var kv in result.MatchCounts.OrderByDescending(x => x.Value).Take(10))
            {
                Console.WriteLine($"  {Path.GetFileName(kv.Key)}: {kv.Value} placements");
            }
        }

        return 0;
    }

    /// <summary>
    /// Batch extract WMO collision geometry from MPQ archives.
    /// </summary>
    private static int RunWmoBatchExtract(Dictionary<string, string> opts)
    {
        var clientPath = opts.GetValueOrDefault("client-path", "");
        var outDir = opts.GetValueOrDefault("out", "");
        var pattern = opts.GetValueOrDefault("pattern", "");
        var limitStr = opts.GetValueOrDefault("limit", "0");

        if (string.IsNullOrEmpty(clientPath))
        {
            Console.WriteLine("Usage: wmo-batch-extract --client-path <path> [--out <dir>] [--pattern <filter>] [--limit <n>]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --client-path   Path to WoW client (with MPQ files)");
            Console.WriteLine("  --out           Output directory for collision OBJs");
            Console.WriteLine("  --pattern       Filter WMO paths (e.g., 'stormwind')");
            Console.WriteLine("  --limit         Max WMOs to extract (0 = unlimited)");
            Console.WriteLine();
            Console.WriteLine("Extracts collision geometry from all WMOs in the client to build");
            Console.WriteLine("a reference library for PM4-to-MODF reconstruction.");
            return 1;
        }

        if (!Directory.Exists(clientPath))
        {
            Console.Error.WriteLine($"[ERROR] Client path not found: {clientPath}");
            return 1;
        }

        if (string.IsNullOrEmpty(outDir))
        {
            outDir = Path.Combine(clientPath, "wmo_collision_library");
        }
        Directory.CreateDirectory(outDir);

        int limit = 0;
        int.TryParse(limitStr, out limit);

        Console.WriteLine("=== WMO Batch Extractor ===");
        Console.WriteLine($"Client: {clientPath}");
        Console.WriteLine($"Output: {outDir}");
        Console.WriteLine($"Pattern: {(string.IsNullOrEmpty(pattern) ? "(all)" : pattern)}");
        Console.WriteLine($"Limit: {(limit == 0 ? "unlimited" : limit.ToString())}");
        Console.WriteLine();

        EnsureStormLibOnPath();

        // Locate MPQs
        var mpqs = ArchiveLocator.LocateMpqs(clientPath);
        Console.WriteLine($"[INFO] Found {mpqs.Count} MPQ archives");

        if (mpqs.Count == 0)
        {
            Console.Error.WriteLine("[ERROR] No MPQ archives found");
            return 1;
        }

        using var archiveSource = new PrioritizedArchiveSource(clientPath, mpqs);

        // Find all WMO root files
        var wmoFiles = new List<string>();
        
        // Scan common WMO paths
        var searchPaths = new[]
        {
            "World\\wmo\\",
            "World\\WMO\\"
        };

        Console.WriteLine("[INFO] Scanning for WMO files...");

        // Check for listfile in multiple locations
        var listfilePaths = new[]
        {
            opts.GetValueOrDefault("listfile", ""),
            Path.Combine(clientPath, "listfile.txt"),
            @"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\World of Warcraft 3x.txt",
            @"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\wmo_roots_3x.txt"
        };

        string? listfilePath = listfilePaths.FirstOrDefault(p => !string.IsNullOrEmpty(p) && File.Exists(p));

        if (!string.IsNullOrEmpty(listfilePath))
        {
            Console.WriteLine($"[INFO] Using listfile: {listfilePath}");
            foreach (var line in File.ReadLines(listfilePath))
            {
                var path = line.Trim();
                // Skip group files (e.g., _000.wmo, _001.wmo)
                if (!path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (System.Text.RegularExpressions.Regex.IsMatch(path, @"_\d{3}\.wmo$", System.Text.RegularExpressions.RegexOptions.IgnoreCase))
                    continue;
                    
                if (string.IsNullOrEmpty(pattern) || path.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                {
                    if (archiveSource.FileExists(path))
                        wmoFiles.Add(path);
                }
            }
        }
        else
        {
            Console.WriteLine("[WARN] No listfile found. Use --listfile to specify one.");
            Console.WriteLine("[WARN] Expected: 3.x listfile from test_data/World of Warcraft 3x.txt");
        }

        Console.WriteLine($"[INFO] Found {wmoFiles.Count} WMO root files");

        if (limit > 0 && wmoFiles.Count > limit)
        {
            wmoFiles = wmoFiles.Take(limit).ToList();
            Console.WriteLine($"[INFO] Limited to {limit} WMOs");
        }

        var extractor = new WmoWalkableSurfaceExtractor();
        int success = 0;
        int failed = 0;

        foreach (var wmoPath in wmoFiles)
        {
            try
            {
                var wmoName = Path.GetFileNameWithoutExtension(wmoPath);
                var outputPath = Path.Combine(outDir, $"{wmoName}_collision.obj");

                if (File.Exists(outputPath))
                {
                    Console.WriteLine($"  [SKIP] {wmoName} (already exists)");
                    success++;
                    continue;
                }

                Console.Write($"  Extracting {wmoName}...");

                // Get group count from root file
                using var rootStream = archiveSource.OpenFile(wmoPath);
                var rootBytes = new byte[rootStream.Length];
                rootStream.Read(rootBytes, 0, rootBytes.Length);

                // Parse root to get group count
                int groupCount = ParseWmoGroupCount(rootBytes);
                if (groupCount == 0)
                {
                    Console.WriteLine(" [WARN] No groups found");
                    failed++;
                    continue;
                }

                // Create group loader function
                var basePath = wmoPath.Substring(0, wmoPath.Length - 4); // Remove .wmo
                var archiveRef = archiveSource; // Capture for closure
                
                Func<string, byte[]?> groupLoader = (groupPath) =>
                {
                    if (archiveRef.FileExists(groupPath))
                    {
                        using var groupStream = archiveRef.OpenFile(groupPath);
                        var groupBytes = new byte[groupStream.Length];
                        groupStream.Read(groupBytes, 0, groupBytes.Length);
                        return groupBytes;
                    }
                    return null;
                };

                // Extract collision geometry
                var result = extractor.ExtractFromBytes(rootBytes, wmoPath, groupLoader);
                
                if (result.AllTriangles.Count == 0)
                {
                    Console.WriteLine(" [WARN] No triangles extracted");
                    failed++;
                    continue;
                }
                
                extractor.ExportToObj(result, outputPath);

                Console.WriteLine($" OK ({result.AllTriangles.Count} faces)");
                success++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($" [ERROR] {ex.Message}");
                failed++;
            }
        }

        Console.WriteLine();
        Console.WriteLine($"=== Complete ===");
        Console.WriteLine($"Success: {success}");
        Console.WriteLine($"Failed: {failed}");

        return 0;
    }

    /// <summary>
    /// Parse WMO root file to get group count from MOHD chunk.
    /// v17 MOHD structure: nTextures (4), nGroups (4), nPortals (4), ...
    /// </summary>
    private static int ParseWmoGroupCount(byte[] data)
    {
        // Scan for MOHD chunk (forward or reversed)
        for (int i = 0; i < data.Length - 12; i++)
        {
            // Check for "MOHD" (forward)
            if (data[i] == 'M' && data[i + 1] == 'O' && data[i + 2] == 'H' && data[i + 3] == 'D')
            {
                // v17: nTextures at +0, nGroups at +4
                if (i + 8 + 8 <= data.Length)
                {
                    return BitConverter.ToInt32(data, i + 8 + 4); // nGroups at offset 4 in MOHD data
                }
            }
            // Check for "DHOM" (reversed, little-endian storage)
            if (data[i] == 'D' && data[i + 1] == 'H' && data[i + 2] == 'O' && data[i + 3] == 'M')
            {
                if (i + 8 + 8 <= data.Length)
                {
                    return BitConverter.ToInt32(data, i + 8 + 4);
                }
            }
        }

        return 0;
    }

    /// <summary>
    /// Export MODF binary chunks from reconstruction results.
    /// </summary>
    private static int RunPm4ExportModf(Dictionary<string, string> opts)
    {
        var modfCsv = opts.GetValueOrDefault("modf-csv", "");
        var mwmoCsv = opts.GetValueOrDefault("mwmo-csv", "");
        var outDir = opts.GetValueOrDefault("out", "");

        if (string.IsNullOrEmpty(modfCsv))
        {
            Console.WriteLine("Usage: pm4-export-modf --modf-csv <path> [--mwmo-csv <path>] [--out <dir>]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --modf-csv    Path to modf_entries.csv from pm4-reconstruct-modf");
            Console.WriteLine("  --mwmo-csv    Path to mwmo_names.csv (optional, derived from modf-csv)");
            Console.WriteLine("  --out         Output directory for binary chunks");
            Console.WriteLine();
            Console.WriteLine("Exports MWMO and MODF binary chunks for ADT reconstruction.");
            return 1;
        }

        if (!File.Exists(modfCsv))
        {
            Console.Error.WriteLine($"[ERROR] MODF CSV not found: {modfCsv}");
            return 1;
        }

        if (string.IsNullOrEmpty(mwmoCsv))
        {
            mwmoCsv = Path.Combine(Path.GetDirectoryName(modfCsv)!, "mwmo_names.csv");
        }

        if (string.IsNullOrEmpty(outDir))
        {
            outDir = Path.Combine(Path.GetDirectoryName(modfCsv)!, "modf_binary");
        }
        Directory.CreateDirectory(outDir);

        Console.WriteLine("=== PM4 MODF Binary Exporter ===");
        Console.WriteLine($"MODF CSV: {modfCsv}");
        Console.WriteLine($"MWMO CSV: {mwmoCsv}");
        Console.WriteLine($"Output: {outDir}");
        Console.WriteLine();

        // Load MODF entries from CSV
        var modfEntries = new List<Pm4ModfReconstructor.ModfEntry>();
        var wmoNames = new List<string>();
        var wmoNameSet = new HashSet<string>();

        foreach (var line in File.ReadLines(modfCsv).Skip(1)) // Skip header
        {
            var parts = line.Split(',');
            if (parts.Length < 12) continue;

            var entry = new Pm4ModfReconstructor.ModfEntry(
                NameId: uint.Parse(parts[2]),
                UniqueId: uint.Parse(parts[3]),
                Position: new Vector3(float.Parse(parts[4]), float.Parse(parts[5]), float.Parse(parts[6])),
                Rotation: new Vector3(float.Parse(parts[7]), float.Parse(parts[8]), float.Parse(parts[9])),
                BoundsMin: Vector3.Zero, // Will be computed
                BoundsMax: Vector3.Zero,
                Flags: 0,
                DoodadSet: 0,
                NameSet: 0,
                Scale: 1024,
                WmoPath: parts[1],
                Ck24: parts[0],
                MatchConfidence: float.Parse(parts[11])
            );

            modfEntries.Add(entry);

            if (!wmoNameSet.Contains(entry.WmoPath))
            {
                wmoNameSet.Add(entry.WmoPath);
                wmoNames.Add(entry.WmoPath);
            }
        }

        Console.WriteLine($"[INFO] Loaded {modfEntries.Count} MODF entries, {wmoNames.Count} unique WMOs");

        // Create reconstruction result for the writer
        var result = new Pm4ModfReconstructor.ReconstructionResult(
            modfEntries,
            wmoNames,
            new List<string>(),
            new Dictionary<string, int>()
        );

        // Export per-tile
        var writer = new ModfChunkWriter();
        writer.ExportPerTile(result, outDir);

        return 0;
    }

    /// <summary>
    /// Create 3.3.5 ADT files with MODF data from reconstruction results.
    /// </summary>
    private static int RunPm4CreateAdt(Dictionary<string, string> opts)
    {
        var modfBinaryDir = opts.GetValueOrDefault("modf-dir", "");
        var outDir = opts.GetValueOrDefault("out", "");
        var mapName = opts.GetValueOrDefault("map", "development");
        var tileFilter = opts.GetValueOrDefault("tile", "");
        var sourceAdtDir = opts.GetValueOrDefault("source-adt", "");

        if (string.IsNullOrEmpty(modfBinaryDir))
        {
            Console.WriteLine("Usage: pm4-create-adt --modf-dir <path> [--source-adt <dir>] [--out <dir>] [--map <name>] [--tile X_Y]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --modf-dir    Path to modf_binary output from pm4-export-modf");
            Console.WriteLine("  --source-adt  Path to source split ADT files (root + _obj0 + _tex0)");
            Console.WriteLine("  --out         Output directory for ADT files");
            Console.WriteLine("  --map         Map name (default: development)");
            Console.WriteLine("  --tile        Filter to specific tile (e.g., 15_36)");
            Console.WriteLine();
            Console.WriteLine("Creates 3.3.5 monolithic ADT files with WMO placements.");
            Console.WriteLine("If --source-adt is provided, merges split ADTs and patches MODF.");
            return 1;
        }

        if (!Directory.Exists(modfBinaryDir))
        {
            Console.Error.WriteLine($"[ERROR] MODF binary directory not found: {modfBinaryDir}");
            return 1;
        }

        if (string.IsNullOrEmpty(outDir))
        {
            outDir = Path.Combine(modfBinaryDir, "..", "adt_335");
        }
        Directory.CreateDirectory(outDir);

        Console.WriteLine("=== PM4 ADT Creator (3.3.5 Format) ===");
        Console.WriteLine($"MODF Dir: {modfBinaryDir}");
        if (!string.IsNullOrEmpty(sourceAdtDir))
            Console.WriteLine($"Source ADT: {sourceAdtDir}");
        Console.WriteLine($"Output: {outDir}");
        Console.WriteLine($"Map: {mapName}");
        if (!string.IsNullOrEmpty(tileFilter))
            Console.WriteLine($"Tile filter: {tileFilter}");
        Console.WriteLine();

        var injector = new AdtModfInjector();
        var merger = new SplitAdtMerger();
        int created = 0;
        int merged = 0;

        // Find all tile directories
        var tileDirs = Directory.GetDirectories(modfBinaryDir, "tile_*");
        
        foreach (var tileDir in tileDirs)
        {
            var dirName = Path.GetFileName(tileDir);
            var parts = dirName.Replace("tile_", "").Split('_');
            if (parts.Length != 2) continue;

            if (!int.TryParse(parts[0], out int tileX) || !int.TryParse(parts[1], out int tileY))
                continue;

            // Apply tile filter if specified
            if (!string.IsNullOrEmpty(tileFilter) && $"{tileX}_{tileY}" != tileFilter)
                continue;

            var mwmoPath = Path.Combine(tileDir, "MWMO.bin");
            var modfPath = Path.Combine(tileDir, "MODF.bin");

            if (!File.Exists(mwmoPath) || !File.Exists(modfPath))
            {
                Console.WriteLine($"  [SKIP] Tile {tileX}_{tileY}: Missing MWMO or MODF");
                continue;
            }

            Console.Write($"  Creating {mapName}_{tileX}_{tileY}.adt...");

            try
            {
                // Read MWMO data and extract WMO names
                var mwmoData = File.ReadAllBytes(mwmoPath);
                var wmoNames = ParseMwmoChunk(mwmoData);

                // Read MODF data and parse entries
                var modfData = File.ReadAllBytes(modfPath);
                var modfEntries = ParseModfChunk(modfData, wmoNames, tileX, tileY);

                // Convert to AdtLkFactory.ModfEntry format
                var factoryEntries = modfEntries.Select(e => new AdtLkFactory.ModfEntry
                {
                    NameId = e.NameId,
                    UniqueId = e.UniqueId,
                    Position = e.Position,
                    Rotation = e.Rotation,
                    BoundsMin = e.BoundsMin,
                    BoundsMax = e.BoundsMax,
                    Flags = e.Flags,
                    DoodadSet = e.DoodadSet,
                    NameSet = e.NameSet,
                    Scale = e.Scale
                }).ToList();

                // Create proper LK ADT using the factory
                var adtName = $"{mapName}_{tileX}_{tileY}.adt";
                var adt = AdtLkFactory.CreateMinimalAdt(adtName, tileX, tileY, wmoNames, factoryEntries);

                // Write ADT file using the proper LK writer
                var adtPath = Path.Combine(outDir, adtName);
                adt.ToFile(adtPath);

                // Get file size for reporting
                var fileInfo = new FileInfo(adtPath);
                Console.WriteLine($" OK ({factoryEntries.Count} WMOs, {fileInfo.Length} bytes)");
                created++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($" [ERROR] {ex.Message}");
            }
        }

        // Create WDT file
        if (created > 0)
        {
            Console.WriteLine();
            Console.WriteLine("[INFO] Creating WDT file...");
            var wdtWriter = new Wdt335Writer();
            wdtWriter.WriteWdtFromAdtFiles(outDir, mapName);
        }

        Console.WriteLine();
        Console.WriteLine($"=== Complete: {created} ADT files created ===");

        return 0;
    }

    private static List<string> ParseMwmoChunk(byte[] data)
    {
        var names = new List<string>();
        
        // Skip chunk header (MWMO + size = 8 bytes)
        if (data.Length < 8) return names;
        
        int pos = 8;
        int dataSize = BitConverter.ToInt32(data, 4);
        int end = Math.Min(8 + dataSize, data.Length);

        while (pos < end)
        {
            int start = pos;
            while (pos < end && data[pos] != 0) pos++;
            
            if (pos > start)
            {
                names.Add(Encoding.ASCII.GetString(data, start, pos - start));
            }
            pos++; // Skip null terminator
        }

        return names;
    }

    private static List<AdtModfInjector.ModfEntry335> ParseModfChunk(byte[] data, List<string> wmoNames, int tileX, int tileY)
    {
        var entries = new List<AdtModfInjector.ModfEntry335>();
        
        // Skip chunk header (MODF + size = 8 bytes)
        if (data.Length < 8) return entries;
        
        int dataSize = BitConverter.ToInt32(data, 4);
        int entryCount = dataSize / 64;
        int pos = 8;

        // Generate unique IDs starting from a high base to avoid conflicts
        // Format: 0xPPXXYYII where PP=0x80 (PM4 marker), XX=tileX, YY=tileY, II=index
        uint baseUniqueId = 0x80000000u | ((uint)tileX << 16) | ((uint)tileY << 8);

        for (int i = 0; i < entryCount && pos + 64 <= data.Length; i++)
        {
            // Read the entry from our binary format
            uint nameOffset = BitConverter.ToUInt32(data, pos);
            // Generate globally unique ID based on tile + index
            uint uniqueId = baseUniqueId | (uint)i;
            
            // Position is in server coordinates - convert to ADT world coordinates
            var serverPos = new Vector3(
                BitConverter.ToSingle(data, pos + 8),
                BitConverter.ToSingle(data, pos + 12),
                BitConverter.ToSingle(data, pos + 16)
            );
            var worldPos = AdtModfInjector.ServerToAdtPosition(serverPos);
            
            var rotation = new Vector3(
                BitConverter.ToSingle(data, pos + 20),
                BitConverter.ToSingle(data, pos + 24),
                BitConverter.ToSingle(data, pos + 28)
            );

            // Find the WMO name index from offset
            uint nameId = 0;
            uint currentOffset = 0;
            for (int j = 0; j < wmoNames.Count; j++)
            {
                if (currentOffset == nameOffset)
                {
                    nameId = (uint)j;
                    break;
                }
                currentOffset += (uint)(Encoding.ASCII.GetByteCount(wmoNames[j]) + 1);
            }

            // Compute bounding box (approximate from WMO size - would need actual WMO data for accuracy)
            var boundsMin = worldPos - new Vector3(50, 50, 50);
            var boundsMax = worldPos + new Vector3(50, 50, 50);

            entries.Add(new AdtModfInjector.ModfEntry335
            {
                NameId = nameId, // Index into MWID (not byte offset!)
                UniqueId = uniqueId,
                Position = worldPos,
                Rotation = rotation,
                BoundsMin = boundsMin,
                BoundsMax = boundsMax,
                Flags = 0,
                DoodadSet = 0,
                NameSet = 0,
                Scale = 1024 // Always 1.0 for WMOs
            });

            pos += 64;
        }

        return entries;
    }
}
