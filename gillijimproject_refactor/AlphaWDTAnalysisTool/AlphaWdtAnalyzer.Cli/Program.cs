using System;
using System.IO;
using System.Text;
using AlphaWdtAnalyzer.Core;
using AlphaWdtAnalyzer.Core.Export;

namespace AlphaWdtAnalyzer.Cli;

public static class Program
{
    // Minimal console tee to mirror stdout/stderr to a log file (AWDT only)
    private sealed class TeeTextWriter : TextWriter
    {
        private readonly TextWriter _primary;
        private readonly StreamWriter _file;
        public TeeTextWriter(TextWriter primary, string filePath)
        {
            _primary = primary;
            Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
            _file = new StreamWriter(filePath, append: false, Encoding.UTF8) { AutoFlush = true };
        }
        public override Encoding Encoding => Encoding.UTF8;
        public override void Write(char value) { _primary.Write(value); _file.Write(value); }
        public override void Write(string? value) { _primary.Write(value); _file.Write(value); }
        public override void WriteLine(string? value) { _primary.WriteLine(value); _file.WriteLine(value); }
        protected override void Dispose(bool disposing)
        {
            if (disposing) { try { _file.Flush(); _file.Dispose(); } catch { } }
            base.Dispose(disposing);
        }
    }

    private sealed class ConsoleTee : IDisposable
    {
        private readonly TextWriter _prevOut;
        private readonly TextWriter _prevErr;
        private readonly TeeTextWriter _teeOut;
        private readonly TeeTextWriter _teeErr;
        private ConsoleTee(TextWriter prevOut, TextWriter prevErr, TeeTextWriter teeOut, TeeTextWriter teeErr)
        { _prevOut = prevOut; _prevErr = prevErr; _teeOut = teeOut; _teeErr = teeErr; }
        public static ConsoleTee Start(string logPath)
        {
            var prevOut = Console.Out; var prevErr = Console.Error;
            var teeOut = new TeeTextWriter(prevOut, logPath);
            var teeErr = new TeeTextWriter(prevErr, logPath);
            Console.SetOut(teeOut); Console.SetError(teeErr);
            Console.WriteLine($"[Log] Mirroring console to {logPath}");
            return new ConsoleTee(prevOut, prevErr, teeOut, teeErr);
        }
        public void Dispose()
        {
            try { Console.SetOut(_prevOut); Console.SetError(_prevErr); } catch { }
            try { _teeOut.Dispose(); _teeErr.Dispose(); } catch { }
        }
    }

    private static int Usage()
    {
        Console.WriteLine("AlphaWdtAnalyzer");
        Console.WriteLine("Usage:");
        Console.WriteLine("  Single map: AlphaWdtAnalyzer --input <path/to/map.wdt> --listfile <community_listfile.csv> [--lk-listfile <3x.txt>] --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--web] [--extract-mcnk-terrain] [--extract-mcnk-shadows] [--export-adt --export-dir <dir> [--fallback-tileset <blp>] [--fallback-wmo <wmo>] [--fallback-m2 <m2>] [--fallback-blp <blp>] [--no-mh2o] [--asset-fuzzy on|off] [--profile preserve|modified] [--no-fallbacks] [--no-fixups] [--remap <remap.json>] [--dbd-dir <dir>] [--dbctool-out-root <dir>] [--dbctool-src-alias <053|055|060>] [--dbctool-src-dir <dir>] [--dbctool-lk-dir <dir>] [--dbctool-patch-dir <dir>] [--dbctool-patch-file <file>] [--patch-only] [--no-zone-fallback] [--viz-svg] [--viz-html] [--viz-dir <dir>] [--verbose] [--track-assets]]");
        Console.WriteLine("  Batch maps:  AlphaWdtAnalyzer --input-dir <root_of_wdts> --listfile <community_listfile.csv> [--lk-listfile <3x.txt>] --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--web] [--extract-mcnk-terrain] [--extract-mcnk-shadows] [--export-adt --export-dir <dir> [--fallback-tileset <blp>] [--fallback-wmo <wmo>] [--fallback-m2 <m2>] [--fallback-blp <blp>] [--no-mh2o] [--asset-fuzzy on|off] [--profile preserve|modified] [--no-fallbacks] [--no-fixups] [--remap <remap.json>] [--dbd-dir <dir>] [--dbctool-out-root <dir>] [--dbctool-src-alias <053|055|060>] [--dbctool-src-dir <dir>] [--dbctool-lk-dir <dir>] [--dbctool-patch-dir <dir>] [--dbctool-patch-file <file>] [--patch-only] [--no-zone-fallback] [--viz-svg] [--viz-html] [--viz-dir <dir>] [--verbose] [--track-assets]]");
        Console.WriteLine("  Count tiles: AlphaWdtAnalyzer --count-tiles --input <path/to/map.wdt>");
        Console.WriteLine("");
        Console.WriteLine("New Terrain Extraction Flags:");
        Console.WriteLine("  --extract-mcnk-terrain   Extract complete MCNK terrain data to CSV (all flags, liquids, holes, AreaID)");
        Console.WriteLine("  --extract-mcnk-shadows   Extract MCSH shadow map bitmaps to CSV (64×64 bitmaps per chunk)");
        Console.WriteLine("  --count-tiles            Print number of ADT tiles and exit (for cache validation)");
        return 2;
    }

    public static int Main(string[] args)
    {
        string? wdt = null;
        string? inputDir = null;
        string? listfile = null;
        string? lkListfile = null;
        string? outDir = null;
        bool web = false; // default off
        int? clusterThreshold = null;
        int? clusterGap = null;
        // export flags
        bool exportAdt = false;
        string? exportDir = null;
        string fallbackTileset = @"Tileset\Generic\Checkers.blp";
        string fallbackWmo = @"wmo\Dungeon\test\missingwmo.wmo";
        string fallbackM2 = @"World\Scale\HumanMaleScale.mdx";
        string fallbackNonTilesetBlp = @"Dungeons\Textures\temp\64.blp";
        bool mh2o = true;
        bool assetFuzzy = true;
        // profiles/toggles
        string profile = "modified"; // modified|preserve
        bool useFallbacks = true;
        bool enableFixups = true;
        string? remap = null;
        bool verbose = false;
        bool trackAssets = false;
        string? dbdDir = null; string? dbctoolOutRoot = null; string? dbctoolSrcAlias = null; string? dbctoolSrcDir = null; string? dbctoolLkDir = null;
        string? dbctoolPatchDir = null; string? dbctoolPatchFile = null;
        bool vizSvg = false; bool vizHtml = false; bool patchOnly = false; bool noZoneFallback = false; string? vizDir = null; int? mdp = null;
        bool extractMcnkTerrain = false; bool extractMcnkShadows = false;
        bool countTiles = false;

        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            switch (a)
            {
                case "--input":
                    if (i + 1 >= args.Length) return Usage();
                    wdt = args[++i];
                    break;
                case "--input-dir":
                    if (i + 1 >= args.Length) return Usage();
                    inputDir = args[++i];
                    break;
                case "--listfile":
                    if (i + 1 >= args.Length) return Usage();
                    listfile = args[++i];
                    break;
                case "--lk-listfile":
                    if (i + 1 >= args.Length) return Usage();
                    lkListfile = args[++i];
                    break;
                case "--out":
                    if (i + 1 >= args.Length) return Usage();
                    outDir = args[++i];
                    break;
                case "--cluster-threshold":
                    if (i + 1 >= args.Length) return Usage();
                    if (!int.TryParse(args[++i], out var ct)) return Usage();
                    clusterThreshold = ct;
                    break;
                case "--cluster-gap":
                    if (i + 1 >= args.Length) return Usage();
                    if (!int.TryParse(args[++i], out var cg)) return Usage();
                    clusterGap = cg;
                    break;
                case "--web":
                    web = true;
                    break;
                case "--no-web":
                    web = false;
                    break;
                case "--export-adt":
                    exportAdt = true;
                    break;
                case "--export-dir":
                    if (i + 1 >= args.Length) return Usage();
                    exportDir = args[++i];
                    break;
                case "--fallback-tileset":
                    if (i + 1 >= args.Length) return Usage();
                    fallbackTileset = args[++i];
                    break;
                case "--fallback-wmo":
                    if (i + 1 >= args.Length) return Usage();
                    fallbackWmo = args[++i];
                    break;
                case "--fallback-m2":
                    if (i + 1 >= args.Length) return Usage();
                    fallbackM2 = args[++i];
                    break;
                case "--fallback-blp":
                    if (i + 1 >= args.Length) return Usage();
                    fallbackNonTilesetBlp = args[++i];
                    break;
                case "--no-mh2o":
                    mh2o = false;
                    break;
                case "--asset-fuzzy":
                    if (i + 1 >= args.Length) return Usage();
                    var v = args[++i];
                    assetFuzzy = !string.Equals(v, "off", StringComparison.OrdinalIgnoreCase);
                    break;
                case "--profile":
                    if (i + 1 >= args.Length) return Usage();
                    profile = args[++i];
                    break;
                case "--no-fallbacks":
                    useFallbacks = false;
                    break;
                case "--no-fixups":
                    enableFixups = false;
                    break;
                case "--remap":
                    if (i + 1 >= args.Length) return Usage();
                    remap = args[++i];
                    break;
                case "--dbd-dir":
                    if (i + 1 >= args.Length) return Usage();
                    dbdDir = args[++i];
                    break;
                case "--dbctool-out-root":
                    if (i + 1 >= args.Length) return Usage();
                    dbctoolOutRoot = args[++i];
                    break;
                case "--dbctool-src-alias":
                    if (i + 1 >= args.Length) return Usage();
                    dbctoolSrcAlias = args[++i];
                    break;
                case "--dbctool-src-dir":
                    if (i + 1 >= args.Length) return Usage();
                    dbctoolSrcDir = args[++i];
                    break;
                case "--dbctool-lk-dir":
                    if (i + 1 >= args.Length) return Usage();
                    dbctoolLkDir = args[++i];
                    break;
                case "--dbctool-patch-dir":
                    if (i + 1 >= args.Length) return Usage();
                    dbctoolPatchDir = args[++i];
                    break;
                case "--dbctool-patch-file":
                    if (i + 1 >= args.Length) return Usage();
                    dbctoolPatchFile = args[++i];
                    break;
                case "--viz-svg":
                    vizSvg = true;
                    break;
                case "--viz-html":
                    vizHtml = true;
                    break;
                case "--patch-only":
                    patchOnly = true;
                    break;
                case "--no-zone-fallback":
                    noZoneFallback = true;
                    break;
                case "--viz-dir":
                    if (i + 1 >= args.Length) return Usage();
                    vizDir = args[++i];
                    break;
                case "--mdp":
                    if (i + 1 >= args.Length) return Usage();
                    if (int.TryParse(args[++i], out var mdpVal) && mdpVal > 0) mdp = mdpVal; else return Usage();
                    break;
                case "--verbose":
                    verbose = true;
                    break;
                case "--track-assets":
                    trackAssets = true;
                    break;
                case "--extract-mcnk-terrain":
                    extractMcnkTerrain = true;
                    break;
                case "--extract-mcnk-shadows":
                    extractMcnkShadows = true;
                    break;
                case "--count-tiles":
                    countTiles = true;
                    break;
                case "-h":
                case "--help":
                    return Usage();
            }
        }

        // Quick tile count mode (for cache validation in rebuild scripts)
        if (countTiles)
        {
            if (string.IsNullOrWhiteSpace(wdt))
            {
                Console.Error.WriteLine("--count-tiles requires --input <path/to/map.wdt>");
                return 1;
            }
            if (!File.Exists(wdt))
            {
                Console.Error.WriteLine($"WDT not found: {wdt}");
                return 1;
            }

            try
            {
                var scanner = new WdtAlphaScanner(wdt);
                Console.WriteLine(scanner.AdtNumbers.Count);
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error reading WDT: {ex.Message}");
                return 1;
            }
        }

        // Apply profile
        if (string.Equals(profile, "preserve", StringComparison.OrdinalIgnoreCase))
        {
            assetFuzzy = false;
            useFallbacks = false;
            enableFixups = false;
        }

        // Allow directory passed via --input
        if (!string.IsNullOrWhiteSpace(wdt) && Directory.Exists(wdt))
        {
            inputDir = wdt;
            wdt = null;
        }

        var isBatch = !string.IsNullOrWhiteSpace(inputDir);

        if ((isBatch && (string.IsNullOrWhiteSpace(listfile) || string.IsNullOrWhiteSpace(outDir))) ||
            (!isBatch && (string.IsNullOrWhiteSpace(wdt) || string.IsNullOrWhiteSpace(listfile) || string.IsNullOrWhiteSpace(outDir))))
        {
            return Usage();
        }

        if (!string.IsNullOrWhiteSpace(listfile) && !File.Exists(listfile))
        {
            Console.Error.WriteLine($"Listfile not found: {listfile}");
            return 1;
        }
        if (!string.IsNullOrWhiteSpace(lkListfile) && !File.Exists(lkListfile))
        {
            Console.Error.WriteLine($"LK listfile not found: {lkListfile}");
            return 1;
        }
        if (exportAdt)
        {
            if (string.IsNullOrWhiteSpace(exportDir))
            {
                Console.Error.WriteLine("--export-adt requires --export-dir <dir>");
                return 1;
            }
        }

        // Determine log root (AWDT only): prefer exportDir when exporting ADTs, else outDir
        string? logRoot = null;
        if (!string.IsNullOrWhiteSpace(exportDir)) logRoot = exportDir;
        else if (!string.IsNullOrWhiteSpace(outDir)) logRoot = outDir;

        try
        {
            // Start console tee if we have a destination
            string? logPath = null; IDisposable? teeScope = null;
            if (!string.IsNullOrWhiteSpace(logRoot))
            {
                try
                {
                    var stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    logPath = Path.Combine(logRoot!, $"awdt_run_{stamp}.log");
                    teeScope = ConsoleTee.Start(logPath);
                }
                catch { /* best-effort */ }
            }

            try
            {
                if (isBatch)
                {
                    BatchAnalysis.Run(new BatchAnalysis.Options
                    {
                        InputRoot = inputDir!,
                        ListfilePath = listfile!,
                        OutDir = outDir!,
                        ClusterThreshold = clusterThreshold ?? 10,
                        ClusterGap = clusterGap ?? 1000,
                        Web = web,
                        ExtractMcnkTerrain = extractMcnkTerrain,
                        ExtractMcnkShadows = extractMcnkShadows
                    });

                    if (exportAdt)
                    {
                        AdtExportPipeline.ExportBatch(new AdtExportPipeline.Options
                        {
                            InputRoot = inputDir!,
                            CommunityListfilePath = listfile!,
                            LkListfilePath = lkListfile,
                            ExportDir = exportDir!,
                            FallbackTileset = fallbackTileset,
                            FallbackNonTilesetBlp = fallbackNonTilesetBlp,
                            FallbackWmo = fallbackWmo,
                            FallbackM2 = fallbackM2,
                            ConvertToMh2o = mh2o,
                            AssetFuzzy = assetFuzzy,
                            UseFallbacks = useFallbacks,
                            EnableFixups = enableFixups,
                            RemapPath = remap,
                            Verbose = verbose,
                            TrackAssets = trackAssets,
                            DbdDir = dbdDir,
                            DbctoolOutRoot = dbctoolOutRoot,
                            DbctoolSrcAlias = dbctoolSrcAlias,
                            DbctoolSrcDir = dbctoolSrcDir,
                            DbctoolLkDir = dbctoolLkDir,
                            DbctoolPatchDir = dbctoolPatchDir,
                            DbctoolPatchFile = dbctoolPatchFile,
                            VizSvg = vizSvg,
                            VizHtml = vizHtml,
                            PatchOnly = patchOnly,
                            NoZoneFallback = noZoneFallback,
                            MaxDegreeOfParallelism = mdp,
                            VizDir = vizDir,
                        });
                    }
                }
                else
                {
                    if (!File.Exists(wdt!))
                    {
                        Console.Error.WriteLine($"WDT not found: {wdt}");
                        return 1;
                    }

                    // Export ADTs FIRST if requested (needed for LK AreaIDs in terrain extraction)
                    string? lkAdtDirectory = null;
                    var wdtScanner = new WdtAlphaScanner(wdt!);
                    
                    if (exportAdt)
                    {
                        AdtExportPipeline.ExportSingle(new AdtExportPipeline.Options
                        {
                            SingleWdtPath = wdt!,
                            CommunityListfilePath = listfile!,
                            LkListfilePath = lkListfile,
                            ExportDir = exportDir!,
                            FallbackTileset = fallbackTileset,
                            FallbackNonTilesetBlp = fallbackNonTilesetBlp,
                            FallbackWmo = fallbackWmo,
                            FallbackM2 = fallbackM2,
                            ConvertToMh2o = mh2o,
                            AssetFuzzy = assetFuzzy,
                            UseFallbacks = useFallbacks,
                            EnableFixups = enableFixups,
                            RemapPath = remap,
                            Verbose = verbose,
                            TrackAssets = trackAssets,
                            DbdDir = dbdDir,
                            DbctoolOutRoot = dbctoolOutRoot,
                            DbctoolSrcAlias = dbctoolSrcAlias,
                            DbctoolSrcDir = dbctoolSrcDir,
                            DbctoolLkDir = dbctoolLkDir,
                            DbctoolPatchDir = dbctoolPatchDir,
                            DbctoolPatchFile = dbctoolPatchFile,
                            VizSvg = vizSvg,
                            VizHtml = vizHtml,
                            PatchOnly = patchOnly,
                            NoZoneFallback = noZoneFallback,
                            VizDir = vizDir,
                        });
                    }
                    
                    // Determine LK ADT directory (check even if we didn't export, they might already exist)
                    lkAdtDirectory = Path.Combine(exportDir!, "World", "Maps", wdtScanner.MapName);
                    
                    if (Directory.Exists(lkAdtDirectory))
                    {
                        Console.WriteLine($"[area] Using LK ADTs from: {lkAdtDirectory}");
                    }
                    else
                    {
                        Console.WriteLine($"[area:warn] LK ADT directory not found: {lkAdtDirectory}");
                        Console.WriteLine($"[area:warn] Area names will show as 'Unknown' (using Alpha AreaIDs)");
                        lkAdtDirectory = null;
                    }
                    
                    // Run analysis pipeline (after ADT export so we have LK AreaIDs)
                    AnalysisPipeline.Run(new AnalysisPipeline.Options
                    {
                        WdtPath = wdt!,
                        ListfilePath = listfile!,
                        OutDir = outDir!,
                        ClusterThreshold = clusterThreshold ?? 10,
                        ClusterGap = clusterGap ?? 1000,
                        ExtractMcnkTerrain = extractMcnkTerrain,
                        ExtractMcnkShadows = extractMcnkShadows,
                        LkAdtDirectory = lkAdtDirectory  // Pass LK ADT directory for area name mapping
                    });

                    if (web)
                    {
                        WebAssetsWriter.Write(outDir!);
                        Console.WriteLine($"Web UI written to {Path.Combine(outDir!, "web")}. Open index.html in a browser.");
                    }
                }

                Console.WriteLine("Analysis complete.");
                return 0;
            }
            finally
            {
                teeScope?.Dispose();
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }
}
