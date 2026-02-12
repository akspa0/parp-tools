using System;
using System.Collections.Generic;
using System.IO;
using AlphaLkToAlphaStandalone.Core;
using AlphaWdtAnalyzer.Core.Export;
using WoWRollback.Core.Services.Archive;

internal static class Program
{
    private const string DefaultOutRoot = "out";

    private static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintHelp();
            return 1;
        }

        var cmd = args[0].ToLowerInvariant();
        var argv = args.Length > 1 ? args[1..] : Array.Empty<string>();

        if (cmd == "-h" || cmd == "--help")
        {
            PrintHelp();
            return 0;
        }

        return cmd switch
        {
            "convert" => Convert(argv),
            "roundtrip" => Roundtrip(argv),
            _ => Unknown(cmd)
        };
    }

    private static int Unknown(string cmd)
    {
        Console.Error.WriteLine($"Unknown command: {cmd}");
        PrintHelp();
        return 2;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("AlphaLkToAlphaStandalone - Commands:");
        Console.WriteLine("  convert   --lk-root <lk_tree_root> --map <MapName> [--out <dir>]");
        Console.WriteLine();
        Console.WriteLine("Example:");
        Console.WriteLine("  AlphaLkToAlphaStandalone convert \\");
        Console.WriteLine("    --lk-root test_data/3.3.5/tree \\");
        Console.WriteLine("    --map Kalidar \\");
        Console.WriteLine("    --out out");
    }

    private static int Convert(string[] argv)
    {
        string? lkRoot = null;
        string? mapName = null;
        string? outRoot = null;

        for (int i = 0; i < argv.Length; i++)
        {
            var a = argv[i];
            switch (a)
            {
                case "--lk-root":
                    if (i + 1 >= argv.Length)
                    {
                        Console.Error.WriteLine("--lk-root requires a value");
                        return 2;
                    }
                    lkRoot = argv[++i];
                    break;
                case "--map":
                    if (i + 1 >= argv.Length)
                    {
                        Console.Error.WriteLine("--map requires a value");
                        return 2;
                    }
                    mapName = argv[++i];
                    break;
                case "--out":
                    if (i + 1 >= argv.Length)
                    {
                        Console.Error.WriteLine("--out requires a value");
                        return 2;
                    }
                    outRoot = argv[++i];
                    break;
                case "-h":
                case "--help":
                    PrintHelp();
                    return 0;
            }
        }

        if (string.IsNullOrWhiteSpace(lkRoot))
        {
            Console.Error.WriteLine("Missing required --lk-root <dir>");
            return 2;
        }
        if (string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("Missing required --map <name>");
            return 2;
        }

        if (string.IsNullOrWhiteSpace(outRoot))
        {
            outRoot = DefaultOutRoot;
        }

        return RunConvertCore(lkRoot, mapName, outRoot);
    }

    private static int RunConvertCore(string lkRoot, string mapName, string outRoot)
    {
        var lkRootFull = Path.GetFullPath(lkRoot);
        if (!Directory.Exists(lkRootFull))
        {
            Console.Error.WriteLine($"LK root directory not found: {lkRootFull}");
            return 3;
        }

        var worldMapsRoot = Path.Combine(lkRootFull, "World", "Maps");
        var lkMapDir = Path.Combine(worldMapsRoot, mapName);
        var dataRoot = Path.Combine(lkRootFull, "Data");

        bool hasFsMaps = Directory.Exists(lkMapDir);
        bool hasDataRoot = Directory.Exists(dataRoot);
        bool useMpq = !hasFsMaps && hasDataRoot;

        if (!hasFsMaps && !hasDataRoot)
        {
            Console.Error.WriteLine($"LK root does not contain World/Maps or Data: {lkRootFull}");
            return 3;
        }

        var stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var sessionDir = Path.Combine(outRoot, $"session_{stamp}");
        var alphaOutDir = Path.Combine(sessionDir, "alpha");
        var diagnosticsDir = Path.Combine(sessionDir, "diagnostics");

        try
        {
            Directory.CreateDirectory(sessionDir);
            Directory.CreateDirectory(alphaOutDir);
            Directory.CreateDirectory(diagnosticsDir);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to create output directories: {ex.Message}");
            return 5;
        }

        Console.WriteLine("AlphaLkToAlphaStandalone convert");
        Console.WriteLine($"  LK root           : {lkRootFull}");
        Console.WriteLine($"  Map name          : {mapName}");
        Console.WriteLine($"  Source mode       : {(useMpq ? "mpq" : "filesystem")}");
        Console.WriteLine($"  LK map dir        : {Path.GetFullPath(lkMapDir)}");
        Console.WriteLine($"  Output root       : {Path.GetFullPath(outRoot)}");
        Console.WriteLine($"  Session directory : {Path.GetFullPath(sessionDir)}");
        Console.WriteLine($"  Alpha out dir     : {Path.GetFullPath(alphaOutDir)}");
        Console.WriteLine($"  Diagnostics dir   : {Path.GetFullPath(diagnosticsDir)}");

        var summaryPath = Path.Combine(diagnosticsDir, "run_summary.txt");
        try
        {
            File.WriteAllLines(summaryPath, new[]
            {
                $"LK root          : {lkRootFull}",
                $"Map name         : {mapName}",
                $"Source mode      : {(useMpq ? "mpq" : "filesystem")}",
                $"LK map dir       : {Path.GetFullPath(lkMapDir)}",
                $"Output root      : {Path.GetFullPath(outRoot)}",
                $"Session directory: {Path.GetFullPath(sessionDir)}",
                $"Alpha out dir    : {Path.GetFullPath(alphaOutDir)}",
                $"Diagnostics dir  : {Path.GetFullPath(diagnosticsDir)}"
            });
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: failed to write run_summary.txt: {ex.Message}");
        }

        var inputsCsvPath = Path.Combine(diagnosticsDir, "input_files.csv");
        try
        {
            var inputRows = new List<string>
            {
                "kind,path",
                $"lk_root,{lkRootFull}",
                $"lk_map_dir,{Path.GetFullPath(lkMapDir)}",
                $"source_mode,{(useMpq ? "mpq" : "filesystem")}",
                $"map_name,{mapName}",
                $"session_dir,{Path.GetFullPath(sessionDir)}",
                $"alpha_out_dir,{Path.GetFullPath(alphaOutDir)}",
                $"diagnostics_dir,{Path.GetFullPath(diagnosticsDir)}"
            };

            File.WriteAllLines(inputsCsvPath, inputRows);
            Console.WriteLine($"Wrote input_files.csv to {inputsCsvPath}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: failed to write input_files.csv: {ex.Message}");
        }

        int matchedAdtCount = 0;
        int ignoredAdtCount = 0;
        try
        {
            var tiles = new bool[64, 64];

            if (!useMpq)
            {
                if (!Directory.Exists(lkMapDir))
                {
                    Console.Error.WriteLine($"LK map directory not found when scanning: {lkMapDir}");
                }
                else
                {
                    foreach (var path in Directory.EnumerateFiles(lkMapDir, "*.adt", SearchOption.TopDirectoryOnly))
                    {
                        var name = Path.GetFileNameWithoutExtension(path);
                        if (string.IsNullOrEmpty(name))
                        {
                            ignoredAdtCount++;
                            continue;
                        }

                        var parts = name.Split('_');
                        if (parts.Length < 3)
                        {
                            ignoredAdtCount++;
                            continue;
                        }

                        if (!parts[0].Equals(mapName, StringComparison.OrdinalIgnoreCase))
                        {
                            ignoredAdtCount++;
                            continue;
                        }

                        if (!int.TryParse(parts[^2], out var x) || !int.TryParse(parts[^1], out var y))
                        {
                            ignoredAdtCount++;
                            continue;
                        }

                        if (x < 0 || x >= 64 || y < 0 || y >= 64)
                        {
                            ignoredAdtCount++;
                            continue;
                        }

                        matchedAdtCount++;
                        tiles[x, y] = true;
                    }
                }
            }
            else
            {
                var mpqs = ArchiveLocator.LocateMpqs(lkRootFull);
                if (mpqs.Count == 0)
                {
                    Console.Error.WriteLine($"No MPQ archives found under {lkRootFull}");
                }
                else
                {
                    using var archiveSource = new MpqArchiveSource(mpqs);
                    for (var y = 0; y < 64; y++)
                    {
                        for (var x = 0; x < 64; x++)
                        {
                            var virtualPath = $"World/Maps/{mapName}/{mapName}_{x}_{y}.adt";
                            if (archiveSource.FileExists(virtualPath))
                            {
                                matchedAdtCount++;
                                tiles[x, y] = true;
                            }
                        }
                    }
                }
            }

            var rows = new List<string> { "tileX,tileY,adt_present" };
            for (var y = 0; y < 64; y++)
            {
                for (var x = 0; x < 64; x++)
                {
                    rows.Add($"{x},{y},{(tiles[x, y] ? 1 : 0)}");
                }
            }

            var csvPath = Path.Combine(diagnosticsDir, "tiles_summary.csv");
            File.WriteAllLines(csvPath, rows);
            Console.WriteLine($"Wrote tiles_summary.csv to {csvPath}");
            Console.WriteLine($"  matched ADT files={matchedAdtCount}, ignored ADT files={ignoredAdtCount}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: failed to write tiles_summary.csv: {ex.Message}");
        }

        try
        {
            var options = new LkToAlphaConversionOptions(
                lkRootFull,
                Path.GetFullPath(lkMapDir),
                alphaOutDir,
                diagnosticsDir,
                mapName);

            LkToAlphaWriter.Plan(options, matchedAdtCount);
            LkToAlphaWriter.WriteAlphaWdtStub(options);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: failed to plan Alpha output: {ex.Message}");
        }

        try
        {
            var extraSummary = new[]
            {
                $"Matched ADT files : {matchedAdtCount}",
                $"Ignored ADT files : {ignoredAdtCount}"
            };
            File.AppendAllLines(summaryPath, extraSummary);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: failed to append ADT counts to run_summary.txt: {ex.Message}");
        }

        Console.WriteLine();
        Console.WriteLine("Stub Alpha WDT has been written based on tile occupancy; no Alpha ADT files have been written yet.");

        return 0;
    }

    private static int Roundtrip(string[] argv)
    {
        string? alphaWdt = null;
        string? lkClient = null;
        string? outRoot = null;

        for (int i = 0; i < argv.Length; i++)
        {
            var a = argv[i];
            switch (a)
            {
                case "--alpha-wdt":
                    if (i + 1 >= argv.Length)
                    {
                        Console.Error.WriteLine("--alpha-wdt requires a value");
                        return 2;
                    }
                    alphaWdt = argv[++i];
                    break;
                case "--lk-client":
                    if (i + 1 >= argv.Length)
                    {
                        Console.Error.WriteLine("--lk-client requires a value");
                        return 2;
                    }
                    lkClient = argv[++i];
                    break;
                case "--out":
                    if (i + 1 >= argv.Length)
                    {
                        Console.Error.WriteLine("--out requires a value");
                        return 2;
                    }
                    outRoot = argv[++i];
                    break;
                case "-h":
                case "--help":
                    PrintHelp();
                    return 0;
            }
        }

        if (string.IsNullOrWhiteSpace(alphaWdt))
        {
            Console.Error.WriteLine("Missing required --alpha-wdt <file>");
            return 2;
        }
        if (string.IsNullOrWhiteSpace(lkClient))
        {
            Console.Error.WriteLine("Missing required --lk-client <dir>");
            return 2;
        }

        var alphaWdtFull = Path.GetFullPath(alphaWdt);
        if (!File.Exists(alphaWdtFull))
        {
            Console.Error.WriteLine($"Alpha WDT not found: {alphaWdtFull}");
            return 3;
        }

        var lkClientFull = Path.GetFullPath(lkClient);
        if (!Directory.Exists(lkClientFull))
        {
            Console.Error.WriteLine($"LK client directory not found: {lkClientFull}");
            return 3;
        }

        if (string.IsNullOrWhiteSpace(outRoot))
        {
            outRoot = DefaultOutRoot;
        }

        var repoRoot = TryFindRepoRoot();
        if (string.IsNullOrWhiteSpace(repoRoot))
        {
            Console.Error.WriteLine("Unable to locate repo root (expected test_data and DBCTool.V2 folders).");
            return 3;
        }

        var dbdDir = Path.Combine(repoRoot, "lib", "WoWDBDefs", "definitions");
        var alias = InferAlphaAlias(alphaWdtFull);

        var alphaDbcDir = Path.Combine(repoRoot, "test_data", alias, "tree", "DBFilesClient");
        if (!Directory.Exists(alphaDbcDir))
        {
            var fallbackTree = Path.Combine(repoRoot, "test_data", alias, "tree");
            var fallbackRoot = Path.Combine(repoRoot, "test_data", alias);
            if (Directory.Exists(fallbackTree)) alphaDbcDir = fallbackTree;
            else if (Directory.Exists(fallbackRoot)) alphaDbcDir = fallbackRoot;
        }

        var lkDbcDir = Path.Combine(repoRoot, "test_data", "3.3.5", "tree", "DBFilesClient");
        if (!Directory.Exists(lkDbcDir))
        {
            var fallbackTree = Path.Combine(repoRoot, "test_data", "3.3.5", "tree");
            var fallbackRoot = Path.Combine(repoRoot, "test_data", "3.3.5");
            if (Directory.Exists(fallbackTree)) lkDbcDir = fallbackTree;
            else if (Directory.Exists(fallbackRoot)) lkDbcDir = fallbackRoot;
        }

        var dbctoolOutRoot = Path.Combine(repoRoot, "DBCTool.V2", "dbctool_outputs");

        var communityListfile = Path.Combine(repoRoot, "test_data", "community-listfile-withcapitals.csv");
        if (!File.Exists(communityListfile)) communityListfile = string.Empty;
        var lkListfile = Path.Combine(repoRoot, "test_data", "World of Warcraft 3x.txt");
        if (!File.Exists(lkListfile)) lkListfile = string.Empty;

        var mapName = Path.GetFileNameWithoutExtension(alphaWdtFull);
        var stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var sessionDir = Path.Combine(outRoot, $"session_{stamp}");
        var lkExportDir = Path.Combine(sessionDir, "lk_export");

        try
        {
            Directory.CreateDirectory(sessionDir);
            Directory.CreateDirectory(lkExportDir);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to create roundtrip output directories: {ex.Message}");
            return 5;
        }

        Console.WriteLine("AlphaLkToAlphaStandalone roundtrip");
        Console.WriteLine($"  Alpha WDT        : {alphaWdtFull}");
        Console.WriteLine($"  Map name         : {mapName}");
        Console.WriteLine($"  LK client        : {lkClientFull}");
        Console.WriteLine($"  Repo root        : {repoRoot}");
        Console.WriteLine($"  DBD dir          : {dbdDir}");
        Console.WriteLine($"  Alpha DBC dir    : {alphaDbcDir}");
        Console.WriteLine($"  LK DBC dir       : {lkDbcDir}");
        Console.WriteLine($"  DBCTool out root : {dbctoolOutRoot}");
        Console.WriteLine($"  Session dir      : {sessionDir}");
        Console.WriteLine($"  LK export dir    : {lkExportDir}");

        try
        {
            var exportOpts = new AdtExportPipeline.Options
            {
                SingleWdtPath = alphaWdtFull,
                CommunityListfilePath = string.IsNullOrWhiteSpace(communityListfile) ? null : communityListfile,
                LkListfilePath = string.IsNullOrWhiteSpace(lkListfile) ? null : lkListfile,
                ExportDir = lkExportDir,
                FallbackTileset = @"Tileset\\Generic\\Checkers.blp",
                FallbackNonTilesetBlp = @"Dungeons\\Textures\\temp\\64.blp",
                FallbackWmo = @"wmo\\Dungeon\\test\\missingwmo.wmo",
                FallbackM2 = @"World\\Scale\\HumanMaleScale.mdx",
                ConvertToMh2o = true,
                AssetFuzzy = true,
                UseFallbacks = true,
                EnableFixups = true,
                RemapPath = null,
                Verbose = false,
                TrackAssets = false,
                DbdDir = Directory.Exists(dbdDir) ? dbdDir : null,
                DbctoolOutRoot = Directory.Exists(dbctoolOutRoot) ? dbctoolOutRoot : null,
                DbctoolSrcAlias = alias,
                DbctoolSrcDir = Directory.Exists(alphaDbcDir) ? alphaDbcDir : null,
                DbctoolLkDir = Directory.Exists(lkDbcDir) ? lkDbcDir : null,
                DbctoolPatchDir = null,
                DbctoolPatchFile = null,
                VizSvg = false,
                VizHtml = false,
                PatchOnly = false,
                NoZoneFallback = false,
                MaxDegreeOfParallelism = null,
                VizDir = null,
                AssetRoots = null,
                LogExact = false
            };

            AdtExportPipeline.ExportSingle(exportOpts);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Roundtrip: Alpha -> LK export failed: {ex.Message}");
            return 6;
        }

        // AreaID roundtrip diagnostics: compare Alpha MCNK area ids vs LK MCNK AreaId
        try
        {
            var diagnosticsDir = Path.Combine(sessionDir, "diagnostics");
            Directory.CreateDirectory(diagnosticsDir);
            var areaCsv = Path.Combine(diagnosticsDir, "areaid_roundtrip.csv");
            AreaIdRoundtripWriter.WriteAreaIdRoundtripCsv(alphaWdtFull, mapName, lkExportDir, areaCsv);
            Console.WriteLine($"Wrote areaid_roundtrip.csv to {areaCsv}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Roundtrip: failed to write AreaID roundtrip CSV: {ex.Message}");
        }

        // Second leg: use the exported LK ADTs as input to the existing LK->Alpha pipeline
        try
        {
            return RunConvertCore(lkExportDir, mapName, sessionDir);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Roundtrip: LK -> Alpha convert failed: {ex.Message}");
            return 7;
        }
    }

    private static string? TryFindRepoRoot()
    {
        string baseDir = AppContext.BaseDirectory;
        var dir = new DirectoryInfo(baseDir);
        for (int i = 0; i < 6 && dir != null; i++)
        {
            var testData = Path.Combine(dir.FullName, "test_data");
            var dbctool = Path.Combine(dir.FullName, "DBCTool.V2");
            if (Directory.Exists(testData) && Directory.Exists(dbctool))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }

        // Fallback to current working directory
        var cwd = new DirectoryInfo(Directory.GetCurrentDirectory());
        dir = cwd;
        for (int i = 0; i < 6 && dir != null; i++)
        {
            var testData = Path.Combine(dir.FullName, "test_data");
            var dbctool = Path.Combine(dir.FullName, "DBCTool.V2");
            if (Directory.Exists(testData) && Directory.Exists(dbctool))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }

        return null;
    }

    private static string InferAlphaAlias(string alphaWdtPath)
    {
        var corpus = (alphaWdtPath ?? string.Empty).ToLowerInvariant();
        if (corpus.Contains("0.6.0") || corpus.Contains("\\060\\") || corpus.Contains("/060/") || corpus.Contains("0_6_0")) return "0.6.0";
        if (corpus.Contains("0.5.5") || corpus.Contains("\\055\\") || corpus.Contains("/055/") || corpus.Contains("0_5_5")) return "0.5.5";
        if (corpus.Contains("0.5.3") || corpus.Contains("\\053\\") || corpus.Contains("/053/") || corpus.Contains("0_5_3")) return "0.5.3";
        return "0.5.3";
    }
}
