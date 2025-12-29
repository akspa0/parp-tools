using WoWMapConverter.Core.Converters;
using WoWMapConverter.Core.Dbc;
using WoWMapConverter.Core.Services;

namespace WoWMapConverter.Cli;

/// <summary>
/// Unified CLI for WoW Alpha → LK 3.3.5 conversion.
/// Includes WDT/ADT, WMO v14→v17, MDX→M2, and DBC crosswalks.
/// </summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            ShowUsage();
            return 0;
        }

        var command = args[0].ToLowerInvariant();

        return command switch
        {
            "convert" => await RunConvertAsync(args.Skip(1).ToArray()),
            "convert-lk-to-alpha" => await RunConvertLkToAlphaAsync(args.Skip(1).ToArray()),
            "convert-wmo" => RunConvertWmo(args.Skip(1).ToArray()),
            "convert-wmo-to-alpha" => RunConvertWmoToAlpha(args.Skip(1).ToArray()),
            "convert-mdx" => RunConvertMdx(args.Skip(1).ToArray()),
            "convert-m2-to-mdx" => RunConvertM2ToMdx(args.Skip(1).ToArray()),
            "pm4-export" => RunPm4Export(args.Skip(1).ToArray()),
            "analyze" => await RunAnalyzeAsync(args.Skip(1).ToArray()),
            "batch" => await RunBatchAsync(args.Skip(1).ToArray()),
            _ => await RunDefaultConvertAsync(args)
        };
    }

    private static async Task<int> RunConvertAsync(string[] args)
    {
        string? inputPath = null;
        string? outputDir = null;
        string? crosswalkDir = null;
        string? communityListfile = null;
        string? lkListfile = null;
        string? wmoDir = null;
        bool fuzzy = false;
        bool verbose = false;
        bool convertWmos = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputDir = args[++i];
                    break;
                case "--crosswalk":
                    if (i + 1 < args.Length) crosswalkDir = args[++i];
                    break;
                case "--listfile":
                    if (i + 1 < args.Length) communityListfile = args[++i];
                    break;
                case "--lk-listfile":
                    if (i + 1 < args.Length) lkListfile = args[++i];
                    break;
                case "--fuzzy":
                    fuzzy = true;
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
                    break;
                case "--convert-wmos":
                    convertWmos = true;
                    break;
                case "--wmo-dir":
                    if (i + 1 < args.Length) wmoDir = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input path required");
            return 1;
        }

        outputDir ??= Path.Combine(Directory.GetCurrentDirectory(), "output");

        var options = new ConversionOptions
        {
            CrosswalkDirectory = crosswalkDir,
            CommunityListfile = communityListfile,
            LkListfile = lkListfile,
            FuzzyAssetMatching = fuzzy,
            Verbose = verbose,
            ConvertWmos = convertWmos,
            AlphaWmoDirectory = wmoDir
        };

        var converter = new AlphaToLkConverter(options);

        Console.WriteLine("WoW Map Converter v3");
        Console.WriteLine("====================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputDir)}");
        Console.WriteLine();

        var result = await converter.ConvertWdtAsync(inputPath, outputDir);

        if (result.Success)
        {
            Console.WriteLine($"✓ Conversion completed in {result.ElapsedMs}ms");
            Console.WriteLine($"  Map: {result.MapName}");
            Console.WriteLine($"  Tiles: {result.TilesConverted}/{result.TotalTiles}");
            return 0;
        }
        else
        {
            Console.Error.WriteLine($"✗ Conversion failed: {result.Error}");
            return 1;
        }
    }

    private static int RunConvertWmo(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input WMO path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".v17.wmo");

        Console.WriteLine("WMO v14 → v17 Converter");
        Console.WriteLine("=======================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var converter = new WmoV14ToV17Converter();
            var textures = converter.Convert(inputPath, outputPath);
            
            // Auto-copy textures
            CopyTextures(inputPath, outputPath, textures);
            
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static void CopyTextures(string inputWmoPath, string outputWmoPath, List<string> textures)
    {
        if (textures == null || textures.Count == 0) return;

        string inputDir = Path.GetDirectoryName(Path.GetFullPath(inputWmoPath))!;
        string outputDir = Path.GetDirectoryName(Path.GetFullPath(outputWmoPath))!;

        // Attempt to find the "Root" data directory by looking for the first part of the texture path
        // e.g. if tex is "World/wmos/...", we look for a "World" folder in inputDir or its parents.
        
        Console.WriteLine($"[INFO] wmo references {textures.Count} textures. Copying...");
        foreach (var t in textures) Console.WriteLine($"  - {t}");

        // Simple heuristic: Try to find the file relative to inputDir, then check parents
        foreach (var tex in textures)
        {
            var cleanTex = tex.Replace('/', '\\');
            string srcPath = null;
            
            // 1. Try relative to wmo file itself (unlikely but possible)
            var p1 = Path.Combine(inputDir, cleanTex);
            if (File.Exists(p1)) srcPath = p1;
            else
            {
                // 2. Walk up 5 levels to find the root
                var curr = new DirectoryInfo(inputDir);
                DirectoryInfo rootDir = null;
                for (int i = 0; i < 5 && curr != null; i++)
                {
                   var p2 = Path.Combine(curr.FullName, cleanTex);
                   if (File.Exists(p2))
                   {
                       srcPath = p2;
                       break;
                   }
                   if (Directory.Exists(Path.Combine(curr.FullName, "DUNGEONS")) || 
                       Directory.Exists(Path.Combine(curr.FullName, "World")) ||
                       Directory.Exists(Path.Combine(curr.FullName, "Textures")))
                   {
                       rootDir = curr;
                   }
                   curr = curr.Parent;
                }

                // 3. Fallback: Recursive search in Root if identified, or InputDir parents
                if (srcPath == null)
                {
                     var searchRoot = rootDir ?? new DirectoryInfo(inputDir).Parent?.Parent;
                     if (searchRoot != null && searchRoot.Exists)
                     {
                         var filename = Path.GetFileName(cleanTex);
                         var found = Directory.EnumerateFiles(searchRoot.FullName, filename, SearchOption.AllDirectories).FirstOrDefault();
                         if (found != null)
                         {
                             srcPath = found;
                             Console.WriteLine($"    Found via search: {srcPath}");
                         }
                     }
                }
            } // End else

            if (srcPath != null)
            {
                // Copy to output
                // Preserve directory structure matches WMO path
                string targetRelPath = cleanTex;
                var destPath = Path.Combine(outputDir, targetRelPath);
                
                Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);
                if (!File.Exists(destPath)) 
                {
                    File.Copy(srcPath, destPath, true);
                }
            }
            else
            {
                Console.WriteLine($"  [WARN] Missing texture: {cleanTex}");
            }
        }
    }

    private static int RunConvertMdx(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input MDX path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".m2");

        Console.WriteLine("MDX → M2 Converter");
        Console.WriteLine("==================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var converter = new MdxToM2Converter();
            converter.Convert(inputPath, outputPath);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> RunConvertLkToAlphaAsync(string[] args)
    {
        string? wdtPath = null;
        string? mapDir = null;
        string? outputPath = null;
        bool verbose = false;
        bool skipM2 = false;
        bool skipWmo = false;
        bool convertLiquids = true;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--wdt":
                    if (i + 1 < args.Length) wdtPath = args[++i];
                    break;
                case "--map-dir":
                    if (i + 1 < args.Length) mapDir = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
                    break;
                case "--skip-m2":
                    skipM2 = true;
                    break;
                case "--skip-wmo":
                    skipWmo = true;
                    break;
                case "--no-liquids":
                    convertLiquids = false;
                    break;
                default:
                    if (!args[i].StartsWith("-") && wdtPath == null)
                        wdtPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(wdtPath))
        {
            Console.Error.WriteLine("Error: LK WDT path required (--wdt or first positional arg)");
            return 1;
        }

        mapDir ??= Path.GetDirectoryName(wdtPath) ?? ".";
        outputPath ??= Path.Combine(Directory.GetCurrentDirectory(), "alpha_output", 
            Path.GetFileNameWithoutExtension(wdtPath) + ".wdt");

        var options = new LkToAlphaOptions
        {
            Verbose = verbose,
            SkipM2 = skipM2,
            SkipWmo = skipWmo,
            ConvertLiquids = convertLiquids
        };

        var converter = new LkToAlphaConverter(options);

        Console.WriteLine("LK → Alpha Converter");
        Console.WriteLine("====================");
        Console.WriteLine($"WDT:     {Path.GetFullPath(wdtPath)}");
        Console.WriteLine($"Map Dir: {Path.GetFullPath(mapDir)}");
        Console.WriteLine($"Output:  {Path.GetFullPath(outputPath)}");
        Console.WriteLine();

        var result = await converter.ConvertAsync(wdtPath, mapDir, outputPath);

        if (result.Success)
        {
            Console.WriteLine($"✓ Conversion completed in {result.ElapsedMs}ms");
            Console.WriteLine($"  Map: {result.MapName}");
            Console.WriteLine($"  Tiles: {result.TilesConverted}/{result.TotalTiles}");
            if (result.Warnings.Count > 0)
            {
                Console.WriteLine($"  Warnings: {result.Warnings.Count}");
                foreach (var w in result.Warnings.Take(5))
                    Console.WriteLine($"    - {w}");
                if (result.Warnings.Count > 5)
                    Console.WriteLine($"    ... and {result.Warnings.Count - 5} more");
            }
            return 0;
        }
        else
        {
            Console.Error.WriteLine($"✗ Conversion failed: {result.Error}");
            return 1;
        }
    }

    private static int RunConvertWmoToAlpha(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input WMO v17 path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".v14.wmo");

        Console.WriteLine("WMO v17 → v14 Converter");
        Console.WriteLine("=======================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var converter = new WmoV17ToV14Converter();
            converter.Convert(inputPath, outputPath);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunConvertM2ToMdx(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input M2 path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".mdx");

        Console.WriteLine("M2 → MDX Converter");
        Console.WriteLine("==================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var converter = new M2ToMdxConverter();
            converter.Convert(inputPath, outputPath);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunPm4Export(string[] args)
    {
        string? inputPath = null;
        string? outputPath = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                default:
                    if (!args[i].StartsWith("-") && inputPath == null)
                        inputPath = args[i];
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input PM4 path required");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".obj");

        Console.WriteLine("PM4 → OBJ Exporter");
        Console.WriteLine("==================");
        Console.WriteLine($"Input:  {Path.GetFullPath(inputPath)}");
        Console.WriteLine($"Output: {Path.GetFullPath(outputPath)}");

        try
        {
            var pm4 = WoWMapConverter.Core.Formats.PM4.Pm4File.FromFile(inputPath);
            Console.WriteLine($"  Version: {pm4.Version}");
            Console.WriteLine($"  Mesh Vertices: {pm4.MeshVertices.Count}");
            Console.WriteLine($"  Mesh Indices: {pm4.MeshIndices.Count}");
            Console.WriteLine($"  Surfaces: {pm4.Surfaces.Count}");
            Console.WriteLine($"  Links: {pm4.LinkEntries.Count}");
            Console.WriteLine($"  Position Refs: {pm4.PositionRefs.Count}");
            Console.WriteLine($"  Path Vertices: {pm4.PathVertices.Count}");
            Console.WriteLine($"  Exterior Vertices: {pm4.ExteriorVertices.Count}");
            Console.WriteLine($"  Chunks: {string.Join(", ", pm4.ChunkSizes.Select(kv => $"{kv.Key}:{kv.Value}"))}");

            if (pm4.MeshVertices.Count == 0)
            {
                Console.WriteLine("[WARN] No mesh vertices found - PM4 may be empty or use different chunk names");
                File.WriteAllText(outputPath, "# Empty PM4 - no mesh data\n");
                return 0;
            }

            var obj = pm4.ExportToObj();
            File.WriteAllText(outputPath, obj);
            Console.WriteLine($"[SUCCESS] Exported to: {outputPath}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            Console.Error.WriteLine($"Stack: {ex.StackTrace}");
            return 1;
        }
    }

    private static async Task<int> RunAnalyzeAsync(string[] args)
    {
        Console.WriteLine("Analyze command - not yet implemented");
        await Task.CompletedTask;
        return 0;
    }

    private static async Task<int> RunBatchAsync(string[] args)
    {
        Console.WriteLine("Batch command - not yet implemented");
        await Task.CompletedTask;
        return 0;
    }

    private static async Task<int> RunDefaultConvertAsync(string[] args)
    {
        // Treat first arg as input path for backward compatibility
        return await RunConvertAsync(args);
    }

    private static void ShowUsage()
    {
        Console.WriteLine("WoW Map Converter v3 - Bidirectional Alpha ↔ LK 3.3.5 Conversion");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  wowmapconverter convert <input.wdt> [options]           Convert Alpha WDT → LK ADT");
        Console.WriteLine("  wowmapconverter convert-lk-to-alpha <wdt> [options]     Convert LK ADT → Alpha WDT");
        Console.WriteLine("  wowmapconverter convert-wmo <input.wmo> [options]       Convert WMO v14 → v17");
        Console.WriteLine("  wowmapconverter convert-wmo-to-alpha <wmo> [options]    Convert WMO v17 → v14");
        Console.WriteLine("  wowmapconverter convert-mdx <input.mdx> [options]       Convert MDX → M2");
        Console.WriteLine("  wowmapconverter convert-m2-to-mdx <m2> [options]        Convert M2 → MDX");
        Console.WriteLine("  wowmapconverter pm4-export <pm4> [options]              Export PM4 to OBJ");
        Console.WriteLine("  wowmapconverter batch --input-dir <dir> [options]       Batch convert directory");
        Console.WriteLine();
        Console.WriteLine("Alpha → LK Conversion Options:");
        Console.WriteLine("  --input, -i <path>      Input Alpha WDT file path");
        Console.WriteLine("  --output, -o <dir>      Output directory (default: ./output)");
        Console.WriteLine("  --crosswalk <dir>       AreaID crosswalk CSV directory");
        Console.WriteLine("  --listfile <csv>        Community listfile CSV for asset fixups");
        Console.WriteLine("  --convert-wmos          Convert WMO v14 files to v17 with _alpha suffix");
        Console.WriteLine("  --wmo-dir <dir>         Alpha WMO source directory (e.g., test_data/0.5.3/tree)");
        Console.WriteLine("  --verbose, -v           Verbose output");
        Console.WriteLine();
        Console.WriteLine("LK → Alpha Conversion Options:");
        Console.WriteLine("  --wdt <path>            Input LK WDT file path");
        Console.WriteLine("  --map-dir <dir>         Directory containing LK ADT files");
        Console.WriteLine("  --output, -o <path>     Output Alpha WDT path");
        Console.WriteLine("  --skip-m2               Skip M2 doodad placements");
        Console.WriteLine("  --skip-wmo              Skip WMO placements");
        Console.WriteLine("  --no-liquids            Disable MH2O → MCLQ conversion");
        Console.WriteLine("  --verbose, -v           Verbose output");
        Console.WriteLine();
        Console.WriteLine("WMO Conversion Options:");
        Console.WriteLine("  --input, -i <path>      Input WMO v14 file");
        Console.WriteLine("  --output, -o <path>     Output WMO v17 path (creates root + _XXX.wmo groups)");
        Console.WriteLine();
        Console.WriteLine("MDX Conversion Options:");
        Console.WriteLine("  --input, -i <path>      Input MDX file");
        Console.WriteLine("  --output, -o <path>     Output M2 path (also creates .skin file)");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  wowmapconverter convert World/Maps/Azeroth/Azeroth.wdt -o ./out");
        Console.WriteLine("  wowmapconverter convert-lk-to-alpha development.wdt --map-dir ./maps -o alpha.wdt");
        Console.WriteLine("  wowmapconverter convert-wmo castle01.wmo -o castle01_v17.wmo");
        Console.WriteLine("  wowmapconverter convert-mdx Human/HumanMale.mdx -o HumanMale.m2");
    }
}
