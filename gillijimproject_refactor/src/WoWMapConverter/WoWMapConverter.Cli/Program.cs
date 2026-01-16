using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.Converters;
using WoWMapConverter.Core.Dbc;
using WoWMapConverter.Core.Services;
using WoWMapConverter.Core.VLM;

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
            "wmo-info" => RunWmoInfo(args.Skip(1).ToArray()),
            "vlm-export" => await RunVlmExportAsync(args.Skip(1).ToArray()),
            "vlm-decode" => await RunVlmDecodeAsync(args.Skip(1).ToArray()),
            "vlm-bake" => await RunVlmBakeAsync(args.Skip(1).ToArray()),
            "vlm-bake-heightmap" => await RunVlmBakeHeightmapAsync(args.Skip(1).ToArray()),
            "vlm-synth" => await RunVlmSynthAsync(args.Skip(1).ToArray()),
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
        bool extended = false;

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
                case "--extended":
                    extended = true;
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
            List<string> textures;
            if (extended)
            {
                var converter = new WmoV14ToV17ExtendedConverter();
                textures = converter.Convert(inputPath, outputPath);
            }
            else
            {
                var converter = new WmoV14ToV17Converter();
                textures = converter.Convert(inputPath, outputPath);
            }
            
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

    private static int RunWmoInfo(string[] args)
    {
        string? inputPath = null;
        bool verbose = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
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

        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Error: File not found: {inputPath}");
            return 1;
        }

        Console.WriteLine();
        Console.WriteLine($"WMO Analysis: {Path.GetFileName(inputPath)}");
        Console.WriteLine(new string('=', 70));

        try
        {
            // Use WmoV14ToV17Converter's ParseWmoV14 to get data
            var converter = new WmoV14ToV17Converter();
            var wmoData = converter.ParseWmoV14(inputPath);

            Console.WriteLine($"Version: v{wmoData.Version}");
            Console.WriteLine($"Groups:  {wmoData.Groups.Count}");
            Console.WriteLine($"Materials: {wmoData.Materials.Count}");
            Console.WriteLine($"Textures: {wmoData.Textures.Count}");
            Console.WriteLine($"Doodad Sets: {wmoData.DoodadSets.Count}");
            Console.WriteLine($"Portals: {wmoData.Portals.Count}");
            Console.WriteLine();

            // List groups
            Console.WriteLine("Groups:");
            Console.WriteLine(new string('-', 70));
            Console.WriteLine($"{"#",-4} {"Name",-35} {"Verts",-8} {"Faces",-8} {"Flags",-14}");
            Console.WriteLine(new string('-', 70));

            int totalVerts = 0, totalFaces = 0, emptyGroups = 0;

            for (int i = 0; i < wmoData.Groups.Count; i++)
            {
                var g = wmoData.Groups[i];
                int faceCount = g.Indices?.Count / 3 ?? 0;
                int vertCount = g.Vertices?.Count ?? 0;
                string flagsHex = $"0x{g.Flags:X8}";

                // Decode common flags
                var flagNotes = new List<string>();
                if ((g.Flags & 0x01) != 0) flagNotes.Add("BSP");
                if ((g.Flags & 0x02) != 0) flagNotes.Add("Light");
                if ((g.Flags & 0x04) != 0) flagNotes.Add("Doodads");
                if ((g.Flags & 0x08) != 0) flagNotes.Add("Liquid");
                if ((g.Flags & 0x40) != 0) flagNotes.Add("Exterior");
                if ((g.Flags & 0x2000) != 0) flagNotes.Add("ExtLit");
                if ((g.Flags & 0x80000) != 0) flagNotes.Add("Indoor");

                string displayName = string.IsNullOrEmpty(g.Name) ? $"(group_{i})" : g.Name;
                if (displayName.Length > 35) displayName = displayName.Substring(0, 32) + "...";

                Console.WriteLine($"{i,-4} {displayName,-35} {vertCount,-8} {faceCount,-8} {flagsHex}");

                if (verbose && flagNotes.Count > 0)
                {
                    Console.WriteLine($"     Flags: {string.Join(", ", flagNotes)}");
                }

                if (vertCount == 0)
                {
                    emptyGroups++;
                    if (verbose)
                        Console.WriteLine($"     ⚠️  EMPTY GROUP");
                }

                totalVerts += vertCount;
                totalFaces += faceCount;
            }

            Console.WriteLine(new string('-', 70));
            Console.WriteLine($"Totals: {totalVerts} vertices, {totalFaces} faces");

            if (emptyGroups > 0)
            {
                Console.WriteLine();
                Console.WriteLine($"⚠️  Warning: {emptyGroups} empty groups (no vertices)");
            }

            // Show group name details in verbose mode
            if (verbose && wmoData.GroupInfos.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("MOGI Group Info (parsed from root file):");
                Console.WriteLine(new string('-', 70));
                for (int i = 0; i < wmoData.GroupInfos.Count; i++)
                {
                    var info = wmoData.GroupInfos[i];
                    Console.WriteLine($"  [{i}] Flags: 0x{info.Flags:X8}, NameOfs: 0x{info.NameOffset:X4}");
                }
            }

            // Texture list
            if (verbose && wmoData.Textures.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("Textures:");
                Console.WriteLine(new string('-', 70));
                for (int i = 0; i < wmoData.Textures.Count; i++)
                {
                    Console.WriteLine($"  [{i}] {wmoData.Textures[i]}");
                }
            }

            Console.WriteLine();
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error parsing WMO: {ex.Message}");
            if (verbose)
                Console.Error.WriteLine(ex.StackTrace);
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

    private static async Task<int> RunVlmExportAsync(string[] args)
    {
        string? clientPath = null;
        string? mapName = null;
        string? outputDir = null;
        string? listfilePath = null;
        int limit = int.MaxValue;
        bool generateDepth = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--client":
                case "-c":
                    if (i + 1 < args.Length) clientPath = args[++i];
                    break;
                case "--map":
                case "-m":
                    if (i + 1 < args.Length) mapName = args[++i];
                    break;
                case "--out":
                case "-o":
                    if (i + 1 < args.Length) outputDir = args[++i];
                    break;
                case "--listfile":
                case "-l":
                    if (i + 1 < args.Length) listfilePath = args[++i];
                    break;
                case "--limit":
                case "-n":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out int n))
                        limit = n;
                    break;
                case "--depth":
                case "-d":
                    generateDepth = true;
                    break;
            }
        }

        if (string.IsNullOrEmpty(clientPath) || string.IsNullOrEmpty(mapName) || string.IsNullOrEmpty(outputDir))
        {
            Console.WriteLine("VLM Export - Generate training dataset from Alpha ADT files");
            Console.WriteLine();
            Console.WriteLine("Usage: vlm-export --client <path> --map <name> --out <dir> [options]");
            Console.WriteLine();
            Console.WriteLine("Required:");
            Console.WriteLine("  --client, -c <path>   Path to Alpha 0.5.3 client Data folder");
            Console.WriteLine("  --map, -m <name>      Map name (e.g., 'development')");
            Console.WriteLine("  --out, -o <dir>       Output directory for dataset");
            Console.WriteLine();
            Console.WriteLine("Optional:");
            Console.WriteLine("  --listfile, -l <csv>  Path to listfile for name resolution");
            Console.WriteLine("  --limit, -n <N>       Export only first N tiles");
            Console.WriteLine("  --depth, -d           Generate depth maps (requires DepthAnything3)");
            return 1;
        }

        Console.WriteLine($"VLM Export: {mapName}");
        Console.WriteLine($"  Client: {clientPath}");
        Console.WriteLine($"  Output: {outputDir}");
        if (limit < int.MaxValue) Console.WriteLine($"  Limit: {limit} tiles");
        Console.WriteLine();

        var exporter = new VlmDatasetExporter();
        var progress = new Progress<string>(msg => Console.WriteLine(msg));

        try
        {
            var result = await exporter.ExportMapAsync(clientPath, mapName, outputDir, progress, limit, listfilePath, generateDepth);
            
            Console.WriteLine();
            Console.WriteLine($"Export complete:");
            Console.WriteLine($"  Tiles exported: {result.TilesExported}");
            Console.WriteLine($"  Tiles skipped: {result.TilesSkipped}");
            Console.WriteLine($"  Unique textures: {result.UniqueTextures}");
            Console.WriteLine($"  Output: {result.OutputDirectory}");
            
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> RunVlmDecodeAsync(string[] args)
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
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("VLM Decode - Reconstruct ADT from VLM JSON output");
            Console.WriteLine();
            Console.WriteLine("Usage: vlm-decode --input <json> --output <adt>");
            Console.WriteLine();
            Console.WriteLine("Required:");
            Console.WriteLine("  --input, -i <json>    VLM dataset JSON file");
            Console.WriteLine("  --output, -o <adt>    Output ADT file path");
            return 1;
        }

        outputPath ??= Path.ChangeExtension(inputPath, ".adt");

        Console.WriteLine($"VLM Decode: {Path.GetFileName(inputPath)} → {Path.GetFileName(outputPath)}");

        var decoder = new VlmAdtDecoder();

        try
        {
            var success = await decoder.DecodeAsync(inputPath, outputPath);
            
            if (success)
            {
                Console.WriteLine($"Decoded successfully: {outputPath}");
                return 0;
            }
            else
            {
                Console.WriteLine("Decode failed - invalid input data");
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }


    private static async Task<int> RunVlmBakeAsync(string[] args)
    {
        string? datasetDir = null;
        string? inputPath = null;
        string? outputPath = null;
        string? minimapPath = null;
        bool withShadows = true;
        bool debakeShadows = false;
        bool exportLayers = false;
        bool invertAlpha = true;  // Default true for correct layer blending
        float shadowIntensity = 0.5f;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--dataset":
                case "-d":
                    if (i + 1 < args.Length) datasetDir = args[++i];
                    break;
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                case "--minimap":
                case "-m":
                    if (i + 1 < args.Length) minimapPath = args[++i];
                    break;
                case "--shadows":
                    withShadows = true;
                    break;
                case "--no-shadows":
                    withShadows = false;
                    break;
                case "--debake":
                    debakeShadows = true;
                    break;
                case "--export-layers":
                case "-l":
                    exportLayers = true;
                    break;
                case "--invert-alpha":
                    invertAlpha = true;
                    break;
                case "--no-invert-alpha":
                    invertAlpha = false;
                    break;
                case "--shadow-intensity":
                    if (i + 1 < args.Length && float.TryParse(args[++i], out var intensity))
                        shadowIntensity = Math.Clamp(intensity, 0f, 1f);
                    break;
            }
        }

        if (string.IsNullOrEmpty(datasetDir) && string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("VLM Bake - Reconstruct high-resolution minimap tiles");
            Console.WriteLine();
            Console.WriteLine("Usage: vlm-bake --dataset <dir> [--input <json>] [--output <png>]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --dataset, -d <dir>       Path to the VLM dataset root (containing tilesets and masks)");
            Console.WriteLine("  --input, -i <json>        Specific VLM dataset JSON file (default: all in dataset/dataset folder)");
            Console.WriteLine("  --output, -o <png>        Output PNG path (for single input) or output directory");
            Console.WriteLine();
            Console.WriteLine("Shadow Options:");
            Console.WriteLine("  --shadows                 Apply shadow maps to output (default)");
            Console.WriteLine("  --no-shadows              Disable shadow map application");
            Console.WriteLine("  --shadow-intensity <0-1>  Shadow darkness (default: 0.5)");
            Console.WriteLine("  --debake                  Remove shadows from existing minimap (requires --minimap)");
            Console.WriteLine("  --minimap, -m <png>       Source minimap for debaking shadows");
            Console.WriteLine();
            Console.WriteLine("Layer Options:");
            Console.WriteLine("  --export-layers, -l       Export individual texture layers as separate PNGs");
            Console.WriteLine("  --no-invert-alpha         Disable alpha inversion (default: inverted for correct blending)");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  vlm-bake -d ./vlm_output -i dataset/Azeroth_0_0.json --shadows");
            Console.WriteLine("  vlm-bake -d ./vlm_output -i dataset/Azeroth_0_0.json --export-layers");
            Console.WriteLine("  vlm-bake -d ./vlm_output --debake -m minimap.png -i tile.json -o clean.png");
            return 1;
        }

        // If datasetDir is null, infer it from inputPath
        if (string.IsNullOrEmpty(datasetDir) && !string.IsNullOrEmpty(inputPath))
        {
            datasetDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetFullPath(inputPath))) ?? ".";
        }

        Console.WriteLine($"VLM Bake: High-Resolution Reconstruction");
        Console.WriteLine($"  Dataset: {datasetDir}");
        Console.WriteLine($"  Shadows: {(withShadows ? "enabled" : "disabled")} (intensity: {shadowIntensity:F2})");
        if (invertAlpha) Console.WriteLine($"  Alpha: INVERTED");
        if (debakeShadows) Console.WriteLine($"  Mode: De-bake (remove shadows)");
        if (exportLayers) Console.WriteLine($"  Mode: Export individual layers");

        var baker = new MinimapBakeService(datasetDir!) { ShadowIntensity = shadowIntensity, InvertAlpha = invertAlpha };
        var filesToProcess = new List<string>();

        if (!string.IsNullOrEmpty(inputPath))
        {
            filesToProcess.Add(inputPath);
        }
        else
        {
            var datasetFolder = Path.Combine(datasetDir!, "dataset");
            if (Directory.Exists(datasetFolder))
            {
                filesToProcess.AddRange(Directory.EnumerateFiles(datasetFolder, "*.json"));
            }
        }

        if (filesToProcess.Count == 0)
        {
            Console.WriteLine("Error: No JSON files found to process.");
            return 1;
        }

        var outputBase = outputPath ?? Path.Combine(datasetDir!, "reconstructed_minimaps");
        Directory.CreateDirectory(outputBase);

        foreach (var file in filesToProcess)
        {
            try
            {
                Console.Write($"  Processing {Path.GetFileName(file)}... ");
                var timer = System.Diagnostics.Stopwatch.StartNew();
                
                Image<Rgba32> image;
                string extraInfo = "";
                
                if (exportLayers)
                {
                    // Export individual layers + composite
                    var (composite, layerCount, stats) = await baker.BakeTileWithLayersAsync(file, outputBase);
                    image = composite;
                    extraInfo = $" [{stats}]";
                }
                else if (debakeShadows && !string.IsNullOrEmpty(minimapPath))
                {
                    // De-bake: remove shadows from existing minimap
                    image = await baker.DebakeShadowsFromMinimapAsync(minimapPath, file);
                }
                else if (withShadows)
                {
                    // Bake with shadows
                    image = await baker.BakeTileWithShadowsAsync(file, applyShadows: true);
                }
                else
                {
                    // Bake without shadows
                    image = await baker.BakeTileAsync(file);
                }
                
                var outName = Path.GetFileNameWithoutExtension(file) + "_highres.png";
                var outPath = Path.IsPathRooted(outputPath) && filesToProcess.Count == 1 
                    ? outputPath 
                    : Path.Combine(outputBase, outName);

                await image.SaveAsPngAsync(outPath);
                image.Dispose();
                
                timer.Stop();
                Console.WriteLine($"done ({timer.ElapsedMilliseconds}ms){extraInfo}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"failed: {ex.Message}");
            }
        }

        return 0;
    }

    private static async Task<int> RunVlmBakeHeightmapAsync(string[] args)
    {
        string? datasetDir = null;
        string? inputPath = null;
        string? outputPath = null;
        bool fullRes = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--dataset":
                case "-d":
                    datasetDir = args[++i];
                    break;
                case "--input":
                case "-i":
                    inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    outputPath = args[++i];
                    break;
                case "--full-res":
                    fullRes = true;
                    break;
                case "--help":
                case "-h":
                    Console.WriteLine("VLM Bake Heightmap - Generate heightmaps from VLM JSON data");
                    Console.WriteLine();
                    Console.WriteLine("Usage: vlm-bake-heightmap --dataset <dir> [--input <json>] [--output <dir>]");
                    Console.WriteLine();
                    Console.WriteLine("Options:");
                    Console.WriteLine("  --dataset, -d <dir>   VLM dataset root directory");
                    Console.WriteLine("  --input, -i <json>    Specific JSON file (or process all if omitted)");
                    Console.WriteLine("  --output, -o <dir>    Output directory (default: dataset/heightmaps)");
                    Console.WriteLine("  --full-res            Generate 4096x4096 instead of 256x256");
                    return 0;
            }
        }

        if (string.IsNullOrEmpty(datasetDir) && string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("Error: Specify --dataset or --input");
            return 1;
        }

        if (string.IsNullOrEmpty(datasetDir) && !string.IsNullOrEmpty(inputPath))
        {
            datasetDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetFullPath(inputPath))) ?? ".";
        }

        Console.WriteLine($"VLM Bake Heightmap");
        Console.WriteLine($"  Dataset: {datasetDir}");
        Console.WriteLine($"  Resolution: {(fullRes ? "4096x4096" : "256x256")}");

        var baker = new HeightmapBakeService(datasetDir!);
        var outputBase = outputPath ?? Path.Combine(datasetDir!, "heightmaps");

        // If no specific input, use map-wide export (scans for global bounds first)
        if (string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("  Mode: MAP-WIDE (global height bounds)");
            var progress = new Progress<string>(msg => Console.WriteLine($"  {msg}"));
            await baker.ExportMapHeightmapsAsync(datasetDir!, outputBase, progress);
            return 0;
        }

        // Single tile mode (per-tile bounds - not recommended)
        Console.WriteLine("  Mode: SINGLE TILE (per-tile bounds)");
        Directory.CreateDirectory(outputBase);

        try
        {
            var tileName = Path.GetFileNameWithoutExtension(inputPath);
            Console.Write($"  {tileName}... ");

            if (fullRes)
            {
                var (heightmap, min, max) = await baker.BakeHeightmap4096Async(inputPath);
                var outPath = Path.Combine(outputBase, $"{tileName}_heightmap_4096.png");
                await heightmap.SaveAsPngAsync(outPath);
                heightmap.Dispose();
                Console.WriteLine($"OK [{min:F1} to {max:F1}]");
            }
            else
            {
                await baker.ExportWithMetadataAsync(inputPath, outputBase);
                Console.WriteLine("OK");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAILED: {ex.Message}");
            return 1;
        }

        return 0;
    }

    private static async Task<int> RunVlmSynthAsync(string[] args)
    {
        string? datasetDir = null;
        string? inputPath = null;
        string? outputPath = null;
        int resolution = 256;
        bool withVariations = false;
        float hillshade = 0.4f;
        float ao = 0.2f;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--dataset":
                case "-d":
                    datasetDir = args[++i];
                    break;
                case "--input":
                case "-i":
                    inputPath = args[++i];
                    break;
                case "--output":
                case "-o":
                    outputPath = args[++i];
                    break;
                case "--resolution":
                case "-r":
                    resolution = int.Parse(args[++i]);
                    break;
                case "--variations":
                    withVariations = true;
                    break;
                case "--hillshade":
                    hillshade = float.Parse(args[++i]);
                    break;
                case "--ao":
                    ao = float.Parse(args[++i]);
                    break;
                case "--help":
                case "-h":
                    Console.WriteLine("VLM Synth - Generate synthesized training pairs");
                    Console.WriteLine();
                    Console.WriteLine("Creates perfectly matched minimap/heightmap pairs where the minimap");
                    Console.WriteLine("is deformed based on the heightmap (hillshading, ambient occlusion).");
                    Console.WriteLine();
                    Console.WriteLine("Usage: vlm-synth --dataset <dir> [--input <json>] [--output <dir>]");
                    Console.WriteLine();
                    Console.WriteLine("Options:");
                    Console.WriteLine("  --dataset, -d <dir>     VLM dataset root directory");
                    Console.WriteLine("  --input, -i <json>      Specific JSON file (or process all)");
                    Console.WriteLine("  --output, -o <dir>      Output directory (default: synthesized/)");
                    Console.WriteLine("  --resolution, -r <n>    Output resolution (default: 256)");
                    Console.WriteLine("  --variations            Generate 4 lighting variations per tile");
                    Console.WriteLine("  --hillshade <0-1>       Hillshade strength (default: 0.4)");
                    Console.WriteLine("  --ao <0-1>              Ambient occlusion strength (default: 0.2)");
                    return 0;
            }
        }

        if (string.IsNullOrEmpty(datasetDir) && string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("Error: Specify --dataset or --input");
            return 1;
        }

        if (string.IsNullOrEmpty(datasetDir) && !string.IsNullOrEmpty(inputPath))
        {
            datasetDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetFullPath(inputPath))) ?? ".";
        }

        Console.WriteLine($"VLM Synthesized Training Pair Generator");
        Console.WriteLine($"  Dataset: {datasetDir}");
        Console.WriteLine($"  Resolution: {resolution}x{resolution}");
        Console.WriteLine($"  Hillshade: {hillshade:F2}, AO: {ao:F2}");
        if (withVariations) Console.WriteLine($"  Mode: 4 lighting variations per tile");

        var synth = new SynthesizedTrainingService(datasetDir!)
        {
            HillshadeStrength = hillshade,
            AmbientOcclusion = ao
        };

        var filesToProcess = new List<string>();

        if (!string.IsNullOrEmpty(inputPath))
        {
            filesToProcess.Add(inputPath);
        }
        else
        {
            var datasetFolder = Path.Combine(datasetDir!, "dataset");
            if (Directory.Exists(datasetFolder))
            {
                filesToProcess.AddRange(Directory.EnumerateFiles(datasetFolder, "*.json"));
            }
        }

        if (filesToProcess.Count == 0)
        {
            Console.WriteLine("Error: No JSON files found.");
            return 1;
        }

        var outputBase = outputPath ?? Path.Combine(datasetDir!, "synthesized");
        Directory.CreateDirectory(outputBase);

        int processed = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        foreach (var file in filesToProcess)
        {
            try
            {
                var tileName = Path.GetFileNameWithoutExtension(file);
                Console.Write($"  {tileName}... ");

                if (withVariations)
                {
                    await synth.ExportWithVariationsAsync(file, outputBase, resolution);
                    Console.WriteLine("OK (4 variations)");
                }
                else
                {
                    await synth.ExportPairAsync(file, outputBase, resolution);
                    Console.WriteLine("OK");
                }
                processed++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FAILED: {ex.Message}");
            }
        }

        sw.Stop();
        Console.WriteLine($"Processed {processed}/{filesToProcess.Count} tiles in {sw.Elapsed.TotalSeconds:F1}s");
        return 0;
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
        Console.WriteLine("  wowmapconverter wmo-info <wmo> [options]                List WMO groups and structure info");
        Console.WriteLine("  wowmapconverter vlm-export [options]                    Export VLM training dataset");
        Console.WriteLine("  wowmapconverter vlm-decode [options]                    Decode VLM JSON to ADT");
        Console.WriteLine("  wowmapconverter vlm-bake [options]                       Bake high-resolution minimap");
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
        Console.WriteLine("  --extended              Use extended converter (for v15/hybrid variants)");
        Console.WriteLine();
        Console.WriteLine("MDX Conversion Options:");
        Console.WriteLine("  --input, -i <path>      Input MDX file");
        Console.WriteLine("  --output, -o <path>     Output M2 path (also creates .skin file)");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  wowmapconverter convert World/Maps/Azeroth/Azeroth.wdt -o ./out");
        Console.WriteLine("  wowmapconverter convert-lk-to-alpha development.wdt --map-dir ./maps -o alpha.wdt");
        Console.WriteLine("  wowmapconverter convert-wmo castle01.wmo -o castle01_v17.wmo");
        Console.WriteLine("  wowmapconverter wmo-info ironforge.wmo -v");
        Console.WriteLine("  wowmapconverter convert-mdx Human/HumanMale.mdx -o HumanMale.m2");
    }
}
