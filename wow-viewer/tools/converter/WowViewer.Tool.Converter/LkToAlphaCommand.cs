using System.Diagnostics;
using System.IO;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tool.Converter;

internal static class LkToAlphaCommand
{
    public static void Run(string[] args)
    {
        try
        {
            LkToAlphaOptions options = ParseOptions(args);

            bool useMpq = !string.IsNullOrEmpty(options.ClientRoot) && !string.IsNullOrEmpty(options.MapName);

            if (!useMpq && string.IsNullOrEmpty(options.InputDir))
            {
                Console.Error.WriteLine("Error: Provide either --input <dir> (loose ADT files) or --client-root <dir> --map <name> (MPQ archives).");
                Environment.ExitCode = 1;
                return;
            }

            if (string.IsNullOrEmpty(options.OutputPath))
            {
                Console.Error.WriteLine("Error: --output <path> is required (output .wdt file path).");
                Environment.ExitCode = 1;
                return;
            }

            string outputPath = Path.GetFullPath(options.OutputPath);
            string outputWdlPath = Path.GetFullPath(options.OutputWdlPath ?? Path.ChangeExtension(outputPath, ".wdl"));
            string mapName = useMpq ? options.MapName! : Path.GetFileNameWithoutExtension(outputPath);

            Console.WriteLine("WowViewer.Tool.Converter convert-lk-to-alpha report");

            // Load target client for asset existence checks (filtering missing placements)
            HashSet<string>? targetFileSet = null;
            string? targetRoot = !string.IsNullOrEmpty(options.TargetClientRoot)
                ? Path.GetFullPath(options.TargetClientRoot)
                : null;
            if (targetRoot != null)
            {
                if (!Directory.Exists(targetRoot))
                {
                    Console.Error.WriteLine($"Error: Target client root not found: {targetRoot}");
                    Environment.ExitCode = 1;
                    return;
                }

                // Try standard MPQ catalog first
                using var targetCatalog = new NativeMpqService();
                targetCatalog.LoadArchives([targetRoot]);
                var knownFiles = targetCatalog.GetAllKnownFiles();

                if (knownFiles.Count > 0)
                {
                    targetFileSet = new HashSet<string>(knownFiles, StringComparer.OrdinalIgnoreCase);
                }
                else
                {
                    // Alpha clients use per-asset .ext.MPQ wrappers — scan for those
                    targetFileSet = ScanAlphaClientFiles(targetRoot);
                }

                Console.WriteLine($"  Target:   {targetRoot} ({targetFileSet.Count} files)");
            }

            var sw = Stopwatch.StartNew();
            var tiles = new Dictionary<(int, int), AlphaTileData>();
            int converted = 0;
            int failed = 0;
            int totalTiles = 0;
            var warnings = new List<string>();
            int missingModelNames = 0;
            int missingWmoNames = 0;

            if (useMpq)
            {
                string clientRoot = Path.GetFullPath(options.ClientRoot!);
                if (!Directory.Exists(clientRoot))
                {
                    Console.Error.WriteLine($"Error: Client root not found: {clientRoot}");
                    Environment.ExitCode = 1;
                    return;
                }

                Console.WriteLine($"  Client:   {clientRoot}");
                Console.WriteLine($"  Map:      {mapName}");
                Console.WriteLine($"  Output:   {outputPath}");
                Console.WriteLine($"  WDL:      {outputWdlPath}");
                Console.WriteLine($"  Verbose:  {options.Verbose}");
                if (options.TerrainOnly)
                    Console.WriteLine($"  Mode:     terrain-only (no placements)");

                using var catalog = new NativeMpqService();
                catalog.LoadArchives([clientRoot]);

                string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
                byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
                if (wdtBytes is null)
                {
                    Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
                    Environment.ExitCode = 1;
                    return;
                }

                using var wdtStream = new MemoryStream(wdtBytes, writable: false);

                int? limit = GetIntOption(args, "--limit", "-n");

                for (int ty = 0; ty < 64; ty++)
                {
                    for (int tx = 0; tx < 64; tx++)
                    {

                        string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tx}_{ty}.adt";
                        string tex0Virtual = $"World\\Maps\\{mapName}\\{mapName}_{tx}_{ty}_tex0.adt";
                        string obj0Virtual = $"World\\Maps\\{mapName}\\{mapName}_{tx}_{ty}_obj0.adt";

                        byte[]? adtBytes = catalog.ReadFile(adtVirtual);
                        if (adtBytes is null)
                            continue;

                        byte[]? tex0Bytes = catalog.ReadFile(tex0Virtual);
                        byte[]? obj0Bytes = catalog.ReadFile(obj0Virtual);

                        totalTiles++;

                        try
                        {
                            LkAdtData adtData = LkAdtReader.Read(adtBytes, tex0Bytes, obj0Bytes, tx, ty);

                            // Filter placements against target client if provided
                            if (targetFileSet != null)
                            {
                                var (filteredModels, filteredWmos, skippedModels, skippedWmos) =
                                    FilterPlacements(adtData, targetFileSet);
                                adtData = new LkAdtData
                                {
                                    MapName = adtData.MapName,
                                    TileX = adtData.TileX,
                                    TileY = adtData.TileY,
                                    TextureNames = adtData.TextureNames,
                                    ModelNames = filteredModels.Keys.ToList(),
                                    WorldModelNames = filteredWmos.Keys.ToList(),
                                    ModelPlacements = filteredModels.Values.ToList(),
                                    WorldModelPlacements = filteredWmos.Values.ToList(),
                                    Chunks = adtData.Chunks,
                                    MhdrFlags = adtData.MhdrFlags,
                                    MfboFlightBounds = adtData.MfboFlightBounds
                                };
                                missingModelNames += skippedModels;
                                missingWmoNames += skippedWmos;
                            }

                            AlphaTileData tileData = LkToAlphaConverter.ConvertTile(adtData, tx, ty);
                            tiles[(tx, ty)] = tileData;
                            converted++;

                            if (options.Verbose)
                                Console.WriteLine($"  Converted: {mapName}_{tx}_{ty} ({adtBytes.Length:N0} bytes)");

                            if (limit.HasValue && converted >= limit.Value)
                                break;
                        }
                        catch (Exception ex)
                        {
                            failed++;
                            warnings.Add($"{mapName}_{tx}_{ty}: {ex.Message}");
                            if (options.Verbose)
                                Console.Error.WriteLine($"  Error converting {mapName}_{tx}_{ty}: {ex}");
                        }
                    }

                    if (limit.HasValue && converted >= limit.Value)
                        break;
                }
            }
            else
            {
                string inputDir = Path.GetFullPath(options.InputDir!);

                if (!Directory.Exists(inputDir))
                {
                    Console.Error.WriteLine($"Error: Directory not found: {inputDir}");
                    Environment.ExitCode = 1;
                    return;
                }

                Console.WriteLine($"  Input:    {inputDir}");
                Console.WriteLine($"  Output:   {outputPath}");
                Console.WriteLine($"  WDL:      {outputWdlPath}");
                Console.WriteLine($"  Verbose:  {options.Verbose}");

                var adtFiles = Directory.GetFiles(inputDir, "*_*.adt", SearchOption.TopDirectoryOnly)
                    .Where(f => !f.Contains("_obj", StringComparison.OrdinalIgnoreCase) && !f.Contains("_tex", StringComparison.OrdinalIgnoreCase) && !f.Contains("_lod", StringComparison.OrdinalIgnoreCase))
                    .ToList();

                if (adtFiles.Count == 0)
                {
                    Console.Error.WriteLine("Error: No ADT files found in directory.");
                    Environment.ExitCode = 1;
                    return;
                }

                Console.WriteLine($"  Found {adtFiles.Count} ADT files.");
                totalTiles = adtFiles.Count;

                foreach (var adtFile in adtFiles)
                {
                    string fileName = Path.GetFileNameWithoutExtension(adtFile);
                    string[] parts = fileName.Split('_');
                    if (parts.Length < 3 || !int.TryParse(parts[^2], out int tileX) || !int.TryParse(parts[^1], out int tileY))
                    {
                        warnings.Add($"Cannot parse tile coords from: {fileName}");
                        continue;
                    }

                    try
                    {
                        byte[] adtBytes = File.ReadAllBytes(adtFile);
                        
                        string tex0File = Path.ChangeExtension(adtFile, "_tex0.adt");
                        byte[]? tex0Bytes = File.Exists(tex0File) ? File.ReadAllBytes(tex0File) : null;
                        
                        string obj0File = Path.ChangeExtension(adtFile, "_obj0.adt");
                        byte[]? obj0Bytes = File.Exists(obj0File) ? File.ReadAllBytes(obj0File) : null;

                        LkAdtData adtData = LkAdtReader.Read(adtBytes, tex0Bytes, obj0Bytes, tileX, tileY);

                        if (targetFileSet != null)
                        {
                            var (filteredModels, filteredWmos, skippedModels, skippedWmos) =
                                FilterPlacements(adtData, targetFileSet);
                            adtData = new LkAdtData
                            {
                                MapName = adtData.MapName,
                                TileX = adtData.TileX,
                                TileY = adtData.TileY,
                                TextureNames = adtData.TextureNames,
                                ModelNames = filteredModels.Keys.ToList(),
                                WorldModelNames = filteredWmos.Keys.ToList(),
                                ModelPlacements = filteredModels.Values.ToList(),
                                WorldModelPlacements = filteredWmos.Values.ToList(),
                                Chunks = adtData.Chunks,
                                MhdrFlags = adtData.MhdrFlags,
                                MfboFlightBounds = adtData.MfboFlightBounds
                            };
                            missingModelNames += skippedModels;
                            missingWmoNames += skippedWmos;
                        }

                        AlphaTileData tileData = LkToAlphaConverter.ConvertTile(adtData, tileX, tileY);
                        tiles[(tileX, tileY)] = tileData;
                        converted++;

                        if (options.Verbose)
                            Console.WriteLine($"  Converted: {fileName} ({adtBytes.Length:N0} bytes)");
                    }
                    catch (Exception ex)
                    {
                        failed++;
                        warnings.Add($"{fileName}: {ex.Message}");
                        if (options.Verbose)
                            Console.Error.WriteLine($"  Error converting {fileName}: {ex}");
                    }
                }
            }

            if (tiles.Count == 0)
            {
                Console.Error.WriteLine("Error: No tiles were successfully converted.");
                Environment.ExitCode = 1;
                return;
            }

            string outputDir = Path.GetDirectoryName(outputPath) ?? ".";
            Directory.CreateDirectory(outputDir);
            string wdlOutputDir = Path.GetDirectoryName(outputWdlPath) ?? ".";
            Directory.CreateDirectory(wdlOutputDir);

            if (options.TerrainOnly)
            {
                tiles = tiles.ToDictionary(
                    kvp => kvp.Key,
                    kvp => new AlphaTileData(
                        kvp.Value.SourcePath,
                        kvp.Value.Heightmap,
                        kvp.Value.McalAlphaPack,
                        kvp.Value.MclyTextureIds,
                        kvp.Value.MclyLayerMask,
                        kvp.Value.HoleMask,
                        kvp.Value.TextureNames,
                        modelPlacements: [],
                        worldModelPlacements: [],
                        liquidChunks: kvp.Value.LiquidChunks,
                        mcnrNormalXyz: kvp.Value.McnrNormalXyz,
                        mcshShadowMask256: kvp.Value.McshShadowMask256,
                        mcshShadowMask1024: kvp.Value.McshShadowMask1024,
                        areaIds: kvp.Value.AreaIds,
                        mccvRgb: kvp.Value.MccvRgb,
                        mclvLightingBytes: kvp.Value.MclvLightingBytes));
            }

            byte[] wdtData = AlphaWdtWriter.Build(mapName, tiles);
            File.WriteAllBytes(outputPath, wdtData);
            var wdlTiles = tiles
                .OrderBy(t => t.Key.Item2 * 64 + t.Key.Item1)
                .Select(t => WdlWriter.ExtractTileHeightsFromAlpha(t.Value.Heightmap, t.Key.Item1, t.Key.Item2))
                .ToList();
            byte[] wdlData = WdlWriter.Build(wdlTiles);
            File.WriteAllBytes(outputWdlPath, wdlData);

            sw.Stop();
            Console.WriteLine($"  Converted: {converted}/{totalTiles} tiles");
            Console.WriteLine($"  Failed:    {failed} tiles");
            Console.WriteLine($"  Output:    {outputPath} ({wdtData.Length:N0} bytes)");
            Console.WriteLine($"  WDL:       {outputWdlPath} ({wdlData.Length:N0} bytes, {wdlTiles.Count} tiles)");
            Console.WriteLine($"  Elapsed:   {sw.ElapsedMilliseconds}ms");

            if (targetFileSet != null)
                Console.WriteLine($"  Assets:   {missingModelNames} MDX + {missingWmoNames} WMO placements filtered (missing in target)");

            if (warnings.Count > 0)
            {
                Console.WriteLine($"  Warnings:  {warnings.Count}");
                foreach (var w in warnings.Take(10))
                    Console.WriteLine($"    {w}");
                if (warnings.Count > 10)
                    Console.WriteLine($"    ... and {warnings.Count - 10} more");
            }

            if (options.BundleTilesets)
            {
                string tilesetRoot = Path.Combine(outputDir, "tilesets", mapName);
                string sourceForTextures = options.ClientRoot ?? options.InputDir ?? ".";
                BundleTilesets(tiles, sourceForTextures, mapName, tilesetRoot);

                // Fix up texture names to point to local tilesets directory
                string tilesetPrefix = $"tilesets\\{mapName}\\";
                tiles = tiles.ToDictionary(
                    kvp => kvp.Key,
                    kvp =>
                    {
                        var fixedTextures = kvp.Value.TextureNames
                            .Select(t => tilesetPrefix + t.TrimStart('\\'))
                            .ToList();
                        return new AlphaTileData(
                            kvp.Value.SourcePath, kvp.Value.Heightmap,
                            kvp.Value.McalAlphaPack, kvp.Value.MclyTextureIds,
                            kvp.Value.MclyLayerMask, kvp.Value.HoleMask,
                            fixedTextures, kvp.Value.ModelPlacements,
                            kvp.Value.WorldModelPlacements, kvp.Value.LiquidChunks,
                            mcnrNormalXyz: kvp.Value.McnrNormalXyz,
                            mcshShadowMask256: kvp.Value.McshShadowMask256,
                            mcshShadowMask1024: kvp.Value.McshShadowMask1024,
                            areaIds: kvp.Value.AreaIds,
                            mccvRgb: kvp.Value.MccvRgb,
                            mclvLightingBytes: kvp.Value.MclvLightingBytes);
                    });
                Console.WriteLine($"  Tilesets: extracted and fixed up paths in {tilesetRoot}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            if (args.Contains("--verbose") || args.Contains("-v"))
                Console.Error.WriteLine(ex.StackTrace);
            Environment.ExitCode = 1;
        }
    }


    private static (Dictionary<string, LkMddfEntry> models, Dictionary<string, LkModfEntry> wmos, int skippedModels, int skippedWmos)
        FilterPlacements(LkAdtData adtData, HashSet<string> targetFileSet)
    {
        var filteredModelPaths = new Dictionary<string, LkMddfEntry>(StringComparer.OrdinalIgnoreCase);
        var filteredWmoPaths = new Dictionary<string, LkModfEntry>(StringComparer.OrdinalIgnoreCase);
        int skippedModels = 0, skippedWmos = 0;

        for (int i = 0; i < adtData.ModelPlacements.Count; i++)
        {
            var p = adtData.ModelPlacements[i];
            string path = p.NameId >= 0 && p.NameId < adtData.ModelNames.Count
                ? adtData.ModelNames[p.NameId] : $"unknown_{p.NameId}";

            if (targetFileSet.Contains(path) || targetFileSet.Contains(path.Replace('\\', '/')))
                filteredModelPaths.TryAdd(path, p);
            else
                skippedModels++;
        }

        for (int i = 0; i < adtData.WorldModelPlacements.Count; i++)
        {
            var p = adtData.WorldModelPlacements[i];
            string path = p.NameId >= 0 && p.NameId < adtData.WorldModelNames.Count
                ? adtData.WorldModelNames[p.NameId] : $"unknown_{p.NameId}";

            if (targetFileSet.Contains(path) || targetFileSet.Contains(path.Replace('\\', '/')))
                filteredWmoPaths.TryAdd(path, p);
            else
                skippedWmos++;
        }

        return (filteredModelPaths, filteredWmoPaths, skippedModels, skippedWmos);
    }

    private static HashSet<string> ScanAlphaClientFiles(string clientRoot)
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string dataDir = Path.Combine(clientRoot, "Data");
        if (!Directory.Exists(dataDir)) return set;

        string[] alphaSuffixes = [".wmo.mpq", ".wmo.MPQ", ".mdx.mpq", ".mdx.MPQ",
                                   ".mdl.mpq", ".mdl.MPQ", ".m2.mpq", ".m2.MPQ",
                                   ".wdt.mpq", ".wdt.MPQ", ".wdl.mpq", ".wdl.MPQ"];

        try
        {
            foreach (string mpqFile in Directory.EnumerateFiles(dataDir, "*.mpq", SearchOption.AllDirectories)
                .Concat(Directory.EnumerateFiles(dataDir, "*.MPQ", SearchOption.AllDirectories)))
            {
                string fileName = Path.GetFileName(mpqFile);
                string matchedSuffix = "";
                foreach (var suffix in alphaSuffixes)
                {
                    if (fileName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                    {
                        matchedSuffix = suffix;
                        break;
                    }
                }
                if (matchedSuffix == "") continue;

                // Strip .MPQ suffix to get virtual path
                string relative = Path.GetRelativePath(dataDir, mpqFile);
                string virtualPath = relative[..^4]; // remove .MPQ

                // Also strip the extension variant to get the base path
                // e.g., World/wmo/Azeroth/Building/test.wmo.MPQ → World/wmo/Azeroth/Building/test.wmo
                set.Add(virtualPath);
                set.Add(virtualPath.Replace('\\', '/'));
            }
        }
        catch { }

        return set;
    }

    private static void BundleTilesets(
        Dictionary<(int, int), AlphaTileData> tiles,
        string sourceRoot, string mapName, string tilesetRoot)
    {
        var texturePaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var tile in tiles.Values)
        {
            foreach (string tex in tile.TextureNames)
                texturePaths.Add(tex);
        }

        if (texturePaths.Count == 0) return;

        bool isMpq = Directory.Exists(sourceRoot) &&
            (Directory.Exists(Path.Combine(sourceRoot, "Data")) ||
             Directory.GetFiles(sourceRoot, "*.mpq", SearchOption.TopDirectoryOnly).Length > 0);

        byte[]? ReadTexture(string path)
        {
            if (isMpq)
            {
                using var cat = new NativeMpqService();
                cat.LoadArchives([sourceRoot]);
                return cat.ReadFile(path) ?? cat.ReadFile(path.Replace('\\', '/'));
            }
            else
            {
                foreach (string candidate in new[] {
                    Path.Combine(sourceRoot, path),
                    Path.Combine(sourceRoot, path.Replace('\\', '/'))})
                {
                    if (File.Exists(candidate))
                        return File.ReadAllBytes(candidate);
                }
                return null;
            }
        }

        Directory.CreateDirectory(tilesetRoot);

        int extracted = 0, failed = 0;
        foreach (string texPath in texturePaths)
        {
            string normPath = texPath.Replace('/', '\\').TrimStart('\\');
            string localPath = Path.Combine(tilesetRoot, normPath);
            string? localDir = Path.GetDirectoryName(localPath);
            if (localDir != null) Directory.CreateDirectory(localDir);

            byte[]? data = ReadTexture(normPath);
            if (data != null)
            {
                File.WriteAllBytes(localPath, data);
                extracted++;
            }
            else
            {
                failed++;
            }
        }

        Console.WriteLine($"    Extracted: {extracted}/{texturePaths.Count} textures to {tilesetRoot}" +
            (failed > 0 ? $" ({failed} missing)" : ""));
    }

    private static LkToAlphaOptions ParseOptions(string[] args)
    {
            return new LkToAlphaOptions(
                InputDir: GetOption(args, "--input", "-i"),
                OutputPath: GetOption(args, "--output", "-o"),
                OutputWdlPath: GetOption(args, "--output-wdl", "--wdl"),
                ClientRoot: GetOption(args, "--client-root", "-c"),
                TargetClientRoot: GetOption(args, "--target-client-root", "-tcr"),
                MapName: GetOption(args, "--map", "-m"),
                Verbose: HasFlag(args, "--verbose") || HasFlag(args, "-v"),
                TerrainOnly: HasFlag(args, "--terrain-only") || HasFlag(args, "-to"),
                BundleTilesets: HasFlag(args, "--bundle-tilesets") || HasFlag(args, "-bt"));
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], longName, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(args[i], shortName, StringComparison.OrdinalIgnoreCase))
            {
                return args[i + 1];
            }
        }
        return null;
    }

    private static int? GetIntOption(string[] args, string longName, string shortName)
    {
        string? value = GetOption(args, longName, shortName);
        return int.TryParse(value, out int result) ? result : null;
    }

    private static bool HasFlag(string[] args, string name)
    {
        foreach (var arg in args)
        {
            if (string.Equals(arg, name, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    private readonly record struct LkToAlphaOptions(
        string? InputDir,
        string? OutputPath,
        string? OutputWdlPath,
        string? ClientRoot,
        string? TargetClientRoot,
        string? MapName,
        bool Verbose,
        bool TerrainOnly,
        bool BundleTilesets);
}
