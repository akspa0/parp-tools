using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.M2;
using WowViewer.Core.Mdx;
using WowViewer.Core.Wmo;

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
            AreaIdMapper areaIdMapper = new();

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

                targetFileSet = BuildTargetClientFileSet(targetRoot);
                Console.WriteLine($"  Target:   {targetRoot} ({targetFileSet.Count} files from target archives + wrapper scan)");
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
                                var (mdlNames, mdlPlacements, wmoNames, wmoPlacements, filteredChunks, skippedModels, skippedWmos) =
                                    FilterPlacements(adtData, targetFileSet, options.BundleM2s, options.BundleWmos);
                                adtData = new LkAdtData
                                {
                                    MapName = adtData.MapName,
                                    TileX = adtData.TileX,
                                    TileY = adtData.TileY,
                                    TextureNames = adtData.TextureNames,
                                    ModelNames = mdlNames,
                                    WorldModelNames = wmoNames,
                                    ModelPlacements = mdlPlacements,
                                    WorldModelPlacements = wmoPlacements,
                                    Chunks = filteredChunks,
                                    MhdrFlags = adtData.MhdrFlags,
                                    MfboFlightBounds = adtData.MfboFlightBounds
                                };
                                missingModelNames += skippedModels;
                                missingWmoNames += skippedWmos;
                            }

                            AlphaTileData tileData = LkToAlphaConverter.ConvertTile(adtData, tx, ty, areaIdMapper, mapName);
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
                            var (mdlNames, mdlPlacements, wmoNames, wmoPlacements, filteredChunks, skippedModels, skippedWmos) =
                                FilterPlacements(adtData, targetFileSet, options.BundleM2s, options.BundleWmos);
                            adtData = new LkAdtData
                            {
                                MapName = adtData.MapName,
                                TileX = adtData.TileX,
                                TileY = adtData.TileY,
                                TextureNames = adtData.TextureNames,
                                ModelNames = mdlNames,
                                WorldModelNames = wmoNames,
                                ModelPlacements = mdlPlacements,
                                WorldModelPlacements = wmoPlacements,
                                Chunks = filteredChunks,
                                MhdrFlags = adtData.MhdrFlags,
                                MfboFlightBounds = adtData.MfboFlightBounds
                            };
                            missingModelNames += skippedModels;
                            missingWmoNames += skippedWmos;
                        }

                        AlphaTileData tileData = LkToAlphaConverter.ConvertTile(adtData, tileX, tileY, areaIdMapper, mapName);
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
                        mclvLightingBytes: kvp.Value.MclvLightingBytes,
                        holeFullMasks: kvp.Value.HoleFullMasks));
            }

            string sourceForAssets = options.ClientRoot ?? options.InputDir ?? ".";
            string? tilesetRoot = null;
            string? mdxRoot = null;
            string? wmoRoot = null;
            if (options.BundleTilesets)
            {
                tilesetRoot = Path.Combine(outputDir, "tilesets", mapName);
                BundleTilesets(tiles, sourceForAssets, mapName, tilesetRoot);
            }

            Dictionary<string, string>? bundledMdxPaths = null;
            if (options.BundleM2s && !options.TerrainOnly)
            {
                mdxRoot = Path.Combine(outputDir, "mdxs", mapName);
                bundledMdxPaths = BundleModels(tiles, sourceForAssets, mapName, mdxRoot, targetFileSet);
            }

            Dictionary<string, string>? bundledWmoPaths = null;
            if (options.BundleWmos && !options.TerrainOnly)
            {
                wmoRoot = Path.Combine(outputDir, "wmos", mapName);
                bundledWmoPaths = BundleWorldModels(tiles, sourceForAssets, mapName, wmoRoot, targetFileSet, options.Verbose);
            }

            if (tilesetRoot is not null || bundledMdxPaths is not null || bundledWmoPaths is not null)
            {
                string? tilesetPrefix = tilesetRoot is null ? null : $"World\\Maps\\{mapName}\\tilesets\\{mapName}\\";
                tiles = tiles.ToDictionary(
                    kvp => kvp.Key,
                    kvp => RewriteBundledAssetPaths(kvp.Value, tilesetPrefix, bundledMdxPaths, bundledWmoPaths, targetFileSet));
            }

            (int finalPlaceholderModels, int finalPlaceholderWmos) = CountPlaceholderPlacements(tiles);

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
                Console.WriteLine($"  Assets:   {finalPlaceholderModels} MDX + {finalPlaceholderWmos} WMO placements mapped to placeholders ({PlaceholderMdx}, {PlaceholderWmo})");

            if (warnings.Count > 0)
            {
                Console.WriteLine($"  Warnings:  {warnings.Count}");
                foreach (var w in warnings.Take(10))
                    Console.WriteLine($"    {w}");
                if (warnings.Count > 10)
                    Console.WriteLine($"    ... and {warnings.Count - 10} more");
            }

            if (tilesetRoot != null)
                Console.WriteLine($"  Tilesets: extracted and fixed up paths in {tilesetRoot}");
            if (mdxRoot != null)
                Console.WriteLine($"  MDXs:     converted/copied and bundled in {mdxRoot}");
            if (wmoRoot != null)
                Console.WriteLine($"  WMOs:     converted and bundled in {wmoRoot}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            if (args.Contains("--verbose") || args.Contains("-v"))
                Console.Error.WriteLine(ex.StackTrace);
            Environment.ExitCode = 1;
        }
    }


    private const string PlaceholderMdx = "World\\ArtTest\\Boxtest\\xyz.mdx";
    private const string PlaceholderWmo = "World\\wmo\\Dungeon\\test\\missingwmo.wmo";

    private static (List<string> names, List<LkMddfEntry> placements, List<string> wmoNames, List<LkModfEntry> wmoPlacements, List<LkMcnkData> chunks, int mappedModels, int mappedWmos)
        FilterPlacements(LkAdtData adtData, HashSet<string> targetFileSet, bool preserveSourceModelPaths = false, bool preserveSourceWmoPaths = false)
    {
        var names = new List<string>();
        var nameIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var placementList = new List<LkMddfEntry>();
        int[] filteredModelIndexBySourceIndex = Enumerable.Repeat(-1, adtData.ModelPlacements.Count).ToArray();
        var seenModelUniqueIds = new HashSet<int>();
        var wmoNames = new List<string>();
        var wmoNameIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var wmoPlacementList = new List<LkModfEntry>();
        int[] filteredWmoIndexBySourceIndex = Enumerable.Repeat(-1, adtData.WorldModelPlacements.Count).ToArray();
        var seenWmoUniqueIds = new HashSet<int>();
        int mappedModels = 0, mappedWmos = 0;

        static int GetOrAddNameIndex(string path, List<string> names, Dictionary<string, int> index)
        {
            if (index.TryGetValue(path, out int existing))
                return existing;

            int next = names.Count;
            names.Add(path);
            index[path] = next;
            return next;
        }

        string ResolveName(int nameId, IReadOnlyList<string> nameTable)
        {
            return nameId >= 0 && nameId < nameTable.Count
                ? nameTable[nameId] : $"unknown_{nameId}";
        }

        bool PathExists(string path) =>
            targetFileSet.Contains(path) || targetFileSet.Contains(path.Replace('\\', '/'));

        string ResolveModelPath(string path)
        {
            string normalized = NormalizeVirtualPath(path);
            if (PathExists(normalized))
                return normalized;

            string extension = Path.GetExtension(normalized);
            if (!extension.Equals(".m2", StringComparison.OrdinalIgnoreCase)
                && !extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
                && !extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
            {
                return PlaceholderMdx;
            }

            string pathWithoutExtension = normalized[..^extension.Length];
            string[] candidateExtensions = [".mdx", ".mdl", ".m2"];
            foreach (string candidateExtension in candidateExtensions)
            {
                string candidatePath = pathWithoutExtension + candidateExtension;
                if (PathExists(candidatePath))
                    return candidatePath;
            }

            return PlaceholderMdx;
        }

        for (int i = 0; i < adtData.ModelPlacements.Count; i++)
        {
            var p = adtData.ModelPlacements[i];
            if (!seenModelUniqueIds.Add(p.UniqueId))
                continue;

            string path = ResolveName(p.NameId, adtData.ModelNames);
            string mappedPath = preserveSourceModelPaths
                ? ResolveBundledModelSourcePath(path)
                : ResolveModelPath(path);
            int mappedNameId = GetOrAddNameIndex(mappedPath, names, nameIndex);

            if (!preserveSourceModelPaths && !ReferenceEquals(mappedPath, path) && !string.Equals(mappedPath, path, StringComparison.OrdinalIgnoreCase))
                mappedModels++;

            filteredModelIndexBySourceIndex[i] = placementList.Count;
            placementList.Add(new LkMddfEntry(
                mappedNameId, p.UniqueId, p.Position, p.Rotation, p.Scale));
        }

        for (int i = 0; i < adtData.WorldModelPlacements.Count; i++)
        {
            var p = adtData.WorldModelPlacements[i];
            if (!seenWmoUniqueIds.Add(p.UniqueId))
                continue;

            string path = NormalizeVirtualPath(ResolveName(p.NameId, adtData.WorldModelNames));
            string mappedPath = preserveSourceWmoPaths
                ? (PathExists(path) ? path : path)
                : (PathExists(path) ? path : PlaceholderWmo);
            int mappedNameId = GetOrAddNameIndex(mappedPath, wmoNames, wmoNameIndex);

            if (!preserveSourceWmoPaths && !ReferenceEquals(mappedPath, path) && !string.Equals(mappedPath, path, StringComparison.OrdinalIgnoreCase))
                mappedWmos++;

            filteredWmoIndexBySourceIndex[i] = wmoPlacementList.Count;
            wmoPlacementList.Add(new LkModfEntry(
                mappedNameId, p.UniqueId, p.Position, p.Rotation,
                p.BoundsMin, p.BoundsMax, p.Flags, p.DoodadSet, p.NameSet, p.Scale));
        }

        List<LkMcnkData> filteredChunks = adtData.Chunks
            .Select(chunk => new LkMcnkData
            {
                IndexX = chunk.IndexX,
                IndexY = chunk.IndexY,
                Flags = chunk.Flags,
                AreaId = chunk.AreaId,
                NLayers = chunk.NLayers,
                HoleMask = chunk.HoleMask,
                BaseHeight = chunk.BaseHeight,
                Heights = chunk.Heights,
                Normals = chunk.Normals,
                ShadowMap = chunk.ShadowMap,
                AlphaMapData = chunk.AlphaMapData,
                AlphaMapSize = chunk.AlphaMapSize,
                Layers = chunk.Layers,
                DoodadRefs = RemapChunkRefs(chunk.DoodadRefs, filteredModelIndexBySourceIndex),
                WorldModelRefs = RemapChunkRefs(chunk.WorldModelRefs, filteredWmoIndexBySourceIndex),
                LiquidData = chunk.LiquidData,
                MccvColors = chunk.MccvColors,
                MclvLighting = chunk.MclvLighting,
                PosX = chunk.PosX,
                PosY = chunk.PosY,
                PosZ = chunk.PosZ,
            })
            .ToList();

        return (names, placementList, wmoNames, wmoPlacementList, filteredChunks, mappedModels, mappedWmos);

        string ResolveBundledModelSourcePath(string path)
        {
            string normalized = NormalizeVirtualPath(path);
            if (PathExists(normalized))
                return ResolveModelPath(path);

            string extension = Path.GetExtension(normalized);
            if (!extension.Equals(".m2", StringComparison.OrdinalIgnoreCase)
                && !extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
                && !extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
            {
                return PlaceholderMdx;
            }

            return normalized;
        }
    }

    private static IReadOnlyList<int> RemapChunkRefs(IReadOnlyList<int> refs, IReadOnlyList<int> filteredIndexBySourceIndex)
    {
        if (refs.Count == 0)
            return [];

        List<int> remapped = [];
        foreach (int refIndex in refs)
        {
            if ((uint)refIndex >= (uint)filteredIndexBySourceIndex.Count)
                continue;

            int filteredIndex = filteredIndexBySourceIndex[refIndex];
            if (filteredIndex >= 0)
                remapped.Add(filteredIndex);
        }

        return remapped.Count > 0 ? remapped : [];
    }

    private static HashSet<string> BuildTargetClientFileSet(string clientRoot)
    {
        HashSet<string> files = new(StringComparer.OrdinalIgnoreCase);

        AddLooseTargetFiles(files, clientRoot);
        AddAlphaWrapperTargetFiles(files, clientRoot);
        AddArchivedTargetFiles(files, clientRoot);

        return files;
    }

    private static void AddArchivedTargetFiles(HashSet<string> files, string clientRoot)
    {
        List<string> searchRoots = BuildArchiveSearchRoots(clientRoot);

        try
        {
            using IArchiveCatalog archiveCatalog = new NativeMpqServiceFactory().Create();
            ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(
                archiveCatalog,
                searchRoots,
                new ArchiveCatalogBootstrapOptions(LoadCachedEntries: false, PersistListfileCache: false));

            foreach (string file in bootstrap.AllFiles)
            {
                if (string.IsNullOrWhiteSpace(file))
                    continue;

                AddVirtualPath(files, file);
            }
        }
        catch
        {
        }
    }

    private static void AddLooseTargetFiles(HashSet<string> files, string clientRoot)
    {
        string dataDir = Path.Combine(clientRoot, "Data");
        if (!Directory.Exists(clientRoot))
            return;

        try
        {
            foreach (string filePath in Directory.EnumerateFiles(clientRoot, "*", SearchOption.AllDirectories))
            {
                if (filePath.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
                    continue;

                string relativePath = Directory.Exists(dataDir) && filePath.StartsWith(dataDir, StringComparison.OrdinalIgnoreCase)
                    ? Path.GetRelativePath(dataDir, filePath)
                    : Path.GetRelativePath(clientRoot, filePath);

                AddVirtualPath(files, relativePath);
            }
        }
        catch
        {
        }
    }

    private static void AddAlphaWrapperTargetFiles(HashSet<string> files, string clientRoot)
    {
        string dataDir = Path.Combine(clientRoot, "Data");
        if (!Directory.Exists(dataDir))
            return;

        string[] alphaSuffixes = [".wmo.mpq", ".wmo.MPQ", ".mdx.mpq", ".mdx.MPQ",
                                   ".mdl.mpq", ".mdl.MPQ", ".m2.mpq", ".m2.MPQ",
                                   ".wdt.mpq", ".wdt.MPQ", ".wdl.mpq", ".wdl.MPQ"];

        try
        {
            foreach (string mpqFile in Directory.EnumerateFiles(dataDir, "*.mpq", SearchOption.AllDirectories)
                .Concat(Directory.EnumerateFiles(dataDir, "*.MPQ", SearchOption.AllDirectories)))
            {
                string fileName = Path.GetFileName(mpqFile);
                bool isWrapper = alphaSuffixes.Any(suffix => fileName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase));
                if (!isWrapper)
                    continue;

                string relativePath = Path.GetRelativePath(dataDir, mpqFile);
                string virtualPath = relativePath[..^4];
                AddVirtualPath(files, virtualPath);
            }
        }
        catch
        {
        }
    }

    private static List<string> BuildArchiveSearchRoots(string clientRoot)
    {
        List<string> roots = [];
        string dataRoot = Path.Combine(clientRoot, "Data");
        if (Directory.Exists(dataRoot))
            roots.Add(dataRoot);

        if (!string.Equals(clientRoot, dataRoot, StringComparison.OrdinalIgnoreCase))
            roots.Add(clientRoot);

        return roots.Count > 0 ? roots : [clientRoot];
    }

    private static string? ResolveTargetListfilePath()
    {
        List<string> candidates =
        [
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer", "community-listfile-withcapitals.csv"),
            Path.Combine(AppContext.BaseDirectory, "community-listfile-withcapitals.csv"),
            Path.Combine(AppContext.BaseDirectory, "listfile.csv"),
        ];

        string? current = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(current))
        {
            candidates.Add(Path.Combine(current, "wow-viewer", "libs", "wowdev", "wow-listfile", "listfile.txt"));
            candidates.Add(Path.Combine(current, "libs", "wowdev", "wow-listfile", "listfile.txt"));
            candidates.Add(Path.Combine(current, "community-listfile-withcapitals.csv"));
            candidates.Add(Path.Combine(current, "listfile.csv"));
            current = Directory.GetParent(current)?.FullName;
        }

        foreach (string candidate in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (File.Exists(candidate))
                return candidate;
        }

        return null;
    }

    private static string NormalizeVirtualPath(string path)
    {
        return path.Replace('/', '\\').Trim().TrimStart('\\');
    }

    private static string ChangeVirtualExtension(string path, string extension)
    {
        string currentExtension = Path.GetExtension(path);
        return string.IsNullOrEmpty(currentExtension)
            ? path + extension
            : path[..^currentExtension.Length] + extension;
    }

    private static string BuildDefaultM2SkinPath(string modelPath)
    {
        string currentExtension = Path.GetExtension(modelPath);
        return string.IsNullOrEmpty(currentExtension)
            ? modelPath + "00.skin"
            : modelPath[..^currentExtension.Length] + "00.skin";
    }

    private static void WriteFixedAscii(BinaryWriter writer, string value, int size)
    {
        byte[] bytes = Encoding.ASCII.GetBytes(value ?? string.Empty);
        if (bytes.Length >= size)
        {
            writer.Write(bytes, 0, size - 1);
            writer.Write((byte)0);
            return;
        }

        writer.Write(bytes);
        writer.Write(new byte[size - bytes.Length]);
    }

    private static void AddVirtualPath(HashSet<string> files, string path)
    {
        string normalized = NormalizeVirtualPath(path);
        if (string.IsNullOrWhiteSpace(normalized))
            return;

        files.Add(normalized);
        files.Add(normalized.Replace('\\', '/'));
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

        bool useArchives = IsArchiveBackedSource(sourceRoot);
        using NativeMpqService? catalog = useArchives ? CreateSourceCatalog(sourceRoot) : null;

        Directory.CreateDirectory(tilesetRoot);

        int extracted = 0, failed = 0, normalized = 0, resized = 0, specularReencoded = 0;
        foreach (string texPath in texturePaths)
        {
            string normPath = texPath.Replace('/', '\\').TrimStart('\\');
            string localPath = Path.Combine(tilesetRoot, normPath);
            string? localDir = Path.GetDirectoryName(localPath);
            if (localDir != null) Directory.CreateDirectory(localDir);

            byte[]? data = ReadSourceAsset(sourceRoot, catalog, normPath);
            if (data != null)
            {
                AlphaBlpCompatibilityResult compatibility = AlphaBlpCompatibilityService.NormalizeForAlphaClient(normPath, data);
                File.WriteAllBytes(localPath, compatibility.Data);
                extracted++;
                if (compatibility.Rewritten)
                {
                    normalized++;
                    if (compatibility.Resized)
                        resized++;
                    if (compatibility.SpecularReencoded)
                        specularReencoded++;
                }
            }
            else
            {
                failed++;
            }
        }

        Console.WriteLine($"    Extracted: {extracted}/{texturePaths.Count} textures to {tilesetRoot}" +
            (failed > 0 ? $" ({failed} missing)" : ""));
        if (normalized > 0)
            Console.WriteLine($"    Re-encoded: {normalized} textures for 0.5.x compatibility ({specularReencoded} specular, {resized} resized)");
    }

    private static Dictionary<string, string> BundleWorldModels(
        Dictionary<(int, int), AlphaTileData> tiles,
        string sourceRoot,
        string mapName,
        string wmoRoot,
        HashSet<string>? targetFileSet,
        bool verbose)
    {
        HashSet<string> wmoPaths = new(StringComparer.OrdinalIgnoreCase);
        foreach (AlphaTileData tile in tiles.Values)
        {
            foreach (AlphaWorldModelPlacement placement in tile.WorldModelPlacements)
            {
                string normalized = NormalizeVirtualPath(placement.ModelPath);
                if (string.IsNullOrWhiteSpace(normalized) || string.Equals(normalized, PlaceholderWmo, StringComparison.OrdinalIgnoreCase))
                    continue;

                if (targetFileSet is not null && (targetFileSet.Contains(normalized) || targetFileSet.Contains(normalized.Replace('\\', '/'))))
                    continue;

                wmoPaths.Add(normalized);
            }
        }

        if (wmoPaths.Count == 0)
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        bool useArchives = IsArchiveBackedSource(sourceRoot);
        using NativeMpqService? catalog = useArchives ? CreateSourceCatalog(sourceRoot) : null;

        Directory.CreateDirectory(wmoRoot);

        Dictionary<string, string> bundledPaths = new(StringComparer.OrdinalIgnoreCase);
        int converted = 0;
        int copied = 0;
        int missing = 0;
        int failed = 0;
        int texturesExtracted = 0;
        int texturesMissing = 0;
        int texturesNormalized = 0;
        int texturesResized = 0;
        int texturesSpecularReencoded = 0;

        foreach (string wmoPath in wmoPaths)
        {
            try
            {
                byte[]? rootBytes = ReadSourceAsset(sourceRoot, catalog, wmoPath);
                if (rootBytes is null)
                {
                    missing++;
                    bundledPaths[wmoPath] = PlaceholderWmo;
                    continue;
                }

                byte[] outputBytes;
                using (MemoryStream rootStream = new(rootBytes, writable: false))
                {
                    WmoSummary summary = WmoSummaryReader.Read(rootStream, wmoPath);
                    if (summary.Version == 17)
                    {
                        int groupCount = summary.ReportedGroupCount > 0 ? summary.ReportedGroupCount : summary.GroupInfoCount;
                        if (groupCount <= 0)
                            throw new InvalidDataException($"WMO '{wmoPath}' does not report any groups for v17 conversion.");

                        List<byte[]> groupBytes = new(groupCount);
                        for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
                        {
                            string groupPath = BuildWmoGroupPath(wmoPath, groupIndex);
                            byte[]? groupData = ReadSourceAsset(sourceRoot, catalog, groupPath);
                            if (groupData is null)
                                throw new FileNotFoundException($"Missing WMO group '{groupPath}' for root '{wmoPath}'.", groupPath);

                            groupBytes.Add(groupData);
                        }

                        outputBytes = WmoV17ToV14Converter.Convert(rootBytes, groupBytes, wmoPath);
                        converted++;
                    }
                    else if (summary.Version == 14)
                    {
                        outputBytes = rootBytes;
                        copied++;
                    }
                    else
                    {
                        throw new InvalidDataException($"WMO '{wmoPath}' uses unsupported root version '{summary.Version?.ToString() ?? "unknown"}' for 0.5.x bundling.");
                    }
                }

                string bundledVirtualPath = NormalizeVirtualPath(Path.Combine("World", "Maps", mapName, "wmos", mapName, wmoPath));
                string localPath = Path.Combine(wmoRoot, wmoPath);
                string? localDir = Path.GetDirectoryName(localPath);
                if (!string.IsNullOrWhiteSpace(localDir))
                    Directory.CreateDirectory(localDir);

                outputBytes = BundleWorldModelTexturesAndRewriteRoot(
                    outputBytes,
                    sourceRoot,
                    catalog,
                    wmoPath,
                    bundledVirtualPath,
                    localPath,
                    ref texturesExtracted,
                    ref texturesMissing,
                    ref texturesNormalized,
                    ref texturesResized,
                    ref texturesSpecularReencoded);

                File.WriteAllBytes(localPath, outputBytes);
                bundledPaths[wmoPath] = bundledVirtualPath;
            }
            catch (Exception ex)
            {
                failed++;
                bundledPaths[wmoPath] = PlaceholderWmo;
                if (verbose)
                    Console.WriteLine($"      WMO bundle failed: {wmoPath} ({ex.GetType().Name}: {ex.Message})");
            }
        }

        Console.WriteLine($"    WMOs: converted {converted}, copied {copied}, missing {missing}, failed {failed} into {wmoRoot}");
        if (texturesExtracted > 0 || texturesMissing > 0)
        {
            Console.WriteLine($"    WMO textures: extracted {texturesExtracted}" +
                (texturesMissing > 0 ? $", missing {texturesMissing}" : string.Empty) +
                (texturesNormalized > 0 ? $", re-encoded {texturesNormalized} ({texturesSpecularReencoded} specular, {texturesResized} resized)" : string.Empty));
        }
        return bundledPaths;
    }

    private static Dictionary<string, string> BundleModels(
        Dictionary<(int, int), AlphaTileData> tiles,
        string sourceRoot,
        string mapName,
        string mdxRoot,
        HashSet<string>? targetFileSet)
    {
        HashSet<string> modelPaths = new(StringComparer.OrdinalIgnoreCase);
        foreach (AlphaTileData tile in tiles.Values)
        {
            foreach (AlphaModelPlacement placement in tile.ModelPlacements)
            {
                string normalized = NormalizeVirtualPath(placement.ModelPath);
                if (string.IsNullOrWhiteSpace(normalized) || string.Equals(normalized, PlaceholderMdx, StringComparison.OrdinalIgnoreCase))
                    continue;

                if (targetFileSet is not null && (targetFileSet.Contains(normalized) || targetFileSet.Contains(normalized.Replace('\\', '/'))))
                    continue;

                modelPaths.Add(normalized);
            }
        }

        if (modelPaths.Count == 0)
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        bool useArchives = IsArchiveBackedSource(sourceRoot);
        using NativeMpqService? catalog = useArchives ? CreateSourceCatalog(sourceRoot) : null;

        Directory.CreateDirectory(mdxRoot);

        Dictionary<string, string> bundledPaths = new(StringComparer.OrdinalIgnoreCase);
        int converted = 0;
        int copied = 0;
        int missing = 0;
        int failed = 0;
        int texturesExtracted = 0;
        int texturesMissing = 0;
        int texturesNormalized = 0;
        int texturesResized = 0;
        int texturesSpecularReencoded = 0;

        foreach (string modelPath in modelPaths)
        {
            try
            {
                string extension = Path.GetExtension(modelPath);
                string bundledRelativePath = extension.Equals(".m2", StringComparison.OrdinalIgnoreCase)
                    ? ChangeVirtualExtension(modelPath, ".mdx")
                    : modelPath;
                string bundledVirtualPath = NormalizeVirtualPath(Path.Combine("World", "Maps", mapName, "mdxs", mapName, bundledRelativePath));
                string localPath = Path.Combine(mdxRoot, bundledRelativePath);
                string? localDir = Path.GetDirectoryName(localPath);
                if (!string.IsNullOrWhiteSpace(localDir))
                    Directory.CreateDirectory(localDir);

                byte[]? modelBytes = ReadSourceAsset(sourceRoot, catalog, modelPath);
                if (modelBytes is null)
                {
                    missing++;
                    bundledPaths[modelPath] = PlaceholderMdx;
                    continue;
                }

                byte[] outputBytes;
                if (extension.Equals(".m2", StringComparison.OrdinalIgnoreCase))
                {
                    string skinPath = BuildDefaultM2SkinPath(modelPath);
                    byte[]? skinBytes = ReadSourceAsset(sourceRoot, catalog, skinPath);
                    if (skinBytes is null)
                    {
                        missing++;
                        continue;
                    }

                    using MemoryStream geometryStream = new(modelBytes, writable: false);
                    using MemoryStream skinStream = new(skinBytes, writable: false);
                    var geometry = M2GeometryReader.Read(geometryStream, modelPath);
                    var skin = M2SkinReader.Read(skinStream, skinPath);
                    IReadOnlyDictionary<string, M2ExternalAnimationDocument> externalAnimations = LoadM2ExternalAnimations(geometry.Model, sourceRoot, catalog);
                    IReadOnlyDictionary<string, string> rewrittenTexturePaths = BundleModelTextures(
                        geometry.Textures.Select(static texture => texture.Filename).Where(static filename => !string.IsNullOrWhiteSpace(filename)).Cast<string>(),
                        sourceRoot,
                        catalog,
                        bundledVirtualPath,
                        localPath,
                        ref texturesExtracted,
                        ref texturesMissing,
                        ref texturesNormalized,
                        ref texturesResized,
                        ref texturesSpecularReencoded);

                    outputBytes = M2ToMdxConverter.Convert(geometry, skin, rewrittenTexturePaths, externalAnimations);
                    converted++;
                }
                else if (extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase) || extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
                {
                    outputBytes = BundleLegacyMdxTexturesAndRewriteModel(
                        modelBytes,
                        sourceRoot,
                        catalog,
                        modelPath,
                        bundledVirtualPath,
                        localPath,
                        ref texturesExtracted,
                        ref texturesMissing,
                        ref texturesNormalized,
                        ref texturesResized,
                        ref texturesSpecularReencoded);
                    copied++;
                }
                else
                {
                    failed++;
                    continue;
                }

                File.WriteAllBytes(localPath, outputBytes);
                bundledPaths[modelPath] = bundledVirtualPath;
            }
            catch
            {
                failed++;
                bundledPaths[modelPath] = PlaceholderMdx;
            }
        }

        Console.WriteLine($"    MDXs: converted {converted}, copied {copied}, missing {missing}, failed {failed} into {mdxRoot}");
        if (texturesExtracted > 0 || texturesMissing > 0)
        {
            Console.WriteLine($"    MDX textures: extracted {texturesExtracted}" +
                (texturesMissing > 0 ? $", missing {texturesMissing}" : string.Empty) +
                (texturesNormalized > 0 ? $", re-encoded {texturesNormalized} ({texturesSpecularReencoded} specular, {texturesResized} resized)" : string.Empty));
        }

        return bundledPaths;
    }

    private static IReadOnlyDictionary<string, M2ExternalAnimationDocument> LoadM2ExternalAnimations(
        M2ModelDocument model,
        string sourceRoot,
        NativeMpqService? catalog)
    {
        Dictionary<string, M2ExternalAnimationDocument> animations = new(StringComparer.OrdinalIgnoreCase);
        HashSet<string> companionPaths = new(StringComparer.OrdinalIgnoreCase);

        for (int sequenceIndex = 0; sequenceIndex < model.Sequences.Count; sequenceIndex++)
        {
            int sourceSequenceIndex = ResolveM2SourceSequenceIndex(model, sequenceIndex);
            if (sourceSequenceIndex < 0 || sourceSequenceIndex >= model.Sequences.Count)
                continue;

            M2SequenceDefinition sequence = model.Sequences[sourceSequenceIndex];
            if (!sequence.UsesExternalAnimationFile)
                continue;

            companionPaths.Add(model.Identity.BuildAnimationPath(sequence.AnimationId, sequence.VariationIndex));
        }

        foreach (string companionPath in companionPaths)
        {
            byte[]? animationBytes = ReadSourceAsset(sourceRoot, catalog, companionPath);
            if (animationBytes is null)
                continue;

            using MemoryStream stream = new(animationBytes, writable: false);
            animations[companionPath] = M2AnimationReader.Read(stream, companionPath);
        }

        return animations;
    }

    private static int ResolveM2SourceSequenceIndex(M2ModelDocument model, int sequenceIndex)
    {
        int resolvedSequenceIndex = sequenceIndex;
        HashSet<int> visited = new();
        while (resolvedSequenceIndex >= 0 && resolvedSequenceIndex < model.Sequences.Count)
        {
            if (!visited.Add(resolvedSequenceIndex))
                break;

            M2SequenceDefinition sequence = model.Sequences[resolvedSequenceIndex];
            if (!sequence.IsAlias || sequence.AliasNext == ushort.MaxValue)
                break;

            if (sequence.AliasNext >= model.Sequences.Count)
                break;

            resolvedSequenceIndex = sequence.AliasNext;
        }

        return resolvedSequenceIndex;
    }

    private static byte[] BundleLegacyMdxTexturesAndRewriteModel(
        byte[] modelBytes,
        string sourceRoot,
        NativeMpqService? catalog,
        string sourceModelPath,
        string bundledModelVirtualPath,
        string bundledModelLocalPath,
        ref int texturesExtracted,
        ref int texturesMissing,
        ref int texturesNormalized,
        ref int texturesResized,
        ref int texturesSpecularReencoded)
    {
        MdxSummary summary;
        using (MemoryStream summaryStream = new(modelBytes, writable: false))
        {
            summary = MdxSummaryReader.Read(summaryStream, sourceModelPath);
        }

        if (summary.TextureCount == 0)
            return modelBytes;

        IReadOnlyDictionary<string, string> rewrittenTexturePaths = BundleModelTextures(
            summary.Textures.Select(static texture => texture.Path).Where(static path => !string.IsNullOrWhiteSpace(path)).Cast<string>(),
            sourceRoot,
            catalog,
            bundledModelVirtualPath,
            bundledModelLocalPath,
            ref texturesExtracted,
            ref texturesMissing,
            ref texturesNormalized,
            ref texturesResized,
            ref texturesSpecularReencoded);

        return rewrittenTexturePaths.Count == 0
            ? modelBytes
            : RewriteBundledMdxTextureReferences(modelBytes, summary.Textures, rewrittenTexturePaths);
    }

    private static IReadOnlyDictionary<string, string> BundleModelTextures(
        IEnumerable<string> texturePaths,
        string sourceRoot,
        NativeMpqService? catalog,
        string bundledModelVirtualPath,
        string bundledModelLocalPath,
        ref int texturesExtracted,
        ref int texturesMissing,
        ref int texturesNormalized,
        ref int texturesResized,
        ref int texturesSpecularReencoded)
    {
        string bundledModelDirectoryVirtualPath = NormalizeVirtualPath(Path.GetDirectoryName(bundledModelVirtualPath) ?? string.Empty);
        string bundledModelDirectoryLocalPath = Path.GetDirectoryName(bundledModelLocalPath) ?? ".";
        Dictionary<string, string> rewrittenTexturePaths = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, string> assignedFileNamesBySourceTexture = new(StringComparer.OrdinalIgnoreCase);
        HashSet<string> usedLocalFileNames = new(StringComparer.OrdinalIgnoreCase);

        foreach (string texturePath in texturePaths)
        {
            string normalizedSourceTexturePath = NormalizeVirtualPath(texturePath);
            if (rewrittenTexturePaths.ContainsKey(normalizedSourceTexturePath))
                continue;

            byte[]? textureBytes = ReadSourceAsset(sourceRoot, catalog, normalizedSourceTexturePath);
            if (textureBytes is null)
            {
                texturesMissing++;
                rewrittenTexturePaths[normalizedSourceTexturePath] = normalizedSourceTexturePath;
                continue;
            }

            string localFileName = GetOrAssignUniqueTextureFileName(normalizedSourceTexturePath);
            string bundledTextureVirtualPath = NormalizeVirtualPath(Path.Combine(bundledModelDirectoryVirtualPath, localFileName));
            string bundledTextureLocalPath = Path.Combine(bundledModelDirectoryLocalPath, localFileName);

            AlphaBlpCompatibilityResult compatibility = AlphaBlpCompatibilityService.NormalizeForAlphaClient(normalizedSourceTexturePath, textureBytes);
            File.WriteAllBytes(bundledTextureLocalPath, compatibility.Data);
            texturesExtracted++;

            if (compatibility.Rewritten)
            {
                texturesNormalized++;
                if (compatibility.Resized)
                    texturesResized++;
                if (compatibility.SpecularReencoded)
                    texturesSpecularReencoded++;
            }

            rewrittenTexturePaths[normalizedSourceTexturePath] = bundledTextureVirtualPath;
        }

        return rewrittenTexturePaths;

        string GetOrAssignUniqueTextureFileName(string normalizedSourceTexturePath)
        {
            if (assignedFileNamesBySourceTexture.TryGetValue(normalizedSourceTexturePath, out string? existingFileName))
                return existingFileName;

            string extension = Path.GetExtension(normalizedSourceTexturePath);
            string baseName = Path.GetFileNameWithoutExtension(normalizedSourceTexturePath);
            if (string.IsNullOrWhiteSpace(baseName))
                baseName = "texture";

            string candidate = baseName + extension;
            int suffix = 1;
            while (!usedLocalFileNames.Add(candidate))
            {
                suffix++;
                candidate = $"{baseName}_{suffix}{extension}";
            }

            assignedFileNamesBySourceTexture[normalizedSourceTexturePath] = candidate;
            return candidate;
        }
    }

    private static byte[] BundleWorldModelTexturesAndRewriteRoot(
        byte[] rootBytes,
        string sourceRoot,
        NativeMpqService? catalog,
        string sourceWmoPath,
        string bundledWmoVirtualPath,
        string bundledWmoLocalPath,
        ref int texturesExtracted,
        ref int texturesMissing,
        ref int texturesNormalized,
        ref int texturesResized,
        ref int texturesSpecularReencoded)
    {
        int extractedCount = 0;
        int missingCount = 0;
        int normalizedCount = 0;
        int resizedCount = 0;
        int specularReencodedCount = 0;

        IReadOnlyList<WmoMaterialDetail> materials;
        using (MemoryStream materialStream = new(rootBytes, writable: false))
        {
            materials = WmoMaterialDetailReader.Read(materialStream, sourceWmoPath);
        }

        if (materials.Count == 0)
            return rootBytes;

        string bundledWmoDirectoryVirtualPath = NormalizeVirtualPath(Path.GetDirectoryName(bundledWmoVirtualPath) ?? string.Empty);
        string bundledWmoDirectoryLocalPath = Path.GetDirectoryName(bundledWmoLocalPath) ?? Path.GetDirectoryName(bundledWmoLocalPath) ?? ".";

        Dictionary<string, string> rewrittenTexturePaths = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, string> assignedFileNamesBySourceTexture = new(StringComparer.OrdinalIgnoreCase);
        HashSet<string> usedLocalFileNames = new(StringComparer.OrdinalIgnoreCase);

        foreach (WmoMaterialDetail material in materials)
        {
            BundleWorldModelTexture(material.Texture1Name);
            BundleWorldModelTexture(material.Texture2Name);
            BundleWorldModelTexture(material.Texture3Name);
        }

        byte[] rewrittenRootBytes = rewrittenTexturePaths.Count == 0
            ? rootBytes
            : RewriteBundledWmoTextureReferences(rootBytes, sourceWmoPath, rewrittenTexturePaths);

        void BundleWorldModelTexture(string texturePath)
        {
            if (string.IsNullOrWhiteSpace(texturePath))
                return;

            string normalizedSourceTexturePath = NormalizeVirtualPath(texturePath);
            if (rewrittenTexturePaths.ContainsKey(normalizedSourceTexturePath))
                return;

            byte[]? textureBytes = ReadSourceAsset(sourceRoot, catalog, normalizedSourceTexturePath);
            if (textureBytes is null)
            {
                missingCount++;
                rewrittenTexturePaths[normalizedSourceTexturePath] = normalizedSourceTexturePath;
                return;
            }

            string localFileName = GetOrAssignUniqueTextureFileName(normalizedSourceTexturePath);
            string bundledTextureVirtualPath = NormalizeVirtualPath(Path.Combine(bundledWmoDirectoryVirtualPath, localFileName));
            string bundledTextureLocalPath = Path.Combine(bundledWmoDirectoryLocalPath, localFileName);

            AlphaBlpCompatibilityResult compatibility = AlphaBlpCompatibilityService.NormalizeForAlphaClient(normalizedSourceTexturePath, textureBytes);
            File.WriteAllBytes(bundledTextureLocalPath, compatibility.Data);
            extractedCount++;

            if (compatibility.Rewritten)
            {
                normalizedCount++;
                if (compatibility.Resized)
                    resizedCount++;
                if (compatibility.SpecularReencoded)
                    specularReencodedCount++;
            }

            rewrittenTexturePaths[normalizedSourceTexturePath] = bundledTextureVirtualPath;
        }

        string GetOrAssignUniqueTextureFileName(string normalizedSourceTexturePath)
        {
            if (assignedFileNamesBySourceTexture.TryGetValue(normalizedSourceTexturePath, out string? existingFileName))
                return existingFileName;

            string extension = Path.GetExtension(normalizedSourceTexturePath);
            string baseName = Path.GetFileNameWithoutExtension(normalizedSourceTexturePath);
            if (string.IsNullOrWhiteSpace(baseName))
                baseName = "texture";

            string candidate = baseName + extension;
            int suffix = 1;
            while (!usedLocalFileNames.Add(candidate))
            {
                suffix++;
                candidate = $"{baseName}_{suffix}{extension}";
            }

            assignedFileNamesBySourceTexture[normalizedSourceTexturePath] = candidate;
            return candidate;
        }

        texturesExtracted += extractedCount;
        texturesMissing += missingCount;
        texturesNormalized += normalizedCount;
        texturesResized += resizedCount;
        texturesSpecularReencoded += specularReencodedCount;

        return rewrittenRootBytes;
    }

    private static byte[] RewriteBundledWmoTextureReferences(
        byte[] rootBytes,
        string sourceWmoPath,
        IReadOnlyDictionary<string, string> rewrittenTexturePaths)
    {
        List<ChunkSpan> chunks = ReadWmoRootChunks(rootBytes);
        ChunkSpan? momoChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Momo);

        ChunkSpan? momtChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Momt);
        ChunkSpan? motxChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Motx);

        IReadOnlyList<WmoMaterialDetail> materials;
        using (MemoryStream materialStream = new(rootBytes, writable: false))
        {
            materials = WmoMaterialDetailReader.Read(materialStream, sourceWmoPath);
        }

        if (momoChunk is not null)
        {
            byte[] rewrittenMomoPayload = RewriteBundledWmoTextureReferencesInContainer(
                CopyChunkPayload(rootBytes, momoChunk.Value),
                materials,
                rewrittenTexturePaths);

            if (ReferenceEquals(rewrittenMomoPayload, rootBytes))
                return rootBytes;

            using MemoryStream rewrittenRoot = new(rootBytes.Length + rewrittenMomoPayload.Length);
            foreach (ChunkSpan chunk in chunks)
            {
                byte[] payload = chunk.Header.Id == WmoChunkIds.Momo
                    ? rewrittenMomoPayload
                    : CopyChunkPayload(rootBytes, chunk);

                rewrittenRoot.Write(chunk.Header.Id.ToFileBytes());
                rewrittenRoot.Write(BitConverter.GetBytes(payload.Length));
                rewrittenRoot.Write(payload);
            }

            return rewrittenRoot.ToArray();
        }

        if (momtChunk is null || motxChunk is null)
            return rootBytes;

        byte[] momtPayload = CopyChunkPayload(rootBytes, momtChunk.Value);
        byte[] motxPayload = BuildBundledWmoTextureTable(materials, rewrittenTexturePaths, momtPayload);

        using MemoryStream output = new(rootBytes.Length + motxPayload.Length);
        foreach (ChunkSpan chunk in chunks)
        {
            byte[] payload = chunk.Header.Id switch
            {
                _ when chunk.Header.Id == WmoChunkIds.Momt => momtPayload,
                _ when chunk.Header.Id == WmoChunkIds.Motx => motxPayload,
                _ => CopyChunkPayload(rootBytes, chunk)
            };

            output.Write(chunk.Header.Id.ToFileBytes());
            output.Write(BitConverter.GetBytes(payload.Length));
            output.Write(payload);
        }

        return output.ToArray();
    }

    private static byte[] RewriteBundledWmoTextureReferencesInContainer(
        byte[] containerBytes,
        IReadOnlyList<WmoMaterialDetail> materials,
        IReadOnlyDictionary<string, string> rewrittenTexturePaths)
    {
        List<ChunkSpan> chunks = ReadWmoRootChunks(containerBytes);
        ChunkSpan? momtChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Momt);
        ChunkSpan? motxChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Motx);
        if (momtChunk is null || motxChunk is null)
            return containerBytes;

        byte[] momtPayload = CopyChunkPayload(containerBytes, momtChunk.Value);
        byte[] motxPayload = BuildBundledWmoTextureTable(materials, rewrittenTexturePaths, momtPayload);

        using MemoryStream output = new(containerBytes.Length + motxPayload.Length);
        foreach (ChunkSpan chunk in chunks)
        {
            byte[] payload = chunk.Header.Id switch
            {
                _ when chunk.Header.Id == WmoChunkIds.Momt => momtPayload,
                _ when chunk.Header.Id == WmoChunkIds.Motx => motxPayload,
                _ => CopyChunkPayload(containerBytes, chunk)
            };

            output.Write(chunk.Header.Id.ToFileBytes());
            output.Write(BitConverter.GetBytes(payload.Length));
            output.Write(payload);
        }

        return output.ToArray();
    }

    private static byte[] RewriteBundledMdxTextureReferences(
        byte[] modelBytes,
        IReadOnlyList<MdxTextureSummary> textures,
        IReadOnlyDictionary<string, string> rewrittenTexturePaths)
    {
        if (modelBytes.Length < 4 || !Encoding.ASCII.GetString(modelBytes, 0, 4).Equals("MDLX", StringComparison.Ordinal))
            return modelBytes;

        int offset = 4;
        while (offset + 8 <= modelBytes.Length)
        {
            string chunkId = Encoding.ASCII.GetString(modelBytes, offset, 4);
            int chunkSize = BinaryPrimitives.ReadInt32LittleEndian(modelBytes.AsSpan(offset + 4, 4));
            int payloadOffset = offset + 8;
            int nextOffset = payloadOffset + chunkSize;
            if (chunkSize < 0 || nextOffset > modelBytes.Length)
                return modelBytes;

            if (string.Equals(chunkId, "TEXS", StringComparison.Ordinal))
            {
                byte[] texsPayload = BuildBundledMdxTextureTable(textures, rewrittenTexturePaths, chunkSize);
                using MemoryStream output = new(modelBytes.Length - chunkSize + texsPayload.Length);
                output.Write(modelBytes, 0, offset);
                output.Write(Encoding.ASCII.GetBytes("TEXS"));
                output.Write(BitConverter.GetBytes(texsPayload.Length));
                output.Write(texsPayload);
                output.Write(modelBytes, nextOffset, modelBytes.Length - nextOffset);
                return output.ToArray();
            }

            offset = nextOffset;
        }

        return modelBytes;
    }

    private static byte[] BuildBundledMdxTextureTable(
        IReadOnlyList<MdxTextureSummary> textures,
        IReadOnlyDictionary<string, string> rewrittenTexturePaths,
        int originalPayloadSize)
    {
        (int entrySize, int pathSize) = ResolveMdxTexsLayout(originalPayloadSize);
        using MemoryStream payload = new(textures.Count * entrySize);
        using BinaryWriter writer = new(payload, Encoding.ASCII, leaveOpen: true);

        foreach (MdxTextureSummary texture in textures)
        {
            writer.Write(texture.ReplaceableId);
            string normalizedPath = string.IsNullOrWhiteSpace(texture.Path)
                ? string.Empty
                : NormalizeVirtualPath(texture.Path);
            string remappedPath = string.IsNullOrEmpty(normalizedPath)
                ? string.Empty
                : rewrittenTexturePaths.TryGetValue(normalizedPath, out string? bundledPath)
                    ? bundledPath
                    : normalizedPath;
            WriteFixedAscii(writer, remappedPath, pathSize);
            writer.Write(texture.Flags);
        }

        writer.Flush();
        return payload.ToArray();
    }

    private static (int EntrySize, int PathSize) ResolveMdxTexsLayout(int payloadSize)
    {
        if (payloadSize % 0x10C == 0)
            return (0x10C, 0x104);

        if (payloadSize % 0x108 == 0)
            return (0x108, 0x100);

        throw new InvalidDataException($"Invalid MDX TEXS payload size 0x{payloadSize:X}.");
    }

    private static byte[] BuildBundledWmoTextureTable(
        IReadOnlyList<WmoMaterialDetail> materials,
        IReadOnlyDictionary<string, string> rewrittenTexturePaths,
        byte[] momtPayload)
    {
        Dictionary<string, uint> offsetsByTexturePath = new(StringComparer.OrdinalIgnoreCase)
        {
            [string.Empty] = 0,
        };

        using MemoryStream motx = new();
        motx.WriteByte(0);

        foreach (WmoMaterialDetail material in materials)
        {
            WriteTextureOffset(material.PayloadOffset + 12, material.Texture1Name);
            WriteTextureOffset(material.PayloadOffset + 24, material.Texture2Name);
            WriteTextureOffset(material.PayloadOffset + 36, material.Texture3Name);
        }

        return motx.ToArray();

        void WriteTextureOffset(int payloadOffset, string sourceTexturePath)
        {
            string normalizedSourceTexturePath = NormalizeVirtualPath(sourceTexturePath);
            string remappedTexturePath = string.IsNullOrWhiteSpace(normalizedSourceTexturePath)
                ? string.Empty
                : rewrittenTexturePaths.TryGetValue(normalizedSourceTexturePath, out string? bundledTexturePath)
                    ? bundledTexturePath
                    : normalizedSourceTexturePath;

            uint offset = GetOrAddTextureOffset(remappedTexturePath);
            BinaryPrimitives.WriteUInt32LittleEndian(momtPayload.AsSpan(payloadOffset, 4), offset);
        }

        uint GetOrAddTextureOffset(string texturePath)
        {
            if (offsetsByTexturePath.TryGetValue(texturePath, out uint existingOffset))
                return existingOffset;

            uint offset = checked((uint)motx.Position);
            byte[] bytes = Encoding.UTF8.GetBytes(texturePath);
            motx.Write(bytes);
            motx.WriteByte(0);
            offsetsByTexturePath[texturePath] = offset;
            return offset;
        }
    }

    private static List<ChunkSpan> ReadWmoRootChunks(byte[] rootBytes)
    {
        List<ChunkSpan> chunks = [];
        int offset = 0;
        while (offset + ChunkHeader.SizeInBytes <= rootBytes.Length)
        {
            FourCC id = FourCC.FromFileBytes(rootBytes.AsSpan(offset, 4));
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(rootBytes.AsSpan(offset + 4, 4));
            int dataOffset = offset + ChunkHeader.SizeInBytes;
            int endOffset = checked(dataOffset + (int)size);
            if (endOffset > rootBytes.Length)
                throw new InvalidDataException($"WMO root chunk '{id}' overruns the supplied byte buffer.");

            chunks.Add(new ChunkSpan(new ChunkHeader(id, size), offset, dataOffset));
            offset = endOffset;
        }

        return chunks;
    }

    private static byte[] CopyChunkPayload(byte[] rootBytes, ChunkSpan chunk)
    {
        byte[] payload = new byte[chunk.Header.Size];
        Buffer.BlockCopy(rootBytes, checked((int)chunk.DataOffset), payload, 0, checked((int)chunk.Header.Size));
        return payload;
    }

    private static AlphaTileData RewriteBundledAssetPaths(
        AlphaTileData tile,
        string? tilesetPrefix,
        IReadOnlyDictionary<string, string>? bundledMdxPaths,
        IReadOnlyDictionary<string, string>? bundledWmoPaths,
        HashSet<string>? targetFileSet)
    {
        bool IsTargetPath(string path) =>
            targetFileSet is not null
            && (targetFileSet.Contains(path) || targetFileSet.Contains(path.Replace('\\', '/')));

        IReadOnlyList<string> textureNames = tilesetPrefix is null
            ? tile.TextureNames
            : tile.TextureNames.Select(t => tilesetPrefix + t.TrimStart('\\')).ToList();

        IReadOnlyList<AlphaModelPlacement> modelPlacements = bundledMdxPaths is null
            ? tile.ModelPlacements
            : tile.ModelPlacements.Select(placement =>
            {
                string normalized = NormalizeVirtualPath(placement.ModelPath);
                string mappedPath = string.Equals(normalized, PlaceholderMdx, StringComparison.OrdinalIgnoreCase)
                    ? PlaceholderMdx
                    : bundledMdxPaths.TryGetValue(normalized, out string? bundledPath)
                        ? bundledPath
                        : IsTargetPath(normalized)
                            ? normalized
                            : PlaceholderMdx;

                return new AlphaModelPlacement(
                    placement.NameId,
                    mappedPath,
                    placement.UniqueId,
                    placement.Position,
                    placement.Rotation,
                    placement.Scale);
            }).ToList();

        IReadOnlyList<AlphaWorldModelPlacement> worldModelPlacements = bundledWmoPaths is null
            ? tile.WorldModelPlacements
            : tile.WorldModelPlacements.Select(placement =>
            {
                string normalized = NormalizeVirtualPath(placement.ModelPath);
                string mappedPath = string.Equals(normalized, PlaceholderWmo, StringComparison.OrdinalIgnoreCase)
                    ? PlaceholderWmo
                    : bundledWmoPaths.TryGetValue(normalized, out string? bundledPath)
                        ? bundledPath
                        : IsTargetPath(normalized)
                            ? normalized
                            : PlaceholderWmo;

                return new AlphaWorldModelPlacement(
                    placement.NameId,
                    mappedPath,
                    placement.UniqueId,
                    placement.Position,
                    placement.Rotation,
                    placement.BoundsMin,
                    placement.BoundsMax,
                    placement.Flags);
                    }).ToList();

        return new AlphaTileData(
            tile.SourcePath,
            tile.Heightmap,
            tile.McalAlphaPack,
            tile.MclyTextureIds,
            tile.MclyLayerMask,
            tile.HoleMask,
            textureNames,
            modelPlacements,
            worldModelPlacements,
            tile.LiquidChunks,
            diagnostics: tile.Diagnostics,
            mcnrNormalXyz: tile.McnrNormalXyz,
            mcshShadowMask256: tile.McshShadowMask256,
            mclqSurfaceHeight: tile.MclqSurfaceHeight,
            mclqTypeMask: tile.MclqTypeMask,
            mcshShadowMask1024: tile.McshShadowMask1024,
            rawChunks: tile.RawChunks,
            areaIds: tile.AreaIds,
            mfboFlightBounds: tile.MfboFlightBounds,
            mccvRgb: tile.MccvRgb,
            mclvLightingBytes: tile.MclvLightingBytes,
            holeFullMasks: tile.HoleFullMasks,
            mcrfDoodadRefsByChunk: tile.McrfDoodadRefsByChunk,
            mcrfWorldModelRefsByChunk: tile.McrfWorldModelRefsByChunk,
            mcrfDoodadUniqueIdsByChunk: tile.McrfDoodadUniqueIdsByChunk,
            mcrfWorldModelUniqueIdsByChunk: tile.McrfWorldModelUniqueIdsByChunk);
    }

    private static (int PlaceholderModels, int PlaceholderWmos) CountPlaceholderPlacements(Dictionary<(int, int), AlphaTileData> tiles)
    {
        int placeholderModels = 0;
        int placeholderWmos = 0;

        foreach (AlphaTileData tile in tiles.Values)
        {
            placeholderModels += tile.ModelPlacements.Count(static placement => string.Equals(placement.ModelPath, PlaceholderMdx, StringComparison.OrdinalIgnoreCase));
            placeholderWmos += tile.WorldModelPlacements.Count(static placement => string.Equals(placement.ModelPath, PlaceholderWmo, StringComparison.OrdinalIgnoreCase));
        }

        return (placeholderModels, placeholderWmos);
    }

    private static bool IsArchiveBackedSource(string sourceRoot)
    {
        return Directory.Exists(sourceRoot)
            && (Directory.Exists(Path.Combine(sourceRoot, "Data"))
                || Directory.GetFiles(sourceRoot, "*.mpq", SearchOption.TopDirectoryOnly).Length > 0);
    }

    private static NativeMpqService CreateSourceCatalog(string sourceRoot)
    {
        NativeMpqService catalog = new();
        catalog.LoadArchives([sourceRoot]);
        return catalog;
    }

    private static byte[]? ReadSourceAsset(string sourceRoot, NativeMpqService? catalog, string path)
    {
        string normalized = NormalizeVirtualPath(path);
        if (catalog is not null)
            return catalog.ReadFile(normalized) ?? catalog.ReadFile(normalized.Replace('\\', '/'));

        string[] candidates =
        [
            Path.Combine(sourceRoot, normalized),
            Path.Combine(sourceRoot, normalized.Replace('\\', Path.DirectorySeparatorChar)),
            Path.Combine(sourceRoot, "Data", normalized),
            Path.Combine(sourceRoot, "Data", normalized.Replace('\\', Path.DirectorySeparatorChar)),
        ];

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
                return File.ReadAllBytes(candidate);
        }

        return null;
    }

    private static string BuildWmoGroupPath(string rootPath, int groupIndex)
    {
        string directory = Path.GetDirectoryName(rootPath) ?? string.Empty;
        string baseName = Path.GetFileNameWithoutExtension(rootPath);
        string fileName = $"{baseName}_{groupIndex:D3}.wmo";
        return string.IsNullOrWhiteSpace(directory) ? fileName : $"{directory}\\{fileName}";
    }

    private static LkToAlphaOptions ParseOptions(string[] args)
    {
            return new LkToAlphaOptions(
                InputDir: GetOption(args, "--input", "-i"),
                OutputPath: GetOption(args, "--output", "-o"),
                OutputWdlPath: GetOption(args, "--output-wdl", "--wdl"),
                ClientRoot: GetOption(args, "--client-root", "-c"),
                TargetClientRoot: GetOption(args, "--target-client-root", "-tcr") ?? GetOption(args, "--target-client-route", "--target-client-route"),
                MapName: GetOption(args, "--map", "-m"),
                Verbose: HasFlag(args, "--verbose") || HasFlag(args, "-v"),
                TerrainOnly: HasFlag(args, "--terrain-only") || HasFlag(args, "-to"),
                BundleTilesets: !HasFlag(args, "--no-bundle-tilesets") && !HasFlag(args, "-nbt"),
                BundleM2s: HasFlag(args, "--bundle-m2s") || HasFlag(args, "-bm"),
                BundleWmos: HasFlag(args, "--bundle-wmos") || HasFlag(args, "-bw"));
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
        bool BundleTilesets,
        bool BundleM2s,
        bool BundleWmos);
}
