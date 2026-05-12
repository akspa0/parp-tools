using System.Diagnostics;
using System.IO;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
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
                                    FilterPlacements(adtData, targetFileSet, options.BundleWmos);
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
                                FilterPlacements(adtData, targetFileSet, options.BundleWmos);
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
            string? wmoRoot = null;
            if (options.BundleTilesets)
            {
                tilesetRoot = Path.Combine(outputDir, "tilesets", mapName);
                BundleTilesets(tiles, sourceForAssets, mapName, tilesetRoot);
            }

            Dictionary<string, string>? bundledWmoPaths = null;
            if (options.BundleWmos && !options.TerrainOnly)
            {
                wmoRoot = Path.Combine(outputDir, "wmos", mapName);
                bundledWmoPaths = BundleWorldModels(tiles, sourceForAssets, mapName, wmoRoot);
            }

            if (tilesetRoot is not null || bundledWmoPaths is not null)
            {
                string? tilesetPrefix = tilesetRoot is null ? null : $"tilesets\\{mapName}\\";
                tiles = tiles.ToDictionary(
                    kvp => kvp.Key,
                    kvp => RewriteBundledAssetPaths(kvp.Value, tilesetPrefix, bundledWmoPaths));
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
                Console.WriteLine($"  Assets:   {missingModelNames} MDX + {missingWmoNames} WMO placements mapped to placeholders ({PlaceholderMdx}, {PlaceholderWmo})");

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
        FilterPlacements(LkAdtData adtData, HashSet<string> targetFileSet, bool preserveSourceWmoPaths = false)
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
            string mappedPath = ResolveModelPath(path);
            int mappedNameId = GetOrAddNameIndex(mappedPath, names, nameIndex);

            if (!ReferenceEquals(mappedPath, path) && !string.Equals(mappedPath, path, StringComparison.OrdinalIgnoreCase))
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
                ? path
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
        string wmoRoot)
    {
        HashSet<string> wmoPaths = new(StringComparer.OrdinalIgnoreCase);
        foreach (AlphaTileData tile in tiles.Values)
        {
            foreach (AlphaWorldModelPlacement placement in tile.WorldModelPlacements)
            {
                string normalized = NormalizeVirtualPath(placement.ModelPath);
                if (string.IsNullOrWhiteSpace(normalized) || string.Equals(normalized, PlaceholderWmo, StringComparison.OrdinalIgnoreCase))
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

        foreach (string wmoPath in wmoPaths)
        {
            try
            {
                byte[]? rootBytes = ReadSourceAsset(sourceRoot, catalog, wmoPath);
                if (rootBytes is null)
                {
                    missing++;
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

                string localPath = Path.Combine(wmoRoot, wmoPath);
                string? localDir = Path.GetDirectoryName(localPath);
                if (!string.IsNullOrWhiteSpace(localDir))
                    Directory.CreateDirectory(localDir);

                File.WriteAllBytes(localPath, outputBytes);
                bundledPaths[wmoPath] = NormalizeVirtualPath(Path.Combine("World", "Maps", mapName, "wmos", mapName, wmoPath));
            }
            catch
            {
                failed++;
            }
        }

        Console.WriteLine($"    WMOs: converted {converted}, copied {copied}, missing {missing}, failed {failed} into {wmoRoot}");
        return bundledPaths;
    }

    private static AlphaTileData RewriteBundledAssetPaths(
        AlphaTileData tile,
        string? tilesetPrefix,
        IReadOnlyDictionary<string, string>? bundledWmoPaths)
    {
        IReadOnlyList<string> textureNames = tilesetPrefix is null
            ? tile.TextureNames
            : tile.TextureNames.Select(t => tilesetPrefix + t.TrimStart('\\')).ToList();

        IReadOnlyList<AlphaWorldModelPlacement> worldModelPlacements = bundledWmoPaths is null
            ? tile.WorldModelPlacements
            : tile.WorldModelPlacements.Select(placement =>
            {
                string normalized = NormalizeVirtualPath(placement.ModelPath);
                string mappedPath = string.Equals(normalized, PlaceholderWmo, StringComparison.OrdinalIgnoreCase)
                    ? PlaceholderWmo
                    : bundledWmoPaths.TryGetValue(normalized, out string? bundledPath)
                        ? bundledPath
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
            tile.ModelPlacements,
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
                BundleTilesets: HasFlag(args, "--bundle-tilesets") || HasFlag(args, "-bt"),
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
        bool BundleWmos);
}
