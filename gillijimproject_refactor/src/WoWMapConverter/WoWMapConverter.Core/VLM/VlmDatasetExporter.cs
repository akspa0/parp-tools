using System.Collections.Concurrent;
using System.Buffers.Binary;
using System.Numerics;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WoWMapConverter.Core.Formats.Liquids;
using WoWMapConverter.Core.Formats.PM4;
using WoWMapConverter.Core.Services;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.M2;
using WowViewer.Core.Mdx;
using WowViewer.Core.Wmo;
using GillijimProject.WowFiles.Alpha;
using WdtAlpha = GillijimProject.WowFiles.Alpha.WdtAlpha;
using SharedMd5TranslateIndex = WowViewer.Core.IO.Files.Md5TranslateIndex;

namespace WoWMapConverter.Core.VLM;

using System.IO;

/// <summary>
/// VLM Dataset Exporter - extracts ADT data for VLM training.
/// Uses AdtAlpha parser and McnkAlpha sub-chunk access.
/// </summary>
public class VlmDatasetExporter
{
    private const float TileSize = 533.33333f;
    private const float MapOrigin = 32f * TileSize;
    private const float ObjectMaskMarginTiles = 0.25f;
    private const int MaxFootprintSamplesPerSource = 2048;

    private enum ObjectProjectionAxis
    {
        Y,
        Z,
    }

    private readonly record struct ObjectProjectionMode(ObjectProjectionAxis SecondaryAxis, bool UseMapOrigin, bool UseNormalized);
    private readonly record struct ProjectionCandidate(ObjectProjectionMode Mode, float U, float V);

    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
    };
    
    private readonly ConcurrentDictionary<string, (float[] Min, float[] Max)?> _modelBoundsCache = new();
    private readonly ConcurrentDictionary<string, Vector2[][]?> _modelFootprintCache = new();

    private static readonly HashSet<FourCC> KnownWmoGroupSubchunkIds =
    [
        WmoChunkIds.Mopy,
        WmoChunkIds.Movi,
        WmoChunkIds.Moin,
        WmoChunkIds.Movt,
        WmoChunkIds.Monr,
        WmoChunkIds.Motv,
        WmoChunkIds.Moba,
        WmoChunkIds.Molr,
        WmoChunkIds.Mobn,
        WmoChunkIds.Mobr,
        WmoChunkIds.Mocv,
        WmoChunkIds.Mliq,
        WmoChunkIds.Modr,
    ];

    public async Task ExportBatchAsync(VlmBatchExportConfig config, IProgress<string>? progress = null)
    {
        foreach (var client in config.Clients)
        {
            progress?.Report($"Processing Client: {client.ClientPath} ({client.ClientVersion})");
            foreach (var map in client.Maps)
            {
                var mapOut = Path.Combine(client.OutputRoot, map);
                await ExportMapAsync(client.ClientPath, map, mapOut, progress, generateDepth: client.GenerateDepth);
            }
        }
    }

    public async Task<VlmExportResult> ExportMapAsync(
        string clientPath,
        string mapName,
        string outputDir,
        IProgress<string>? progress = null,
        int limit = int.MaxValue,
        string? listfilePath = null,
        bool generateDepth = false,
        string? minimapRoot = null,
        string? tileFilter = null,
        bool skipDerivedAssets = false,
        bool interestingOnly = false,
        int interestingMinScore = 1)
    {
        progress?.Report($"Starting VLM export for map: {mapName}");

        // Create output directories
        var imagesDir = Path.Combine(outputDir, "images");
        var shadowsDir = string.Empty;
        var masksDir = string.Empty;
        var liquidsDir = Path.Combine(outputDir, "liquids");
        var datasetDir = Path.Combine(outputDir, "dataset");
        
        Directory.CreateDirectory(imagesDir);
        Directory.CreateDirectory(liquidsDir);
        Directory.CreateDirectory(datasetDir);
        
        var depthsDir = Path.Combine(outputDir, "depths");
        if (generateDepth)
            Directory.CreateDirectory(depthsDir);

        // Normalize client path
        var dataPath = clientPath;
        if (!Directory.Exists(Path.Combine(clientPath, "World")) &&
            Directory.Exists(Path.Combine(clientPath, "Data", "World")))
        {
            dataPath = Path.Combine(clientPath, "Data");
            progress?.Report($"Using Data subfolder: {dataPath}");
            progress?.Report($"Using Data subfolder: {dataPath}");
        }

        var searchPaths = new List<string> { dataPath };
        if (!string.Equals(clientPath, dataPath, StringComparison.OrdinalIgnoreCase))
        {
            searchPaths.Add(clientPath);
        }

        var minimapSearchPaths = string.IsNullOrWhiteSpace(minimapRoot)
            ? searchPaths
            : BuildSearchPaths(minimapRoot);

        if (!string.IsNullOrWhiteSpace(minimapRoot))
        {
            if (minimapSearchPaths.Count == 0)
            {
                progress?.Report($"Explicit minimap root does not expose a readable World tree: {minimapRoot}");
            }
            else
            {
                progress?.Report($"Using explicit minimap root: {minimapRoot}");
            }
        }

        // Resolve directory names before any map file lookup so clients whose on-disk
        // folder differs from the requested map label still find WDT/ADT/WDL payloads.
        string[] wdtPaths;
        string[] wdtArchivePaths;

        string? wdtPath = null;
        byte[]? wdtData = null;

        // Initialize the shared archive catalog early so we can search MPQ archives for WDT.
        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();

        string[] listfileSearchPaths =
        {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer", "community-listfile-withcapitals.csv"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "community-listfile-withcapitals.csv"),
            "community-listfile-withcapitals.csv",
            "listfile.csv",
        };
        string? resolvedListfile = !string.IsNullOrWhiteSpace(listfilePath) && File.Exists(listfilePath)
            ? listfilePath
            : listfileSearchPaths.FirstOrDefault(File.Exists);
        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, searchPaths, resolvedListfile);

        var mapDirectoryLookup = new MapDirectoryLookup();
        mapDirectoryLookup.Load(searchPaths, archiveCatalog);

        string? resolvedMapDirectory = mapDirectoryLookup.ResolveDirectory(mapName);
        string mapDirectory = resolvedMapDirectory
            ?? TryResolveArchiveMapDirectoryAlias(mapName, archiveCatalog.GetAllKnownFiles())
            ?? mapName;
        if (!string.Equals(mapDirectory, mapName, StringComparison.Ordinal))
        {
            string resolutionSource = resolvedMapDirectory is not null ? "Map.dbc" : "archive file names";
            Console.WriteLine($"Resolved map '{mapName}' to directory '{mapDirectory}' via {resolutionSource}");
        }

        string[] mapPathCandidates = DistinctMapNames(mapName, mapDirectory).ToArray();
        wdtPaths = mapPathCandidates
            .SelectMany(directoryName => new[]
            {
                Path.Combine(dataPath, "World", "Maps", directoryName, $"{directoryName}.wdt"),
                Path.Combine(clientPath, "Data", "World", "Maps", directoryName, $"{directoryName}.wdt"),
                Path.Combine(clientPath, "World", "Maps", directoryName, $"{directoryName}.wdt"),
            })
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        wdtArchivePaths = mapPathCandidates
            .Select(directoryName => $"World\\Maps\\{directoryName}\\{directoryName}.wdt")
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        foreach (var tryPath in wdtPaths)
        {
            // Try flat file first
            if (File.Exists(tryPath))
            {
                wdtPath = tryPath;
                progress?.Report($"Found WDT: {wdtPath}");
                break;
            }

            // Try per-asset MPQ (file.wdt.MPQ) - Alpha 0.5.3 style
            wdtData = AlphaArchiveReader.ReadWithMpqFallback(tryPath);
            if (wdtData != null)
            {
                var tempWdt = Path.Combine(outputDir, $"{mapName}.wdt");
                await File.WriteAllBytesAsync(tempWdt, wdtData);
                wdtPath = tempWdt;
                progress?.Report($"Found WDT in MPQ at: {tryPath}.MPQ");
                break;
            }
        }

        // Fallback: Try reading from large MPQ archives (3.3.5+, world.mpq, etc.)
        if (wdtPath == null)
        {
            foreach (var wdtInternalPath in wdtArchivePaths)
            {
                if (!archiveCatalog.FileExists(wdtInternalPath))
                    continue;

                wdtData = archiveCatalog.ReadFile(wdtInternalPath);
                if (wdtData != null)
                {
                    var tempWdt = Path.Combine(outputDir, $"{mapName}.wdt");
                    await File.WriteAllBytesAsync(tempWdt, wdtData);
                    wdtPath = tempWdt;
                    progress?.Report($"Found WDT in MPQ archive: {wdtInternalPath}");
                    break;
                }
            }
        }

        if (wdtPath == null)
        {
            progress?.Report($"WDT not found for map '{mapName}'.");
            progress?.Report($"Searched paths:");
            foreach (var p in wdtPaths)
                progress?.Report($"  - {p} (and {p}.MPQ)");
            foreach (var p in wdtArchivePaths)
                progress?.Report($"  - archive: {p}");
            progress?.Report("Ensure MPQ archives (world.mpq, terrain.mpq, etc.) are in the Data folder.");
            return new VlmExportResult(0, 0, 0, outputDir);
        }

        // Initialize MD5 Translate Service (Legacy)
        SharedMd5TranslateIndex? md5Index = null;
        
        // Also check map-specific TRS file (often found in newer clients)
        var mapTrs = $"World\\Maps\\{mapDirectory}\\md5translate.trs";
        var extraCandidates = new[] { mapTrs };

        if (WowViewer.Core.IO.Files.Md5TranslateResolver.TryLoad(
            searchPaths,
            archiveCatalog.FileExists,
            archiveCatalog.ReadFile,
            out var loadedIndex,
            extraCandidates))
        {
            md5Index = loadedIndex;
            Console.WriteLine($"Loaded MD5 Translate Index with {md5Index?.HashToPlain.Count} entries.");
            
            // md5Index loaded successfully
        }

        var groundEffectLookup = new GroundEffectLookup();
        groundEffectLookup.Load(searchPaths, archiveCatalog);

        // Detect WDT format using file size:
        // - Alpha 0.5.3 WDT: Large file (contains embedded ADT data, typically several MB)
        // - LK 3.3.5+ WDT: Small file (~32KB, only tile existence flags, ADTs are separate files)
        long wdtFileSize = new FileInfo(wdtPath).Length;
        bool isAlphaFormat = wdtFileSize > 100_000; // Alpha WDTs are typically > 1MB
        if (isAlphaFormat)
        {
            progress?.Report($"Detected Alpha format WDT ({wdtFileSize:N0} bytes - embedded ADT data)");
        }
        else
        {
            progress?.Report($"Detected LK format WDT ({wdtFileSize:N0} bytes - separate ADT files in MPQ)");
        }

        // Enumerate existing tiles based on WDT format
        List<int> existingTiles;
        WdtAlpha? wdt = null;
        List<int>? adtOffsets = null;
        List<string>? mdnmNames = null;
        List<string>? monmNames = null;
        
        try
        {
            if (isAlphaFormat)
            {
                // Alpha 0.5.3: Use WdtAlpha parser (monolithic WDT with embedded ADTs)
                wdt = new WdtAlpha(wdtPath);
                existingTiles = wdt.GetExistingAdtsNumbers();
                adtOffsets = wdt.GetAdtOffsetsInMain();
                mdnmNames = wdt.GetMdnmFileNames();
                monmNames = wdt.GetMonmFileNames();
                progress?.Report($"[Alpha WDT] Found {existingTiles.Count} embedded tiles");
            }
            else
            {
                // LK 3.0.1+: Read MAIN chunk to enumerate tiles
                var wdtBytes = wdtData ?? await File.ReadAllBytesAsync(wdtPath);
                existingTiles = ReadLkWdtTiles(wdtBytes);
                progress?.Report($"[LK WDT] Found {existingTiles.Count} tiles from MAIN chunk");
            }
        }
        catch (Exception ex)
        {
            progress?.Report($"Failed to enumerate WDT tiles: {ex.Message}");
            return new VlmExportResult(0, 0, 0, outputDir);
        }

        if (!isAlphaFormat)
        {
            int reachableBeforeFilter = existingTiles.Count;
            existingTiles = FilterReachableLkTiles(existingTiles, searchPaths, archiveCatalog, mapDirectory);
            if (existingTiles.Count != reachableBeforeFilter)
            {
                progress?.Report($"Filtered LK tile list to {existingTiles.Count} reachable root ADTs (from {reachableBeforeFilter} WDT-flagged tiles)");
            }
        }

        if (!string.IsNullOrWhiteSpace(tileFilter))
        {
            if (!TryParseTileFilter(tileFilter, out int tileX, out int tileY))
                throw new ArgumentException($"Invalid tile filter '{tileFilter}'. Expected format x_y.", nameof(tileFilter));

            int requestedTileIndex = tileY * 64 + tileX;
            existingTiles = existingTiles.Where(tileIndex => tileIndex == requestedTileIndex).ToList();
            progress?.Report($"Tile filter active: {tileX}_{tileY} ({existingTiles.Count} matching tile(s))");
        }

        // Load WDL if available
        WdlParser.WdlData? wdlData = null;
        try
        {
            string wdlPath = Path.ChangeExtension(wdtPath, ".wdl");
            
            // 1. Try flat file next to WDT
            if (File.Exists(wdlPath))
            {
                var wdlBytes = await File.ReadAllBytesAsync(wdlPath);
                wdlData = WdlParser.Parse(wdlBytes);
                progress?.Report($"Loaded WDL data from {wdlPath}");
            }
            // 2. Try WDL.MPQ (Alpha 0.5.3 style) across search paths
            else
            {
                 bool loaded = false;
                 // Try to find .wdl.MPQ in all search paths
                 foreach (var path in searchPaths)
                 {
                     foreach (string directoryName in mapPathCandidates)
                     {
                         var wdlMpqDiscovered = Path.Combine(path, "World", "Maps", directoryName, $"{directoryName}.wdl.MPQ");
                         if (!File.Exists(wdlMpqDiscovered))
                            wdlMpqDiscovered = Path.Combine(path, "World", "Maps", directoryName, $"{directoryName}.wdl.mpq");

                         if (!File.Exists(wdlMpqDiscovered))
                            continue;

                         var wdlExpectedPath = Path.Combine(path, "World", "Maps", directoryName, $"{directoryName}.wdl");
                         var wdlBytes = AlphaArchiveReader.ReadFromMpq(
                             wdlMpqDiscovered,
                             AlphaArchiveReader.BuildInternalNameCandidates(wdlExpectedPath));
                         if (wdlBytes != null)
                         {
                             wdlData = WdlParser.Parse(wdlBytes);
                             progress?.Report($"Loaded WDL from Alpha MPQ: {wdlMpqDiscovered}");
                             loaded = true;
                             break;
                         }
                     }

                     if (loaded)
                         break;
                 }

                 // 3. Try standard MPQ path (Modern/Legacy)
                 if (!loaded)
                 {
                    foreach (string directoryName in mapPathCandidates)
                    {
                        var wdlInternalPath = $"World\\Maps\\{directoryName}\\{directoryName}.wdl";
                        if (!archiveCatalog.FileExists(wdlInternalPath))
                            continue;

                        var wdlBytes = archiveCatalog.ReadFile(wdlInternalPath);
                        if (wdlBytes != null)
                        {
                            wdlData = WdlParser.Parse(wdlBytes);
                            progress?.Report($"Loaded WDL data from MPQ internal: {wdlInternalPath}");
                            break;
                        }
                    }
                 }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Warning] Failed to load WDL: {ex.Message}");
        }

        progress?.Report($"Found {existingTiles.Count} tiles in WDT");
        
        // Extract MPHD flags from WDT for LK format (needed for useBigAlphamaps)
        uint wdtMphdFlags = 0;
        if (!isAlphaFormat)
        {
            try
            {
                // Read WDT and find MPHD chunk
                var wdtBytes = wdtData ?? (wdtPath != null ? await File.ReadAllBytesAsync(wdtPath) : null);
                if (wdtBytes != null)
                {
                    int mphdOffset = FindLkChunk(wdtBytes, "MPHD");
                    if (mphdOffset >= 0 && mphdOffset + 12 < wdtBytes.Length)
                    {
                        wdtMphdFlags = BitConverter.ToUInt32(wdtBytes, mphdOffset + 8);
                        progress?.Report($"WDT MPHD flags: 0x{wdtMphdFlags:X} (useBigAlphamaps={(wdtMphdFlags & 0x4) != 0})");
                    }
                }
            }
            catch { }
        }
        int tilesExported = 0;
        int tilesSkipped = 0;
        var allTextures = new ConcurrentDictionary<string, byte>(StringComparer.OrdinalIgnoreCase);

        // Parallel tile processing with configurable degree of parallelism
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        var tilesToProcess = SelectTilesForProcessing(
            existingTiles,
            limit,
            isAlphaFormat,
            mapDirectory,
            archiveCatalog,
            searchPaths,
            interestingOnly,
            interestingMinScore);
        
        await Parallel.ForEachAsync(tilesToProcess, parallelOptions, async (tileIndex, ct) =>
        {
            int x = tileIndex % 64;
            int y = tileIndex / 64;
            var tileName = $"{mapName}_{x}_{y}";

            try
            {
                VlmTerrainData? sample = null;
                string? imageRelPath = null;
                
                // Try to find minimap (common to both formats)
                var minimapPath = FindMinimapTile(minimapSearchPaths, archiveCatalog, md5Index, mapDirectory, x, y);
                if (minimapPath != null)
                {
                    var imageFileName = $"{tileName}.png";
                    var outputImagePath = Path.Combine(imagesDir, imageFileName);
                    if (ConvertBlpToPng(minimapPath, outputImagePath, archiveCatalog))
                    {
                        imageRelPath = $"images/{imageFileName}";
                    }
                }

                // Lookup WDL tile
                var wdlTile = wdlData?.Tiles[tileIndex];
                
                if (isAlphaFormat)
                {
                    // Alpha format: ADT data is embedded in WDT at offset
                    int adtOffset = tileIndex < adtOffsets!.Count ? adtOffsets[tileIndex] : 0;
                    if (adtOffset <= 0)
                    {
                        Interlocked.Increment(ref tilesSkipped);
                        return;
                    }

                    // Parse ADT using AdtAlpha (proven parser)
                    AdtAlpha adtAlpha;
                    try
                    {
                        adtAlpha = new AdtAlpha(wdtPath, adtOffset, tileIndex);
                    }
                    catch (Exception ex)
                    {
                        progress?.Report($"Failed to parse ADT {tileName}: {ex.Message}");
                        Interlocked.Increment(ref tilesSkipped);
                        return;
                    }

                    // Extract terrain data using AdtAlpha's methods
                    sample = await ExtractFromAdtAlpha(adtAlpha, wdtPath, adtOffset, tileIndex, tileName, outputDir,
                        shadowsDir, masksDir, mdnmNames!, monmNames!, allTextures, groundEffectLookup, wdlTile, clientPath);
                }
                else
                {
                    // LK/Cata format: Read ADT from MPQ or loose files on disk
                    // Check for Split ADT files (Cataclysm+)
                    var adtBase = $"World\\Maps\\{mapDirectory}\\{mapDirectory}_{x}_{y}";
                    var rootAdtPath = $"{adtBase}.adt";
                    var texAdtPath = $"{adtBase}_tex0.adt";
                    var objAdtPath = $"{adtBase}_obj0.adt";

                    // Try loose files on disk first, then fall back to archive catalog
                    byte[]? adtBytes = null;
                    byte[]? texBytes = null;
                    byte[]? objBytes = null;

                    foreach (var bp in searchPaths)
                    {
                        var diskRoot = Path.Combine(bp, rootAdtPath);
                        if (File.Exists(diskRoot))
                        {
                            adtBytes = await File.ReadAllBytesAsync(diskRoot);
                            var diskTex = Path.Combine(bp, texAdtPath);
                            if (File.Exists(diskTex))
                            {
                                texBytes = await File.ReadAllBytesAsync(diskTex);
                                progress?.Report($"[Split ADT] Found tex0 for {tileName}");
                            }
                            var diskObj = Path.Combine(bp, objAdtPath);
                            if (File.Exists(diskObj))
                            {
                                objBytes = await File.ReadAllBytesAsync(diskObj);
                                progress?.Report($"[Split ADT] Found obj0 for {tileName}");
                            }
                            break;
                        }
                    }

                    // Fall back to archive catalog (MPQ) if not found on disk
                    if (adtBytes == null || adtBytes.Length == 0)
                    {
                        adtBytes = archiveCatalog.ReadFile(rootAdtPath);

                        if (archiveCatalog.FileExists(texAdtPath))
                        {
                            texBytes = archiveCatalog.ReadFile(texAdtPath);
                            progress?.Report($"[Split ADT] Found tex0 for {tileName}");
                        }

                        if (archiveCatalog.FileExists(objAdtPath))
                        {
                            objBytes = archiveCatalog.ReadFile(objAdtPath);
                            progress?.Report($"[Split ADT] Found obj0 for {tileName}");
                        }
                    }
                    
                    if (adtBytes == null || adtBytes.Length == 0)
                    {
                        Interlocked.Increment(ref tilesSkipped);
                        return;
                    }

                    // Extract terrain data using LK/Modern ADT parsing
                    sample = await ExtractFromLkAdt(adtBytes, texBytes, objBytes, tileIndex, tileName, outputDir,
                        shadowsDir, masksDir, allTextures, archiveCatalog, searchPaths, groundEffectLookup, wdlTile, wdtMphdFlags);
                }

                if (sample == null)
                {
                    Interlocked.Increment(ref tilesSkipped);
                    return;
                }

                var finalSample = new VlmTrainingSample(
                    imageRelPath ?? "",
                    null,
                    sample
                );

                var jsonPath = Path.Combine(datasetDir, $"{tileName}.json");
                var json = JsonSerializer.Serialize(finalSample, _jsonOptions);
                await File.WriteAllTextAsync(jsonPath, json);

                Interlocked.Increment(ref tilesExported);
                var currentCount = tilesExported;
                if (currentCount % 50 == 0)
                    progress?.Report($"Exported {currentCount} tiles...");
            }
            catch (Exception ex)
            {
                progress?.Report($"Error processing {tileName}: {ex.Message}");
                Interlocked.Increment(ref tilesSkipped);
            }
        });

        var textureDbPath = Path.Combine(outputDir, "texture_database.json");
        var textureDb = new { count = allTextures.Count, textures = allTextures.Keys.ToList() };
        await File.WriteAllTextAsync(textureDbPath, JsonSerializer.Serialize(textureDb, _jsonOptions));

        if (!skipDerivedAssets && allTextures.Count > 0)
        {
            progress?.Report($"Exporting {allTextures.Count} unique tileset textures...");
            int textureCount = ExportTilesetTextures(outputDir, allTextures.Keys, archiveCatalog, searchPaths);
            progress?.Report($"Exported {textureCount} textures");
        }

        // Stitch chunk data into tile-level images
        if (!skipDerivedAssets && tilesExported > 0)
        {
            progress?.Report("Stitching tile images...");
            var stitchedDir = Path.Combine(outputDir, "stitched");
            var semanticDir = Path.Combine(outputDir, "semantic");
            // liquidsDir already declared/created at start
            Directory.CreateDirectory(stitchedDir);
            Directory.CreateDirectory(semanticDir);

            var minimapBakeService = new MinimapBakeService(outputDir);
            
            var jsonFiles = Directory.GetFiles(datasetDir, "*.json");
            int stitchedCount = 0;
            foreach (var jsonPath in jsonFiles)
            {
                var tileName = Path.GetFileNameWithoutExtension(jsonPath);
                try
                {
                    // Load JSON to update with stitched paths
                    var json = await File.ReadAllTextAsync(jsonPath);
                    var sample = JsonSerializer.Deserialize<VlmTrainingSample>(json);
                    
                    if (sample != null && sample.TerrainData != null)
                    {
                        // Stitch shadows and alpha layers from serialized per-chunk data.
                        var (shadowPath, alphaPaths, alphaAtlasPath) = await TileStitchingService.StitchTileWithPackedAtlasAsync(
                            sample.TerrainData, tileName, stitchedDir);

                        // Stitch Liquids
                        string? lHeightPath = null;
                        string? lMaskPath = null;
                        string? noLiquidMinimapPath = null;
                        string? noMccvMinimapPath = null;
                        string? objectVisibilityMaskPath = null;
                        string? pm4MaskPath = null;
                        string? noObjectMinimapPath = null;
                        string? terrainOnlyMinimapPath = null;
                        string? holesMaskPath = null;
                        string? areaIdMapPath = null;
                        string? chunkFlagsMapPath = null;
                        string? liquidTypeMapPath = null;
                        string? dominantEffectIdMapPath = null;
                        float lMin = 0f, lMax = 0f;
                        byte[] objectMaskBytes = Array.Empty<byte>();
                        byte[] pm4MaskBytes = Array.Empty<byte>();
                        byte[] liquidMaskBytes = Array.Empty<byte>();
                        int semanticWidth = 256;
                        int semanticHeight = 256;

                        string? sourceMinimapPath = sample.ImagePath;
                        if (!string.IsNullOrWhiteSpace(sourceMinimapPath) && !Path.IsPathRooted(sourceMinimapPath))
                            sourceMinimapPath = Path.Combine(outputDir, sourceMinimapPath);

                        string? cleanedTerrainMinimapPath = sourceMinimapPath;

                        if (!string.IsNullOrWhiteSpace(sourceMinimapPath)
                            && File.Exists(sourceMinimapPath)
                            && !string.IsNullOrWhiteSpace(sample.TerrainData.MccvMapPath))
                        {
                            string absoluteMccvPath = sample.TerrainData.MccvMapPath!;
                            if (!Path.IsPathRooted(absoluteMccvPath))
                                absoluteMccvPath = Path.Combine(outputDir, absoluteMccvPath);

                            if (File.Exists(absoluteMccvPath))
                            {
                                byte[] noMccvBytes = VlmMinimapCleanupService.RemoveMccvTint(sourceMinimapPath, absoluteMccvPath);
                                if (noMccvBytes.Length > 0)
                                {
                                    noMccvMinimapPath = $"images/{tileName}_no_mccv.png";
                                    cleanedTerrainMinimapPath = Path.Combine(outputDir, noMccvMinimapPath);
                                    await File.WriteAllBytesAsync(cleanedTerrainMinimapPath, noMccvBytes);
                                }
                            }
                        }

                        if (!string.IsNullOrWhiteSpace(sourceMinimapPath) && File.Exists(sourceMinimapPath))
                        {
                            using Image<Rgba32> minimapForMask = Image.Load<Rgba32>(sourceMinimapPath);
                            if (minimapForMask.Width > 0 && minimapForMask.Height > 0)
                            {
                                semanticWidth = minimapForMask.Width;
                                semanticHeight = minimapForMask.Height;

                                if (TryParseTileCoordinates(tileName, out int tileX, out int tileY))
                                {
                                    IReadOnlyList<MprlEntry> positionRefs = LoadPm4PositionRefs(
                                        searchPaths,
                                        archiveCatalog,
                                        mapName,
                                        mapDirectory,
                                        tileName,
                                        tileX,
                                        tileY);
                                    if (positionRefs.Count > 0)
                                    {
                                        pm4MaskBytes = VlmPm4MaskService.BuildPm4Mask(
                                            tileName,
                                            positionRefs,
                                            minimapForMask.Width,
                                            minimapForMask.Height);
                                        if (pm4MaskBytes.Length > 0)
                                        {
                                            pm4MaskPath = $"images/{tileName}_pm4_mask.png";
                                            await File.WriteAllBytesAsync(Path.Combine(outputDir, pm4MaskPath), pm4MaskBytes);
                                        }
                                    }
                                }

                                objectMaskBytes = BuildObjectVisibilityMask(sample.TerrainData, minimapForMask.Width, minimapForMask.Height, archiveCatalog, searchPaths);
                                if (objectMaskBytes.Length > 0)
                                {
                                    objectVisibilityMaskPath = $"images/{tileName}_object_visibility_mask.png";
                                    await File.WriteAllBytesAsync(Path.Combine(outputDir, objectVisibilityMaskPath), objectMaskBytes);
                                }

                                byte[] removalMask = CombineMaskPngBytes(objectMaskBytes, pm4MaskBytes);
                                if (removalMask.Length > 0)
                                {
                                    byte[] noObjectBytes = await SynthesizeTerrainMaskedMinimapAsync(
                                        cleanedTerrainMinimapPath ?? sourceMinimapPath,
                                        removalMask,
                                        sample.TerrainData,
                                        minimapBakeService);
                                    if (noObjectBytes.Length > 0)
                                    {
                                        noObjectMinimapPath = $"images/{tileName}_no_objects.png";
                                        await File.WriteAllBytesAsync(Path.Combine(outputDir, noObjectMinimapPath), noObjectBytes);
                                    }
                                }
                            }
                        }

                        if (sample.TerrainData.Liquids != null)
                        {
                            var liquidsList = sample.TerrainData.Liquids.ToList();
                            
                            // Heights
                            var (liqImg, min, max) = TileStitchingService.StitchLiquidHeights(liquidsList, tileName);
                            if (liqImg.Length > 0)
                            {
                                lHeightPath = $"liquids/{tileName}_liq_height.png";
                                await File.WriteAllBytesAsync(Path.Combine(outputDir, lHeightPath), liqImg);
                                lMin = min;
                                lMax = max;
                            }

                            // Mask
                            var liqMask = TileStitchingService.StitchLiquidMask(liquidsList, tileName);
                            if (liqMask.Length > 0)
                            {
                                liquidMaskBytes = liqMask;
                                lMaskPath = $"liquids/{tileName}_liq_mask.png";
                                await File.WriteAllBytesAsync(Path.Combine(outputDir, lMaskPath), liqMask);

                                // Generate a synthetic no-liquid minimap for training.
                                if (!string.IsNullOrWhiteSpace(sourceMinimapPath) && File.Exists(sourceMinimapPath))
                                {
                                    var noLiqBytes = SynthesizeMaskedMinimap(cleanedTerrainMinimapPath ?? sourceMinimapPath, liqMask);
                                    if (noLiqBytes.Length > 0)
                                    {
                                        noLiquidMinimapPath = $"images/{tileName}_no_liquid.png";
                                        await File.WriteAllBytesAsync(Path.Combine(outputDir, noLiquidMinimapPath), noLiqBytes);
                                    }
                                }
                            }
                        }

                        if (!string.IsNullOrWhiteSpace(sourceMinimapPath) && File.Exists(sourceMinimapPath))
                        {
                            byte[] terrainOnlyRemovalMask = await BuildCombinedTerrainOnlyMaskAsync(
                                GetTerrainOnlyMaskPaths(alphaPaths, shadowPath),
                                objectMaskBytes,
                                pm4MaskBytes,
                                liquidMaskBytes);
                            if (terrainOnlyRemovalMask.Length > 0)
                            {
                                byte[] terrainOnlyBytes = await SynthesizeTerrainMaskedMinimapAsync(
                                    cleanedTerrainMinimapPath ?? sourceMinimapPath,
                                    terrainOnlyRemovalMask,
                                    sample.TerrainData,
                                    minimapBakeService);
                                if (terrainOnlyBytes.Length > 0)
                                {
                                    terrainOnlyMinimapPath = $"images/{tileName}_terrain_only.png";
                                    await File.WriteAllBytesAsync(Path.Combine(outputDir, terrainOnlyMinimapPath), terrainOnlyBytes);
                                }
                            }
                        }

                        byte[] holesMaskBytes = RenderHolesMask(sample.TerrainData.Holes, semanticWidth, semanticHeight);
                        if (holesMaskBytes.Length > 0)
                        {
                            holesMaskPath = $"semantic/{tileName}_holes_mask.png";
                            await File.WriteAllBytesAsync(Path.Combine(outputDir, holesMaskPath), holesMaskBytes);
                        }

                        byte[] areaIdMapBytes = RenderChunkValueMap(BuildChunkAreaIdValues(sample.TerrainData.ChunkLayers), semanticWidth, semanticHeight);
                        if (areaIdMapBytes.Length > 0)
                        {
                            areaIdMapPath = $"semantic/{tileName}_area_id_map.png";
                            await File.WriteAllBytesAsync(Path.Combine(outputDir, areaIdMapPath), areaIdMapBytes);
                        }

                        byte[] chunkFlagsMapBytes = RenderChunkFlagMap(BuildChunkFlagValues(sample.TerrainData.ChunkLayers), semanticWidth, semanticHeight);
                        if (chunkFlagsMapBytes.Length > 0)
                        {
                            chunkFlagsMapPath = $"semantic/{tileName}_chunk_flags_map.png";
                            await File.WriteAllBytesAsync(Path.Combine(outputDir, chunkFlagsMapPath), chunkFlagsMapBytes);
                        }

                        byte[] liquidTypeMapBytes = RenderChunkValueMap(BuildLiquidTypeValues(sample.TerrainData.Liquids), semanticWidth, semanticHeight);
                        if (liquidTypeMapBytes.Length > 0)
                        {
                            liquidTypeMapPath = $"semantic/{tileName}_liquid_type_map.png";
                            await File.WriteAllBytesAsync(Path.Combine(outputDir, liquidTypeMapPath), liquidTypeMapBytes);
                        }

                        byte[] dominantEffectIdMapBytes = RenderChunkValueMap(BuildDominantEffectIdValues(sample.TerrainData.ChunkLayers), semanticWidth, semanticHeight);
                        if (dominantEffectIdMapBytes.Length > 0)
                        {
                            dominantEffectIdMapPath = $"semantic/{tileName}_dominant_effect_id_map.png";
                            await File.WriteAllBytesAsync(Path.Combine(outputDir, dominantEffectIdMapPath), dominantEffectIdMapBytes);
                        }

                        // Update Terrain Data
                        var updatedTerrain = sample.TerrainData with
                        {
                            ShadowMaps = shadowPath != null ? new[] { Path.GetRelativePath(outputDir, shadowPath).Replace("\\", "/") } : null,
                            AlphaMasks = alphaPaths.Select(p => Path.GetRelativePath(outputDir, p).Replace("\\", "/")).ToArray(),
                            AlphaAtlasPath = alphaAtlasPath != null ? Path.GetRelativePath(outputDir, alphaAtlasPath).Replace("\\", "/") : null,
                            LiquidHeightPath = lHeightPath,
                            LiquidMaskPath = lMaskPath,
                            NoLiquidMinimapPath = noLiquidMinimapPath,
                            NoMccvMinimapPath = noMccvMinimapPath,
                            ObjectVisibilityMaskPath = objectVisibilityMaskPath,
                            Pm4MaskPath = pm4MaskPath,
                            NoObjectMinimapPath = noObjectMinimapPath,
                            TerrainOnlyMinimapPath = terrainOnlyMinimapPath,
                            HolesMaskPath = holesMaskPath,
                            AreaIdMapPath = areaIdMapPath,
                            ChunkFlagsMapPath = chunkFlagsMapPath,
                            LiquidTypeMapPath = liquidTypeMapPath,
                            DominantEffectIdMapPath = dominantEffectIdMapPath,
                            LiquidMinHeight = lMin,
                            LiquidMaxHeight = lMax
                        };

                        var updatedSample = sample with { TerrainData = updatedTerrain };
                        await File.WriteAllTextAsync(jsonPath, JsonSerializer.Serialize(updatedSample, _jsonOptions));
                        stitchedCount++;
                    }
                }
                catch { }
            }
            progress?.Report($"Stitched images and updated JSON for {stitchedCount} tiles");
        }

        // Generate global heightmaps for each tile (per-map min/max)
        if (tilesExported > 0)
        {
            progress?.Report("Generating global-normalized heightmaps...");
            await GenerateGlobalHeightmapsAsync(datasetDir, outputDir, progress);
        }

        // Stitch full world map images
        if (!skipDerivedAssets && tilesExported > 0)
        {
            progress?.Report("Stitching full world map images...");
            var stitchedDir = Path.Combine(outputDir, "stitched");
            Directory.CreateDirectory(stitchedDir);
            
            // Stitch minimaps (256 resolution typical for minimaps)
            var minimapOutput = Path.Combine(stitchedDir, $"{mapName}_full_minimap.png");
            var minimapBounds = TileStitchingService.StitchFullMap(imagesDir, mapName, 256, minimapOutput);
            if (minimapBounds.HasValue)
            {
                progress?.Report($"Created full minimap: {minimapOutput}");
            }

            var noObjectMinimapOutput = Path.Combine(stitchedDir, $"{mapName}_full_minimap_no_objects.png");
            var noObjectMinimapBounds = TileStitchingService.StitchFullMap(
                imagesDir, mapName, 256, noObjectMinimapOutput, "_no_objects.png");
            if (noObjectMinimapBounds.HasValue)
            {
                progress?.Report($"Created full no-object minimap: {noObjectMinimapOutput}");
            }

            var objectMaskOutput = Path.Combine(stitchedDir, $"{mapName}_full_object_visibility_mask.png");
            var objectMaskBounds = TileStitchingService.StitchFullMap(
                imagesDir, mapName, 256, objectMaskOutput, "_object_visibility_mask.png");
            if (objectMaskBounds.HasValue)
            {
                progress?.Report($"Created full object visibility mask: {objectMaskOutput}");
            }

            var pm4MaskOutput = Path.Combine(stitchedDir, $"{mapName}_full_pm4_mask.png");
            var pm4MaskBounds = TileStitchingService.StitchFullMap(
                imagesDir, mapName, 256, pm4MaskOutput, "_pm4_mask.png");
            if (pm4MaskBounds.HasValue)
            {
                progress?.Report($"Created full PM4 mask: {pm4MaskOutput}");
            }

            // Stitch shadow maps (1024 resolution)
            var shadowOutput = Path.Combine(stitchedDir, $"{mapName}_full_shadows.png");
            var shadowBounds = TileStitchingService.StitchFullMap(
                stitchedDir, mapName, 1024, shadowOutput, "_shadow.png");
            if (shadowBounds.HasValue)
            {
                progress?.Report($"Created full shadow map: {shadowOutput}");
            }

            // Stitch alpha masks (Layers 1-4)
            for (int l = 1; l <= 4; l++)
            {
                var alphaOutput = Path.Combine(stitchedDir, $"{mapName}_full_alpha_l{l}.png");
                var alphaBounds = TileStitchingService.StitchFullMap(
                    stitchedDir, mapName, 1024, alphaOutput, $"_alpha_l{l}.png");
                if (alphaBounds.HasValue)
                {
                    progress?.Report($"Created full alpha map L{l}: {alphaOutput}");
                }
            }

            var alphaAtlasOutput = Path.Combine(stitchedDir, $"{mapName}_full_alpha_atlas.png");
            var alphaAtlasBounds = TileStitchingService.StitchFullMap(
                stitchedDir, mapName, 1024, alphaAtlasOutput, "_alpha_atlas.png");
            if (alphaAtlasBounds.HasValue)
            {
                progress?.Report($"Created full alpha atlas: {alphaAtlasOutput}");
            }
            
            // Stitch heightmaps into full world map (PNG)
            var heightmapOutput = Path.Combine(stitchedDir, $"{mapName}_full_heightmap.png");
            var heightmapBounds = StitchHeightmapsToPng(imagesDir, mapName, heightmapOutput, progress, "_heightmap");
            if (heightmapBounds.HasValue)
            {
                progress?.Report($"Created full heightmap: {heightmapOutput} ({heightmapBounds.Value.width}x{heightmapBounds.Value.height})");
            }

            var heightmapGlobalOutput = Path.Combine(stitchedDir, $"{mapName}_full_heightmap_global.png");
            var heightmapGlobalBounds = StitchHeightmapsToPng(imagesDir, mapName, heightmapGlobalOutput, progress, "_heightmap_global");
            if (heightmapGlobalBounds.HasValue)
            {
                progress?.Report($"Created full global heightmap: {heightmapGlobalOutput} ({heightmapGlobalBounds.Value.width}x{heightmapGlobalBounds.Value.height})");
            }
        }

        if (skipDerivedAssets)
        {
            progress?.Report("Skipped derived tileset, stitched, and semantic assets by request.");
        }

        progress?.Report($"Export complete: {tilesExported} tiles exported, {tilesSkipped} skipped");
        return new VlmExportResult(tilesExported, tilesSkipped, allTextures.Count, outputDir);
    }

    private static List<string> BuildSearchPaths(string rootPath)
    {
        var paths = new List<string>();

        if (string.IsNullOrWhiteSpace(rootPath))
            return paths;

        if (Directory.Exists(Path.Combine(rootPath, "World")))
            paths.Add(rootPath);

        var dataRoot = Path.Combine(rootPath, "Data");
        if (Directory.Exists(Path.Combine(dataRoot, "World")))
            paths.Add(dataRoot);

        if (Directory.Exists(rootPath))
            paths.Add(rootPath);

        return paths
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<int> FilterReachableLkTiles(
        IEnumerable<int> tiles,
        IEnumerable<string> searchPaths,
        IArchiveReader archiveReader,
        string mapDirectory)
    {
        var reachable = new List<int>();

        foreach (int tileIndex in tiles)
        {
            int x = tileIndex % 64;
            int y = tileIndex / 64;
            string rootAdtPath = $"World\\Maps\\{mapDirectory}\\{mapDirectory}_{x}_{y}.adt";

            bool foundOnDisk = searchPaths.Any(basePath => File.Exists(Path.Combine(basePath, rootAdtPath)));
            bool foundInArchive = archiveReader.FileExists(rootAdtPath);

            if (foundOnDisk || foundInArchive)
                reachable.Add(tileIndex);
        }

        return reachable;
    }

    private async Task UpdateJsonWithDepthPaths(string datasetDir, IProgress<string>? progress)
    {
        var jsonFiles = Directory.GetFiles(datasetDir, "*.json");
        int updated = 0;
        
        foreach (var jsonPath in jsonFiles)
        {
            try
            {
                var json = await File.ReadAllTextAsync(jsonPath);
                var sample = JsonSerializer.Deserialize<VlmTrainingSample>(json);
                if (sample == null) continue;
                
                var baseName = Path.GetFileNameWithoutExtension(jsonPath);
                var depthRelPath = $"depths/{baseName}_depth.png";
                
                var depthAbsPath = Path.Combine(Path.GetDirectoryName(datasetDir)!, depthRelPath);
                if (File.Exists(depthAbsPath))
                {
                    var updatedSample = sample with { DepthPath = depthRelPath };
                    var updatedJson = JsonSerializer.Serialize(updatedSample, _jsonOptions);
                    await File.WriteAllTextAsync(jsonPath, updatedJson);
                    updated++;
                }
            }
            catch { }
        }
        
        progress?.Report($"Updated {updated} JSON files with depth paths");
    }

    private async Task<VlmTerrainData?> ExtractFromAdtAlpha(
        AdtAlpha adt, string wdtPath, int adtOffset, int tileIndex, string tileName,
        string outputDir, string shadowsDir, string masksDir,
        List<string> mdnmNames, List<string> monmNames,
        ConcurrentDictionary<string, byte> textureCollector, GroundEffectLookup? groundEffectLookup = null,
        WdlParser.WdlTile? wdlTile = null, string? clientPath = null)
    {
        var heights = new List<VlmChunkHeights>();

        // Prepare WDL data if available
        VlmWdlData? wdlHeights = null;
        if (wdlTile != null && wdlTile.HasData)
        {
            // Flatten arrays
            var h17 = new short[17 * 17];
            for (int r = 0; r < 17; r++)
                for (int c = 0; c < 17; c++)
                    h17[r * 17 + c] = wdlTile.Height17[r, c];

            var h16 = new short[16 * 16];
            for (int r = 0; r < 16; r++)
                for (int c = 0; c < 16; c++)
                    h16[r * 16 + c] = wdlTile.Height16[r, c];

            wdlHeights = new VlmWdlData(h17, h16);
        }
        var chunkPositions = new float[256 * 3];
        var holes = new int[256];
        var textures = new List<string>();
        var chunkLayers = new List<VlmChunkLayers>();
        var liquids = new List<VlmLiquidData>();
        var objects = new List<VlmObjectPlacement>();
        var shadowPaths = new List<string>();
        var shadowBits = new List<VlmChunkShadowBits>();
        var alphaPaths = new List<string>();
        
        float heightMin = float.MaxValue;
        float heightMax = float.MinValue;

        // Get textures from MTEX
        var mtexNames = adt.GetMtexTextureNames();
        if (mtexNames != null)
        {
            textures.AddRange(mtexNames);
            foreach (var t in mtexNames) textureCollector.TryAdd(t, 0);
        }

        // Process MCNKs by reading directly with McnkAlpha
        // Same approach as AdtAlpha.ToAdtLk
        try
        {
            // Create a temp AdtAlpha to get MCIN offsets
            // We already have adt, but need to access internal _mcin - so we recreate the pattern
            using var fs = File.OpenRead(wdtPath);
            
            // Read MHDR to get MCIN offset
            fs.Seek(adtOffset + 8, SeekOrigin.Begin); // Skip chunk header
            var mhdrBuf = new byte[64];
            fs.Read(mhdrBuf, 0, 64);
            int mcinOffsetRel = BitConverter.ToInt32(mhdrBuf, 0);
            int mcinAbsolute = adtOffset + 8 + mcinOffsetRel;
            
            // Read MCIN
            fs.Seek(mcinAbsolute + 8, SeekOrigin.Begin); // Skip MCIN chunk header
            var mcinBuf = new byte[256 * 16];
            fs.Read(mcinBuf, 0, mcinBuf.Length);
            
            var mcnkOffsets = new int[256];
            for (int i = 0; i < 256; i++)
            {
                mcnkOffsets[i] = BitConverter.ToInt32(mcinBuf, i * 16);
            }
            
            // Process each MCNK using McnkAlpha - now with public accessors!
            for (int i = 0; i < 256; i++)
            {
                int off = mcnkOffsets[i];
                if (off <= 0) continue;
                
                try
                {
                    // Use off directly - MCIN stores absolute offsets in Alpha WDT
                    var mcnk = new McnkAlpha(fs, off, 0, tileIndex);
                    
                    // Use public accessors instead of manual header parsing
                    int idxX = mcnk.IndexX;
                    int idxY = mcnk.IndexY;
                    int chunkIndex = idxY * 16 + idxX;
                    if (chunkIndex < 0 || chunkIndex >= 256) continue;
                    int nLayers = mcnk.NLayers;
                    
                    // Extract heights from McvtData (145 floats = 580 bytes)
                    var mcvtBuf = mcnk.McvtData;
                    var chunkHeights = new float[145];
                    for (int h = 0; h < 145 && h * 4 + 3 < mcvtBuf.Length; h++)
                    {
                        chunkHeights[h] = BitConverter.ToSingle(mcvtBuf, h * 4);
                        if (float.IsNaN(chunkHeights[h]) || float.IsInfinity(chunkHeights[h]))
                            chunkHeights[h] = 0;
                        else
                        {
                            if (chunkHeights[h] < heightMin) heightMin = chunkHeights[h];
                            if (chunkHeights[h] > heightMax) heightMax = chunkHeights[h];
                        }
                    }
                    heights.Add(new VlmChunkHeights(chunkIndex, chunkHeights));
                    
                    // Positions - compute from tile/chunk indices  
                    float posX = (32 - (tileIndex / 64)) * 533.33333f - idxX * 33.33333f;
                    float posY = (32 - (tileIndex % 64)) * 533.33333f - idxY * 33.33333f;
                    float posZ = 0; // Base height
                    chunkPositions[chunkIndex * 3] = posX;
                    chunkPositions[chunkIndex * 3 + 1] = posY;
                    chunkPositions[chunkIndex * 3 + 2] = posZ;
                    holes[chunkIndex] = mcnk.Holes;
                    
                    // Extract shadow from McshData (64x64 bits = 512 bytes raw, but check McshSize from header)
                    var mcshBuf = mcnk.McshData;
                    int mcshSize = mcnk.McshSize;
                    
                    // MCSH needs at least 64 bytes (512 bits minimum for partial shadow)
                    // Full shadow is 512 bytes (64 rows × 8 bytes/row)
                    if (mcshBuf.Length > 0 && mcshSize > 0)
                    {
                        try
                        {
                            // Store raw shadow bits (full 512 bytes = 64 rows × 8 bytes/row)
                            int shadowByteCount = Math.Min(512, mcshBuf.Length);
                            var rawShadowBytes = new byte[shadowByteCount];
                            Array.Copy(mcshBuf, rawShadowBytes, shadowByteCount);
                            shadowBits.Add(new VlmChunkShadowBits(chunkIndex, Convert.ToBase64String(rawShadowBytes)));
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[MCSH] Error processing shadow for {tileName}_c{chunkIndex}: {ex.Message}");
                        }
                    }
                    
                    // Extract alpha layers from McalData + MclyData
                    var mcalBuf = mcnk.McalData;
                    var mclyBuf = mcnk.MclyData;
                    
                    // Collect raw alpha data per layer for storage
                    var layerAlphaBits = new Dictionary<int, string>(); // layer index -> Base64
                    
                    if (mcalBuf.Length > 0 && nLayers > 1)
                    {
                        try
                        {
                            // Parse MCLY to get layer flags (16 bytes per layer)
                            int alphaOffset = 0;
                            for (int layer = 1; layer < nLayers && layer < 4; layer++)
                            {
                                if (layer * 16 > mclyBuf.Length) break;
                                uint layerFlags = BitConverter.ToUInt32(mclyBuf, layer * 16 + 4);
                                bool isCompressed = (layerFlags & 0x200) != 0;
                                
                                // Read this layer's alpha
                                var alphaData = AlphaMapService.ReadAlpha(mcalBuf, alphaOffset, layerFlags, false, false);
                                layerAlphaBits[layer] = Convert.ToBase64String(alphaData);
                                
                                int alphaSize = isCompressed ? 4096 : 2048;
                                
                                // Advance offset (2048 for uncompressed 4-bit, varies for compressed)
                                alphaOffset += alphaSize;
                            }
                        }
                        catch { }
                    }
                    
                    // Store layer info for this chunk with resolved texture paths
                    var layerList = new List<VlmTextureLayer>();
                    
                    // Try to parse MCLY if it has data
                    if (mclyBuf.Length >= 16)
                    {
                        for (int layer = 0; layer < nLayers && layer < 4 && layer * 16 + 15 < mclyBuf.Length; layer++)
                        {
                            uint textureId = BitConverter.ToUInt32(mclyBuf, layer * 16);
                            uint flags = BitConverter.ToUInt32(mclyBuf, layer * 16 + 4);
                            uint alphaoffs = BitConverter.ToUInt32(mclyBuf, layer * 16 + 8);
                            uint effectId = BitConverter.ToUInt32(mclyBuf, layer * 16 + 12);
                            
                            // Resolve texture path from MTEX index
                            string? texturePath = textureId < textures.Count ? textures[(int)textureId] : null;
                            
                            string[]? groundEffects = null;
                            if (effectId > 0 && groundEffectLookup != null)
                            {
                                groundEffects = groundEffectLookup.GetDoodadsEffect(effectId);
                            }
                            
                            // Get raw alpha bits if available (only for layers > 0)
                            string? alphaBitsBase64 = layerAlphaBits.TryGetValue(layer, out var bits) ? bits : null;
                            
                            // Alpha path for this layer (layer > 0)
                            string? alphaPath = null;

                            layerList.Add(new VlmTextureLayer(textureId, texturePath, flags, alphaoffs, effectId, groundEffects, alphaBitsBase64, alphaPath));
                        }
                    }
                    
                    // Fallback: if no layers parsed but we have textures, create layers from nLayers count
                    if (layerList.Count == 0 && nLayers > 0 && textures.Count > 0)
                    {
                        for (int layer = 0; layer < nLayers && layer < 4 && layer < textures.Count; layer++)
                        {
                            string? alphaPath = null;
                            layerList.Add(new VlmTextureLayer((uint)layer, textures[layer], 0, 0, 0, null, null, alphaPath));
                        }
                    }
                    
                    // Shadow path for this chunk
                    string? chunkShadowPath = null;
                    
                    // Extract normals (MCNR - 448 bytes)
                    sbyte[]? normalsArray = null;
                    var mcnrBuf = mcnk.McnrData;
                    if (mcnrBuf != null && mcnrBuf.Length > 0)
                    {
                        normalsArray = new sbyte[mcnrBuf.Length];
                        for (int n = 0; n < mcnrBuf.Length; n++)
                            normalsArray[n] = (sbyte)mcnrBuf[n];
                    }
                    
                    // Get area_id and flags from MCNK header
                    uint areaId = (uint)mcnk.Header.Unknown3;  // Unknown3 is area ID in Alpha
                    uint chunkFlags = (uint)mcnk.Header.Flags;
                    
                    // Extract MCCV vertex colors (if present - not in Alpha, added in later versions)
                    byte[]? mccvColors = null;
                    var mccvBuf = mcnk.MccvData;
                    if (mccvBuf != null && mccvBuf.Length > 0)
                    {
                        mccvColors = mccvBuf;
                    }
                    
                    chunkLayers.Add(new VlmChunkLayers(chunkIndex, layerList.ToArray(), chunkShadowPath, normalsArray, mccvColors, areaId, chunkFlags));

                    // Extract Liquid Data (MCLQ - Legacy)
                    var mclqData = mcnk.MclqData;
                    if (mclqData != null && mclqData.Length > 0)
                    {
                        var liquid = LiquidService.ExtractMCLQ(mclqData, chunkIndex);
                        if (liquid != null)
                        {
                            liquids.Add(liquid);
                        }
                    }
                }
                catch { }
            }
        }
        catch
        {
            return null;
        }

        // Extract objects using MDDF/MODF raw data
        try
        {
            var mddfRaw = adt.GetMddfRaw();
            const int mddfEntrySize = 36;
            for (int i = 0; i + mddfEntrySize <= mddfRaw.Length; i += mddfEntrySize)
            {
                uint nameId = BitConverter.ToUInt32(mddfRaw, i);
                uint uniqueId = BitConverter.ToUInt32(mddfRaw, i + 4);
                float px = BitConverter.ToSingle(mddfRaw, i + 8);
                float py = BitConverter.ToSingle(mddfRaw, i + 12);
                float pz = BitConverter.ToSingle(mddfRaw, i + 16);
                float rx = BitConverter.ToSingle(mddfRaw, i + 20);
                float ry = BitConverter.ToSingle(mddfRaw, i + 24);
                float rz = BitConverter.ToSingle(mddfRaw, i + 28);
                ushort scale = BitConverter.ToUInt16(mddfRaw, i + 32);
                
                // Get full model path and extract bounds from MDX file
                string fullPath = nameId < mdnmNames.Count ? mdnmNames[(int)nameId] : "";
                string name = Path.GetFileNameWithoutExtension(fullPath);
                
                // Extract bounding box from MDX via AlphaArchiveReader
                float[]? boundsMin = null;
                float[]? boundsMax = null;
                if (!string.IsNullOrEmpty(clientPath) && !string.IsNullOrEmpty(fullPath))
                {
                    var modelMpqPath = Path.Combine(clientPath, "Data", fullPath + ".MPQ");
                    var bounds = GetMdxBounds(modelMpqPath);
                    if (bounds != null)
                    {
                        boundsMin = bounds.Value.Min;
                        boundsMax = bounds.Value.Max;
                    }
                }
                
                objects.Add(new VlmObjectPlacement(name, nameId, uniqueId, px, py, pz, rx, ry, rz, scale / 1024f, "m2", boundsMin, boundsMax, fullPath));
            }
            
            var modfRaw = adt.GetModfRaw();
            const int modfEntrySize = 64;
            for (int i = 0; i + modfEntrySize <= modfRaw.Length; i += modfEntrySize)
            {
                uint nameId = BitConverter.ToUInt32(modfRaw, i);
                uint uniqueId = BitConverter.ToUInt32(modfRaw, i + 4);
                float px = BitConverter.ToSingle(modfRaw, i + 8);
                float py = BitConverter.ToSingle(modfRaw, i + 12);
                float pz = BitConverter.ToSingle(modfRaw, i + 16);
                float rx = BitConverter.ToSingle(modfRaw, i + 20);
                float ry = BitConverter.ToSingle(modfRaw, i + 24);
                float rz = BitConverter.ToSingle(modfRaw, i + 28);
                ushort scale = BitConverter.ToUInt16(modfRaw, i + 60);
                
                // Get full model path and extract bounds from WMO file
                string fullPath = nameId < monmNames.Count ? monmNames[(int)nameId] : "";
                string name = Path.GetFileNameWithoutExtension(fullPath);
                
                // Extract bounding box from WMO via AlphaArchiveReader
                float[]? boundsMin = null;
                float[]? boundsMax = null;
                if (!string.IsNullOrEmpty(clientPath) && !string.IsNullOrEmpty(fullPath))
                {
                    var modelMpqPath = Path.Combine(clientPath, "Data", fullPath + ".MPQ");
                    var bounds = GetWmoBounds(modelMpqPath);
                    if (bounds != null)
                    {
                        boundsMin = bounds.Value.Min;
                        boundsMax = bounds.Value.Max;
                    }
                }
                
                objects.Add(new VlmObjectPlacement(name, nameId, uniqueId, px, py, pz, rx, ry, rz, scale / 1024f, "wmo", boundsMin, boundsMax, fullPath));
            }
        }
        catch { }

        if (heights.Count == 0)
            return null;

        VlmChunkShadowAnalysis[]? shadowAnalysis = shadowBits.Count > 0
            ? VlmShadowAssociationService.AnalyzeTile(shadowBits, chunkPositions, objects)
            : null;

        var heightmapPath = await GenerateHeightmap(heights, tileName, outputDir, isInterleaved: false);
        
        var normalmapPath = await GenerateNormalmap(heights, holes, tileName, outputDir, isInterleaved: false);
        var mccvMapPath = await GenerateMccvMap(chunkLayers, tileName, outputDir);
        
        return new VlmTerrainData(
            AdtTile: tileName,
            Heights: heights.ToArray(),
            ChunkPositions: chunkPositions,
            Holes: holes,
            HeightmapPath: heightmapPath,
            HeightmapLocalPath: heightmapPath,
            HeightmapGlobalPath: null,
            NormalmapPath: normalmapPath,
            MccvMapPath: mccvMapPath,
            ShadowMaps: shadowPaths.Count > 0 ? shadowPaths.ToArray() : null,
            ShadowBits: shadowBits.Count > 0 ? shadowBits.ToArray() : null,
            ShadowAnalysis: shadowAnalysis,
            AlphaMasks: alphaPaths.Count > 0 ? alphaPaths.ToArray() : null,
            AlphaAtlasPath: null,
            LiquidMaskPath: null,
            LiquidHeightPath: null,
            LiquidMinHeight: 0f,
            LiquidMaxHeight: 0f,
            NoLiquidMinimapPath: null,
            NoMccvMinimapPath: null,
            ObjectVisibilityMaskPath: null,
            Pm4MaskPath: null,
            NoObjectMinimapPath: null,
            TerrainOnlyMinimapPath: null,
            HolesMaskPath: null,
            AreaIdMapPath: null,
            ChunkFlagsMapPath: null,
            LiquidTypeMapPath: null,
            DominantEffectIdMapPath: null,
            Textures: textures,
            ChunkLayers: chunkLayers.Count > 0 ? chunkLayers.ToArray() : null,
            Liquids: liquids.Count > 0 ? liquids.ToArray() : null,
            Objects: objects,
            WdlHeights: wdlHeights,
            HeightMin: heightMin == float.MaxValue ? 0 : heightMin,
            HeightMax: heightMax == float.MinValue ? 0 : heightMax,
            HeightGlobalMin: 0,
            HeightGlobalMax: 0,
            IsInterleaved: false);
    }

    /// <summary>
    /// Extract terrain data from LK/Modern ADT bytes.
    /// Supports Split ADTs (_tex0, _obj0) via optional buffers.
    /// </summary>
    private async Task<VlmTerrainData?> ExtractFromLkAdt(
        byte[] adtBytes, byte[]? texBytes, byte[]? objBytes, int tileIndex, string tileName,
        string outputDir, string shadowsDir, string masksDir,
        ConcurrentDictionary<string, byte> textureCollector, IArchiveReader archiveReader,
        IReadOnlyList<string> searchPaths,
        GroundEffectLookup? groundEffectLookup = null, WdlParser.WdlTile? wdlTile = null,
        uint wdtMphdFlags = 0)
    {
        float heightMin = float.MaxValue;
        float heightMax = float.MinValue;

        // Parse X and Y from tileName (e.g. "Azeroth_30_20")
        int x = 0;
        int y = 0;
        try 
        {
            // Assuming format MapName_X_Y
            var parts = tileName.Split('_');
            if (parts.Length >= 2)
            {
                if (int.TryParse(parts[parts.Length - 2], out int px) && int.TryParse(parts[parts.Length - 1], out int py))
                {
                   x = px;
                   y = py;
                }
            }
        }
        catch { /* ignore, default to 0,0 */ }

        var heights = new List<VlmChunkHeights>();
        var chunkLayers = new List<VlmChunkLayers>();
        var chunkPositions = new float[256 * 3];
        var holes = new int[256];
        var shadowPaths = new List<string>();
        var shadowBits = new List<VlmChunkShadowBits>();
        var liquids = new List<VlmLiquidData>();
        var objectPlacements = new List<VlmObjectPlacement>();
        
        // Prepare WDL data if available
        VlmWdlData? wdlHeights = null;
        if (wdlTile != null && wdlTile.HasData)
        {
            var h17 = new short[17 * 17];
            for (int r = 0; r < 17; r++)
                for (int c = 0; c < 17; c++)
                    h17[r * 17 + c] = wdlTile.Height17[r, c];

            var h16 = new short[16 * 16];
            for (int r = 0; r < 16; r++)
                for (int c = 0; c < 16; c++)
                    h16[r * 16 + c] = wdlTile.Height16[r, c];

            wdlHeights = new VlmWdlData(h17, h16);
        }

        try
        {
            // Use MCIN offsets to locate MCNK chunks (gillijimproject approach)
            var textures = new List<string>();
            var rootM2Names = new List<string>();
            var rootWmoNames = new List<string>();
            byte[]? mh2oData = null;
            byte[]? rootMddfRaw = null;
            byte[]? rootModfRaw = null;
            var legacyMclqLiquids = new List<VlmLiquidData>();
            var shadowMapData = new byte[256][];
            
            // Find MHDR chunk (on-disk 'RDHM')
            int mhdrOffset = -1;
            for (int i = 0; i + 8 <= adtBytes.Length;)
            {
                string fcc = System.Text.Encoding.ASCII.GetString(adtBytes, i, 4);
                int sz = BitConverter.ToInt32(adtBytes, i + 4);
                if (sz < 0) break;
                int next = i + 8 + sz + ((sz & 1) == 1 ? 1 : 0);
                if (fcc == "RDHM") { mhdrOffset = i; break; }
                if (i + 8 + sz > adtBytes.Length) break;
                if (next <= i) break;
                i = next;
            }
            
            if (mhdrOffset < 0)
            {
                Console.WriteLine($"[LK ADT] MHDR not found in {tileName}");
                return null;
            }
            
            // Use gillijimproject Mhdr and Mcin classes
            var mhdr = new GillijimProject.WowFiles.Mhdr(adtBytes, mhdrOffset);
            int mhdrStart = mhdrOffset + 8;
            int mcinOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.McinOffset);
            
            List<int> mcnkOffsets;
            if (mcinOff != 0)
            {
                var mcin = new GillijimProject.WowFiles.Mcin(adtBytes, mhdrStart + mcinOff);
                mcnkOffsets = mcin.GetMcnkOffsets();
                Console.WriteLine($"[DEBUG] Found {mcnkOffsets.Count} MCNK offsets via MCIN for {tileName}");
            }
            else
            {
                // Cata 4.0.0+ split-ADT: no MCIN chunk, scan sequentially for MCNK (on-disk 'KNCM')
                mcnkOffsets = new List<int>();
                for (int i = 0; i + 8 <= adtBytes.Length;)
                {
                    string fcc = System.Text.Encoding.ASCII.GetString(adtBytes, i, 4);
                    int sz = BitConverter.ToInt32(adtBytes, i + 4);
                    if (sz < 0 || i + 8 + sz > adtBytes.Length) break;
                    if (fcc == "KNCM")
                        mcnkOffsets.Add(i);
                    int next = i + 8 + sz;
                    if (next <= i) break;
                    i = next;
                }
                Console.WriteLine($"[Split ADT] Found {mcnkOffsets.Count} MCNK offsets via sequential scan for {tileName}");
            }
            
            CollectTopLevelChunkData(adtBytes, textures, rootM2Names, rootWmoNames, ref mh2oData, ref rootMddfRaw, ref rootModfRaw);

            // Split-ADT: parse _tex0 for texture names and per-chunk layer/alpha data
            Dictionary<int, WoWMapConverter.Core.Formats.LichKing.Mcnk>? texMcnkByChunkIndex = null;
            if (texBytes is { Length: > 16 })
            {
                // Get MTEX from _tex0 (in split-ADT, texture names live here)
                var tex0Textures = new List<string>();
                byte[]? ignoreMh2o = null;
                byte[]? ignoreMddf = null;
                byte[]? ignoreModf = null;
                CollectTopLevelChunkData(texBytes, tex0Textures, null, null, ref ignoreMh2o, ref ignoreMddf, ref ignoreModf);
                if (tex0Textures.Count > 0 && textures.Count == 0)
                {
                    textures.AddRange(tex0Textures);
                    Console.WriteLine($"[Split ADT] Loaded {tex0Textures.Count} textures from _tex0 for {tileName}");
                }

                // Build MCNK index map from _tex0 for per-chunk layer data
                texMcnkByChunkIndex = new Dictionary<int, WoWMapConverter.Core.Formats.LichKing.Mcnk>();
                var texOffsets = new List<int>();
                for (int i = 0; i + 8 <= texBytes.Length;)
                {
                    string fcc = System.Text.Encoding.ASCII.GetString(texBytes, i, 4);
                    int sz = BitConverter.ToInt32(texBytes, i + 4);
                    if (sz < 0 || i + 8 + sz > texBytes.Length) break;
                    if (fcc == "KNCM")
                        texOffsets.Add(i);
                    int next = i + 8 + sz;
                    if (next <= i) break;
                    i = next;
                }

                // Cata+ _tex0 MCNK chunks are headerless — sub-chunks start at offset 0
                var tex0ParseOptions = new WoWMapConverter.Core.Formats.LichKing.Mcnk.ParseOptions { SkipHeader = true };
                for (int ci = 0; ci < Math.Min(256, texOffsets.Count); ci++)
                {
                    int off = texOffsets[ci];
                    if (off + 8 > texBytes.Length) continue;
                    int mcnkSize = BitConverter.ToInt32(texBytes, off + 4);
                    if (mcnkSize <= 0 || off + 8 + mcnkSize > texBytes.Length) continue;
                    var mcnkBody = new byte[mcnkSize];
                    Array.Copy(texBytes, off + 8, mcnkBody, 0, mcnkSize);
                    try
                    {
                        var texMcnk = new WoWMapConverter.Core.Formats.LichKing.Mcnk(mcnkBody, tex0ParseOptions);
                        texMcnkByChunkIndex[ci] = texMcnk;
                    }
                    catch { /* skip unparseable tex0 chunks */ }
                }
                Console.WriteLine($"[Split ADT] Parsed {texMcnkByChunkIndex.Count} texture MCNK chunks from _tex0 for {tileName}");
            }

            IReadOnlyList<string> m2NamesForPlacements = rootM2Names;
            IReadOnlyList<string> wmoNamesForPlacements = rootWmoNames;
            byte[]? mddfRawForPlacements = rootMddfRaw;
            byte[]? modfRawForPlacements = rootModfRaw;

            if (objBytes is { Length: > 0 })
            {
                var objM2Names = new List<string>();
                var objWmoNames = new List<string>();
                byte[]? objMddfRaw = null;
                byte[]? objModfRaw = null;

                CollectTopLevelChunkData(objBytes, null, objM2Names, objWmoNames, ref mh2oData, ref objMddfRaw, ref objModfRaw);

                if (objMddfRaw is { Length: > 0 })
                {
                    mddfRawForPlacements = objMddfRaw;
                    if (objM2Names.Count > 0)
                        m2NamesForPlacements = objM2Names;
                }

                if (objModfRaw is { Length: > 0 })
                {
                    modfRawForPlacements = objModfRaw;
                    if (objWmoNames.Count > 0)
                        wmoNamesForPlacements = objWmoNames;
                }
            }
            
            // Parse MCNK chunks using LichKing.Mcnk (ported from Warcraft.NET)
            for (int chunkIndex = 0; chunkIndex < 256 && chunkIndex < mcnkOffsets.Count; chunkIndex++)
            {
                int off = mcnkOffsets[chunkIndex];
                if (off <= 0) continue;
                
                // Get MCNK size from file
                if (off + 8 > adtBytes.Length) continue;
                int mcnkSize = BitConverter.ToInt32(adtBytes, off + 4);
                if (off + 8 + mcnkSize > adtBytes.Length) continue;
                
                // Read MCNK Body (excluding 8 byte header)
                byte[] mcnkBody = new byte[mcnkSize];
                Array.Copy(adtBytes, off + 8, mcnkBody, 0, mcnkSize);

                try 
                {
                    var mcnk = new WoWMapConverter.Core.Formats.LichKing.Mcnk(mcnkBody);

                    // 1. Position
                    if (mcnk.Header.Position != null && mcnk.Header.Position.Length == 3)
                    {
                        // DEBUG: Check coordinate values
                        if (chunkIndex == 0) // Print once per tile loop
                        {
                            // Console.WriteLine($"[POS-DEBUG] IdxX:{mcnk.Header.IndexX} IdxY:{mcnk.Header.IndexY} P[0](Z):{mcnk.Header.Position[0]} P[1](X):{mcnk.Header.Position[1]} P[2](Y):{mcnk.Header.Position[2]}");
                        }

                         // Standard Mapping Attempt based on wiki (X, Y, Z)
                         // But old code used P[1]->X, P[2]->Y, P[0]->Z.
                         // Let's stick to OLD code's mapping for now but log values to verify.
                         
                         chunkPositions[chunkIndex * 3 + 0] = mcnk.Header.Position[1]; // Old X mapping
                         // Calculate absolute World Coordinates from indices
                         // The MCNK header Position[1] (X) and Position[2] (Y) are unreliable.
                         // WoW Coords: X (North+), Y (West+), Z (Up+)
                         // Center (32,32) is (0,0). Origin (0,0) is Top-Left (Max X, Max Y).
                         
                         float TileSize = 533.33333f;
                         float ChunkSize = TileSize / 16.0f;
                         float Origin = 32.0f * TileSize; // 17066.66656
 
                         // ADT 'x' corresponds to World Y axis (West-East columns).
                         // ADT 'y' corresponds to World X axis (North-South rows).
                         // MCNK 'IndexX' is the COLUMN index (West-East).
                         // MCNK 'IndexY' is the ROW index (North-South).
                         
                         // World X (North-South) should depend on Row Indices (y and IndexY).
                         // World Y (West-East) should depend on Col Indices (x and IndexX).
 
                         float worldX = Origin - (y * TileSize) - (mcnk.Header.IndexY * ChunkSize);
                         float worldY = Origin - (x * TileSize) - (mcnk.Header.IndexX * ChunkSize);
 
                         chunkPositions[chunkIndex * 3 + 0] = worldX; 
                         chunkPositions[chunkIndex * 3 + 1] = worldY;
                         chunkPositions[chunkIndex * 3 + 2] = mcnk.Header.Position[0]; // Z (Base Height)
                    }

                    // 2. Heights
                    if (mcnk.Heightmap != null)
                    {
                        var mcvtHeights = mcnk.Heightmap;
                        float baseZ = 0f;
                        
                        // WotLK/Cata MCVT is often relative to MCNK Position Z.
                        // Position[0] = Z, Position[1] = X, Position[2] = Y (Verified by old code mapping)
                        if (mcnk.Header.Position != null && mcnk.Header.Position.Length >= 1)
                            baseZ = mcnk.Header.Position[0]; // Z is first float in MCNK header

                        // float minH = float.MaxValue;
                        // float maxH = float.MinValue;
                        // for (int h = 0; h < mcvtHeights.Length; h++) {
                        //     if (mcvtHeights[h] < minH) minH = mcvtHeights[h];
                        //     if (mcvtHeights[h] > maxH) maxH = mcvtHeights[h];
                        // }
                        // if (chunkIndex == 0) Console.WriteLine($"[HEIGHT-DEBUG] Raw Range: {minH} to {maxH} | BaseZ: {baseZ}");

                        // Create a NEW array to apply the offset
                        var absHeights = new float[mcvtHeights.Length];
                        for (int h = 0; h < mcvtHeights.Length; h++)
                        {
                            float rawVal = mcvtHeights[h];
                            if (float.IsNaN(rawVal) || float.IsInfinity(rawVal)) rawVal = 0;
                            
                            absHeights[h] = rawVal + baseZ;
                            
                            if (Math.Abs(absHeights[h]) > 50000f) absHeights[h] = baseZ; // Sanitize garbage
                            
                            // Calc global min/max for normalization
                            if (absHeights[h] < heightMin) heightMin = absHeights[h];
                            if (absHeights[h] > heightMax) heightMax = absHeights[h];
                        }
                        
                        heights.Add(new VlmChunkHeights(chunkIndex, absHeights));
                    }

                    // 3. Layers & Alpha Maps — use _tex0 MCNK if split-ADT
                    var layerSource = mcnk;
                    if (texMcnkByChunkIndex != null && texMcnkByChunkIndex.TryGetValue(chunkIndex, out var texMcnk))
                    {
                        layerSource = texMcnk;
                    }
                    
                    var layers = new List<VlmTextureLayer>();
                    if (layerSource.TextureLayers != null)
                    {
                        foreach (var layer in layerSource.TextureLayers)
                        {
                            string texPath = ((int)layer.TextureId) < textures.Count ? textures[(int)layer.TextureId] : "";
                            
                            byte[]? alphaInfo = null;
                            if (layerSource.AlphaMaps != null)
                            {
                                alphaInfo = layerSource.AlphaMaps.GetAlphaMapForLayer(layer, false);
                            }

                            layers.Add(new VlmTextureLayer(
                                layer.TextureId, 
                                texPath, 
                                (uint)layer.Flags, 
                                layer.AlphaMapOffset, 
                                layer.EffectId,
                                null, // GroundEffects
                                null, // AlphaBitsBase64
                                null, // AlphaPath
                                alphaInfo // AlphaData (byte[])
                            ));
                        }
                    }

                    sbyte[]? normalsArray = null;
                    var mcnrBuf = mcnk.McnrData;
                    if (mcnrBuf != null && mcnrBuf.Length > 0)
                    {
                        normalsArray = new sbyte[mcnrBuf.Length];
                        for (int n = 0; n < mcnrBuf.Length; n++)
                            normalsArray[n] = (sbyte)mcnrBuf[n];
                    }

                    byte[]? mccvColors = null;
                    var mccvBuf = mcnk.MccvData;
                    if (mccvBuf != null && mccvBuf.Length > 0)
                    {
                        mccvColors = mccvBuf;
                    }

                    if (layers.Count > 0 || normalsArray != null || mccvColors != null)
                    {
                        uint areaId = mcnk.Header.AreaId;
                        uint chunkFlags = (uint)mcnk.Header.Flags;
                        chunkLayers.Add(new VlmChunkLayers(chunkIndex, layers.ToArray(), null, normalsArray, mccvColors, areaId, chunkFlags));
                    }

                    // 4. Shadows (MCSH) — prefer _tex0 MCNK shadow if base has none (matches MdxViewer)
                    var shadowSource = mcnk.McshData ?? layerSource.McshData;
                    if (shadowSource != null && shadowSource.Length == 512)
                    {
                        shadowMapData[chunkIndex] = shadowSource;
                    }

                    // 5. Early LK/legacy liquids can still be stored as per-chunk MCLQ.
                    if (mcnk.MclqData != null && mcnk.MclqData.Length > 0)
                    {
                        var legacyLiquid = LiquidService.ExtractMCLQ(mcnk.MclqData, chunkIndex);
                        if (legacyLiquid != null)
                            legacyMclqLiquids.Add(legacyLiquid);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Error] Failed to parse MCNK at index {chunkIndex}: {ex.Message}");
                }
            }

            if (mh2oData is { Length: > 0 })
            {
                try
                {
                    var mh2o = Mh2oChunk.Parse(mh2oData);
                    foreach (var instance in mh2o.Instances)
                    {
                        var liquid = LiquidService.CreateMh2oLiquid(instance);
                        if (liquid != null)
                            liquids.Add(liquid);
                    }

                    if (liquids.Count > 0)
                        Console.WriteLine($"[DEBUG] Parsed {liquids.Count} MH2O liquid layers for {tileName}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[LK ADT] Failed to parse MH2O in {tileName}: {ex.Message}");
                }
            }

            if (legacyMclqLiquids.Count > 0)
            {
                int liquidCountBeforeLegacyMerge = liquids.Count;
                liquids.AddRange(MergeLegacyLiquids(liquids, legacyMclqLiquids));

                int mergedLegacyChunkCount = liquids.Count - liquidCountBeforeLegacyMerge;
                if (mergedLegacyChunkCount > 0)
                    Console.WriteLine($"[DEBUG] Added {mergedLegacyChunkCount} legacy MCLQ liquid chunks for {tileName}");
            }

            AppendObjectPlacementsFromRaw(objectPlacements, mddfRawForPlacements, m2NamesForPlacements, "m2", archiveReader, searchPaths);
            AppendObjectPlacementsFromRaw(objectPlacements, modfRawForPlacements, wmoNamesForPlacements, "wmo", archiveReader, searchPaths);
            
            Console.WriteLine($"[DEBUG] Parsed {heights.Count} chunks with heights, range {heightMin:F2} to {heightMax:F2}");
            
            // Collect unique textures
            foreach (var t in textures) textureCollector.TryAdd(t, 0);
            
            // Process shadow maps
            for (int i = 0; i < 256; i++)
            {
                if (shadowMapData[i] != null)
                {
                    try
                    {
                        shadowBits.Add(new VlmChunkShadowBits(i, Convert.ToBase64String(shadowMapData[i])));
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[DEBUG] Chunk {i} Shadow Error: {ex.Message}");
                    }
                }
            }
            
            VlmChunkShadowAnalysis[]? shadowAnalysis = shadowBits.Count > 0
                ? VlmShadowAssociationService.AnalyzeTile(shadowBits, chunkPositions, objectPlacements)
                : null;

            var heightmapPath = await GenerateHeightmap(heights, tileName, outputDir, isInterleaved: true);
            var normalmapPath = await GenerateNormalmap(heights, holes, tileName, outputDir, isInterleaved: true);
            var mccvMapPath = await GenerateMccvMap(chunkLayers, tileName, outputDir);
            
            return new VlmTerrainData(
                AdtTile: tileName,
                Heights: heights.ToArray(),
                ChunkPositions: chunkPositions,
                Holes: holes,
                HeightmapPath: heightmapPath,
                HeightmapLocalPath: heightmapPath,
                HeightmapGlobalPath: null,
                NormalmapPath: normalmapPath,
                MccvMapPath: mccvMapPath,
                ShadowMaps: shadowPaths.Count > 0 ? shadowPaths.ToArray() : null,
                ShadowBits: shadowBits.Count > 0 ? shadowBits.ToArray() : null,
                ShadowAnalysis: shadowAnalysis,
                AlphaMasks: null,
                AlphaAtlasPath: null,
                LiquidMaskPath: null,
                LiquidHeightPath: null,
                LiquidMinHeight: 0f,
                LiquidMaxHeight: 0f,
                NoLiquidMinimapPath: null,
                NoMccvMinimapPath: null,
                ObjectVisibilityMaskPath: null,
                Pm4MaskPath: null,
                NoObjectMinimapPath: null,
                TerrainOnlyMinimapPath: null,
                HolesMaskPath: null,
                AreaIdMapPath: null,
                ChunkFlagsMapPath: null,
                LiquidTypeMapPath: null,
                DominantEffectIdMapPath: null,
                Textures: textures,
                ChunkLayers: chunkLayers.ToArray(),
                Liquids: liquids.Count > 0 ? liquids.ToArray() : null,
                Objects: objectPlacements,
                WdlHeights: wdlHeights,
                HeightMin: heightMin == float.MaxValue ? 0 : heightMin,
                HeightMax: heightMax == float.MinValue ? 0 : heightMax,
                HeightGlobalMin: heightMin == float.MaxValue ? 0 : heightMin,
                HeightGlobalMax: heightMax == float.MinValue ? 0 : heightMax,
                IsInterleaved: true // LK/Modern IS interleaved
            );

        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LK ADT] Error parsing {tileName}: {ex.Message}");
            return null;
        }
    }
    

    
    /// <summary>
    /// Parse null-terminated strings from a byte array block
    /// </summary>
    private static List<string> ParseNullStrings(byte[] data, int offset, int size)
    {
        var list = new List<string>();
        int sStart = offset;
        int end = offset + size;
        while (sStart < end)
        {
            int nullPos = Array.IndexOf(data, (byte)0, sStart, end - sStart);
            if (nullPos == -1) nullPos = end;
            int len = nullPos - sStart;
            if (len > 0)
            {
                string str = System.Text.Encoding.UTF8.GetString(data, sStart, len);
                if (!string.IsNullOrWhiteSpace(str))
                    list.Add(str);
            }
            sStart = nullPos + 1;
        }
        return list;
    }

    private static void CollectTopLevelChunkData(
        byte[] source,
        List<string>? textures,
        List<string> m2Names,
        List<string> wmoNames,
        ref byte[]? mh2oData,
        ref byte[]? mddfRaw,
        ref byte[]? modfRaw)
    {
        for (int i = 0; i + 8 <= source.Length;)
        {
            string fcc = System.Text.Encoding.ASCII.GetString(source, i, 4);
            int sz = BitConverter.ToInt32(source, i + 4);
            if (sz < 0)
                break;

            int dataStart = i + 8;
            int next = dataStart + sz + ((sz & 1) == 1 ? 1 : 0);
            if (dataStart + sz > source.Length)
                break;

            if (textures != null && (fcc == "XETM" || fcc == "MTEX"))
            {
                textures.AddRange(ParseNullStrings(source, dataStart, sz));
            }
            else if (fcc == "XDMM" || fcc == "MMDX")
            {
                m2Names.AddRange(ParseNullStrings(source, dataStart, sz));
            }
            else if (fcc == "OMWM" || fcc == "MWMO")
            {
                wmoNames.AddRange(ParseNullStrings(source, dataStart, sz));
            }
            else if (mh2oData == null && (fcc == "O2HM" || fcc == "MH2O"))
            {
                mh2oData = new byte[sz];
                Array.Copy(source, dataStart, mh2oData, 0, sz);
            }
            else if (fcc == ReverseFourCc("MDDF") || fcc == "MDDF")
            {
                mddfRaw = new byte[sz];
                Array.Copy(source, dataStart, mddfRaw, 0, sz);
            }
            else if (fcc == ReverseFourCc("MODF") || fcc == "MODF")
            {
                modfRaw = new byte[sz];
                Array.Copy(source, dataStart, modfRaw, 0, sz);
            }

            if (next <= i)
                break;

            i = next;
        }
    }

    private void AppendObjectPlacementsFromRaw(
        List<VlmObjectPlacement> objectPlacements,
        byte[]? raw,
        IReadOnlyList<string> names,
        string category,
        IArchiveReader archiveReader,
        IReadOnlyList<string> searchPaths)
    {
        if (raw == null || raw.Length == 0)
            return;

        int entrySize = category == "wmo" ? 64 : 36;
        int scaleOffset = category == "wmo" ? 60 : 32;

        for (int i = 0; i + entrySize <= raw.Length; i += entrySize)
        {
            uint nameId = BitConverter.ToUInt32(raw, i);
            uint uniqueId = BitConverter.ToUInt32(raw, i + 4);
            float px = BitConverter.ToSingle(raw, i + 8);
            float py = BitConverter.ToSingle(raw, i + 12);
            float pz = BitConverter.ToSingle(raw, i + 16);
            float rx = BitConverter.ToSingle(raw, i + 20);
            float ry = BitConverter.ToSingle(raw, i + 24);
            float rz = BitConverter.ToSingle(raw, i + 28);
            ushort scale = BitConverter.ToUInt16(raw, i + scaleOffset);

            string fullPath = nameId < names.Count ? names[(int)nameId] : string.Empty;
            string name = string.IsNullOrWhiteSpace(fullPath)
                ? $"{category}_{nameId}"
                : Path.GetFileNameWithoutExtension(fullPath);

            float[]? boundsMin = null;
            float[]? boundsMax = null;
            if (!string.IsNullOrWhiteSpace(fullPath))
            {
                var bounds = GetModelBounds(fullPath, archiveReader, searchPaths);
                if (bounds.HasValue)
                {
                    boundsMin = bounds.Value.Min;
                    boundsMax = bounds.Value.Max;
                }
            }

            objectPlacements.Add(new VlmObjectPlacement(
                name,
                nameId,
                uniqueId,
                px,
                py,
                pz,
                rx,
                ry,
                rz,
                scale / 1024f,
                category,
                boundsMin,
                boundsMax,
                fullPath));
        }
    }

    /// <summary>
    /// Find a chunk in LK ADT format (reversed FourCC).
    /// </summary>
    private static int FindLkChunk(byte[] bytes, string fourCC)
    {
        // LK uses reversed FourCC on disk
        string reversed = new string(fourCC.Reverse().ToArray());
        
        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = System.Text.Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            
            if (fcc == reversed)
                return i;

            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= i) break;
            i = next;
        }

        return -1;
    }

    /// <summary>
    /// Read the MAIN chunk from an LK WDT and enumerate existing tiles.
    /// LK MAIN chunk: 64x64 grid (4096 entries), 8 bytes each.
    /// Bytes 0-3: flags (0 = no tile, non-zero = tile exists)
    /// Bytes 4-7: async_id (unused for enumeration)
    /// </summary>
    private static List<int> ReadLkWdtTiles(byte[] wdtBytes)
    {
        var tiles = new List<int>();
        
        int mainOffset = FindLkChunk(wdtBytes, "MAIN");
        if (mainOffset < 0)
            throw new InvalidDataException("MAIN chunk not found in LK WDT");
        
        int mainSize = BitConverter.ToInt32(wdtBytes, mainOffset + 4);
        int mainDataStart = mainOffset + 8;
        
        // MAIN should be 64x64 * 8 bytes = 32768 bytes
        if (mainSize < 64 * 64 * 8)
            throw new InvalidDataException($"MAIN chunk too small: {mainSize} bytes (expected {64 * 64 * 8})");
        
        // Read 4096 tile entries
        for (int i = 0; i < 64 * 64; i++)
        {
            int entryOffset = mainDataStart + (i * 8);
            if (entryOffset + 8 > wdtBytes.Length)
                break;
            
            uint flags = BitConverter.ToUInt32(wdtBytes, entryOffset);
            
            // If flags != 0, tile exists
            if (flags != 0)
            {
                tiles.Add(i);
            }
        }
        
        return tiles;
    }

    private static List<int> SelectTilesForProcessing(
        List<int> existingTiles,
        int limit,
        bool isAlphaFormat,
        string mapDirectory,
        IArchiveReader archiveReader,
        IReadOnlyList<string> searchPaths,
        bool interestingOnly,
        int interestingMinScore)
    {
        if (existingTiles.Count == 0)
            return existingTiles;

        var fallbackOrder = OrderTilesByCenter(existingTiles);
        var fallbackIndex = fallbackOrder
            .Select((tileIndex, index) => (tileIndex, index))
            .ToDictionary(entry => entry.tileIndex, entry => entry.index);

        bool shouldRankTiles = interestingOnly || (!isAlphaFormat && limit > 0 && limit < existingTiles.Count) || isAlphaFormat;
        if (!shouldRankTiles)
        {
            if (limit <= 0 || limit >= existingTiles.Count)
                return existingTiles.ToList();

            return fallbackOrder.Take(limit).ToList();
        }

        var rankedTiles = existingTiles
            .Select(tileIndex =>
            {
                int x = tileIndex % 64;
                int y = tileIndex / 64;
                int contentScore = ScoreTileContent(mapDirectory, x, y, isAlphaFormat, archiveReader, searchPaths);
                return (tileIndex, contentScore);
            })
            .ToList();

        if (interestingOnly)
        {
            var interestingTiles = rankedTiles
                .Where(entry => entry.contentScore >= interestingMinScore)
                .OrderByDescending(entry => entry.contentScore)
                .ThenBy(entry => fallbackIndex[entry.tileIndex]);

            var selectedInterestingTiles = limit > 0 && limit < existingTiles.Count
                ? interestingTiles.Take(limit).Select(entry => entry.tileIndex).ToList()
                : interestingTiles.Select(entry => entry.tileIndex).ToList();

            if (selectedInterestingTiles.Count > 0)
                return selectedInterestingTiles;

            return fallbackOrder.Take(1).ToList();
        }

        if (limit <= 0 || limit >= existingTiles.Count)
            return existingTiles.ToList();

        if (rankedTiles.Any(entry => entry.contentScore > 0))
        {
            return rankedTiles
                .OrderByDescending(entry => entry.contentScore)
                .ThenBy(entry => fallbackIndex[entry.tileIndex])
                .Take(limit)
                .Select(entry => entry.tileIndex)
                .ToList();
        }

        return fallbackOrder.Take(limit).ToList();
    }

    private static int ScoreTileContent(string mapDirectory, int x, int y, bool isAlphaFormat, IArchiveReader archiveReader, IReadOnlyList<string> searchPaths)
    {
        string adtBase = $"World/Maps/{mapDirectory}/{mapDirectory}_{x}_{y}";

        int score = 0;

        try
        {
            byte[]? rootAdt = ReadTileAssetBytesForScoring(searchPaths, $"{adtBase}.adt", archiveReader, isAlphaFormat);
            if (rootAdt is { Length: > 0 })
            {
                if (ContainsTopLevelChunk(rootAdt, "MH2O"))
                    score += 100;

                if (ContainsChunkToken(rootAdt, "MCLQ"))
                    score += 80;

                if (ContainsChunkToken(rootAdt, "MCAL"))
                    score += 8;

                if (ContainsChunkToken(rootAdt, "MCLY"))
                    score += 6;

                if (ContainsChunkToken(rootAdt, "MCNK"))
                    score += 4;

                if (ContainsTopLevelChunk(rootAdt, "MDDF") || ContainsTopLevelChunk(rootAdt, "MODF"))
                    score += 10;
            }
        }
        catch
        {
        }

        try
        {
            byte[]? texAdt = ReadTileAssetBytesForScoring(searchPaths, $"{adtBase}_tex0.adt", archiveReader, isAlphaFormat);
            if (texAdt is { Length: > 0 })
            {
                if (ContainsChunkToken(texAdt, "MCAL"))
                    score += 8;

                if (ContainsChunkToken(texAdt, "MCLY"))
                    score += 6;
            }
        }
        catch
        {
        }

        try
        {
            byte[]? objAdt = ReadTileAssetBytesForScoring(searchPaths, $"{adtBase}_obj0.adt", archiveReader, isAlphaFormat);
            if (objAdt is { Length: > 0 } && (ContainsTopLevelChunk(objAdt, "MDDF") || ContainsTopLevelChunk(objAdt, "MODF")))
                score += 10;
        }
        catch
        {
        }

        return score;
    }

    internal static int ScoreTileContentForTesting(string mapDirectory, int x, int y, bool isAlphaFormat, IArchiveReader archiveReader, IReadOnlyList<string> searchPaths)
        => ScoreTileContent(mapDirectory, x, y, isAlphaFormat, archiveReader, searchPaths);

    internal static List<int> SelectTilesForProcessingForTesting(
        List<int> existingTiles,
        int limit,
        bool isAlphaFormat,
        string mapDirectory,
        IArchiveReader archiveReader,
        IReadOnlyList<string> searchPaths,
        bool interestingOnly,
        int interestingMinScore)
        => SelectTilesForProcessing(existingTiles, limit, isAlphaFormat, mapDirectory, archiveReader, searchPaths, interestingOnly, interestingMinScore);

    private static byte[]? ReadTileAssetBytesForScoring(IReadOnlyList<string> searchPaths, string virtualPath, IArchiveReader archiveReader, bool isAlphaFormat)
    {
        byte[]? bytes = ReadVirtualAssetBytes(searchPaths, virtualPath, archiveReader);
        if (bytes is { Length: > 0 } || !isAlphaFormat)
            return bytes;

        string normalizedPath = NormalizeVirtualAssetPath(virtualPath);
        foreach (string basePath in searchPaths)
        {
            string diskCandidate = Path.Combine(basePath, normalizedPath);
            bytes = AlphaArchiveReader.ReadWithMpqFallback(diskCandidate);
            if (bytes is { Length: > 0 })
                return bytes;
        }

        return null;
    }

    private static bool ContainsTopLevelChunk(byte[] fileBytes, string fourCc)
    {
        string reversedFourCc = ReverseFourCc(fourCc);

        for (int offset = 0; offset + 8 <= fileBytes.Length;)
        {
            string chunkId = System.Text.Encoding.ASCII.GetString(fileBytes, offset, 4);
            int size = BitConverter.ToInt32(fileBytes, offset + 4);
            if (size < 0)
                break;

            if (chunkId == fourCc || chunkId == reversedFourCc)
                return true;

            int next = offset + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (offset + 8 + size > fileBytes.Length || next <= offset)
                break;

            offset = next;
        }

        return false;
    }

    private static bool ContainsChunkToken(byte[] fileBytes, string fourCc)
    {
        if (fileBytes.Length < 4)
            return false;

        byte[] token = System.Text.Encoding.ASCII.GetBytes(fourCc);
        for (int index = 0; index <= fileBytes.Length - token.Length; index++)
        {
            bool match = true;
            for (int tokenIndex = 0; tokenIndex < token.Length; tokenIndex++)
            {
                if (fileBytes[index + tokenIndex] != token[tokenIndex])
                {
                    match = false;
                    break;
                }
            }

            if (match)
                return true;
        }

        return false;
    }

    private static string ReverseFourCc(string fourCc)
    {
        char[] chars = fourCc.ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }

    internal static IEnumerable<VlmLiquidData> MergeLegacyLiquids(
        IReadOnlyCollection<VlmLiquidData> mh2oLiquids,
        IReadOnlyCollection<VlmLiquidData> legacyMclqLiquids)
    {
        if (legacyMclqLiquids.Count == 0)
            return Array.Empty<VlmLiquidData>();

        HashSet<int> mh2oChunkIndices = mh2oLiquids.Count > 0
            ? mh2oLiquids.Select(liquid => liquid.ChunkIndex).ToHashSet()
            : [];

        if (mh2oChunkIndices.Count == 0)
            return legacyMclqLiquids;

        return legacyMclqLiquids.Where(liquid => !mh2oChunkIndices.Contains(liquid.ChunkIndex)).ToArray();
    }

    private static List<int> OrderTilesByCenter(List<int> existingTiles)
    {
        if (existingTiles.Count == 0)
            return existingTiles;

        int minX = 63;
        int maxX = 0;
        int minY = 63;
        int maxY = 0;

        foreach (int tileIndex in existingTiles)
        {
            int x = tileIndex % 64;
            int y = tileIndex / 64;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        double centerX = (minX + maxX) / 2.0;
        double centerY = (minY + maxY) / 2.0;

        return existingTiles
            .OrderBy(tileIndex =>
            {
                int x = tileIndex % 64;
                int y = tileIndex / 64;
                double dx = x - centerX;
                double dy = y - centerY;
                return (dx * dx) + (dy * dy);
            })
            .ThenBy(tileIndex => tileIndex / 64)
            .ThenBy(tileIndex => tileIndex % 64)
            .ToList();
    }

    internal static bool TryParseTileFilter(string tileFilter, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;

        if (string.IsNullOrWhiteSpace(tileFilter))
            return false;

        string[] parts = tileFilter.Split(['_', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length != 2)
            return false;

        return int.TryParse(parts[0], out tileX)
            && int.TryParse(parts[1], out tileY)
            && tileX >= 0 && tileX < 64
            && tileY >= 0 && tileY < 64;
    }

    private string? FindMinimapTile(IEnumerable<string> searchPaths, IArchiveReader archiveReader, SharedMd5TranslateIndex? index, string mapName, int x, int y)
    {
        // Generate all possible plain-name candidates for this tile
        // TRS format per wowdev.wiki/TRS.md: map_%d_%02d.blp (x not padded, y 2-digit padded)
        var candidates = new List<string>();
        
        var x2 = x.ToString("D2");  // Zero-padded (legacy)
        var y2 = y.ToString("D2");  // Zero-padded (legacy)
        
        // TRS format: x not padded, y 2-digit padded (map_26_09.blp for x=26, y=9)
        var trsFormat = $"map{x}_{y2}.blp";
        
        // 1. TRS format candidates (highest priority - matches actual TRS file format)
        candidates.Add($"{mapName}\\{trsFormat}");  // Exact TRS format with backslash
        candidates.Add($"{mapName}/{trsFormat}");   // Forward slash variant
        candidates.Add($"textures/minimap/{mapName}/{trsFormat}");
        
        // 2. Legacy formats (both coords padded)
        candidates.Add($"textures/minimap/{mapName}/{mapName}_{x2}_{y2}.blp");
        candidates.Add($"textures/minimap/{mapName}/map{x2}_{y2}.blp");
        candidates.Add($"{mapName}/map{x2}_{y2}.blp");
        
        // 3. Space variants (0.6.0 bug)
        var mapNameSpace = InsertSpaceBeforeCapitals(mapName);
        if (mapNameSpace != mapName)
        {
            candidates.Add($"{mapNameSpace}\\{trsFormat}");
            candidates.Add($"textures/minimap/{mapNameSpace}/{trsFormat}");
            candidates.Add($"textures/minimap/{mapNameSpace}/{mapNameSpace}_{x2}_{y2}.blp");
            candidates.Add($"textures/minimap/{mapNameSpace}/map{x2}_{y2}.blp");
            candidates.Add($"{mapNameSpace}/map{x2}_{y2}.blp");
        }

        // 4. Other Legacy/Release variants
        candidates.Add($"World/Minimaps/{mapName}/map{x2}_{y2}.blp");
        candidates.Add($"World/Minimaps/{mapName}/map{x}_{y}.blp");
        candidates.Add($"Textures/Minimap/{mapName}_{x2}_{y2}.blp");
        candidates.Add($"Textures/Minimap/{mapName}_{x}_{y}.blp");

        // 5. Pre-converted PNG variants (minimaps already converted from BLP)
        candidates.Add($"World/Textures/Minimap/{mapName}_{x}_{y}.png");
        candidates.Add($"World/Textures/Minimap/{mapName}_{x2}_{y2}.png");
        candidates.Add($"Textures/Minimap/{mapName}_{x}_{y}.png");
        candidates.Add($"Textures/Minimap/{mapName}_{x2}_{y2}.png");

        // PRIORITY 1: Check MD5 Index for ANY candidate
        bool debugTile = (x == 18 && y == 10) || (x == 44 && y == 26);  // Only debug specific tiles
        if (debugTile)
        {
            Console.WriteLine($"\n[DEBUG] FindMinimapTile for {mapName} {x}_{y}");
            Console.WriteLine($"[DEBUG] Generated {candidates.Count} candidates:");
            foreach (var c in candidates)
                Console.WriteLine($"  - {c}");
        }
        
        if (index != null)
        {
            if (debugTile)
            {
                Console.WriteLine($"\n[DEBUG] Checking md5Index (Total entries: {index.PlainToHash.Count})");
                
                // Show sample of what's actually stored in the index FOR THIS MAP
                Console.WriteLine($"[DEBUG] Sample PlainToHash entries containing '{mapName}':");
                int sampleCount = 0;
                foreach (var kvp in index.PlainToHash)
                {
                    if (sampleCount >= 10) break;
                    if (kvp.Key.Contains(mapName, StringComparison.OrdinalIgnoreCase))
                    {
                        Console.WriteLine($"  KEY: '{kvp.Key}' => VAL: '{kvp.Value}'");
                        sampleCount++;
                    }
                }
                if (sampleCount == 0)
                {
                    Console.WriteLine($"  [WARNING] No entries found containing '{mapName}'!");
                    // Show a few random entries to understand format
                    Console.WriteLine($"[DEBUG] Sample PlainToHash entries (first 5):");
                    foreach (var kvp in index.PlainToHash.Take(5))
                    {
                        Console.WriteLine($"  KEY: '{kvp.Key}' => VAL: '{kvp.Value}'");
                    }
                }
            }
            
            foreach (var candidate in candidates)
            {
                // Normalize for lookup (legacy uses internal normalization, but our dict is case-insensitive too)
                var lookupKey = candidate.Replace('\\', '/').TrimStart('/');
                var normalizedKey = lookupKey.ToLowerInvariant();
                
                if (debugTile)
                {
                    Console.WriteLine($"\n[DEBUG] Trying candidate: '{candidate}'");
                    Console.WriteLine($"  Lookup key: '{lookupKey}'");
                    Console.WriteLine($"  Normalized: '{normalizedKey}'");
                }
                
                bool found = index.PlainToHash.TryGetValue(lookupKey, out var hashed);
                if (!found)
                {
                    // Try lowercase as fallback (md5translate often uses lowercase)
                    found = index.PlainToHash.TryGetValue(normalizedKey, out hashed);
                    if (debugTile && found)
                    {
                        Console.WriteLine($"  Found via lowercase: YES -> '{hashed}'");
                    }
                }
                else if (debugTile)
                {
                    Console.WriteLine($"  Found directly: YES -> '{hashed}'");
                }

                if (found && !string.IsNullOrWhiteSpace(hashed))
                {
                    // Found a mapping!
                    // The 'hashed' value is the filename in the MPQ.
                    if (TryResolveLooseAssetPath(searchPaths, hashed) is { } looseMappedPath)
                    {
                        if (debugTile)
                            Console.WriteLine($"  Found loose override on disk: {looseMappedPath}");

                        return looseMappedPath;
                    }

                    var mpqKey = NormalizeVirtualAssetPath(hashed);
                    
                    if (debugTile)
                    {
                        Console.WriteLine($"  MPQ key to check: '{mpqKey}'");
                    }
                    
                    if (archiveReader.FileExists(mpqKey))
                    {
                        Console.WriteLine($"[Match] Translated '{candidate}' -> '{hashed}' (Found in MPQ)");
                        return $"MPQ:{mpqKey}";
                    }
                    
                    if (debugTile)
                    {
                        Console.WriteLine($"  File '{mpqKey}' not found in MPQ or disk!");
                    }
                    Console.WriteLine($"[Mapping Found] '{candidate}' -> '{hashed}' but file missing.");
                }
                else
                {
                    if (debugTile)
                    {
                        Console.WriteLine($"  Found: NO");
                    }
                }
            }
        }

        // PRIORITY 2: Check standard candidates on Disk/MPQ (Loose or Plain)
        foreach (var candidate in candidates)
        {
             if (TryResolveLooseAssetPath(searchPaths, candidate) is { } looseCandidatePath)
             {
                 Console.WriteLine($"Found minimap on disk: {looseCandidatePath}");
                 return looseCandidatePath;
             }
             
             // Check MPQ by plain name (fallback)
             var mpqPlainKey = NormalizeVirtualAssetPath(candidate);
             if (archiveReader.FileExists(mpqPlainKey))
             {
                 Console.WriteLine($"Found minimap in Archive (Plain): {mpqPlainKey}");
                 return $"MPQ:{mpqPlainKey}";
             }
        }

        Console.WriteLine($"Minimap not found for {mapName} {x}_{y}");
        return null;
    }

    private static string InsertSpaceBeforeCapitals(string input)
    {
        if (string.IsNullOrEmpty(input) || input.Length < 2) return input;
        var sb = new System.Text.StringBuilder();
        sb.Append(input[0]);
        for (int i = 1; i < input.Length; i++)
        {
            if (char.IsUpper(input[i]) && !char.IsUpper(input[i - 1])) sb.Append(' ');
            sb.Append(input[i]);
        }
        return sb.ToString();
    }

    private async Task<string?> GenerateHeightmap(List<VlmChunkHeights> chunkHeights, string tileName, string outputDir, bool isInterleaved)
    {
        if (chunkHeights == null || chunkHeights.Count == 0) return null;

        var heightsDict = chunkHeights.ToDictionary(k => k.ChunkIndex, v => v.Heights);
        var tileHeightmap = TerrainTileBakeService.BuildTileHeightmap257(heightsDict, isInterleaved);

        var filename = $"{tileName}_heightmap.png";
        var imagesDir = Path.Combine(outputDir, "images");
        Directory.CreateDirectory(imagesDir);
        var path = Path.Combine(imagesDir, filename);
        using (Image<L16> image = TerrainTileBakeService.CreateHeightmapImage(tileHeightmap.Heights, tileHeightmap.MinHeight, tileHeightmap.MaxHeight, 256))
        {
            await image.SaveAsPngAsync(path);
        }

        return $"images/{filename}";
    }

    private (float min, float max) GetHeightRange(Dictionary<int, float[]> heightsDict)
    {
        float minZ = float.MaxValue;
        float maxZ = float.MinValue;
        foreach (var kvp in heightsDict)
        {
            if (kvp.Value == null) continue;
            foreach (var h in kvp.Value)
            {
                if (float.IsNaN(h) || float.IsInfinity(h)) continue;
                if (Math.Abs(h) > 50000f) continue; // Ignore outliers
                if (h < minZ) minZ = h;
                if (h > maxZ) maxZ = h;
            }
        }

        if (minZ >= maxZ)
        {
            minZ = 0;
            maxZ = 1;
        }

        return (minZ, maxZ);
    }

    private byte[] RenderHeightmapImage(Dictionary<int, float[]> heightsDict, float minZ, float maxZ, int size, bool isInterleaved)
    {
        float range = maxZ - minZ;
        if (range < 0.001f) range = 1.0f;

        using var rawMap = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.L16>(size, size);

        // Helper for Barycentric Interpolation on ADT grid (4 triangles per square)
        float SampleHeight(float[] hData, float lx, float ly)
        {
             // lx, ly in [0, 1] within chunk
             float gx = lx * 8;
             float gy = ly * 8;
             
             int ix = Math.Clamp((int)gx, 0, 7);
             int iy = Math.Clamp((int)gy, 0, 7);
             
             float dx = gx - ix;
             float dy = gy - iy;
             
             // Vertices - Standard Interleaved Format
             // Row 0: 9 Outer + 8 Inner = 17 floats
             // Outer(row, col) = row * 17 + col
             // Inner(row, col) = row * 17 + 9 + col

             // Outer Grid (9x9)
             float GetOuter(int r, int c) => isInterleaved ? hData[r * 17 + c] : hData[r * 9 + c];
             
             // Inner Grid (8x8)
             float GetInner(int r, int c) => isInterleaved ? hData[r * 17 + 9 + c] : hData[81 + r * 8 + c];

             float vTL = GetOuter(iy, ix);
             float vTR = GetOuter(iy, ix + 1);
             float vBL = GetOuter(iy + 1, ix);
             float vBR = GetOuter(iy + 1, ix + 1);
             
             float vC = GetInner(iy, ix);
             
             // Determine triangle quadrant
             if (dy < dx && dy < 1.0f - dx) // Top (North) -> TL, TR, C
             {
                 return vTL * (1 - dx - dy) + vTR * (dx - dy) + vC * (2 * dy);
             }
             else if (dy > dx && dy > 1.0f - dx) // Bottom (South) -> BL, BR, C
             {
                 return vBL * (dy - dx) + vBR * (dx + dy - 1) + vC * 2 * (1 - dy);
             }
             else if (dx < dy && dx < 1.0f - dy) // Left (West) -> TL, BL, C
             {
                 return vTL * (1 - dx - dy) + vBL * (dy - dx) + vC * (2 * dx);
             }
             else // Right (East) -> TR, BR, C
             {
                 return vTR * (dx - dy) + vBR * (dy + dx - 1) + vC * 2 * (1 - dx);
             }
        }

        for (int y = 0; y < size; y++)
        {
            float v = y / (float)(size - 1);
            float cy = v * 16;
            int cIy = Math.Clamp((int)cy, 0, 15);

            for (int x = 0; x < size; x++)
            {
                float u = x / (float)(size - 1);
                float cx = u * 16;
                int cIx = Math.Clamp((int)cx, 0, 15);

                int chunkIndex = cIy * 16 + cIx;

                if (!heightsDict.TryGetValue(chunkIndex, out var hData) || hData == null || hData.Length < 145)
                {
                    rawMap[x, y] = new SixLabors.ImageSharp.PixelFormats.L16(0);
                    continue;
                }

                float lx = Math.Clamp(cx - cIx, 0f, 1f);
                float ly = Math.Clamp(cy - cIy, 0f, 1f);
                float z = SampleHeight(hData, lx, ly);
                float norm = Math.Clamp((z - minZ) / range, 0f, 1f);
                rawMap[x, y] = new SixLabors.ImageSharp.PixelFormats.L16((ushort)(norm * 65535));
            }
        }

        using var ms = new MemoryStream();
        rawMap.SaveAsPng(ms);
        return ms.ToArray();
    }

    private async Task<string?> GenerateNormalmap(List<VlmChunkHeights> chunkHeights, int[] holes, string tileName, string outputDir, bool isInterleaved)
    {
        if (chunkHeights == null || chunkHeights.Count == 0) return null;

        var heightsDict = chunkHeights.ToDictionary(k => k.ChunkIndex, v => v.Heights);
        var holeMasks = new Dictionary<int, int>(holes?.Length ?? 0);
        if (holes != null)
        {
            for (int i = 0; i < holes.Length; i++)
                holeMasks[i] = holes[i];
        }

        var tileHeightmap = TerrainTileBakeService.BuildTileHeightmap257(heightsDict, isInterleaved);
        var tileNormals = TerrainTileBakeService.BuildTileNormals257(tileHeightmap.Heights, holeMasks);

        var filename = $"{tileName}_normal.png";
        var imagesDir = Path.Combine(outputDir, "images");
        Directory.CreateDirectory(imagesDir);
        var path = Path.Combine(imagesDir, filename);
        using (Image<Rgba32> image = TerrainTileBakeService.CreateNormalmapImage(tileNormals, 256))
        {
            await image.SaveAsPngAsync(path);
        }

        return $"images/{filename}";
    }

    private byte[] RenderNormalmapImage(Dictionary<int, sbyte[]> normalsDict, int size)
    {
        using var image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(size, size);

        // Helper for Barycentric Interpolation of Normals
        // Returns Vector3 (x,y,z) normalized
             (float x, float y, float z) SampleNormal(sbyte[] nData, float lx, float ly)
        {
             float gx = lx * 8;
             float gy = ly * 8;
             
             int ix = Math.Clamp((int)gx, 0, 7);
             int iy = Math.Clamp((int)gy, 0, 7);
             
             float dx = Math.Clamp(gx - ix, 0f, 1f);
             float dy = Math.Clamp(gy - iy, 0f, 1f);
             
             // Helper to unpack normal at index
             // MCNR format: sbyte X, Y, Z. (145 * 3 bytes)
             (float nx, float ny, float nz) GetN(int index)
             {
                 int baseIdx = index * 3;
                 return (nData[baseIdx] / 127.0f, nData[baseIdx + 1] / 127.0f, nData[baseIdx + 2] / 127.0f);
             }

             // Vertices
             var vTL = GetN(iy * 9 + ix);
             var vTR = GetN(iy * 9 + ix + 1);
             var vBL = GetN((iy + 1) * 9 + ix);
             var vBR = GetN((iy + 1) * 9 + ix + 1);
             var vC = GetN(81 + iy * 8 + ix); // Inner center
             
             (float x, float y, float z) nRes;

             if (dy < dx && dy < 1.0f - dx) // Top (North)
             {
                 float wTL = 1 - dx - dy; float wTR = dx - dy; float wC = 2 * dy;
                 nRes = (
                    vTL.nx * wTL + vTR.nx * wTR + vC.nx * wC,
                    vTL.ny * wTL + vTR.ny * wTR + vC.ny * wC,
                    vTL.nz * wTL + vTR.nz * wTR + vC.nz * wC
                 );
             }
             else if (dy > dx && dy > 1.0f - dx) // Bottom (South)
             {
                 float wBL = dy - dx; float wBR = dx + dy - 1; float wC = 2 * (1 - dy);
                 nRes = (
                    vBL.nx * wBL + vBR.nx * wBR + vC.nx * wC,
                    vBL.ny * wBL + vBR.ny * wBR + vC.ny * wC,
                    vBL.nz * wBL + vBR.nz * wBR + vC.nz * wC
                 );
             }
             else if (dx < dy && dx < 1.0f - dy) // Left (West)
             {
                 float wTL = 1 - dx - dy; float wBL = dy - dx; float wC = 2 * dx;
                 nRes = (
                    vTL.nx * wTL + vBL.nx * wBL + vC.nx * wC,
                    vTL.ny * wTL + vBL.ny * wBL + vC.ny * wC,
                    vTL.nz * wTL + vBL.nz * wBL + vC.nz * wC
                 );
             }
             else // Right (East)
             {
                 float wTR = dx - dy; float wBR = dy + dx - 1; float wC = 2 * (1 - dx);
                 nRes = (
                    vTR.nx * wTR + vBR.nx * wBR + vC.nx * wC,
                    vTR.ny * wTR + vBR.ny * wBR + vC.ny * wC,
                    vTR.nz * wTR + vBR.nz * wBR + vC.nz * wC
                 );
             }
             
             // Normalize result
             float mag = (float)Math.Sqrt(nRes.x * nRes.x + nRes.y * nRes.y + nRes.z * nRes.z);
             if (mag > 1e-6f)
                return (nRes.x / mag, nRes.y / mag, nRes.z / mag);
             return (0, 1, 0); // Default up
        }

        for (int y = 0; y < size; y++)
        {
            float v = y / (float)(size - 1);
            float cy = v * 16;
            int cIy = Math.Clamp((int)cy, 0, 15);

            for (int x = 0; x < size; x++)
            {
                float u = x / (float)(size - 1);
                float cx = u * 16;
                int cIx = Math.Clamp((int)cx, 0, 15);

                int chunkIndex = cIy * 16 + cIx;

                if (!normalsDict.TryGetValue(chunkIndex, out var nData))
                {
                    // Default normal 128,128,255
                    image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(128, 128, 255);
                    continue;
                }

                float lx = Math.Clamp(cx - cIx, 0f, 1f);
                float ly = Math.Clamp(cy - cIy, 0f, 1f);
                var (nx, ny, nz) = SampleNormal(nData, lx, ly);
                
                // Pack to RGB [0, 255]
                // [-1, 1] -> [0, 1] -> [0, 255]
                byte r = (byte)((nx * 0.5f + 0.5f) * 255);
                byte g = (byte)((ny * 0.5f + 0.5f) * 255);
                byte b = (byte)((nz * 0.5f + 0.5f) * 255);
                
                image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(r, g, b);
            }
        }

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();


    }

    private async Task<string?> GenerateMccvMap(List<VlmChunkLayers> chunkLayers, string tileName, string outputDir)
    {
        if (chunkLayers == null || chunkLayers.Count == 0) return null;

        const int Size = 145;
        var mccvDict = chunkLayers
            .Where(c => c.MccvColors != null && c.MccvColors.Length >= 145 * 4)
            .ToDictionary(k => k.ChunkIndex, v => v.MccvColors!);

        if (mccvDict.Count == 0) return null;

        var mapBytes = RenderMccvImage(mccvDict, Size);

        var filename = $"{tileName}_mccv.png";
        var imagesDir = Path.Combine(outputDir, "images");
        Directory.CreateDirectory(imagesDir);
        var path = Path.Combine(imagesDir, filename);
        await File.WriteAllBytesAsync(path, mapBytes);

        return $"images/{filename}";
    }

    internal static byte[] RenderMccvImage(Dictionary<int, byte[]> mccvDict, int size)
    {
        using var image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(size, size);

        (float r, float g, float b, float a) SampleColor(byte[] cData, float lx, float ly)
        {
             float gx = lx * 8;
             float gy = ly * 8;
             
             int ix = Math.Clamp((int)gx, 0, 7);
             int iy = Math.Clamp((int)gy, 0, 7);
             
             float dx = Math.Clamp(gx - ix, 0f, 1f);
             float dy = Math.Clamp(gy - iy, 0f, 1f);
             
             (float r, float g, float b, float a) GetC(int index)
             {
                 int baseIdx = index * 4;
                 return (
                     cData[baseIdx + 0] / 255.0f,
                     cData[baseIdx + 1] / 255.0f,
                     cData[baseIdx + 2] / 255.0f,
                     cData[baseIdx + 3] / 255.0f);
             }

             var vTL = GetC(iy * 9 + ix);
             var vTR = GetC(iy * 9 + ix + 1);
             var vBL = GetC((iy + 1) * 9 + ix);
             var vBR = GetC((iy + 1) * 9 + ix + 1);
             var vC = GetC(81 + iy * 8 + ix); 
             
             (float r, float g, float b, float a) res;

             if (dy < dx && dy < 1.0f - dx) // Top 
             {
                 float wTL = 1 - dx - dy; float wTR = dx - dy; float wC = 2 * dy;
                 res = (vTL.r * wTL + vTR.r * wTR + vC.r * wC, vTL.g * wTL + vTR.g * wTR + vC.g * wC, vTL.b * wTL + vTR.b * wTR + vC.b * wC, vTL.a * wTL + vTR.a * wTR + vC.a * wC);
             }
             else if (dy > dx && dy > 1.0f - dx) // Bottom
             {
                 float wBL = dy - dx; float wBR = dx + dy - 1; float wC = 2 * (1 - dy);
                 res = (vBL.r * wBL + vBR.r * wBR + vC.r * wC, vBL.g * wBL + vBR.g * wBR + vC.g * wC, vBL.b * wBL + vBR.b * wBR + vC.b * wC, vBL.a * wBL + vBR.a * wBR + vC.a * wC);
             }
             else if (dx < dy && dx < 1.0f - dy) // Left
             {
                 float wTL = 1 - dx - dy; float wBL = dy - dx; float wC = 2 * dx;
                 res = (vTL.r * wTL + vBL.r * wBL + vC.r * wC, vTL.g * wTL + vBL.g * wBL + vC.g * wC, vTL.b * wTL + vBL.b * wBL + vC.b * wC, vTL.a * wTL + vBL.a * wBL + vC.a * wC);
             }
             else // Right
             {
                 float wTR = dx - dy; float wBR = dy + dx - 1; float wC = 2 * (1 - dx);
                 res = (vTR.r * wTR + vBR.r * wBR + vC.r * wC, vTR.g * wTR + vBR.g * wBR + vC.g * wC, vTR.b * wTR + vBR.b * wBR + vC.b * wC, vTR.a * wTR + vBR.a * wBR + vC.a * wC);
             }
             return (Math.Clamp(res.r, 0f, 1f), Math.Clamp(res.g, 0f, 1f), Math.Clamp(res.b, 0f, 1f), Math.Clamp(res.a, 0f, 1f));
        }

        for (int y = 0; y < size; y++)
        {
            float v = y / (float)(size - 1);
            float cy = v * 16;
            int cIy = Math.Clamp((int)cy, 0, 15);

            for (int x = 0; x < size; x++)
            {
                float u = x / (float)(size - 1);
                float cx = u * 16;
                int cIx = Math.Clamp((int)cx, 0, 15);

                int chunkIndex = cIy * 16 + cIx;

                if (!mccvDict.TryGetValue(chunkIndex, out var cData))
                {
                    image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(127, 127, 127, 127);
                    continue;
                }

                float lx = Math.Clamp(cx - cIx, 0f, 1f);
                float ly = Math.Clamp(cy - cIy, 0f, 1f);
                var (r, g, b, a) = SampleColor(cData, lx, ly);

                // Intentionally preserve the raw stored MCCV channel order in the PNG.
                // The cleanup path decodes this back to renderer RGB before removing tint.
                image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(r, g, b, a);
            }
        }

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    private byte[] BuildObjectVisibilityMask(VlmTerrainData terrainData, int width, int height, IArchiveReader archiveReader, IReadOnlyList<string> searchPaths)
    {
        if (width <= 0 || height <= 0 || terrainData.Objects.Count == 0)
            return Array.Empty<byte>();

        using Image<L8> objectMask = new(width, height);
        bool hasAnyCoverage = false;

        foreach (VlmObjectPlacement obj in terrainData.Objects)
        {
            if (!TryResolveObjectProjectionMode(obj, terrainData.AdtTile, out int tileX, out int tileY, out ObjectProjectionMode projectionMode))
                continue;

            bool wroteObject = false;

            if (!projectionMode.UseNormalized && !string.IsNullOrWhiteSpace(obj.ModelPath))
            {
                Vector2[][]? footprintPolygons = GetModelFootprintPolygons(obj.ModelPath, obj.Category, archiveReader, searchPaths);
                if (footprintPolygons != null)
                {
                    foreach (Vector2[] localPolygon in footprintPolygons)
                    {
                        if (!TryProjectFootprintPolygon(localPolygon, obj, projectionMode, tileX, tileY, width, height, out Vector2[] projectedPolygon))
                            continue;

                        if (!FillMaskPolygon(objectMask, projectedPolygon))
                            continue;

                        wroteObject = true;
                        hasAnyCoverage = true;
                    }
                }
            }

            if (!wroteObject && TryBuildBoundsFootprintPolygon(obj, out Vector2[] boundsPolygon))
            {
                if (TryProjectFootprintPolygon(boundsPolygon, obj, projectionMode, tileX, tileY, width, height, out Vector2[] projectedBoundsPolygon)
                    && FillMaskPolygon(objectMask, projectedBoundsPolygon))
                {
                    wroteObject = true;
                    hasAnyCoverage = true;
                }
            }

            if (wroteObject)
                continue;

            if (!TryProjectObjectToTilePixel(obj, terrainData.AdtTile, width, height, out int centerX, out int centerY))
                continue;

            EstimateObjectRadiiPixels(obj, width, height, out int radiusX, out int radiusY);
            DrawFilledEllipse(objectMask, centerX, centerY, radiusX, radiusY);
            hasAnyCoverage = true;
        }

        if (!hasAnyCoverage)
            return Array.Empty<byte>();

        using MemoryStream ms = new();
        objectMask.SaveAsPng(ms);
        return ms.ToArray();
    }

    private static IReadOnlyList<MprlEntry> LoadPm4PositionRefs(
        IReadOnlyList<string> searchPaths,
        IArchiveCatalog archiveCatalog,
        string mapName,
        string mapDirectory,
        string tileName,
        int tileX,
        int tileY)
    {
        try
        {
            foreach (string candidate in BuildPm4InternalPathCandidates(mapName, mapDirectory, tileX, tileY))
            {
                foreach (string basePath in searchPaths)
                {
                    string diskCandidate = Path.Combine(basePath, candidate);
                    if (!File.Exists(diskCandidate))
                        continue;

                    Pm4File diskPm4File = Pm4File.FromFile(diskCandidate);
                    return diskPm4File.PositionRefs;
                }

                if (!archiveCatalog.FileExists(candidate))
                    continue;

                byte[]? data = archiveCatalog.ReadFile(candidate);
                if (data == null || data.Length == 0)
                    continue;

                Pm4File archivePm4File = new(data);
                return archivePm4File.PositionRefs;
            }

            foreach (string fileName in BuildPm4FileNameCandidates(mapName, mapDirectory, tileName, tileX, tileY))
            {
                foreach (string basePath in searchPaths)
                {
                    string diskCandidate = Path.Combine(basePath, fileName);
                    if (!File.Exists(diskCandidate))
                        continue;

                    Pm4File loosePm4File = Pm4File.FromFile(diskCandidate);
                    return loosePm4File.PositionRefs;
                }
            }
        }
        catch
        {
        }

        return Array.Empty<MprlEntry>();
    }

    private static IEnumerable<string> BuildPm4InternalPathCandidates(string mapName, string mapDirectory, int tileX, int tileY)
    {
        foreach (string directoryName in DistinctMapNames(mapName, mapDirectory))
            yield return Path.Combine("World", "Maps", directoryName, $"{directoryName}_{tileX}_{tileY}.pm4");
    }

    private static IEnumerable<string> BuildPm4FileNameCandidates(string mapName, string mapDirectory, string tileName, int tileX, int tileY)
    {
        yield return $"{tileName}.pm4";
        foreach (string directoryName in DistinctMapNames(mapName, mapDirectory))
            yield return $"{directoryName}_{tileX}_{tileY}.pm4";
    }

    private static IEnumerable<string> DistinctMapNames(string mapName, string mapDirectory)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (!string.IsNullOrWhiteSpace(mapName) && seen.Add(mapName))
            yield return mapName;
        if (!string.IsNullOrWhiteSpace(mapDirectory) && seen.Add(mapDirectory))
            yield return mapDirectory;
    }

    internal static string? TryResolveArchiveMapDirectoryAlias(string mapName, IReadOnlyList<string> knownFiles)
    {
        if (string.IsNullOrWhiteSpace(mapName) || knownFiles.Count == 0)
            return null;

        string requestedToken = NormalizeMapToken(mapName);
        if (requestedToken.Length == 0)
            return null;

        HashSet<string> candidates = new(StringComparer.OrdinalIgnoreCase);
        foreach (string knownFile in knownFiles)
        {
            if (string.IsNullOrWhiteSpace(knownFile))
                continue;

            string normalizedPath = knownFile.Replace('\\', '/');
            if (!normalizedPath.StartsWith("World/Maps/", StringComparison.OrdinalIgnoreCase)
                || !normalizedPath.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string[] segments = normalizedPath.Split('/', StringSplitOptions.RemoveEmptyEntries);
            if (segments.Length < 4)
                continue;

            string directoryName = segments[2];
            string fileName = Path.GetFileNameWithoutExtension(segments[^1]);
            if (!string.Equals(directoryName, fileName, StringComparison.OrdinalIgnoreCase))
                continue;

            candidates.Add(directoryName);
        }

        foreach (string candidate in candidates)
        {
            if (string.Equals(NormalizeMapToken(candidate), requestedToken, StringComparison.Ordinal))
                return candidate;
        }

        string? bestCandidate = null;
        int bestDistance = int.MaxValue;
        bool isAmbiguous = false;

        foreach (string candidate in candidates)
        {
            string candidateToken = NormalizeMapToken(candidate);
            if (candidateToken.Length == 0 || candidateToken[0] != requestedToken[0])
                continue;

            if (Math.Abs(candidateToken.Length - requestedToken.Length) > 2)
                continue;

            int distance = ComputeLevenshteinDistance(requestedToken, candidateToken, 2);
            if (distance > 2)
                continue;

            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestCandidate = candidate;
                isAmbiguous = false;
            }
            else if (distance == bestDistance && !string.Equals(bestCandidate, candidate, StringComparison.OrdinalIgnoreCase))
            {
                isAmbiguous = true;
            }
        }

        return isAmbiguous ? null : bestCandidate;
    }

    private static string NormalizeMapToken(string value)
    {
        Span<char> buffer = stackalloc char[value.Length];
        int length = 0;
        foreach (char ch in value)
        {
            if (!char.IsLetterOrDigit(ch))
                continue;

            buffer[length++] = char.ToLowerInvariant(ch);
        }

        return length == 0 ? string.Empty : new string(buffer[..length]);
    }

    private static int ComputeLevenshteinDistance(string source, string target, int maxDistance)
    {
        int sourceLength = source.Length;
        int targetLength = target.Length;

        if (sourceLength == 0)
            return targetLength;
        if (targetLength == 0)
            return sourceLength;
        if (Math.Abs(sourceLength - targetLength) > maxDistance)
            return maxDistance + 1;

        int[] previous = new int[targetLength + 1];
        int[] current = new int[targetLength + 1];

        for (int j = 0; j <= targetLength; j++)
            previous[j] = j;

        for (int i = 1; i <= sourceLength; i++)
        {
            current[0] = i;
            int rowMin = current[0];

            for (int j = 1; j <= targetLength; j++)
            {
                int substitutionCost = source[i - 1] == target[j - 1] ? 0 : 1;
                current[j] = Math.Min(
                    Math.Min(previous[j] + 1, current[j - 1] + 1),
                    previous[j - 1] + substitutionCost);
                rowMin = Math.Min(rowMin, current[j]);
            }

            if (rowMin > maxDistance)
                return maxDistance + 1;

            (previous, current) = (current, previous);
        }

        return previous[targetLength];
    }

    private static bool FillMaskPolygon(Image<L8> image, IReadOnlyList<Vector2> polygon)
    {
        if (polygon.Count < 3)
            return false;

        float minYFloat = float.MaxValue;
        float maxYFloat = float.MinValue;
        foreach (Vector2 point in polygon)
        {
            if (!IsFinite(point.X) || !IsFinite(point.Y))
                return false;

            minYFloat = MathF.Min(minYFloat, point.Y);
            maxYFloat = MathF.Max(maxYFloat, point.Y);
        }

        int minY = Math.Max(0, (int)MathF.Floor(minYFloat));
        int maxY = Math.Min(image.Height - 1, (int)MathF.Ceiling(maxYFloat));
        if (maxY < minY)
            return false;

        List<float> intersections = new(polygon.Count);
        bool wroteAny = false;

        for (int y = minY; y <= maxY; y++)
        {
            float scanY = y + 0.5f;
            intersections.Clear();

            for (int index = 0; index < polygon.Count; index++)
            {
                Vector2 a = polygon[index];
                Vector2 b = polygon[(index + 1) % polygon.Count];
                if ((a.Y <= scanY && b.Y > scanY) || (b.Y <= scanY && a.Y > scanY))
                {
                    float x = a.X + ((scanY - a.Y) * (b.X - a.X) / (b.Y - a.Y));
                    intersections.Add(x);
                }
            }

            if (intersections.Count < 2)
                continue;

            intersections.Sort();
            for (int i = 0; i + 1 < intersections.Count; i += 2)
            {
                int startX = Math.Max(0, (int)MathF.Ceiling(intersections[i]));
                int endX = Math.Min(image.Width - 1, (int)MathF.Floor(intersections[i + 1]));
                if (endX < startX)
                    continue;

                for (int x = startX; x <= endX; x++)
                    image[x, y] = new L8(255);

                wroteAny = true;
            }
        }

        return wroteAny;
    }

    private static void DrawFilledEllipse(Image<L8> image, int centerX, int centerY, int radiusX, int radiusY)
    {
        radiusX = Math.Max(2, radiusX);
        radiusY = Math.Max(2, radiusY);

        int minX = Math.Max(0, centerX - radiusX);
        int maxX = Math.Min(image.Width - 1, centerX + radiusX);
        int minY = Math.Max(0, centerY - radiusY);
        int maxY = Math.Min(image.Height - 1, centerY + radiusY);
        float invRadiusXSquared = 1f / (radiusX * radiusX);
        float invRadiusYSquared = 1f / (radiusY * radiusY);

        for (int y = minY; y <= maxY; y++)
        {
            float dy = y - centerY;
            float dyTerm = dy * dy * invRadiusYSquared;
            for (int x = minX; x <= maxX; x++)
            {
                float dx = x - centerX;
                if ((dx * dx * invRadiusXSquared) + dyTerm > 1f)
                    continue;

                image[x, y] = new L8(255);
            }
        }
    }

    private static void EstimateObjectRadiiPixels(VlmObjectPlacement obj, int width, int height, out int radiusX, out int radiusY)
    {
        float pixelsPerWorld = MathF.Min(width, height) / TileSize;
        float scale = float.IsFinite(obj.Scale) && obj.Scale > 0f ? obj.Scale : 1f;

        if (obj.BoundsMin != null && obj.BoundsMax != null && obj.BoundsMin.Length >= 3 && obj.BoundsMax.Length >= 3)
        {
            float halfWidthWorld = MathF.Abs(obj.BoundsMax[0] - obj.BoundsMin[0]) * 0.5f * scale;
            float halfDepthWorld = MathF.Abs(obj.BoundsMax[2] - obj.BoundsMin[2]) * 0.5f * scale;
            radiusX = Math.Max(2, (int)MathF.Round(halfWidthWorld * pixelsPerWorld));
            radiusY = Math.Max(2, (int)MathF.Round(halfDepthWorld * pixelsPerWorld));
            return;
        }

        if (obj.BoundsMin != null && obj.BoundsMax != null && obj.BoundsMin.Length >= 2 && obj.BoundsMax.Length >= 2)
        {
            float halfWidthWorld = MathF.Abs(obj.BoundsMax[0] - obj.BoundsMin[0]) * 0.5f * scale;
            float halfDepthWorld = MathF.Abs(obj.BoundsMax[1] - obj.BoundsMin[1]) * 0.5f * scale;
            radiusX = Math.Max(2, (int)MathF.Round(halfWidthWorld * pixelsPerWorld));
            radiusY = Math.Max(2, (int)MathF.Round(halfDepthWorld * pixelsPerWorld));
            return;
        }

        float baseRadiusWorld = obj.Category.Contains("wmo", StringComparison.OrdinalIgnoreCase) ? 6f : 3f;
        int radius = Math.Max(2, (int)MathF.Round(baseRadiusWorld * scale * pixelsPerWorld));
        radiusX = radius;
        radiusY = radius;
    }

    private static bool TryBuildBoundsFootprintPolygon(VlmObjectPlacement obj, out Vector2[] polygon)
    {
        polygon = Array.Empty<Vector2>();

        if (obj.BoundsMin != null && obj.BoundsMax != null && obj.BoundsMin.Length >= 3 && obj.BoundsMax.Length >= 3)
        {
            polygon =
            [
                new Vector2(obj.BoundsMin[0], obj.BoundsMin[2]),
                new Vector2(obj.BoundsMax[0], obj.BoundsMin[2]),
                new Vector2(obj.BoundsMax[0], obj.BoundsMax[2]),
                new Vector2(obj.BoundsMin[0], obj.BoundsMax[2]),
            ];
            return true;
        }

        if (obj.BoundsMin != null && obj.BoundsMax != null && obj.BoundsMin.Length >= 2 && obj.BoundsMax.Length >= 2)
        {
            polygon =
            [
                new Vector2(obj.BoundsMin[0], obj.BoundsMin[1]),
                new Vector2(obj.BoundsMax[0], obj.BoundsMin[1]),
                new Vector2(obj.BoundsMax[0], obj.BoundsMax[1]),
                new Vector2(obj.BoundsMin[0], obj.BoundsMax[1]),
            ];
            return true;
        }

        return false;
    }

    private static bool TryProjectFootprintPolygon(
        IReadOnlyList<Vector2> localPolygon,
        VlmObjectPlacement obj,
        ObjectProjectionMode projectionMode,
        int tileX,
        int tileY,
        int width,
        int height,
        out Vector2[] projectedPolygon)
    {
        projectedPolygon = Array.Empty<Vector2>();
        if (projectionMode.UseNormalized || localPolygon.Count < 3)
            return false;

        Vector2[] transformedPolygon = TransformFootprintPolygonToWorld(localPolygon, obj, projectionMode);
        if (transformedPolygon.Length < 3)
            return false;

        float minU = float.MaxValue;
        float minV = float.MaxValue;
        float maxU = float.MinValue;
        float maxV = float.MinValue;
        projectedPolygon = new Vector2[transformedPolygon.Length];

        for (int index = 0; index < transformedPolygon.Length; index++)
        {
            Vector2 worldPoint = transformedPolygon[index];
            (float u, float v) = ProjectToTileUv(worldPoint.X, worldPoint.Y, tileX, tileY, projectionMode);
            if (!IsFinite(u) || !IsFinite(v))
                return false;

            minU = MathF.Min(minU, u);
            minV = MathF.Min(minV, v);
            maxU = MathF.Max(maxU, u);
            maxV = MathF.Max(maxV, v);
            projectedPolygon[index] = new Vector2(u * (width - 1), v * (height - 1));
        }

        if (maxU < -ObjectMaskMarginTiles || minU > 1f + ObjectMaskMarginTiles ||
            maxV < -ObjectMaskMarginTiles || minV > 1f + ObjectMaskMarginTiles)
        {
            projectedPolygon = Array.Empty<Vector2>();
            return false;
        }

        return true;
    }

    internal static Vector2[] TransformFootprintPolygonToWorldForTesting(IReadOnlyList<Vector2> localPolygon, VlmObjectPlacement obj, bool secondaryAxisIsZ)
    {
        ObjectProjectionMode projectionMode = new(
            secondaryAxisIsZ ? ObjectProjectionAxis.Z : ObjectProjectionAxis.Y,
            UseMapOrigin: false,
            UseNormalized: false);
        return TransformFootprintPolygonToWorld(localPolygon, obj, projectionMode);
    }

    private static Vector2[] TransformFootprintPolygonToWorld(
        IReadOnlyList<Vector2> localPolygon,
        VlmObjectPlacement obj,
        ObjectProjectionMode projectionMode)
    {
        float scale = float.IsFinite(obj.Scale) && obj.Scale > 0f ? obj.Scale : 1f;
        float rotationDegrees = projectionMode.SecondaryAxis == ObjectProjectionAxis.Z ? obj.RotY : obj.RotZ;
        float angle = rotationDegrees * MathF.PI / 180f;
        float cos = MathF.Cos(angle);
        float sin = MathF.Sin(angle);
        float baseSecondary = GetProjectionSecondaryCoordinate(obj, projectionMode.SecondaryAxis);

        Vector2[] transformedPolygon = new Vector2[localPolygon.Count];

        for (int index = 0; index < localPolygon.Count; index++)
        {
            Vector2 localPoint = localPolygon[index];
            float scaledX = localPoint.X * scale;
            float scaledY = localPoint.Y * scale;
            float worldA = obj.X + (scaledX * cos) - (scaledY * sin);
            float worldB = baseSecondary + (scaledX * sin) + (scaledY * cos);
            transformedPolygon[index] = new Vector2(worldA, worldB);
        }

        return transformedPolygon;
    }

    private static float GetProjectionSecondaryCoordinate(VlmObjectPlacement obj, ObjectProjectionAxis secondaryAxis)
    {
        return secondaryAxis == ObjectProjectionAxis.Z ? obj.Z : obj.Y;
    }

    private static (float U, float V) ProjectToTileUv(float worldA, float worldB, int tileX, int tileY, ObjectProjectionMode projectionMode)
    {
        if (projectionMode.UseNormalized)
            return (((worldB + 1f) * 0.5f), ((worldA + 1f) * 0.5f));

        if (!projectionMode.UseMapOrigin)
            return ((worldA / TileSize) - tileX, (worldB / TileSize) - tileY);

        return (((MapOrigin - worldB) / TileSize) - tileX, ((MapOrigin - worldA) / TileSize) - tileY);
    }

    private static bool TryResolveObjectProjectionMode(
        VlmObjectPlacement obj,
        string adtTile,
        out int tileX,
        out int tileY,
        out ObjectProjectionMode projectionMode)
    {
        tileX = 0;
        tileY = 0;
        projectionMode = default;

        if (!TryParseTileCoordinates(adtTile, out tileX, out tileY))
            return false;

        List<ProjectionCandidate> candidates = new();
        if (MathF.Abs(obj.X) < 2f && MathF.Abs(obj.Y) < 2f)
        {
            candidates.Add(new ProjectionCandidate(
                new ObjectProjectionMode(ObjectProjectionAxis.Y, UseMapOrigin: false, UseNormalized: true),
                (obj.Y + 1f) * 0.5f,
                (obj.X + 1f) * 0.5f));
        }

        candidates.Add(new ProjectionCandidate(
            new ObjectProjectionMode(ObjectProjectionAxis.Z, UseMapOrigin: false, UseNormalized: false),
            (obj.X / TileSize) - tileX,
            (obj.Z / TileSize) - tileY));
        candidates.Add(new ProjectionCandidate(
            new ObjectProjectionMode(ObjectProjectionAxis.Z, UseMapOrigin: true, UseNormalized: false),
            ((MapOrigin - obj.Z) / TileSize) - tileX,
            ((MapOrigin - obj.X) / TileSize) - tileY));
        candidates.Add(new ProjectionCandidate(
            new ObjectProjectionMode(ObjectProjectionAxis.Y, UseMapOrigin: false, UseNormalized: false),
            (obj.X / TileSize) - tileX,
            (obj.Y / TileSize) - tileY));
        candidates.Add(new ProjectionCandidate(
            new ObjectProjectionMode(ObjectProjectionAxis.Y, UseMapOrigin: true, UseNormalized: false),
            ((MapOrigin - obj.Y) / TileSize) - tileX,
            ((MapOrigin - obj.X) / TileSize) - tileY));

        ProjectionCandidate best = candidates[0];
        float bestOverflow = float.PositiveInfinity;
        foreach (ProjectionCandidate candidate in candidates)
        {
            float overflow =
                MathF.Max(0f, -candidate.U) + MathF.Max(0f, candidate.U - 1f) +
                MathF.Max(0f, -candidate.V) + MathF.Max(0f, candidate.V - 1f);
            if (overflow < bestOverflow)
            {
                best = candidate;
                bestOverflow = overflow;
                if (overflow <= 0.000001f)
                    break;
            }
        }

        if (best.U < -ObjectMaskMarginTiles || best.U > 1f + ObjectMaskMarginTiles ||
            best.V < -ObjectMaskMarginTiles || best.V > 1f + ObjectMaskMarginTiles)
        {
            return false;
        }

        projectionMode = best.Mode;
        return true;
    }

    private static bool TryProjectObjectToTilePixel(
        VlmObjectPlacement obj,
        string adtTile,
        int width,
        int height,
        out int centerX,
        out int centerY)
    {
        centerX = 0;
        centerY = 0;

        if (!TryResolveObjectProjectionMode(obj, adtTile, out int tileX, out int tileY, out ObjectProjectionMode projectionMode))
            return false;

        float secondary = GetProjectionSecondaryCoordinate(obj, projectionMode.SecondaryAxis);
        (float U, float V) best = ProjectToTileUv(obj.X, secondary, tileX, tileY, projectionMode);
        centerX = Math.Clamp((int)MathF.Round(Math.Clamp(best.U, 0f, 1f) * (width - 1)), 0, width - 1);
        centerY = Math.Clamp((int)MathF.Round(Math.Clamp(best.V, 0f, 1f) * (height - 1)), 0, height - 1);
        return true;
    }

    private static bool TryParseTileCoordinates(string adtTile, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;

        string[] parts = adtTile.Split('_');
        if (parts.Length < 3)
            return false;

        return int.TryParse(parts[^2], out tileX) && int.TryParse(parts[^1], out tileY);
    }

    private Vector2[][]? GetModelFootprintPolygons(string modelPath, string category, IArchiveReader archiveReader, IReadOnlyList<string> searchPaths)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return null;

        string cacheKey = NormalizeModelPath(modelPath).ToLowerInvariant();
        if (_modelFootprintCache.TryGetValue(cacheKey, out Vector2[][]? cached))
            return cached;

        try
        {
            foreach (string candidatePath in EnumerateModelPathCandidates(modelPath))
            {
                byte[]? data = ReadVirtualAssetBytes(searchPaths, candidatePath, archiveReader);
                if (data is null || data.Length < 16)
                    continue;

                Vector2[][]? polygons = TryReadFootprintPolygonsFromModelBytes(data, candidatePath, category, archiveReader, searchPaths);
                if (polygons is null || polygons.Length == 0)
                    continue;

                _modelFootprintCache[cacheKey] = polygons;
                string candidateCacheKey = NormalizeModelPath(candidatePath).ToLowerInvariant();
                _modelFootprintCache[candidateCacheKey] = polygons;
                return polygons;
            }
        }
        catch
        {
        }

        _modelFootprintCache[cacheKey] = null;
        return null;
    }

    private Vector2[][]? TryReadFootprintPolygonsFromModelBytes(byte[] data, string sourcePath, string category, IArchiveReader archiveReader, IReadOnlyList<string> searchPaths)
    {
        string extension = Path.GetExtension(sourcePath);
        bool preferWmo = category.Contains("wmo", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".wmo", StringComparison.OrdinalIgnoreCase);
        bool preferMdx = extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase);

        if (preferWmo)
        {
            Vector2[][]? wmoPolygons = TryReadWmoFootprintPolygons(data, sourcePath, archiveReader, searchPaths);
            if (wmoPolygons is { Length: > 0 })
                return wmoPolygons;
        }

        if (preferMdx)
        {
            Vector2[][]? mdxPolygons = TryReadMdxFootprintPolygons(data, sourcePath);
            if (mdxPolygons is { Length: > 0 })
                return mdxPolygons;

            Vector2[][]? m2Polygons = TryReadM2FootprintPolygons(data, sourcePath);
            if (m2Polygons is { Length: > 0 })
                return m2Polygons;
        }
        else
        {
            Vector2[][]? m2Polygons = TryReadM2FootprintPolygons(data, sourcePath);
            if (m2Polygons is { Length: > 0 })
                return m2Polygons;

            Vector2[][]? mdxPolygons = TryReadMdxFootprintPolygons(data, sourcePath);
            if (mdxPolygons is { Length: > 0 })
                return mdxPolygons;
        }

        return preferWmo ? null : TryReadWmoFootprintPolygons(data, sourcePath, archiveReader, searchPaths);
    }

    private static Vector2[][]? TryReadM2FootprintPolygons(byte[] data, string sourcePath)
    {
        try
        {
            using MemoryStream stream = new(data, writable: false);
            M2GeometryDocument geometry = M2GeometryReader.Read(stream, sourcePath);
            Vector2[]? hull = BuildFootprintHull(
                geometry.Vertices.Count,
                index => new Vector2(geometry.Vertices[index].Position.X, geometry.Vertices[index].Position.Z));
            return hull is { Length: >= 3 } ? [hull] : null;
        }
        catch
        {
            return null;
        }
    }

    private static Vector2[][]? TryReadMdxFootprintPolygons(byte[] data, string sourcePath)
    {
        try
        {
            using MemoryStream stream = new(data, writable: false);
            MdxGeometryFile geometry = MdxGeometryReader.Read(stream, sourcePath);
            List<Vector2[]> polygons = [];
            foreach (MdxGeosetGeometry geoset in geometry.Geosets)
            {
                Vector2[]? hull = BuildFootprintHull(
                    geoset.Vertices.Count,
                    index => new Vector2(geoset.Vertices[index].X, geoset.Vertices[index].Z));
                if (hull is { Length: >= 3 })
                    polygons.Add(hull);
            }

            return polygons.Count > 0 ? [.. polygons] : null;
        }
        catch
        {
            return null;
        }
    }

    private Vector2[][]? TryReadWmoFootprintPolygons(byte[] data, string sourcePath, IArchiveReader archiveReader, IReadOnlyList<string> searchPaths)
    {
        try
        {
            List<Vector2[]> polygons = [];
            AppendEmbeddedWmoFootprintPolygons(data, polygons);

            using MemoryStream stream = new(data, writable: false);
            WmoSummary summary = WmoSummaryReader.Read(stream, sourcePath);
            if (summary.ReportedGroupCount > 0)
                AppendSplitWmoFootprintPolygons(sourcePath, summary.ReportedGroupCount, archiveReader, searchPaths, polygons);

            return polygons.Count > 0 ? [.. polygons] : null;
        }
        catch
        {
            return null;
        }
    }

    private static void AppendEmbeddedWmoFootprintPolygons(byte[] data, List<Vector2[]> polygons)
    {
        using MemoryStream stream = new(data, writable: false);
        IReadOnlyList<ChunkSpan> chunks = ReadExpandedWmoRootChunks(stream);
        foreach (ChunkSpan chunk in chunks)
        {
            if (chunk.Header.Id != WmoChunkIds.Mogp)
                continue;

            byte[] mogp = ReadChunkPayload(stream, chunk);
            Vector2[]? hull = TryReadWmoGroupHullFromMogpPayload(mogp);
            if (hull is { Length: >= 3 })
                polygons.Add(hull);
        }
    }

    private static void AppendSplitWmoFootprintPolygons(string sourcePath, int groupCount, IArchiveReader archiveReader, IReadOnlyList<string> searchPaths, List<Vector2[]> polygons)
    {
        string normalizedRootPath = NormalizeModelPath(sourcePath);
        string directory = Path.GetDirectoryName(normalizedRootPath) ?? string.Empty;
        string baseName = Path.GetFileNameWithoutExtension(normalizedRootPath);

        for (int index = 0; index < groupCount; index++)
        {
            string groupPath = string.IsNullOrEmpty(directory)
                ? $"{baseName}_{index:D3}.wmo"
                : $"{directory}\\{baseName}_{index:D3}.wmo";
            byte[]? groupBytes = ReadVirtualAssetBytes(searchPaths, groupPath, archiveReader);
            if (groupBytes is null || groupBytes.Length < ChunkHeader.SizeInBytes)
                continue;

            Vector2[]? hull = TryReadWmoGroupHullFromGroupBytes(groupBytes);
            if (hull is { Length: >= 3 })
                polygons.Add(hull);
        }
    }

    private static IReadOnlyList<ChunkSpan> ReadExpandedWmoRootChunks(Stream stream)
    {
        IReadOnlyList<ChunkSpan> topLevelChunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        if (topLevelChunks.Count <= 1 || topLevelChunks[1].Header.Id != WmoChunkIds.Momo)
            return topLevelChunks;

        List<ChunkSpan> expandedChunks = new(topLevelChunks.Count + 8);
        foreach (ChunkSpan chunk in topLevelChunks)
        {
            if (chunk.Header.Id != WmoChunkIds.Momo)
            {
                expandedChunks.Add(chunk);
                continue;
            }

            long previousPosition = stream.Position;
            byte[] headerBytes = new byte[ChunkHeader.SizeInBytes];
            try
            {
                long offset = chunk.DataOffset;
                while (offset + ChunkHeader.SizeInBytes <= chunk.EndOffset)
                {
                    stream.Position = offset;
                    stream.ReadExactly(headerBytes);
                    FourCC id = FourCC.FromFileBytes(headerBytes.AsSpan(0, 4));
                    uint size = BinaryPrimitives.ReadUInt32LittleEndian(headerBytes.AsSpan(4, 4));
                    long dataOffset = offset + ChunkHeader.SizeInBytes;
                    long endOffset = dataOffset + size;
                    if (endOffset > chunk.EndOffset)
                        break;

                    expandedChunks.Add(new ChunkSpan(new ChunkHeader(id, size), offset, dataOffset));
                    offset = endOffset;
                }
            }
            finally
            {
                stream.Position = previousPosition;
            }
        }

        return expandedChunks;
    }

    private static byte[] ReadChunkPayload(Stream stream, ChunkSpan chunk)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            byte[] payload = new byte[chunk.Header.Size];
            stream.ReadExactly(payload);
            return payload;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static Vector2[]? TryReadWmoGroupHullFromGroupBytes(byte[] groupBytes)
    {
        using MemoryStream stream = new(groupBytes, writable: false);
        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
        ChunkSpan mogpChunk = chunks.FirstOrDefault(static chunk => chunk.Header.Id == WmoChunkIds.Mogp);
        if (mogpChunk.Header.Id != WmoChunkIds.Mogp)
            return null;

        byte[] mogp = ReadChunkPayload(stream, mogpChunk);
        return TryReadWmoGroupHullFromMogpPayload(mogp);
    }

    private static Vector2[]? TryReadWmoGroupHullFromMogpPayload(byte[] mogp)
    {
        if (mogp.Length < 0x38)
            return null;

        int headerSize = FindWmoGroupHeaderSize(mogp);
        byte[]? movtPayload = TryReadWmoSubchunkPayload(mogp, headerSize, WmoChunkIds.Movt);
        if (movtPayload is null || movtPayload.Length < 36 || (movtPayload.Length % 12) != 0)
            return null;

        int vertexCount = movtPayload.Length / 12;
        return BuildFootprintHull(
            vertexCount,
            index =>
            {
                int offset = index * 12;
                float x = BitConverter.ToSingle(movtPayload, offset);
                float z = BitConverter.ToSingle(movtPayload, offset + 8);
                return new Vector2(x, z);
            });
    }

    private static int FindWmoGroupHeaderSize(byte[] mogp)
    {
        foreach (int candidate in new[] { 0x44, 0x80 })
        {
            if (HasKnownWmoGroupSubchunkAt(mogp, candidate))
                return candidate;
        }

        for (int candidate = 0x38; candidate <= mogp.Length - ChunkHeader.SizeInBytes; candidate += 4)
        {
            if (HasKnownWmoGroupSubchunkAt(mogp, candidate))
                return candidate;
        }

        return Math.Min(0x80, mogp.Length);
    }

    private static bool HasKnownWmoGroupSubchunkAt(byte[] mogp, int offset)
    {
        if (offset > mogp.Length - ChunkHeader.SizeInBytes)
            return false;

        if (!ChunkHeaderReader.TryRead(mogp.AsSpan(offset, ChunkHeader.SizeInBytes), out ChunkHeader header))
            return false;

        return KnownWmoGroupSubchunkIds.Contains(header.Id)
            && (long)offset + ChunkHeader.SizeInBytes + header.Size <= mogp.Length;
    }

    private static byte[]? TryReadWmoSubchunkPayload(byte[] mogp, int headerSizeBytes, FourCC chunkId)
    {
        int position = headerSizeBytes;
        while (position <= mogp.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(mogp.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                return null;

            int dataOffset = position + ChunkHeader.SizeInBytes;
            long endOffset = (long)dataOffset + header.Size;
            if (endOffset > mogp.Length)
                return null;

            if (header.Id == chunkId)
                return mogp.AsSpan(dataOffset, checked((int)header.Size)).ToArray();

            position = checked((int)endOffset);
        }

        return null;
    }

    private static Vector2[]? BuildFootprintHull(int pointCount, Func<int, Vector2> pointAccessor)
    {
        if (pointCount < 3)
            return null;

        int step = Math.Max(1, (int)Math.Ceiling(pointCount / (double)MaxFootprintSamplesPerSource));
        List<Vector2> points = new(Math.Min(pointCount, MaxFootprintSamplesPerSource) + 1);
        for (int index = 0; index < pointCount; index += step)
        {
            Vector2 point = pointAccessor(index);
            if (IsFinite(point.X) && IsFinite(point.Y))
                points.Add(point);
        }

        int lastIndex = pointCount - 1;
        if ((lastIndex % step) != 0)
        {
            Vector2 point = pointAccessor(lastIndex);
            if (IsFinite(point.X) && IsFinite(point.Y))
                points.Add(point);
        }

        return BuildConvexHull(points);
    }

    private static Vector2[]? BuildConvexHull(List<Vector2> points)
    {
        if (points.Count < 3)
            return null;

        points.Sort(static (left, right) =>
        {
            int compareX = left.X.CompareTo(right.X);
            return compareX != 0 ? compareX : left.Y.CompareTo(right.Y);
        });

        List<Vector2> uniquePoints = new(points.Count);
        foreach (Vector2 point in points)
        {
            if (uniquePoints.Count > 0 && Vector2.DistanceSquared(uniquePoints[^1], point) < 0.000001f)
                continue;

            uniquePoints.Add(point);
        }

        if (uniquePoints.Count < 3)
            return null;

        List<Vector2> hull = new(uniquePoints.Count * 2);
        foreach (Vector2 point in uniquePoints)
        {
            while (hull.Count >= 2 && Cross(hull[^2], hull[^1], point) <= 0f)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(point);
        }

        int lowerCount = hull.Count;
        for (int index = uniquePoints.Count - 2; index >= 0; index--)
        {
            Vector2 point = uniquePoints[index];
            while (hull.Count > lowerCount && Cross(hull[^2], hull[^1], point) <= 0f)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(point);
        }

        if (hull.Count <= 3)
            return null;

        hull.RemoveAt(hull.Count - 1);
        return hull.ToArray();
    }

    private static float Cross(Vector2 origin, Vector2 left, Vector2 right)
    {
        Vector2 a = left - origin;
        Vector2 b = right - origin;
        return (a.X * b.Y) - (a.Y * b.X);
    }

    /// <summary>
    /// Produces a synthesized minimap where masked pixels are replaced with estimated
    /// terrain color sampled from surrounding unmasked pixels.
    /// </summary>
    private static byte[] SynthesizeMaskedMinimap(string sourceMinimapPath, byte[] maskPngBytes)
    {
        try
        {
            using var srcImage = Image.Load<Rgba32>(sourceMinimapPath);
            int w = srcImage.Width;
            int h = srcImage.Height;

            using var maskStream = new MemoryStream(maskPngBytes);
            using var maskImage = Image.Load<L8>(maskStream);
            if (maskImage.Width != w || maskImage.Height != h)
                maskImage.Mutate(ctx => ctx.Resize(w, h));

            var pixels = new Rgba32[w * h];
            var isFilled = new bool[w * h];

            srcImage.ProcessPixelRows(acc =>
            {
                for (int y = 0; y < h; y++)
                {
                    var row = acc.GetRowSpan(y);
                    for (int x = 0; x < w; x++)
                        pixels[y * w + x] = row[x];
                }
            });

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    var maskPixel = maskImage[x, y];
                    isFilled[y * w + x] = maskPixel.PackedValue < 128;
                }
            }

            long rSum = 0;
            long gSum = 0;
            long bSum = 0;
            long count = 0;
            for (int i = 0; i < pixels.Length; i++)
            {
                if (!isFilled[i])
                    continue;

                rSum += pixels[i].R;
                gSum += pixels[i].G;
                bSum += pixels[i].B;
                count++;
            }

            var fallback = count > 0
                ? new Rgba32((byte)(rSum / count), (byte)(gSum / count), (byte)(bSum / count), 255)
                : new Rgba32(100, 120, 80, 255);

            var pending = new bool[w * h];
            for (int i = 0; i < pending.Length; i++)
                pending[i] = !isFilled[i];

            int[] dx = { -1, 1, 0, 0 };
            int[] dy = { 0, 0, -1, 1 };

            for (int pass = 0; pass < 64; pass++)
            {
                bool anyResolved = false;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int idx = y * w + x;
                        if (!pending[idx])
                            continue;

                        long r = 0;
                        long g = 0;
                        long b = 0;
                        int n = 0;
                        for (int d = 0; d < 4; d++)
                        {
                            int nx = x + dx[d];
                            int ny = y + dy[d];
                            if (nx < 0 || nx >= w || ny < 0 || ny >= h)
                                continue;

                            int ni = ny * w + nx;
                            if (!isFilled[ni])
                                continue;

                            r += pixels[ni].R;
                            g += pixels[ni].G;
                            b += pixels[ni].B;
                            n++;
                        }

                        if (n > 0)
                        {
                            pixels[idx] = new Rgba32((byte)(r / n), (byte)(g / n), (byte)(b / n), 255);
                            isFilled[idx] = true;
                            pending[idx] = false;
                            anyResolved = true;
                        }
                    }
                }

                if (!anyResolved)
                    break;
            }

            for (int i = 0; i < pixels.Length; i++)
            {
                if (pending[i])
                    pixels[i] = fallback;
            }

            using var result = new Image<Rgba32>(w, h);
            result.ProcessPixelRows(acc =>
            {
                for (int y = 0; y < h; y++)
                {
                    var row = acc.GetRowSpan(y);
                    pixels.AsSpan(y * w, w).CopyTo(row);
                }
            });

            using var ms = new MemoryStream();
            result.SaveAsPng(ms);
            return ms.ToArray();
        }
        catch
        {
            return Array.Empty<byte>();
        }
    }

    private async Task<byte[]> SynthesizeTerrainMaskedMinimapAsync(
        string sourceMinimapPath,
        byte[] maskPngBytes,
        VlmTerrainData terrainData,
        MinimapBakeService minimapBakeService)
    {
        if (terrainData.ChunkLayers == null || terrainData.ChunkLayers.Length == 0)
            return Array.Empty<byte>();

        try
        {
            using var result = Image.Load<Rgba32>(sourceMinimapPath);
            using var maskStream = new MemoryStream(maskPngBytes);
            using var maskImage = Image.Load<L8>(maskStream);
            if (maskImage.Width != result.Width || maskImage.Height != result.Height)
                maskImage.Mutate(ctx => ctx.Resize(result.Width, result.Height));

            bool anyReplaced = await ReplaceMaskedPixelsWithBakedChunksAsync(
                result,
                maskImage,
                terrainData.ChunkLayers,
                minimapBakeService.TryBakeChunkAsync);

            if (!anyReplaced)
                return Array.Empty<byte>();

            using var output = new MemoryStream();
            result.SaveAsPng(output);
            return output.ToArray();
        }
        catch
        {
            return Array.Empty<byte>();
        }
    }

    internal static async Task<bool> ReplaceMaskedPixelsWithBakedChunksAsync(
        Image<Rgba32> result,
        Image<L8> maskImage,
        VlmChunkLayers[]? chunkLayers,
        Func<VlmChunkLayers, string?, Task<Image<Rgba32>?>> bakeChunkAsync)
    {
        ArgumentNullException.ThrowIfNull(result);
        ArgumentNullException.ThrowIfNull(maskImage);
        ArgumentNullException.ThrowIfNull(bakeChunkAsync);

        if (chunkLayers == null || chunkLayers.Length == 0)
            return false;

        string?[] fallbackTexturePaths = ResolveChunkTextureFallbackPaths(chunkLayers);
        bool anyReplaced = false;

        foreach (VlmChunkLayers chunk in chunkLayers)
        {
            if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= 256)
                continue;

            var bounds = GetChunkBounds(chunk.ChunkIndex, result.Width, result.Height);
            if (!ChunkHasMaskedPixels(maskImage, bounds))
                continue;

            string? fallbackTexturePath = chunk.ChunkIndex < fallbackTexturePaths.Length
                ? fallbackTexturePaths[chunk.ChunkIndex]
                : null;

            Image<Rgba32>? bakedChunk = await bakeChunkAsync(chunk, fallbackTexturePath);
            if (bakedChunk == null)
                continue;

            using (bakedChunk)
            {
                int chunkWidth = bounds.EndX - bounds.StartX;
                int chunkHeight = bounds.EndY - bounds.StartY;
                if (chunkWidth <= 0 || chunkHeight <= 0)
                    continue;

                if (bakedChunk.Width != chunkWidth || bakedChunk.Height != chunkHeight)
                    bakedChunk.Mutate(ctx => ctx.Resize(chunkWidth, chunkHeight));

                for (int y = 0; y < chunkHeight; y++)
                {
                    for (int x = 0; x < chunkWidth; x++)
                    {
                        if (maskImage[bounds.StartX + x, bounds.StartY + y].PackedValue < 128)
                            continue;

                        result[bounds.StartX + x, bounds.StartY + y] = bakedChunk[x, y];
                        anyReplaced = true;
                    }
                }
            }
        }

        return anyReplaced;
    }

    private static byte[] CombineMaskPngBytes(params byte[][] maskPngs)
    {
        byte[][] validMasks = maskPngs
            .Where(mask => mask != null && mask.Length > 0)
            .ToArray();
        if (validMasks.Length == 0)
            return Array.Empty<byte>();

        if (validMasks.Length == 1)
            return validMasks[0];

        try
        {
            using MemoryStream firstStream = new(validMasks[0]);
            using Image<L8> merged = Image.Load<L8>(firstStream);

            for (int index = 1; index < validMasks.Length; index++)
            {
                using MemoryStream overlayStream = new(validMasks[index]);
                using Image<L8> overlay = Image.Load<L8>(overlayStream);
                if (overlay.Width != merged.Width || overlay.Height != merged.Height)
                    overlay.Mutate(ctx => ctx.Resize(merged.Width, merged.Height));

                for (int y = 0; y < merged.Height; y++)
                {
                    for (int x = 0; x < merged.Width; x++)
                    {
                        if (overlay[x, y].PackedValue > merged[x, y].PackedValue)
                            merged[x, y] = overlay[x, y];
                    }
                }
            }

            using MemoryStream output = new();
            merged.SaveAsPng(output);
            return output.ToArray();
        }
        catch
        {
            return validMasks[0];
        }
    }

    internal static IEnumerable<string> GetTerrainOnlyMaskPaths(IEnumerable<string> alphaPaths, string? shadowPath)
    {
        ArgumentNullException.ThrowIfNull(alphaPaths);
        _ = shadowPath;

        // MCSH darkening is part of the baked minimap in the general case, so terrain-only cleanup
        // should only remove actual overlay masks and other non-terrain occluders.
        return alphaPaths.Where(path => !string.IsNullOrWhiteSpace(path));
    }

    private static async Task<byte[]> BuildCombinedTerrainOnlyMaskAsync(IEnumerable<string> maskPaths, params byte[][] inMemoryMasks)
    {
        var masks = new List<byte[]>();

        foreach (byte[] mask in inMemoryMasks)
        {
            if (mask != null && mask.Length > 0)
                masks.Add(mask);
        }

        foreach (string path in maskPaths)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                continue;

            try
            {
                byte[] maskBytes = await File.ReadAllBytesAsync(path);
                if (maskBytes.Length > 0)
                    masks.Add(maskBytes);
            }
            catch
            {
            }
        }

        return masks.Count > 0 ? CombineMaskPngBytes(masks.ToArray()) : Array.Empty<byte>();
    }

    internal static string?[] ResolveChunkTextureFallbackPaths(VlmChunkLayers[]? chunkLayers)
    {
        string?[] fallbackPaths = new string?[256];
        if (chunkLayers == null || chunkLayers.Length == 0)
            return fallbackPaths;

        foreach (VlmChunkLayers chunk in chunkLayers)
        {
            if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= fallbackPaths.Length)
                continue;

            fallbackPaths[chunk.ChunkIndex] = GetPrimaryTexturePath(chunk);
        }

        for (int index = 0; index < fallbackPaths.Length; index++)
        {
            if (!string.IsNullOrWhiteSpace(fallbackPaths[index]))
                continue;

            int chunkX = index % 16;
            int chunkY = index / 16;
            int bestDistance = int.MaxValue;
            string? bestTexture = null;

            for (int candidateIndex = 0; candidateIndex < fallbackPaths.Length; candidateIndex++)
            {
                string? candidateTexture = fallbackPaths[candidateIndex];
                if (string.IsNullOrWhiteSpace(candidateTexture))
                    continue;

                int candidateX = candidateIndex % 16;
                int candidateY = candidateIndex / 16;
                int distance = Math.Abs(candidateX - chunkX) + Math.Abs(candidateY - chunkY);
                if (distance >= bestDistance)
                    continue;

                bestDistance = distance;
                bestTexture = candidateTexture;
            }

            fallbackPaths[index] = bestTexture;
        }

        return fallbackPaths;
    }

    internal static byte[] RenderChunkValueMap(ushort[] chunkValues, int width, int height)
    {
        if (chunkValues == null || chunkValues.Length == 0 || width <= 0 || height <= 0)
            return Array.Empty<byte>();

        using var image = new Image<L16>(width, height);
        for (int chunkIndex = 0; chunkIndex < Math.Min(chunkValues.Length, 256); chunkIndex++)
        {
            ushort value = chunkValues[chunkIndex];
            var bounds = GetChunkBounds(chunkIndex, width, height);
            for (int y = bounds.StartY; y < bounds.EndY; y++)
            {
                for (int x = bounds.StartX; x < bounds.EndX; x++)
                    image[x, y] = new L16(value);
            }
        }

        using var output = new MemoryStream();
        image.SaveAsPng(output);
        return output.ToArray();
    }

    internal static byte[] RenderChunkFlagMap(uint[] chunkFlags, int width, int height)
    {
        if (chunkFlags == null || chunkFlags.Length == 0 || width <= 0 || height <= 0)
            return Array.Empty<byte>();

        using var image = new Image<Rgba32>(width, height);
        for (int chunkIndex = 0; chunkIndex < Math.Min(chunkFlags.Length, 256); chunkIndex++)
        {
            uint value = chunkFlags[chunkIndex];
            var color = new Rgba32(
                (byte)(value & 0xFF),
                (byte)((value >> 8) & 0xFF),
                (byte)((value >> 16) & 0xFF),
                (byte)((value >> 24) & 0xFF));
            var bounds = GetChunkBounds(chunkIndex, width, height);
            for (int y = bounds.StartY; y < bounds.EndY; y++)
            {
                for (int x = bounds.StartX; x < bounds.EndX; x++)
                    image[x, y] = color;
            }
        }

        using var output = new MemoryStream();
        image.SaveAsPng(output);
        return output.ToArray();
    }

    internal static byte[] RenderHolesMask(int[]? holes, int width, int height)
    {
        if (holes == null || holes.Length == 0 || width <= 0 || height <= 0)
            return Array.Empty<byte>();

        using var image = new Image<L8>(width, height);
        for (int chunkIndex = 0; chunkIndex < Math.Min(holes.Length, 256); chunkIndex++)
        {
            int holeMask = holes[chunkIndex];
            if (holeMask == 0)
                continue;

            var bounds = GetChunkBounds(chunkIndex, width, height);
            for (int cellY = 0; cellY < 4; cellY++)
            {
                for (int cellX = 0; cellX < 4; cellX++)
                {
                    int holeBit = 1 << (cellX + (cellY * 4));
                    if ((holeMask & holeBit) == 0)
                        continue;

                    int startX = bounds.StartX + (cellX * (bounds.EndX - bounds.StartX)) / 4;
                    int endX = bounds.StartX + ((cellX + 1) * (bounds.EndX - bounds.StartX)) / 4;
                    int startY = bounds.StartY + (cellY * (bounds.EndY - bounds.StartY)) / 4;
                    int endY = bounds.StartY + ((cellY + 1) * (bounds.EndY - bounds.StartY)) / 4;

                    for (int y = startY; y < endY; y++)
                    {
                        for (int x = startX; x < endX; x++)
                            image[x, y] = new L8(255);
                    }
                }
            }
        }

        using var output = new MemoryStream();
        image.SaveAsPng(output);
        return output.ToArray();
    }

    internal static ushort[] BuildChunkAreaIdValues(VlmChunkLayers[]? chunkLayers)
    {
        ushort[] values = new ushort[256];
        if (chunkLayers == null)
            return values;

        foreach (VlmChunkLayers chunk in chunkLayers)
        {
            if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= values.Length)
                continue;

            values[chunk.ChunkIndex] = (ushort)Math.Min(chunk.AreaId, ushort.MaxValue);
        }

        return values;
    }

    internal static uint[] BuildChunkFlagValues(VlmChunkLayers[]? chunkLayers)
    {
        uint[] values = new uint[256];
        if (chunkLayers == null)
            return values;

        foreach (VlmChunkLayers chunk in chunkLayers)
        {
            if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= values.Length)
                continue;

            values[chunk.ChunkIndex] = chunk.Flags;
        }

        return values;
    }

    internal static ushort[] BuildLiquidTypeValues(VlmLiquidData[]? liquids)
    {
        ushort[] values = new ushort[256];
        if (liquids == null)
            return values;

        foreach (VlmLiquidData liquid in liquids)
        {
            if (liquid.ChunkIndex < 0 || liquid.ChunkIndex >= values.Length)
                continue;

            values[liquid.ChunkIndex] = (ushort)Math.Clamp(liquid.LiquidType, 0, ushort.MaxValue);
        }

        return values;
    }

    internal static ushort[] BuildDominantEffectIdValues(VlmChunkLayers[]? chunkLayers)
    {
        ushort[] values = new ushort[256];
        if (chunkLayers == null)
            return values;

        foreach (VlmChunkLayers chunk in chunkLayers)
        {
            if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= values.Length || chunk.Layers == null || chunk.Layers.Length == 0)
                continue;

            uint effectId = chunk.Layers
                .Where(layer => layer != null)
                .Select(layer => layer.EffectId)
                .FirstOrDefault(id => id != 0);
            if (effectId == 0)
                effectId = chunk.Layers[0].EffectId;

            values[chunk.ChunkIndex] = (ushort)Math.Min(effectId, ushort.MaxValue);
        }

        return values;
    }

    private static bool ChunkHasMaskedPixels(Image<L8> maskImage, (int StartX, int StartY, int EndX, int EndY) bounds)
    {
        for (int y = bounds.StartY; y < bounds.EndY; y++)
        {
            for (int x = bounds.StartX; x < bounds.EndX; x++)
            {
                if (maskImage[x, y].PackedValue >= 128)
                    return true;
            }
        }

        return false;
    }

    private static (int StartX, int StartY, int EndX, int EndY) GetChunkBounds(int chunkIndex, int width, int height)
    {
        int chunkX = chunkIndex % 16;
        int chunkY = chunkIndex / 16;
        int startX = (chunkX * width) / 16;
        int endX = ((chunkX + 1) * width) / 16;
        int startY = (chunkY * height) / 16;
        int endY = ((chunkY + 1) * height) / 16;
        return (startX, startY, endX, endY);
    }

    private static string? GetPrimaryTexturePath(VlmChunkLayers chunk)
    {
        if (chunk.Layers == null)
            return null;

        foreach (VlmTextureLayer layer in chunk.Layers)
        {
            if (!string.IsNullOrWhiteSpace(layer.TexturePath))
                return layer.TexturePath;
        }

        return null;
    }

    private async Task GenerateGlobalHeightmapsAsync(string datasetDir, string outputDir, IProgress<string>? progress)
    {
        var jsonFiles = Directory.GetFiles(datasetDir, "*.json");
        if (jsonFiles.Length == 0) return;

        float globalMin = float.MaxValue;
        float globalMax = float.MinValue;

        foreach (var jsonPath in jsonFiles)
        {
            try
            {
                var json = await File.ReadAllTextAsync(jsonPath);
                var sample = JsonSerializer.Deserialize<VlmTrainingSample>(json);
                var heights = sample?.TerrainData?.Heights;
                if (heights == null || heights.Length == 0) continue;

                foreach (var chunk in heights)
                {
                    if (chunk.Heights == null) continue;
                    foreach (var h in chunk.Heights)
                    {
                        if (float.IsNaN(h) || float.IsInfinity(h)) continue;
                        if (Math.Abs(h) > 50000f) continue; // Ignore outliers
                        if (h < globalMin) globalMin = h;
                        if (h > globalMax) globalMax = h;
                    }
                }
            }
            catch
            {
                // Skip malformed tiles
            }
        }

        if (globalMin >= globalMax)
        {
            globalMin = 0;
            globalMax = 1;
        }

        foreach (var jsonPath in jsonFiles)
        {
            try
            {
                var json = await File.ReadAllTextAsync(jsonPath);
                var sample = JsonSerializer.Deserialize<VlmTrainingSample>(json);
                if (sample?.TerrainData?.Heights == null) continue;

                var heightsDict = sample.TerrainData.Heights
                    .Where(h => h.Heights != null)
                    .ToDictionary(h => h.ChunkIndex, h => h.Heights!);
                if (heightsDict.Count == 0) continue;

                var filename = $"{sample.TerrainData.AdtTile}_heightmap_global.png";
                var imagesDir = Path.Combine(outputDir, "images");
                Directory.CreateDirectory(imagesDir);
                var path = Path.Combine(imagesDir, filename);
                var tileHeightmap = TerrainTileBakeService.BuildTileHeightmap257(heightsDict, sample.TerrainData.IsInterleaved);
                using (Image<L16> image = TerrainTileBakeService.CreateHeightmapImage(tileHeightmap.Heights, globalMin, globalMax, 512))
                {
                    await image.SaveAsPngAsync(path);
                }

                var heightmapGlobalPath = $"images/{filename}";
                var updatedTerrain = sample.TerrainData with
                {
                    HeightmapLocalPath = sample.TerrainData.HeightmapLocalPath ?? sample.TerrainData.HeightmapPath,
                    HeightmapGlobalPath = heightmapGlobalPath,
                    HeightGlobalMin = globalMin,
                    HeightGlobalMax = globalMax,
                    IsInterleaved = sample.TerrainData.IsInterleaved
                };
                var updatedSample = sample with { TerrainData = updatedTerrain };
                await File.WriteAllTextAsync(jsonPath, JsonSerializer.Serialize(updatedSample, _jsonOptions));
            }
            catch
            {
                // Skip malformed tiles
            }
        }

        progress?.Report($"Global heightmaps generated with range {globalMin} to {globalMax}");
    }

    private int ExportTilesetTextures(string outputDir, IEnumerable<string> textures, IArchiveReader archiveCatalog, IReadOnlyList<string> searchPaths)
    {
        var tilesetsDir = Path.Combine(outputDir, "tilesets");
        Directory.CreateDirectory(tilesetsDir);

        int textureCount = 0;
        foreach (string texture in textures)
        {
            var texName = Path.GetFileName(texture);
            var pngName = Path.ChangeExtension(texName, ".png");
            var pngPath = Path.Combine(tilesetsDir, pngName);

            if (File.Exists(pngPath))
                continue;

            string normalizedTexturePath = NormalizeVirtualAssetPath(texture);
            string? looseTexturePath = TryResolveLooseAssetPath(searchPaths, normalizedTexturePath);
            bool converted = looseTexturePath != null
                ? ConvertBlpToPng(looseTexturePath, pngPath, archiveCatalog)
                : ConvertBlpToPng($"MPQ:{normalizedTexturePath}", pngPath, archiveCatalog);

            if (converted)
                textureCount++;
        }

        return textureCount;
    }

    private bool ConvertBlpToPng(string blpPath, string pngPath, IArchiveReader? archiveReader = null)
    {
        try
        {
            // If source is already a PNG, just copy it to the output location
            if (blpPath.EndsWith(".png", StringComparison.OrdinalIgnoreCase) && File.Exists(blpPath))
            {
                File.Copy(blpPath, pngPath, overwrite: true);
                return true;
            }

            byte[]? blpData = null;
            
            if (blpPath.StartsWith("MPQ:"))
            {
                var key = blpPath.Substring(4);
                blpData = archiveReader?.ReadFile(key);
            }
            else if (blpPath.EndsWith(".MPQ", StringComparison.OrdinalIgnoreCase))
            {
                blpData = AlphaArchiveReader.ReadFromMpq(blpPath);
            }
            else if (File.Exists(blpPath))
            {
                blpData = File.ReadAllBytes(blpPath);
            }

            if (blpData == null || blpData.Length == 0)
            {
                Console.WriteLine($"Empty BLP data: {blpPath}");
                if (blpPath.StartsWith("MPQ:") && archiveReader != null)
                {
                     string k = blpPath.Substring(4);
                     if (archiveReader.FileExists(k))
                         Console.WriteLine($"[DEBUG] CRITICAL: File exists in MPQ but ReadFile failed! Key: {k}");
                     else
                         Console.WriteLine($"[DEBUG] File not found in MPQ archives: {k}");
                }
                return false;
            }


            using var ms = new MemoryStream(blpData);
            using var blp = new SereniaBLPLib.BlpFile(ms);
            using var bmp = blp.GetBitmap(0);
            

            
            // Log ALL minimaps to debug
            if (blpPath.Contains("minimap", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine($"[DEBUG] ConvertBlpToPng: {blpPath}");
                Console.WriteLine($"[DEBUG]   BLP Size: {blpData.Length} bytes");
                Console.WriteLine($"[DEBUG]   Bitmap: {bmp.Width}x{bmp.Height} {bmp.PixelFormat}");
                var px = bmp.GetPixel(bmp.Width/2, bmp.Height/2);
                Console.WriteLine($"[DEBUG]   Center Pixel: R={px.R} G={px.G} B={px.B} A={px.A}");
            }

            // V7 Dataset Standard requires 512x512 for terrain tiles, 
            // but older dataset tools expect 256x256 for MINIMAP tiles?
            // User says: "minimap tiles in 4.0.0 are 512x512... it breaks all the dataset tools".
            // So we should specificially RESIZE MINIMAP TILES to 256x256 if they are 512x512.
            
            // NOTE: This function is used for BOTH tileset textures (which we want 512) and minimaps (which might need 256).
            // We need a flag or logic to distinguish?
            // "blpPath" usually contains "minimap" string if it's a minimap.
            
            int targetWidth = 512;
            int targetHeight = 512;
            
            bool isMinimap = blpPath.Contains("minimap", StringComparison.OrdinalIgnoreCase);
            if (isMinimap)
            {
                // Force minimaps to 256x256 to allow stitching tools (which expect 256) to work.
                targetWidth = 256;
                targetHeight = 256;
            }

            if (bmp.Width != targetWidth || bmp.Height != targetHeight)
            {
                var resized = new System.Drawing.Bitmap(targetWidth, targetHeight);
                using (var g = System.Drawing.Graphics.FromImage(resized))
                {
                    // Use HighQualityBicubic for downscaling to preserve detail
                    // Use NearestNeighbor for upscaling (if needed)
                    g.InterpolationMode = bmp.Width > targetWidth 
                        ? System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic 
                        : System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                        
                    g.DrawImage(bmp, 0, 0, targetWidth, targetHeight);
                }
                resized.Save(pngPath, System.Drawing.Imaging.ImageFormat.Png);
            }
            else
            {
                bmp.Save(pngPath, System.Drawing.Imaging.ImageFormat.Png);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error converting {blpPath}: {ex.Message}");
            return false;
        }
    }
    
    private (int minX, int minY, int maxX, int maxY, int width, int height)? StitchHeightmapsToPng(
        string imagesDir, string mapName, string outputPath, IProgress<string>? progress, string tileSuffix)
    {
        try
        {
            // Find all heightmap tiles
            var pattern = $"{mapName}_*_*{tileSuffix}.png";
            var files = Directory.GetFiles(imagesDir, pattern);
            if (files.Length == 0) return null;
            
            // Parse tile coordinates
            var tiles = new List<(int x, int y, string path)>();
            foreach (var file in files)
            {
                var name = Path.GetFileNameWithoutExtension(file);
                var parts = name.Replace($"{mapName}_", "").Replace(tileSuffix, "").Split('_');
                if (parts.Length >= 2 && int.TryParse(parts[0], out int x) && int.TryParse(parts[1], out int y))
                {
                    tiles.Add((x, y, file));
                }
            }
            
            if (tiles.Count == 0) return null;
            
            int minX = tiles.Min(t => t.x);
            int maxX = tiles.Max(t => t.x);
            int minY = tiles.Min(t => t.y);
            int maxY = tiles.Max(t => t.y);
            
            int tilesWide = maxX - minX + 1;
            int tilesHigh = maxY - minY + 1;
            
            // Each tile is 256x256
            int outputWidth = tilesWide * 256;
            int outputHeight = tilesHigh * 256;
            int tileSize = 256;
            
            progress?.Report($"Stitching {tiles.Count} heightmaps into {outputWidth}x{outputHeight} PNG...");
            
            using var canvas = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.L16>(outputWidth, outputHeight);
            
            foreach (var (x, y, path) in tiles)
            {
                try
                {
                    using var tile = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.L16>(path);
                    if (tile.Width != tileSize || tile.Height != tileSize)
                    {
                        tile.Mutate(ctx => ctx.Resize(tileSize, tileSize));
                    }
                    
                    // Copy tile to canvas
                    canvas.Mutate(ctx => ctx.DrawImage(tile, new SixLabors.ImageSharp.Point((x - minX) * tileSize, (y - minY) * tileSize), 1f));
                }
                catch (Exception ex)
                {
                    progress?.Report($"Warning: Failed to load heightmap {path}: {ex.Message}");
                }
            }
            
            canvas.SaveAsPng(outputPath);

            if (outputWidth > 2048 || outputHeight > 2048)
            {
                var dir = Path.GetDirectoryName(outputPath) ?? ".";
                var name = Path.GetFileNameWithoutExtension(outputPath);
                var resizedDir = Path.Combine(dir, "resized");
                Directory.CreateDirectory(resizedDir);
                int w50 = outputWidth / 2;
                int h50 = outputHeight / 2;
                using var scaled50 = canvas.Clone(ctx => ctx.Resize(w50, h50));
                var path50 = Path.Combine(resizedDir, $"{name}_50pct.png");
                scaled50.SaveAsPng(path50);
            }
            return (minX, minY, maxX, maxY, outputWidth, outputHeight);
        }
        catch (Exception ex)
        {
            progress?.Report($"Error stitching heightmaps: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Get model bounding box from MDX or WMO file.
    /// </summary>
    private (float[] Min, float[] Max)? GetModelBounds(string modelPath, IArchiveReader archiveReader, IReadOnlyList<string> searchPaths)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return null;

        string cacheKey = NormalizeModelPath(modelPath).ToLowerInvariant();
        if (_modelBoundsCache.TryGetValue(cacheKey, out var cached))
            return cached;

        try
        {
            foreach (string candidatePath in EnumerateModelPathCandidates(modelPath))
            {
                byte[]? data = ReadVirtualAssetBytes(searchPaths, candidatePath, archiveReader);
                if (data is null || data.Length < 16)
                    continue;

                (float[] Min, float[] Max)? bounds = TryReadBoundsFromModelBytes(data, candidatePath);
                if (!bounds.HasValue)
                    continue;

                _modelBoundsCache[cacheKey] = bounds;
                string candidateCacheKey = NormalizeModelPath(candidatePath).ToLowerInvariant();
                _modelBoundsCache[candidateCacheKey] = bounds;
                return bounds;
            }
        }
        catch
        {
        }

        _modelBoundsCache[cacheKey] = null;
        return null;
    }

    private static (float[] Min, float[] Max)? TryReadBoundsFromModelBytes(byte[] data, string sourcePath)
    {
        string extension = Path.GetExtension(sourcePath);
        bool preferWmo = extension.Equals(".wmo", StringComparison.OrdinalIgnoreCase);

        if (preferWmo)
        {
            (float[] Min, float[] Max)? wmoBounds = TryReadWmoBounds(data, sourcePath);
            if (wmoBounds.HasValue)
                return wmoBounds;

            return TryReadMdxBounds(data, sourcePath);
        }

        (float[] Min, float[] Max)? mdxBounds = TryReadMdxBounds(data, sourcePath);
        if (mdxBounds.HasValue)
            return mdxBounds;

        return TryReadWmoBounds(data, sourcePath);
    }

    private static (float[] Min, float[] Max)? TryReadMdxBounds(byte[] data, string sourcePath)
    {
        try
        {
            using MemoryStream stream = new(data, writable: false);
            var summary = MdxSummaryReader.Read(stream, sourcePath);

            Vector3? min = summary.Collision?.BoundsMin ?? summary.BoundsMin;
            Vector3? max = summary.Collision?.BoundsMax ?? summary.BoundsMax;
            if (!min.HasValue || !max.HasValue)
                return null;

            if (!TryConvertBounds(min.Value, max.Value, out float[] boundsMin, out float[] boundsMax))
                return null;

            return (boundsMin, boundsMax);
        }
        catch
        {
            return null;
        }
    }

    private static (float[] Min, float[] Max)? TryReadWmoBounds(byte[] data, string sourcePath)
    {
        try
        {
            using MemoryStream stream = new(data, writable: false);
            var summary = WmoSummaryReader.Read(stream, sourcePath);
            if (!TryConvertBounds(summary.BoundsMin, summary.BoundsMax, out float[] boundsMin, out float[] boundsMax))
                return null;

            return (boundsMin, boundsMax);
        }
        catch
        {
            return null;
        }
    }

    private static bool TryConvertBounds(Vector3 min, Vector3 max, out float[] boundsMin, out float[] boundsMax)
    {
        boundsMin = [];
        boundsMax = [];

        if (!IsFinite(min.X) || !IsFinite(min.Y) || !IsFinite(min.Z) ||
            !IsFinite(max.X) || !IsFinite(max.Y) || !IsFinite(max.Z))
        {
            return false;
        }

        // Reject obviously corrupt AABBs from malformed assets.
        const float maxAbs = 250_000f;
        if (Math.Abs(min.X) > maxAbs || Math.Abs(min.Y) > maxAbs || Math.Abs(min.Z) > maxAbs ||
            Math.Abs(max.X) > maxAbs || Math.Abs(max.Y) > maxAbs || Math.Abs(max.Z) > maxAbs)
        {
            return false;
        }

        if (min.X > max.X || min.Y > max.Y || min.Z > max.Z)
            return false;

        boundsMin = [min.X, min.Y, min.Z];
        boundsMax = [max.X, max.Y, max.Z];
        return true;
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    internal static string? TryResolveLooseAssetPath(IEnumerable<string> searchPaths, string virtualPath)
    {
        ArgumentNullException.ThrowIfNull(searchPaths);
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);

        string normalizedPath = NormalizeVirtualAssetPath(virtualPath);
        string relativePath = normalizedPath.Replace('\\', Path.DirectorySeparatorChar);

        foreach (string basePath in searchPaths)
        {
            if (string.IsNullOrWhiteSpace(basePath))
                continue;

            string fullPath = Path.Combine(basePath, relativePath);
            if (File.Exists(fullPath))
                return fullPath;

            string mpqPath = fullPath + ".MPQ";
            if (File.Exists(mpqPath))
                return mpqPath;
        }

        return null;
    }

    internal static byte[]? ReadVirtualAssetBytes(IEnumerable<string> searchPaths, string virtualPath, IArchiveReader archiveReader)
    {
        ArgumentNullException.ThrowIfNull(archiveReader);

        if (TryResolveLooseAssetPath(searchPaths, virtualPath) is { } loosePath)
        {
            if (loosePath.EndsWith(".MPQ", StringComparison.OrdinalIgnoreCase))
                return AlphaArchiveReader.ReadFromMpq(loosePath);

            return File.ReadAllBytes(loosePath);
        }

        return archiveReader.ReadFile(NormalizeVirtualAssetPath(virtualPath));
    }

    private static IEnumerable<string> EnumerateModelPathCandidates(string modelPath)
    {
        string normalized = NormalizeModelPath(modelPath);
        var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            normalized
        };

        string extension = Path.GetExtension(normalized);
        if (extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            candidates.Add(Path.ChangeExtension(normalized, ".m2"));
        }
        else if (extension.Equals(".m2", StringComparison.OrdinalIgnoreCase))
        {
            candidates.Add(Path.ChangeExtension(normalized, ".mdx"));
        }
        else if (string.IsNullOrEmpty(extension))
        {
            candidates.Add(normalized + ".m2");
            candidates.Add(normalized + ".mdx");
            candidates.Add(normalized + ".wmo");
        }

        return candidates;
    }

    private static string NormalizeVirtualAssetPath(string path)
    {
        return path.Replace('/', '\\').TrimStart('\\');
    }

    private static string NormalizeModelPath(string path)
    {
        return NormalizeVirtualAssetPath(path);
    }
    
    private int FindChunkOffset(byte[] data, string chunkId)
    {
        if (data.Length < 8) return -1;
        byte[] searchBytes = System.Text.Encoding.ASCII.GetBytes(chunkId);
        for (int i = 0; i <= data.Length - 8; i++)
        {
            if (data[i] == searchBytes[0] && data[i+1] == searchBytes[1] &&
                data[i+2] == searchBytes[2] && data[i+3] == searchBytes[3])
                return i;
        }
        return -1;
    }
    
    /// <summary>
    /// Extract bounding box from MDX file via AlphaArchiveReader (per-asset MPQ).
    /// </summary>
    private (float[] Min, float[] Max)? GetMdxBounds(string mdxMpqPath)
    {
        if (!File.Exists(mdxMpqPath)) return null;
        
        // Check cache
        var key = mdxMpqPath.ToLowerInvariant();
        if (_modelBoundsCache.TryGetValue(key, out var cached))
            return cached;
        
        try
        {
            var data = AlphaArchiveReader.ReadFromMpq(mdxMpqPath);
            if (data == null || data.Length < 100)
            {
                _modelBoundsCache[key] = null;
                return null;
            }
            
            // MDX header: bounding box typically at offset 64-88
            // Format: 6 floats (min xyz, max xyz)
            int bbOffset = 64;
            if (data.Length >= bbOffset + 24)
            {
                var boundsMin = new float[3];
                var boundsMax = new float[3];
                boundsMin[0] = BitConverter.ToSingle(data, bbOffset);
                boundsMin[1] = BitConverter.ToSingle(data, bbOffset + 4);
                boundsMin[2] = BitConverter.ToSingle(data, bbOffset + 8);
                boundsMax[0] = BitConverter.ToSingle(data, bbOffset + 12);
                boundsMax[1] = BitConverter.ToSingle(data, bbOffset + 16);
                boundsMax[2] = BitConverter.ToSingle(data, bbOffset + 20);
                
                // Sanity check
                if (!float.IsNaN(boundsMin[0]) && !float.IsNaN(boundsMax[0]) &&
                    Math.Abs(boundsMin[0]) < 10000 && Math.Abs(boundsMax[0]) < 10000)
                {
                    var result = (boundsMin, boundsMax);
                    _modelBoundsCache[key] = result;
                    return result;
                }
            }
        }
        catch { }
        
        _modelBoundsCache[key] = null;
        return null;
    }
    
    /// <summary>
    /// Extract bounding box from WMO file via AlphaArchiveReader (per-asset MPQ).
    /// </summary>
    private (float[] Min, float[] Max)? GetWmoBounds(string wmoMpqPath)
    {
        if (!File.Exists(wmoMpqPath)) return null;
        
        // Check cache
        var key = wmoMpqPath.ToLowerInvariant();
        if (_modelBoundsCache.TryGetValue(key, out var cached))
            return cached;
        
        try
        {
            var data = AlphaArchiveReader.ReadFromMpq(wmoMpqPath);
            if (data == null || data.Length < 100)
            {
                _modelBoundsCache[key] = null;
                return null;
            }
            
            // WMO: Find MOHD chunk, bounding box at offset 28 from chunk data start
            int mohdOffset = FindChunkOffset(data, "MOHD");
            if (mohdOffset >= 0 && mohdOffset + 8 + 52 <= data.Length)
            {
                int dataStart = mohdOffset + 8; // Skip chunk ID + size
                var boundsMin = new float[3];
                var boundsMax = new float[3];
                boundsMin[0] = BitConverter.ToSingle(data, dataStart + 28);
                boundsMin[1] = BitConverter.ToSingle(data, dataStart + 32);
                boundsMin[2] = BitConverter.ToSingle(data, dataStart + 36);
                boundsMax[0] = BitConverter.ToSingle(data, dataStart + 40);
                boundsMax[1] = BitConverter.ToSingle(data, dataStart + 44);
                boundsMax[2] = BitConverter.ToSingle(data, dataStart + 48);
                
                // Sanity check
                if (!float.IsNaN(boundsMin[0]) && !float.IsNaN(boundsMax[0]) &&
                    Math.Abs(boundsMin[0]) < 100000 && Math.Abs(boundsMax[0]) < 100000)
                {
                    var result = (boundsMin, boundsMax);
                    _modelBoundsCache[key] = result;
                    return result;
                }
            }
        }
        catch { }
        
        _modelBoundsCache[key] = null;
        return null;
    }

}
