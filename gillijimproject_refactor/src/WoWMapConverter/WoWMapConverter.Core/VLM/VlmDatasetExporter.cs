using System.Collections.Concurrent;
using System.Numerics;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WoWMapConverter.Core.Formats.Liquids;
using WoWMapConverter.Core.Services;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
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

    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
    };
    
    private readonly ConcurrentDictionary<string, (float[] Min, float[] Max)?> _modelBoundsCache = new();

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
        bool generateDepth = false)
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

        // Try multiple WDT locations
        var wdtPaths = new[]
        {
            Path.Combine(dataPath, "World", "Maps", mapName, $"{mapName}.wdt"),
            Path.Combine(clientPath, "Data", "World", "Maps", mapName, $"{mapName}.wdt"),
            Path.Combine(clientPath, "World", "Maps", mapName, $"{mapName}.wdt"),
        };

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
            var wdtInternalPath = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            if (archiveCatalog.FileExists(wdtInternalPath))
            {
                wdtData = archiveCatalog.ReadFile(wdtInternalPath);
                if (wdtData != null)
                {
                    var tempWdt = Path.Combine(outputDir, $"{mapName}.wdt");
                    await File.WriteAllBytesAsync(tempWdt, wdtData);
                    wdtPath = tempWdt;
                    progress?.Report($"Found WDT in MPQ archive: {wdtInternalPath}");
                }
            }
        }

        if (wdtPath == null)
        {
            progress?.Report($"WDT not found for map '{mapName}'.");
            progress?.Report($"Searched paths:");
            foreach (var p in wdtPaths)
                progress?.Report($"  - {p} (and {p}.MPQ)");
            progress?.Report($"Also searched in loaded MPQ archives for: World\\Maps\\{mapName}\\{mapName}.wdt");
            progress?.Report("Ensure MPQ archives (world.mpq, terrain.mpq, etc.) are in the Data folder.");
            return new VlmExportResult(0, 0, 0, outputDir);
        }


        var mapDirectoryLookup = new MapDirectoryLookup();
        mapDirectoryLookup.Load(searchPaths, archiveCatalog);

        string mapDirectory = mapDirectoryLookup.ResolveDirectory(mapName) ?? mapName;
        if (!string.Equals(mapDirectory, mapName, StringComparison.Ordinal))
        {
            Console.WriteLine($"Resolved map '{mapName}' to directory '{mapDirectory}' via Map.dbc");
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
                     var wdlMpqDiscovered = Path.Combine(path, "World", "Maps", mapName, $"{mapName}.wdl.MPQ");
                     if (!File.Exists(wdlMpqDiscovered)) 
                        wdlMpqDiscovered = Path.Combine(path, "World", "Maps", mapName, $"{mapName}.wdl.mpq");
                        
                     if (File.Exists(wdlMpqDiscovered))
                     {
                         var wdlExpectedPath = Path.Combine(path, "World", "Maps", mapName, $"{mapName}.wdl");
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
                 }

                 // 3. Try standard MPQ path (Modern/Legacy)
                 if (!loaded)
                 {
                    var wdlInternalPath = $"World\\Maps\\{mapName}\\{mapName}.wdl";
                    if (archiveCatalog.FileExists(wdlInternalPath))
                    {
                        var wdlBytes = archiveCatalog.ReadFile(wdlInternalPath);
                        if (wdlBytes != null)
                        {
                            wdlData = WdlParser.Parse(wdlBytes);
                            progress?.Report($"Loaded WDL data from MPQ internal: {wdlInternalPath}");
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
        var tilesToProcess = SelectTilesForProcessing(existingTiles, limit, isAlphaFormat, mapDirectory, archiveCatalog);
        
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
                var minimapPath = FindMinimapTile(searchPaths, archiveCatalog, md5Index, mapDirectory, x, y);
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
                    var adtBase = $"World\\Maps\\{mapName}\\{mapName}_{x}_{y}";
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
                        shadowsDir, masksDir, allTextures, archiveCatalog, groundEffectLookup, wdlTile, wdtMphdFlags);
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

                await WriteBinaryTile(sample, datasetDir);

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

        // Stitch chunk data into tile-level images
        if (tilesExported > 0)
        {
            progress?.Report("Stitching tile images...");
            var stitchedDir = Path.Combine(outputDir, "stitched");
            // liquidsDir already declared/created at start
            Directory.CreateDirectory(stitchedDir);
            
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
                        string? objectVisibilityMaskPath = null;
                        string? noObjectMinimapPath = null;
                        float lMin = 0f, lMax = 0f;

                        string? sourceMinimapPath = sample.ImagePath;
                        if (!string.IsNullOrWhiteSpace(sourceMinimapPath) && !Path.IsPathRooted(sourceMinimapPath))
                            sourceMinimapPath = Path.Combine(outputDir, sourceMinimapPath);

                        if (!string.IsNullOrWhiteSpace(sourceMinimapPath) && File.Exists(sourceMinimapPath))
                        {
                            using Image<Rgba32> minimapForMask = Image.Load<Rgba32>(sourceMinimapPath);
                            if (minimapForMask.Width > 0 && minimapForMask.Height > 0)
                            {
                                byte[] objectMask = BuildObjectVisibilityMask(sample.TerrainData, minimapForMask.Width, minimapForMask.Height);
                                if (objectMask.Length > 0)
                                {
                                    objectVisibilityMaskPath = $"images/{tileName}_object_visibility_mask.png";
                                    await File.WriteAllBytesAsync(Path.Combine(outputDir, objectVisibilityMaskPath), objectMask);

                                    byte[] noObjectBytes = SynthesizeMaskedMinimap(sourceMinimapPath, objectMask);
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
                                lMaskPath = $"liquids/{tileName}_liq_mask.png";
                                await File.WriteAllBytesAsync(Path.Combine(outputDir, lMaskPath), liqMask);

                                // Generate a synthetic no-liquid minimap for training.
                                if (!string.IsNullOrWhiteSpace(sourceMinimapPath) && File.Exists(sourceMinimapPath))
                                {
                                    var noLiqBytes = SynthesizeMaskedMinimap(sourceMinimapPath, liqMask);
                                    if (noLiqBytes.Length > 0)
                                    {
                                        noLiquidMinimapPath = $"images/{tileName}_no_liquid.png";
                                        await File.WriteAllBytesAsync(Path.Combine(outputDir, noLiquidMinimapPath), noLiqBytes);
                                    }
                                }
                            }
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
                            ObjectVisibilityMaskPath = objectVisibilityMaskPath,
                            NoObjectMinimapPath = noObjectMinimapPath,
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

            // Export Tileset Textures
            progress?.Report($"Exporting {allTextures.Count} unique tileset textures...");
            var tilesetsDir = Path.Combine(outputDir, "tilesets");
            Directory.CreateDirectory(tilesetsDir);
            
            int textureCount = 0;
            foreach (var texture in allTextures.Keys)
            {
                var texName = Path.GetFileName(texture); // e.g. "grass.blp"
                var pngName = Path.ChangeExtension(texName, ".png");
                var pngPath = Path.Combine(tilesetsDir, pngName);
                
                if (!File.Exists(pngPath))
                {
                    // Try to find the BLP in client
                    // 'texture' might be full path or relative. 
                    // Usually it is relative like "Textures\Grass.blp"
                    
                    // Try common locations
                    // Try common locations
                    var candidates = new List<string>();
                    
                    // Try MPQ first (most likely for tilesets)
                    // Ensure texture has backslashes for consistency, though NativeMpq handles it
                    candidates.Add($"MPQ:{texture}");
                    
                    // Also try looking on disk (Data folder)
                    candidates.Add(Path.Combine(dataPath, texture));

                    
                    bool converted = false;
                     foreach (var path in candidates)
                    {
                         if (ConvertBlpToPng(path, pngPath, archiveCatalog))
                         {
                             converted = true;
                             break;
                         }
                    }

                    if (converted) textureCount++;
                }
            }
            progress?.Report($"Exported {textureCount} textures");
        }

        // Generate global heightmaps for each tile (per-map min/max)
        if (tilesExported > 0)
        {
            progress?.Report("Generating global-normalized heightmaps...");
            await GenerateGlobalHeightmapsAsync(datasetDir, outputDir, progress);
        }

        // Stitch full world map images
        if (tilesExported > 0)
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

        progress?.Report($"Export complete: {tilesExported} tiles exported, {tilesSkipped} skipped");
        return new VlmExportResult(tilesExported, tilesSkipped, allTextures.Count, outputDir);
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
                
                objects.Add(new VlmObjectPlacement(name, nameId, uniqueId, px, py, pz, rx, ry, rz, scale / 1024f, "m2", boundsMin, boundsMax));
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
                
                objects.Add(new VlmObjectPlacement(name, nameId, uniqueId, px, py, pz, rx, ry, rz, scale / 1024f, "wmo", boundsMin, boundsMax));
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
            ObjectVisibilityMaskPath: null,
            NoObjectMinimapPath: null,
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

            if (liquids.Count == 0 && legacyMclqLiquids.Count > 0)
            {
                liquids.AddRange(legacyMclqLiquids);
                Console.WriteLine($"[DEBUG] Parsed {legacyMclqLiquids.Count} MCLQ liquid chunks for {tileName}");
            }

            AppendObjectPlacementsFromRaw(objectPlacements, mddfRawForPlacements, m2NamesForPlacements, "m2", archiveReader);
            AppendObjectPlacementsFromRaw(objectPlacements, modfRawForPlacements, wmoNamesForPlacements, "wmo", archiveReader);
            
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
                ObjectVisibilityMaskPath: null,
                NoObjectMinimapPath: null,
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
        IArchiveReader archiveReader)
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
                var bounds = GetModelBounds(fullPath, archiveReader);
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
                boundsMax));
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

    private static List<int> SelectTilesForProcessing(List<int> existingTiles, int limit, bool isAlphaFormat, string mapDirectory, IArchiveReader archiveReader)
    {
        if (existingTiles.Count == 0)
            return existingTiles;

        if (limit <= 0 || limit >= existingTiles.Count)
            return existingTiles.ToList();

        if (!isAlphaFormat)
        {
            var rankedTiles = existingTiles
                .Select(tileIndex =>
                {
                    int x = tileIndex % 64;
                    int y = tileIndex / 64;
                    int contentScore = ScoreLkTileContent(mapDirectory, x, y, archiveReader);
                    return (tileIndex, contentScore);
                })
                .ToList();

            if (rankedTiles.Any(entry => entry.contentScore > 0))
            {
                var fallbackOrder = OrderTilesByCenter(existingTiles);
                var fallbackIndex = fallbackOrder
                    .Select((tileIndex, index) => (tileIndex, index))
                    .ToDictionary(entry => entry.tileIndex, entry => entry.index);

                return rankedTiles
                    .OrderByDescending(entry => entry.contentScore)
                    .ThenBy(entry => fallbackIndex[entry.tileIndex])
                    .Take(limit)
                    .Select(entry => entry.tileIndex)
                    .ToList();
            }
        }

        return OrderTilesByCenter(existingTiles).Take(limit).ToList();
    }

    private static int ScoreLkTileContent(string mapDirectory, int x, int y, IArchiveReader archiveReader)
    {
        string adtBase = $"World/Maps/{mapDirectory}/{mapDirectory}_{x}_{y}";

        int score = 0;

        try
        {
            byte[]? rootAdt = archiveReader.ReadFile($"{adtBase}.adt");
            if (rootAdt is { Length: > 0 })
            {
                if (ContainsTopLevelChunk(rootAdt, "MH2O"))
                    score += 100;

                if (ContainsChunkToken(rootAdt, "MCLQ"))
                    score += 80;

                if (ContainsTopLevelChunk(rootAdt, "MDDF") || ContainsTopLevelChunk(rootAdt, "MODF"))
                    score += 10;
            }
        }
        catch
        {
        }

        try
        {
            byte[]? objAdt = archiveReader.ReadFile($"{adtBase}_obj0.adt");
            if (objAdt is { Length: > 0 } && (ContainsTopLevelChunk(objAdt, "MDDF") || ContainsTopLevelChunk(objAdt, "MODF")))
                score += 10;
        }
        catch
        {
        }

        return score;
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

                if (found)
                {
                    // Found a mapping!
                    // The 'hashed' value is the filename in the MPQ.
                    var mpqKey = hashed.Replace("/", "\\");
                    
                    if (debugTile)
                    {
                        Console.WriteLine($"  MPQ key to check: '{mpqKey}'");
                    }
                    
                    if (archiveReader.FileExists(mpqKey))
                    {
                        Console.WriteLine($"[Match] Translated '{candidate}' -> '{hashed}' (Found in MPQ)");
                        return $"MPQ:{mpqKey}";
                    }
                    
                    // Also check disk
                    foreach (var bp in searchPaths)
                    {
                         var hp = Path.Combine(bp, hashed);
                         if (File.Exists(hp))
                         {
                             if (debugTile) Console.WriteLine($"  Found on disk: {hp}");
                             return hp;
                         }
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
             foreach (var basePath in searchPaths)
             {
                 var fullPath = Path.Combine(basePath, candidate);
                 if (File.Exists(fullPath)) 
                 {
                     Console.WriteLine($"Found minimap on disk: {fullPath}");
                     return fullPath;
                 }
                 if (File.Exists(fullPath + ".MPQ"))
                 {
                     return fullPath + ".MPQ";
                 }
             }
             
             // Check MPQ by plain name (fallback)
             var mpqPlainKey = candidate.Replace("/", "\\");
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

    private byte[] RenderMccvImage(Dictionary<int, byte[]> mccvDict, int size)
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
                 return (cData[baseIdx] / 255.0f, cData[baseIdx + 1] / 255.0f, cData[baseIdx + 2] / 255.0f, cData[baseIdx + 3] / 255.0f);
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
                    image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(127, 127, 127, 255);
                    continue;
                }

                float lx = Math.Clamp(cx - cIx, 0f, 1f);
                float ly = Math.Clamp(cy - cIy, 0f, 1f);
                var (r, g, b, a) = SampleColor(cData, lx, ly);
                
                image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(r, g, b, a);
            }
        }

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    private static byte[] BuildObjectVisibilityMask(VlmTerrainData terrainData, int width, int height)
    {
        if (width <= 0 || height <= 0 || terrainData.Objects.Count == 0)
            return Array.Empty<byte>();

        using Image<L8> objectMask = new(width, height);
        bool hasAnyCoverage = false;
        var projectedWmoObjects = new Dictionary<uint, (int CenterX, int CenterY, int Radius)>();

        foreach (VlmObjectPlacement obj in terrainData.Objects)
        {
            if (!obj.Category.Contains("wmo", StringComparison.OrdinalIgnoreCase))
                continue;

            if (!TryProjectObjectToTilePixel(obj, terrainData.AdtTile, width, height, out int centerX, out int centerY))
                continue;

            int radius = EstimateObjectRadiusPixels(obj, width, height);
            if (radius <= 0)
                continue;

            projectedWmoObjects[obj.UniqueId] = (centerX, centerY, radius);
        }

        if (terrainData.ShadowAnalysis != null && projectedWmoObjects.Count > 0)
        {
            float scaleX = width / 1024f;
            float scaleY = height / 1024f;

            foreach (VlmChunkShadowAnalysis chunk in terrainData.ShadowAnalysis)
            {
                int chunkX = chunk.ChunkIndex % 16;
                int chunkY = chunk.ChunkIndex / 16;

                foreach (VlmShadowRegion region in chunk.Regions)
                {
                    if (region.CandidateObjectIds.Length == 0)
                        continue;

                    if (region.BoundingBoxMinPixels.Length < 2 || region.BoundingBoxMaxPixels.Length < 2)
                        continue;

                    int x0Shadow = (chunkX * 64) + region.BoundingBoxMinPixels[0];
                    int y0Shadow = (chunkY * 64) + region.BoundingBoxMinPixels[1];
                    int x1Shadow = (chunkX * 64) + region.BoundingBoxMaxPixels[0];
                    int y1Shadow = (chunkY * 64) + region.BoundingBoxMaxPixels[1];

                    int x0 = Math.Clamp((int)MathF.Floor(x0Shadow * scaleX), 0, width - 1);
                    int y0 = Math.Clamp((int)MathF.Floor(y0Shadow * scaleY), 0, height - 1);
                    int x1 = Math.Clamp((int)MathF.Ceiling((x1Shadow + 1) * scaleX), 0, width);
                    int y1 = Math.Clamp((int)MathF.Ceiling((y1Shadow + 1) * scaleY), 0, height);

                    if (x1 <= x0 || y1 <= y0)
                        continue;

                    // Keep shadow-derived masking local to each projected WMO footprint.
                    // This avoids wiping large terrain regions when a shadow region is broad.
                    foreach (uint candidateObjectId in region.CandidateObjectIds)
                    {
                        if (!projectedWmoObjects.TryGetValue(candidateObjectId, out var projected))
                            continue;

                        int localRadius = Math.Max(4, projected.Radius * 2);
                        int localX0 = Math.Max(0, projected.CenterX - localRadius);
                        int localY0 = Math.Max(0, projected.CenterY - localRadius);
                        int localX1 = Math.Min(width, projected.CenterX + localRadius + 1);
                        int localY1 = Math.Min(height, projected.CenterY + localRadius + 1);

                        int clippedX0 = Math.Max(x0, localX0);
                        int clippedY0 = Math.Max(y0, localY0);
                        int clippedX1 = Math.Min(x1, localX1);
                        int clippedY1 = Math.Min(y1, localY1);

                        if (clippedX1 <= clippedX0 || clippedY1 <= clippedY0)
                            continue;

                        FillMaskRectangle(objectMask, clippedX0, clippedY0, clippedX1, clippedY1);
                        hasAnyCoverage = true;
                    }
                }
            }
        }

        foreach ((int centerX, int centerY, int radius) in projectedWmoObjects.Values)
        {
            DrawFilledCircle(objectMask, centerX, centerY, radius);
            hasAnyCoverage = true;
        }

        if (!hasAnyCoverage)
            return Array.Empty<byte>();

        using MemoryStream ms = new();
        objectMask.SaveAsPng(ms);
        return ms.ToArray();
    }

    private static void FillMaskRectangle(Image<L8> image, int minX, int minY, int maxXExclusive, int maxYExclusive)
    {
        for (int y = minY; y < maxYExclusive; y++)
        {
            for (int x = minX; x < maxXExclusive; x++)
                image[x, y] = new L8(255);
        }
    }

    private static void DrawFilledCircle(Image<L8> image, int centerX, int centerY, int radius)
    {
        if (radius <= 0)
            return;

        int minX = Math.Max(0, centerX - radius);
        int maxX = Math.Min(image.Width - 1, centerX + radius);
        int minY = Math.Max(0, centerY - radius);
        int maxY = Math.Min(image.Height - 1, centerY + radius);
        int radiusSquared = radius * radius;

        for (int y = minY; y <= maxY; y++)
        {
            int dy = y - centerY;
            for (int x = minX; x <= maxX; x++)
            {
                int dx = x - centerX;
                if ((dx * dx) + (dy * dy) > radiusSquared)
                    continue;

                image[x, y] = new L8(255);
            }
        }
    }

    private static int EstimateObjectRadiusPixels(VlmObjectPlacement obj, int width, int height)
    {
        float pixelsPerWorld = MathF.Min(width, height) / TileSize;
        float scale = float.IsFinite(obj.Scale) && obj.Scale > 0f ? obj.Scale : 1f;

        if (obj.BoundsMin != null && obj.BoundsMax != null && obj.BoundsMin.Length >= 3 && obj.BoundsMax.Length >= 3)
        {
            float halfWidthWorld = MathF.Abs(obj.BoundsMax[0] - obj.BoundsMin[0]) * 0.5f * scale;
            float halfDepthWorld = MathF.Abs(obj.BoundsMax[2] - obj.BoundsMin[2]) * 0.5f * scale;
            float radiusWorld = MathF.Max(halfWidthWorld, halfDepthWorld);
            return Math.Max(2, (int)MathF.Round(radiusWorld * pixelsPerWorld));
        }

        if (obj.BoundsMin != null && obj.BoundsMax != null && obj.BoundsMin.Length >= 2 && obj.BoundsMax.Length >= 2)
        {
            float halfWidthWorld = MathF.Abs(obj.BoundsMax[0] - obj.BoundsMin[0]) * 0.5f * scale;
            float halfDepthWorld = MathF.Abs(obj.BoundsMax[1] - obj.BoundsMin[1]) * 0.5f * scale;
            float radiusWorld = MathF.Max(halfWidthWorld, halfDepthWorld);
            return Math.Max(2, (int)MathF.Round(radiusWorld * pixelsPerWorld));
        }

        float baseRadiusWorld = obj.Category.Contains("wmo", StringComparison.OrdinalIgnoreCase) ? 6f : 3f;
        return Math.Max(2, (int)MathF.Round(baseRadiusWorld * scale * pixelsPerWorld));
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

        if (!TryParseTileCoordinates(adtTile, out int tileX, out int tileY))
            return false;

        List<(float U, float V)> candidates = new();
        if (MathF.Abs(obj.X) < 2f && MathF.Abs(obj.Y) < 2f)
        {
            candidates.Add((((obj.Y + 1f) * 0.5f), ((obj.X + 1f) * 0.5f)));
        }

        candidates.AddRange(BuildTileUvCandidates(obj.X, obj.Z, tileX, tileY));
        candidates.AddRange(BuildTileUvCandidates(obj.X, obj.Y, tileX, tileY));

        if (candidates.Count == 0)
            return false;

        (float U, float V) best = candidates[0];
        float bestOverflow = float.PositiveInfinity;

        foreach ((float u, float v) in candidates)
        {
            float overflow =
                MathF.Max(0f, -u) + MathF.Max(0f, u - 1f) +
                MathF.Max(0f, -v) + MathF.Max(0f, v - 1f);
            if (overflow < bestOverflow)
            {
                best = (u, v);
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

        centerX = Math.Clamp((int)MathF.Round(Math.Clamp(best.U, 0f, 1f) * (width - 1)), 0, width - 1);
        centerY = Math.Clamp((int)MathF.Round(Math.Clamp(best.V, 0f, 1f) * (height - 1)), 0, height - 1);
        return true;
    }

    private static IEnumerable<(float U, float V)> BuildTileUvCandidates(float worldA, float worldB, int tileX, int tileY)
    {
        yield return ((worldA / TileSize) - tileX, (worldB / TileSize) - tileY);
        yield return (((MapOrigin - worldB) / TileSize) - tileX, ((MapOrigin - worldA) / TileSize) - tileY);
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
    private (float[] Min, float[] Max)? GetModelBounds(string modelPath, IArchiveReader archiveReader)
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
                byte[]? data = archiveReader.ReadFile(candidatePath);
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

    private static string NormalizeModelPath(string path)
    {
        return path.Replace('/', '\\').TrimStart('\\');
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

    private async Task<string> WriteBinaryTile(VlmTerrainData data, string outputDir)
    {
        var binFilename = $"{data.AdtTile.Replace(".adt", "")}.bin";
        var binPath = Path.Combine(outputDir, binFilename);
        var relPath = binFilename; // root relative

        await using var fs = new FileStream(binPath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        // 1. Header (16 bytes)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("VLM1")); // Magic
        bw.Write((int)1); // Version
        bw.Write((int)0); // Flags
        bw.Write((int)256); // NumChunks

        // 2. Placeholder for Offset Table (256 * 8 bytes = 2048 bytes)
        long offsetTablePos = fs.Position;
        bw.Write(new byte[256 * 8]); // Fill with zeros

        // 3. Write Chunk Data
        var chunkOffsets = new (int offset, int size)[256];
        var heightDict = data.Heights?.ToDictionary(h => h.ChunkIndex, h => h.Heights) ?? new();
        var chunksDict = data.ChunkLayers?.ToDictionary(c => c.ChunkIndex) ?? new();
        var shadowDict = data.ShadowBits?.ToDictionary(s => s.ChunkIndex, s => Convert.FromBase64String(s.BitsBase64)) ?? new();

        for (int i = 0; i < 256; i++)
        {
            long chunkStart = fs.Position;
            
            // Heights (Required, 0-fill if missing)
            if (heightDict.TryGetValue(i, out var h) && h.Length == 145)
            {
                foreach (var val in h) bw.Write(val);
            }
            else
            {
                bw.Write(new byte[145 * 4]);
            }

            // Normals
            var layers = chunksDict.GetValueOrDefault(i);
            if (layers?.Normals != null && layers.Normals.Length == 145 * 3)
            {
                 foreach (var b in layers.Normals) bw.Write(b);
            }
            else
            {
                bw.Write(new byte[145 * 3]);
            }

            // MCCV
            if (layers?.MccvColors != null && layers.MccvColors.Length == 145 * 4)
            {
                bw.Write(layers.MccvColors);
            }
            else
            {
                 bw.Write(new byte[145 * 4]);
            }

            // Shadows (Packed 64 bytes)
            if (shadowDict.TryGetValue(i, out var sBits) && sBits.Length >= 512)
            {
                 if (sBits.Length == 512) bw.Write(sBits);
                 else bw.Write(sBits.Take(512).ToArray());
            }
            else
            {
                bw.Write(new byte[512]); // Empty shadow
            }

            // Alpha Layers (4 fixed slots, 4096 bytes each)
            if (layers != null && layers.Layers != null)
            {
                for (int l = 0; l < 4; l++)
                {
                    bool wrote = false;
                    if (l < layers.Layers.Length)
                    {
                        var lay = layers.Layers[l];
                        var alpha = lay.AlphaData;
                        
                        // Treat layer > 0 as having alpha. Layer 0 alpha is usually implied or handled by splat map.
                        // VLM protocol: We store 4 alpha channels. 
                        // If alpha is present, write 4096 bytes.
                        // If packed, expand? 
                        // For now we assume AlphaData is populated correctly uncompressed 4096 bytes.
                        // If null, write zeros.
                        if (l > 0 && alpha != null && alpha.Length == 4096)
                        {
                            bw.Write(alpha);
                            wrote = true;
                        }
                    }
                    if (!wrote) bw.Write(new byte[4096]);
                }
            }
            else
            {
                // Write 4 empty alpha slots (16 KB)
                bw.Write(new byte[4096 * 4]);
            }

            long chunkEnd = fs.Position;
            chunkOffsets[i] = ((int)chunkStart, (int)(chunkEnd - chunkStart));
        }

        // 4. Update Offset Table
        fs.Seek(offsetTablePos, SeekOrigin.Begin);
        for (int i = 0; i < 256; i++)
        {
            bw.Write(chunkOffsets[i].offset);
            bw.Write(chunkOffsets[i].size);
        }

        return relPath;
    }
}
