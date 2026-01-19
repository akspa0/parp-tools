using System.Collections.Concurrent;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WoWMapConverter.Core.Services;
using GillijimProject.WowFiles.Alpha;
using WdtAlpha = GillijimProject.WowFiles.Alpha.WdtAlpha;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// VLM Dataset Exporter - extracts ADT data for VLM training.
/// Uses AdtAlpha parser and McnkAlpha sub-chunk access.
/// </summary>
public class VlmDatasetExporter
{
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
    };
    
    // Cache for model bounding boxes: modelPath -> (boundsMin, boundsMax)
    private readonly ConcurrentDictionary<string, (float[] Min, float[] Max)?> _modelBoundsCache = new();
    private MpqArchiveService? _mpqService;

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
        var shadowsDir = Path.Combine(outputDir, "shadows");
        var masksDir = Path.Combine(outputDir, "masks");
        var liquidsDir = Path.Combine(outputDir, "liquids");
        var datasetDir = Path.Combine(outputDir, "dataset");
        
        Directory.CreateDirectory(imagesDir);
        Directory.CreateDirectory(shadowsDir);
        Directory.CreateDirectory(masksDir);
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
        
        // Initialize MPQ Service early so we can search archives for WDT
        using var mpqService = new MpqArchiveService();
        mpqService.LoadArchives(searchPaths);
        
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
            wdtData = AlphaMpqReader.ReadWithMpqFallback(tryPath);
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
            if (mpqService.FileExists(wdtInternalPath))
            {
                wdtData = mpqService.ReadFile(wdtInternalPath);
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

        


        // Initialize MapDbcService to check for strict directory names
        var mapDbcService = new MapDbcService();
        mapDbcService.Load(searchPaths, mpqService);
        
        string mapDirectory = mapDbcService.ResolveDirectory(mapName) ?? mapName;
        if (!string.Equals(mapDirectory, mapName, StringComparison.Ordinal))
        {
            Console.WriteLine($"Resolved map '{mapName}' to directory '{mapDirectory}' via Map.dbc");
        }

        // Initialize MD5 Translate Service (Legacy)
        Md5TranslateIndex? md5Index = null;
        if (Md5TranslateResolver.TryLoad(searchPaths, mpqService, out var loadedIndex))
        {
            md5Index = loadedIndex;
            Console.WriteLine($"Loaded MD5 Translate Index with {md5Index?.HashToPlain.Count} entries.");
        }

        // Initialize GroundEffectService
        var groundEffectService = new GroundEffectService();
        groundEffectService.Load(searchPaths, mpqService);  // Pass mpqService to GroundEffectService (need update)

        WdtAlpha wdt;
        try
        {
            wdt = new WdtAlpha(wdtPath);
        }
        catch (Exception ex)
        {
            progress?.Report($"Failed to parse WDT: {ex.Message}");
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
                         var wdlBytes = AlphaMpqReader.ReadFromMpq(
                             wdlMpqDiscovered,
                             AlphaMpqReader.BuildInternalNameCandidates(wdlExpectedPath));
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
                    if (mpqService.FileExists(wdlInternalPath))
                    {
                        var wdlBytes = mpqService.ReadFile(wdlInternalPath);
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

        var existingTiles = wdt.GetExistingAdtsNumbers();
        var adtOffsets = wdt.GetAdtOffsetsInMain();
        var mdnmNames = wdt.GetMdnmFileNames();
        var monmNames = wdt.GetMonmFileNames();
        
        progress?.Report($"Found {existingTiles.Count} tiles in WDT");
        
        // Detect WDT format using file size:
        // - Alpha 0.5.3 WDT: Large file (contains embedded ADT data, typically several MB)
        // - LK 3.3.5+ WDT: Small file (~32KB, only tile existence flags, ADTs are separate files)
        long wdtFileSize = new FileInfo(wdtPath).Length;
        bool isAlphaFormat = wdtFileSize > 100_000; // Alpha WDTs are typically > 1MB
        if (!isAlphaFormat)
        {
            progress?.Report($"Detected LK format WDT ({wdtFileSize:N0} bytes - separate ADT files in MPQ)");
        }
        else
        {
            progress?.Report($"Detected Alpha format WDT ({wdtFileSize:N0} bytes - embedded ADT data)");
        }

        int tilesExported = 0;
        int tilesSkipped = 0;
        var allTextures = new ConcurrentDictionary<string, byte>(StringComparer.OrdinalIgnoreCase);

        // Parallel tile processing with configurable degree of parallelism
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        var tilesToProcess = existingTiles.Take(limit).ToList();
        
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
                var minimapPath = FindMinimapTile(searchPaths, mpqService, md5Index, mapDirectory, x, y);
                if (minimapPath != null)
                {
                    var imageFileName = $"{tileName}.png";
                    var outputImagePath = Path.Combine(imagesDir, imageFileName);
                    
                    if (ConvertBlpToPng(minimapPath, outputImagePath, mpqService))
                    {
                        imageRelPath = $"images/{imageFileName}";
                    }
                }

                // Lookup WDL tile
                var wdlTile = wdlData?.Tiles[tileIndex];
                
                if (isAlphaFormat)
                {
                    // Alpha format: ADT data is embedded in WDT at offset
                    int adtOffset = tileIndex < adtOffsets.Count ? adtOffsets[tileIndex] : 0;
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
                        shadowsDir, masksDir, mdnmNames, monmNames, allTextures, groundEffectService, wdlTile, clientPath);
                }
                else
                {
                    // LK/Cata format: Read ADT from MPQ
                    // Check for Split ADT files (Cataclysm+)
                    var adtBase = $"World\\Maps\\{mapName}\\{mapName}_{x}_{y}";
                    var rootAdtPath = $"{adtBase}.adt";
                    var texAdtPath = $"{adtBase}_tex0.adt";
                    var objAdtPath = $"{adtBase}_obj0.adt";

                    var adtBytes = mpqService.ReadFile(rootAdtPath);
                    byte[]? texBytes = null;
                    byte[]? objBytes = null;

                    if (mpqService.FileExists(texAdtPath))
                    {
                        texBytes = mpqService.ReadFile(texAdtPath);
                        progress?.Report($"[Split ADT] Found tex0 for {tileName}");
                    }

                    if (mpqService.FileExists(objAdtPath))
                    {
                        objBytes = mpqService.ReadFile(objAdtPath);
                         progress?.Report($"[Split ADT] Found obj0 for {tileName}");
                    }
                    
                    if (adtBytes == null || adtBytes.Length == 0)
                    {
                        Interlocked.Increment(ref tilesSkipped);
                        return;
                    }

                    // Extract terrain data using LK/Modern ADT parsing
                    sample = await ExtractFromLkAdt(adtBytes, texBytes, objBytes, tileIndex, tileName, outputDir,
                        shadowsDir, masksDir, allTextures, mpqService, groundEffectService, wdlTile);
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
                    // Stitch Shadows & Alpha
                    var (shadowPath, alphaPaths) = await TileStitchingService.StitchTileAsync(
                        shadowsDir, masksDir, tileName, stitchedDir);

                    // Load JSON to update with stitched paths
                    var json = await File.ReadAllTextAsync(jsonPath);
                    var sample = JsonSerializer.Deserialize<VlmTrainingSample>(json);
                    
                    if (sample != null && sample.TerrainData != null)
                    {
                        // Stitch Liquids
                        string? lHeightPath = null;
                        string? lMaskPath = null;
                        float lMin = 0f, lMax = 0f;

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
                            }
                        }

                        // Update Terrain Data
                        var updatedTerrain = sample.TerrainData with
                        {
                            ShadowMaps = shadowPath != null ? new[] { Path.GetRelativePath(outputDir, shadowPath).Replace("\\", "/") } : null,
                            AlphaMasks = alphaPaths.Select(p => Path.GetRelativePath(outputDir, p).Replace("\\", "/")).ToArray(),
                            LiquidHeightPath = lHeightPath,
                            LiquidMaskPath = lMaskPath,
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
                    var candidates = new[]
                    {
                        Path.Combine(dataPath, texture)
                    };
                    
                    bool converted = false;
                    foreach (var path in candidates)
                    {
                         if (ConvertBlpToPng(path, pngPath))
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
        ConcurrentDictionary<string, byte> textureCollector, GroundEffectService? groundEffectService = null,
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
                            var shadow = ShadowMapService.ReadShadow(mcshBuf);
                            var shadowPng = ShadowMapService.ToPng(shadow);
                            var shadowFileName = $"{tileName}_c{chunkIndex}.png";
                            File.WriteAllBytes(Path.Combine(shadowsDir, shadowFileName), shadowPng);
                            shadowPaths.Add($"shadows/{shadowFileName}");
                            
                            // Store raw shadow bits
                            var rawShadowBytes = mcshBuf.Length >= 64 ? mcshBuf.Take(64).ToArray() : mcshBuf;
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
                                var alphaPng = AlphaMapService.ToPng(alphaData);
                                var alphaFileName = $"{tileName}_c{chunkIndex}_l{layer}.png";
                                File.WriteAllBytes(Path.Combine(masksDir, alphaFileName), alphaPng);
                                alphaPaths.Add($"masks/{alphaFileName}");
                                
                                // Store raw alpha bits (8-bit = 64x64 = 4096 bytes, or compressed = varies)
                                int alphaSize = isCompressed ? 4096 : 2048;
                                if (alphaOffset + alphaSize <= mcalBuf.Length)
                                {
                                    var rawAlpha = new byte[alphaSize];
                                    Array.Copy(mcalBuf, alphaOffset, rawAlpha, 0, alphaSize);
                                    layerAlphaBits[layer] = Convert.ToBase64String(rawAlpha);
                                }
                                
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
                            if (effectId > 0 && groundEffectService != null)
                            {
                                groundEffects = groundEffectService.GetDoodadsEffect(effectId);
                            }
                            
                            // Get raw alpha bits if available (only for layers > 0)
                            string? alphaBitsBase64 = layerAlphaBits.TryGetValue(layer, out var bits) ? bits : null;
                            
                            // Alpha path for this layer (layer > 0)
                            string? alphaPath = layer > 0 ? $"masks/{tileName}_c{chunkIndex}_l{layer}.png" : null;

                            layerList.Add(new VlmTextureLayer(textureId, texturePath, flags, alphaoffs, effectId, groundEffects, alphaBitsBase64, alphaPath));
                        }
                    }
                    
                    // Fallback: if no layers parsed but we have textures, create layers from nLayers count
                    if (layerList.Count == 0 && nLayers > 0 && textures.Count > 0)
                    {
                        for (int layer = 0; layer < nLayers && layer < 4 && layer < textures.Count; layer++)
                        {
                            string? alphaPath = layer > 0 ? $"masks/{tileName}_c{chunkIndex}_l{layer}.png" : null;
                            layerList.Add(new VlmTextureLayer((uint)layer, textures[layer], 0, 0, 0, null, null, alphaPath));
                        }
                    }
                    
                    // Shadow path for this chunk
                    string? chunkShadowPath = $"shadows/{tileName}_c{chunkIndex}.png";
                    if (!File.Exists(Path.Combine(shadowsDir, $"{tileName}_c{chunkIndex}.png"))) 
                        chunkShadowPath = null;
                    
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
                            string? maskPath = null;
                            
                            // Save heightmap PNG if exists
                            if (liquid.Heights != null)
                            {
                                var liquidsDir = Path.Combine(outputDir, "liquids");
                                Directory.CreateDirectory(liquidsDir);
                                
                                var heightPng = LiquidService.GenerateHeightPng(liquid.Heights, liquid.MinHeight, liquid.MaxHeight);
                                var heightFileName = $"{tileName}_c{chunkIndex}_liq_h.png";
                                var heightPath = Path.Combine(liquidsDir, heightFileName);
                                await File.WriteAllBytesAsync(heightPath, heightPng);
                                
                                // Set the mask path for JSON
                                maskPath = $"liquids/{heightFileName}";
                            }
                            
                            // Create updated liquid record with mask path
                            var liquidWithPath = new VlmLiquidData(
                                liquid.ChunkIndex, liquid.LiquidType, liquid.MinHeight, liquid.MaxHeight,
                                maskPath, liquid.Heights);
                            liquids.Add(liquidWithPath);
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
                
                // Extract bounding box from MDX via AlphaMpqReader
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
                
                // Extract bounding box from WMO via AlphaMpqReader
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

        var heightmapPath = await GenerateHeightmap(heights, tileName, outputDir);
        
        var normalmapPath = await GenerateNormalmap(chunkLayers, tileName, outputDir);
        var mccvMapPath = await GenerateMccvMap(chunkLayers, tileName, outputDir);
        
        return new VlmTerrainData(
            tileName,
            heights.ToArray(),
            chunkPositions,
            holes,
            heightmapPath,
            heightmapPath,
            null,
            normalmapPath, // NormalMapPath
            mccvMapPath,
            shadowPaths.Count > 0 ? shadowPaths.ToArray() : null,
            shadowBits.Count > 0 ? shadowBits.ToArray() : null,  // Raw shadow bit data
            alphaPaths.Count > 0 ? alphaPaths.ToArray() : null,
            null, // LiquidMaskPath
            null, // LiquidHeightPath
            0f,   // LiquidMinHeight
            0f,   // LiquidMaxHeight
            textures,
            chunkLayers.Count > 0 ? chunkLayers.ToArray() : null,
            liquids.Count > 0 ? liquids.ToArray() : null,
            objects,
            wdlHeights, // WDL Data
            heightMin == float.MaxValue ? 0 : heightMin,
            heightMax == float.MinValue ? 0 : heightMax,
            0,
            0
        );
    }

    /// <summary>
    /// Extract terrain data from LK/Modern ADT bytes.
    /// Supports Split ADTs (_tex0, _obj0) via optional buffers.
    /// </summary>
    private async Task<VlmTerrainData?> ExtractFromLkAdt(
        byte[] adtBytes, byte[]? texBytes, byte[]? objBytes, int tileIndex, string tileName,
        string outputDir, string shadowsDir, string masksDir,
        ConcurrentDictionary<string, byte> textureCollector, MpqArchiveService mpqService,
        GroundEffectService? groundEffectService = null, WdlParser.WdlTile? wdlTile = null)
    {
        var heights = new List<VlmChunkHeights>();
        var chunkPositions = new float[256 * 3];
        var holes = new int[256];
        var textures = new List<string>();
        var chunkLayers = new List<VlmChunkLayers>();
        var liquids = new List<VlmLiquidData>();
        var objects = new List<VlmObjectPlacement>();
        var shadowPaths = new List<string>();
        var shadowBits = new List<VlmChunkShadowBits>();
        
        float heightMin = float.MaxValue;
        float heightMax = float.MinValue;

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
            // Find MHDR to get chunk offsets
            int mhdrOffset = FindLkChunk(adtBytes, "MHDR");
            if (mhdrOffset < 0)
            {
                Console.WriteLine($"[LK ADT] MHDR not found in {tileName}");
                return null;
            }

            int mhdrDataStart = mhdrOffset + 8;
            int mhdrFlags = BitConverter.ToInt32(adtBytes, mhdrDataStart);
            bool useBigAlphamaps = (mhdrFlags & 4) != 0; // 0x4 = 4096 byte alpha

            int mcinRelOffset = BitConverter.ToInt32(adtBytes, mhdrDataStart + 4); // MCIN offset (0x04)

            if (mcinRelOffset <= 0)
            {
                Console.WriteLine($"[LK ADT] Invalid MCIN offset in {tileName}");
                return null;
            }

            int mcinOffset = mhdrDataStart + mcinRelOffset;
            
            // Read MCIN to get MCNK offsets
            var mcnkOffsets = new int[256];
            int mcinDataStart = mcinOffset + 8; // skip chunk header
            for (int i = 0; i < 256 && mcinDataStart + i * 16 + 4 <= adtBytes.Length; i++)
            {
                mcnkOffsets[i] = BitConverter.ToInt32(adtBytes, mcinDataStart + i * 16);
            }
            
            // Determine source for Texture Data (MTEX/MCNK tex chunks)
            // If texBytes provided (Split ADT), use that. Otherwise use main ADT.
            byte[] texSource = texBytes ?? adtBytes;
            
            // Extract textures from MTEX
            int mtexOffset = FindLkChunk(texSource, "MTEX");
            if (mtexOffset >= 0)
            {
                int mtexSize = BitConverter.ToInt32(texSource, mtexOffset + 4);
                int pos = mtexOffset + 8;
                int end = pos + mtexSize;
                while (pos < end)
                {
                    int nul = Array.IndexOf(texSource, (byte)0, pos, end - pos);
                    if (nul == -1) nul = end;
                    int len = nul - pos;
                    if (len > 0)
                    {
                        var texName = System.Text.Encoding.UTF8.GetString(texSource, pos, len);
                        textures.Add(texName);
                        textureCollector.TryAdd(texName, 0);
                    }
                    pos = nul + 1;
                }
            }
            
            // Determine offsets for Split ADT chunks if needed
            int[]? texMcnkOffsets = null;
            if (texBytes != null)
            {
                texMcnkOffsets = GetMcnkOffsets(texBytes); // Needs helper
            }

            // Process each MCNK chunk
            for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
            {
                int mcnkOffset = mcnkOffsets[chunkIndex];
                if (mcnkOffset <= 0 || mcnkOffset + 128 > adtBytes.Length)
                    continue;

                try
                {
                    // Read MCNK header and switch contexts if needed
                    int headerStart = mcnkOffset + 8;
                    
                    // Contexts for offsets
                    byte[] texContext = adtBytes;
                    int texHeaderStart = headerStart;
                    
                    if (texBytes != null && texMcnkOffsets != null)
                    {
                        texContext = texBytes;
                        int texOff = texMcnkOffsets[chunkIndex];
                        if (texOff > 0)
                        {
                            texHeaderStart = texOff + 8;
                        }
                    }

                    // READ Header Flags
                    int mcnkFlags = BitConverter.ToInt32(adtBytes, headerStart); // 0x00
                    bool doNotFixAlphaMap = (mcnkFlags & 0x8000) != 0; // 0x8000 = do_not_fix_alpha_map

                    // Position (0x68 in MCNK header)
                    // Z, X, Y -> but verify order. Usually Z is first in ADT coords?
                    // Typically: Z, X, Y.
                    float posZ = BitConverter.ToSingle(adtBytes, headerStart + 0x68);
                    float posX = BitConverter.ToSingle(adtBytes, headerStart + 0x6C);
                    float posY = BitConverter.ToSingle(adtBytes, headerStart + 0x70);
                    
                    chunkPositions[chunkIndex * 3] = posX;
                    chunkPositions[chunkIndex * 3 + 1] = posY;
                    chunkPositions[chunkIndex * 3 + 2] = posZ;
                    
                    // Holes (0x3C = 60)
                    int holesValue = BitConverter.ToInt32(adtBytes, headerStart + 60);
                    holes[chunkIndex] = holesValue;

                    // MCVT (Heights): 0x10 -> +16
                    int mcvtRelOffset = BitConverter.ToInt32(adtBytes, headerStart + 0x10);
                    if (mcvtRelOffset > 0)
                    {
                        int mcvtAbs = headerStart + mcvtRelOffset + 8;
                        if (mcvtAbs + 145 * 4 <= adtBytes.Length)
                        {
                            var chunkHeights = new float[145];
                            for (int h = 0; h < 145; h++)
                            {
                                // Heights are relative to posZ
                                chunkHeights[h] = posZ + BitConverter.ToSingle(adtBytes, mcvtAbs + h * 4);
                                if (chunkHeights[h] < heightMin) heightMin = chunkHeights[h];
                                if (chunkHeights[h] > heightMax) heightMax = chunkHeights[h];
                            }
                            heights.Add(new VlmChunkHeights(chunkIndex, chunkHeights));
                        }
                    }

                    // MCNR (Normals): 0x14 -> +20
                    int mcnrRelOffset = BitConverter.ToInt32(adtBytes, headerStart + 0x14);
                    sbyte[]? normalsArray = null;
                    if (mcnrRelOffset > 0)
                    {
                        int mcnrAbs = headerStart + mcnrRelOffset + 8;
                        if (mcnrAbs + 145 * 3 <= adtBytes.Length)
                        {
                            normalsArray = new sbyte[145 * 3];
                            for (int n = 0; n < 145 * 3; n++)
                                normalsArray[n] = (sbyte)adtBytes[mcnrAbs + n];
                        }
                    }

                    // MCSH (Shadows): 0x28 -> +40 (texContext)
                    string? chunkShadowPath = null;
                    int mcshRelOffset = BitConverter.ToInt32(texContext, texHeaderStart + 0x2C); // 0x2C: ofsShadow
                    // Wait, wiki says 0x28 is ofsShadow?
                    // Let's check offsets carefully.
                    // 0x10 MCVT, 0x14 MCNR, 0x18 MCLY, 0x1C MCRF, 0x20 MCAL, 0x24 sizeAlpha, 0x28 ofsShadow, 0x2C sizeShadow
                    // My previous thought was +36? 
                    // Let's use 0x28 (40).
                    mcshRelOffset = BitConverter.ToInt32(texContext, texHeaderStart + 0x28);
                    
                    if (mcshRelOffset > 0)
                    {
                        int mcshAbs = texHeaderStart + mcshRelOffset;
                        if (mcshAbs + 8 <= texContext.Length)
                        {
                            string mcshFourCC = System.Text.Encoding.ASCII.GetString(texContext, mcshAbs, 4);
                            if (mcshFourCC == "HSMC")
                            {
                                int mcshSize = BitConverter.ToInt32(texContext, mcshAbs + 4);
                                int mcshDataStart = mcshAbs + 8;
                                if (mcshSize >= 512 && mcshDataStart + 512 <= texContext.Length)
                                {
                                    try
                                    {
                                        var mcshBuf = new byte[512];
                                        Array.Copy(texContext, mcshDataStart, mcshBuf, 0, 512);
                                        
                                        var shadow = ShadowMapService.ReadShadow(mcshBuf);
                                        var shadowPng = ShadowMapService.ToPng(shadow);
                                        var shadowFileName = $"{tileName}_c{chunkIndex}.png";
                                        File.WriteAllBytes(Path.Combine(shadowsDir, shadowFileName), shadowPng);
                                        shadowPaths.Add($"shadows/{shadowFileName}");
                                        chunkShadowPath = $"shadows/{shadowFileName}";
                                        
                                        var rawShadowBytes = mcshBuf.Take(512).ToArray();
                                        shadowBits.Add(new VlmChunkShadowBits(chunkIndex, Convert.ToBase64String(rawShadowBytes)));
                                    }
                                    catch { }
                                }
                            }
                        }
                    }

                    // MCCV (Vertex Colors): 0x50 -> +80? Or check flags?
                    // Usually present if MCNK 0x20 flag set.
                    // Subchunk usually at the end.
                    // We'll rely on flags or try to find "MCCV" in the chunk data if needed? No, too slow.
                    // If we use 0x50, let's verify.
                    // For now, let's skip MCCV unless confident. Or try to find it via subchunk scan?
                    // Standard approach: Check MCNK flag 0x20. If set, MCCV follows MCAL?
                    // Or check 0x58?
                    // Actually, let's look for MCCV by searching reversed "MCCV" in the chunk area.
                    // Only search within the localized chunk data to be safe.
                    // Chunk size is not in MCNK header directly? No, it's in MCIN.
                    // We know the chunk ends at next MCNK offset.
                    
                    byte[]? mccvColors = null;
                    if ((mcnkFlags & 0x20) != 0) // MCCV flag
                    {
                        // Search for MCCV in the chunk data range
                        // Limit search?
                        // Just search in texContext (where layers are) starting from texHeaderStart.
                        // MCCV must be in texContext for Split ADT.
                        
                        // We can use FindLkChunk logic but scoped.
                        // Scan from headerStart to next chunk? We don't know exact size here easily without parsing.
                        // Let's scan forward from standard offsets.
                        // It's robust enough to scan for VCCM (reversed MCCV).
                        
                        int mccvAbs = -1;
                        int limit = 16384; // Chunk shouldn't be bigger than 16KB usually.
                        for(int k = texHeaderStart; k < texContext.Length && k < texHeaderStart + limit; k++) {
                            if (texContext[k] == 'V' && texContext[k+1] == 'C' && texContext[k+2] == 'C' && texContext[k+3] == 'M') {
                                mccvAbs = k;
                                break;
                            }
                        }
                        
                        if (mccvAbs >= 0) {
                            // Found MCCV
                            if (mccvAbs + 145 * 4 + 8 <= texContext.Length) {
                                mccvColors = new byte[145 * 4];
                                Array.Copy(texContext, mccvAbs + 8, mccvColors, 0, 145 * 4);
                            }
                        }
                    }

                    // MCLY (0x18 -> +24) and MCAL (0x20 -> +32)
                    int mclyRelOffset = BitConverter.ToInt32(texContext, texHeaderStart + 0x1C); // 0x1C MCLY
                    int mcalRelOffset = BitConverter.ToInt32(texContext, texHeaderStart + 0x20); // 0x20 MCAL?
                    // Offsets check:
                    // 0x10 MCVT, 0x14 MCNR, 0x18 MCLY -> Wait.
                    // 0x10=16. 0x14=20. 0x18=24.
                    // So MCLY IS AT 0x18 (+24).
                    mclyRelOffset = BitConverter.ToInt32(texContext, texHeaderStart + 0x18);
                    mcalRelOffset = BitConverter.ToInt32(texContext, texHeaderStart + 0x20); // 0x20 = 32. MCAL.
                    
                    var layerList = new List<VlmTextureLayer>();
                    if (mclyRelOffset > 0 && mcalRelOffset > 0)
                    {
                        int mclyAbs = texHeaderStart + mclyRelOffset + 8;
                        int mcalAbs = texHeaderStart + mcalRelOffset + 8; 
                        
                        int nLayers = BitConverter.ToInt32(texContext, texHeaderStart + 0x08); // 0x08 = nLayers
                        if (nLayers <= 0) nLayers = 0; if (nLayers > 4) nLayers = 4;
                        
                        for (int l = 0; l < nLayers; l++)
                        {
                            if (mclyAbs + (l + 1) * 16 > texContext.Length) break;
                            
                            uint tId = BitConverter.ToUInt32(texContext, mclyAbs + l * 16);
                            uint flags = BitConverter.ToUInt32(texContext, mclyAbs + l * 16 + 4);
                            uint aOff = BitConverter.ToUInt32(texContext, mclyAbs + l * 16 + 8);
                            uint eff = BitConverter.ToUInt32(texContext, mclyAbs + l * 16 + 12);
                            
                            string? texturePath = tId < textures.Count ? textures[(int)tId] : null;
                            string[]? effects = (eff > 0 && groundEffectService != null) ? groundEffectService.GetDoodadsEffect(eff) : null;
                            
                            string? alphaBase64 = null;
                            string? alphaPath = null;
                            byte[]? alphaBytes = null;
                            
                            if (l > 0)
                            {
                                try {
                                    // Use AlphaMapService logic
                                    // Provide absolute offset to alpha data start for this layer
                                    int layerAlphaStart = mcalAbs + (int)aOff;
                                    
                                    // Validation
                                    if (layerAlphaStart < texContext.Length)
                                    {
                                         var alpha = AlphaMapService.ReadAlpha(texContext, layerAlphaStart, flags, useBigAlphamaps, doNotFixAlphaMap);
                                         if (alpha != null && alpha.Length == 64*64)
                                         {
                                             alphaBytes = alpha;
                                             // Also save PNG/Base64 for debug?
                                             // Saving PNG for visual debug
                                             var png = AlphaMapService.ToPng(alpha);
                                             var aName = $"{tileName}_c{chunkIndex}_l{l}.png";
                                             File.WriteAllBytes(Path.Combine(masksDir, aName), png);
                                             alphaPath = $"masks/{aName}";
                                         }
                                    }
                                } catch {}
                            }
                            
                            layerList.Add(new VlmTextureLayer(tId, texturePath, flags, aOff, eff, effects, alphaBase64, alphaPath, alphaBytes));
                        }
                    }

                    chunkLayers.Add(new VlmChunkLayers(
                        chunkIndex, 
                        layerList.ToArray(), 
                        chunkShadowPath,
                        normalsArray,
                        mccvColors,
                        (uint)holesValue,
                        0
                    ));
                }
                catch { }
            }
            
            // Helper method for offsets
            int[]? GetMcnkOffsets(byte[] d) {
                try {
                    int h = FindLkChunk(d, "MHDR"); 
                    if (h < 0) return null;
                    int cin = BitConverter.ToInt32(d, h + 8 + BitConverter.ToInt32(d, h + 8 + 4)); // MHDR data + 4 -> MCIN offset
                    if (cin <= 0) return null;
                    var o = new int[256];
                    for(int i=0; i<256; i++) o[i] = BitConverter.ToInt32(d, h + 8 + cin + 8 + i*16);
                    return o;
                } catch { return null; }
            }

            var heightmapPath = await GenerateHeightmap(heights, tileName, outputDir);
            var normalmapPath = await GenerateNormalmap(chunkLayers, tileName, outputDir);
            var mccvMapPath = await GenerateMccvMap(chunkLayers, tileName, outputDir);

            return new VlmTerrainData(
                tileName,
                heights.ToArray(),
                chunkPositions,
                holes,
                heightmapPath,
                heightmapPath,
                null,
                normalmapPath,
                mccvMapPath,
                shadowPaths.Count > 0 ? shadowPaths.ToArray() : null,
                shadowBits.Count > 0 ? shadowBits.ToArray() : null,
                null,
                null,
                null,
                0f,
                0f,
                textures,
                chunkLayers.Count > 0 ? chunkLayers.ToArray() : null,
                liquids.Count > 0 ? liquids.ToArray() : null,
                objects,
                wdlHeights,
                heightMin == float.MaxValue ? 0 : heightMin,
                heightMax == float.MinValue ? 0 : heightMax,
                heightMin == float.MaxValue ? 0 : heightMin,
                heightMax == float.MinValue ? 0 : heightMax
            );
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LK ADT] Error parsing {tileName}: {ex.Message}");
            return null;
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

    private string? FindMinimapTile(IEnumerable<string> searchPaths, MpqArchiveService mpqService, Md5TranslateIndex? index, string mapName, int x, int y)
    {
        // Generate all possible plain-name candidates for this tile
        // matching legacy MinimapFileResolver.EnumeratePlainCandidates
        var candidates = new List<string>();
        
        var x2 = x.ToString("D2");
        var y2 = y.ToString("D2");
        
        // 1. Standard full paths
        candidates.Add($"textures/minimap/{mapName}/{mapName}_{x2}_{y2}.blp");
        candidates.Add($"textures/minimap/{mapName}/map{x2}_{y2}.blp");
        
        // 2. Short paths (often used in md5translate)
        candidates.Add($"{mapName}/map{x2}_{y2}.blp");
        
        // 3. Space variants (0.6.0 bug)
        var mapNameSpace = InsertSpaceBeforeCapitals(mapName);
        if (mapNameSpace != mapName)
        {
            candidates.Add($"textures/minimap/{mapNameSpace}/{mapNameSpace}_{x2}_{y2}.blp");
            candidates.Add($"textures/minimap/{mapNameSpace}/map{x2}_{y2}.blp");
            candidates.Add($"{mapNameSpace}/map{x2}_{y2}.blp");
        }

        // 4. Other Legacy/Release variants
        candidates.Add($"World/Minimaps/{mapName}/map{x2}_{y2}.blp");
        candidates.Add($"World/Minimaps/{mapName}/map{x}_{y}.blp");
        candidates.Add($"Textures/Minimap/{mapName}_{x2}_{y2}.blp");
        candidates.Add($"Textures/Minimap/{mapName}_{x}_{y}.blp");

        // PRIORITY 1: Check MD5 Index for ANY candidate
        if (index != null)
        {
            foreach (var candidate in candidates)
            {
                // Normalize for lookup (legacy uses internal normalization, but our dict is case-insensitive too)
                var lookupKey = candidate.Replace('\\', '/').TrimStart('/');
                
                if (index.PlainToHash.TryGetValue(lookupKey, out var hashed))
                {
                    // Found a mapping!
                    // The 'hashed' value is the filename in the MPQ.
                    var mpqKey = hashed.Replace("/", "\\");
                    if (mpqService.FileExists(mpqKey))
                    {
                        Console.WriteLine($"[Match] Translated '{candidate}' -> '{hashed}' (Found in MPQ)");
                        return $"MPQ:{mpqKey}";
                    }
                    
                    // Also check disk
                    foreach (var bp in searchPaths)
                    {
                         var hp = Path.Combine(bp, hashed);
                         if (File.Exists(hp)) return hp;
                    }
                    
                    Console.WriteLine($"[Mapping Found] '{candidate}' -> '{hashed}' but file missing.");
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
             if (mpqService.FileExists(mpqPlainKey))
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

    private async Task<string?> GenerateHeightmap(List<VlmChunkHeights> chunkHeights, string tileName, string outputDir)
    {
        if (chunkHeights == null || chunkHeights.Count == 0) return null;

        const int Size = 145;
        var heightsDict = chunkHeights.ToDictionary(k => k.ChunkIndex, v => v.Heights);
        var (minZ, maxZ) = GetHeightRange(heightsDict);
        var mapBytes = RenderHeightmapImage(heightsDict, minZ, maxZ, Size);

        var filename = $"{tileName}_heightmap.png";
        var imagesDir = Path.Combine(outputDir, "images");
        Directory.CreateDirectory(imagesDir);
        var path = Path.Combine(imagesDir, filename);
        await File.WriteAllBytesAsync(path, mapBytes);

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

    private byte[] RenderHeightmapImage(Dictionary<int, float[]> heightsDict, float minZ, float maxZ, int size)
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
             
             // Vertices
             float vTL = hData[iy * 9 + ix];
             float vTR = hData[iy * 9 + ix + 1];
             float vBL = hData[(iy + 1) * 9 + ix];
             float vBR = hData[(iy + 1) * 9 + ix + 1];
             float vC = hData[81 + iy * 8 + ix]; // Inner center
             
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

    private async Task<string?> GenerateNormalmap(List<VlmChunkLayers> chunkLayers, string tileName, string outputDir)
    {
        if (chunkLayers == null || chunkLayers.Count == 0) return null;

        const int Size = 145;
        var normalsDict = chunkLayers
            .Where(c => c.Normals != null && c.Normals.Length >= 145 * 3)
            .ToDictionary(k => k.ChunkIndex, v => v.Normals!);

        if (normalsDict.Count == 0) return null;

        var mapBytes = RenderNormalmapImage(normalsDict, Size);

        var filename = $"{tileName}_normal.png";
        var imagesDir = Path.Combine(outputDir, "images");
        Directory.CreateDirectory(imagesDir);
        var path = Path.Combine(imagesDir, filename);
        await File.WriteAllBytesAsync(path, mapBytes);

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

                var mapBytes = RenderHeightmapImage(heightsDict, globalMin, globalMax, 512);
                var filename = $"{sample.TerrainData.AdtTile}_heightmap_global.png";
                var imagesDir = Path.Combine(outputDir, "images");
                Directory.CreateDirectory(imagesDir);
                var path = Path.Combine(imagesDir, filename);
                await File.WriteAllBytesAsync(path, mapBytes);

                var heightmapGlobalPath = $"images/{filename}";
                var updatedTerrain = sample.TerrainData with
                {
                    HeightmapLocalPath = sample.TerrainData.HeightmapLocalPath ?? sample.TerrainData.HeightmapPath,
                    HeightmapGlobalPath = heightmapGlobalPath,
                    HeightGlobalMin = globalMin,
                    HeightGlobalMax = globalMax
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

    private bool ConvertBlpToPng(string blpPath, string pngPath, MpqArchiveService? mpqService = null)
    {
        try
        {
            byte[]? blpData = null;
            
            if (blpPath.StartsWith("MPQ:"))
            {
                var key = blpPath.Substring(4);
                blpData = mpqService?.ReadFile(key);
            }
            else if (blpPath.EndsWith(".MPQ", StringComparison.OrdinalIgnoreCase))
            {
                blpData = AlphaMpqReader.ReadFromMpq(blpPath);
            }
            else if (File.Exists(blpPath))
            {
                blpData = File.ReadAllBytes(blpPath);
            }

            if (blpData == null || blpData.Length == 0)
            {
                Console.WriteLine($"Empty BLP data: {blpPath}");
                return false;
            }

            using var ms = new MemoryStream(blpData);
            using var blp = new SereniaBLPLib.BlpFile(ms);
            using var bmp = blp.GetBitmap(0);

            // Upscale to 512x512 to match V7 dataset standard
            if (bmp.Width != 512 || bmp.Height != 512)
            {
                var resized = new System.Drawing.Bitmap(512, 512);
                using (var g = System.Drawing.Graphics.FromImage(resized))
                {
                    // Use NearestNeighbor for crisp, honest upscaling (avoids "mushy" text/edges)
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half; // Aligns pixels correctly
                    g.DrawImage(bmp, 0, 0, 512, 512);
                }
                resized.Save(pngPath, System.Drawing.Imaging.ImageFormat.Png);
                resized.Dispose();
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
    private (float[] Min, float[] Max)? GetModelBounds(string modelPath, MpqArchiveService mpqService)
    {
        if (string.IsNullOrEmpty(modelPath)) return null;
        
        // Check cache first
        if (_modelBoundsCache.TryGetValue(modelPath.ToLowerInvariant(), out var cached))
            return cached;
        
        try
        {
            // Try to read from MPQ
            var data = mpqService.ReadFile(modelPath);
            if (data == null || data.Length < 100) 
            {
                _modelBoundsCache[modelPath.ToLowerInvariant()] = null;
                return null;
            }
            
            float[] boundsMin = null!;
            float[] boundsMax = null!;
            
            // Check if MDX or WMO by extension/signature
            var ext = Path.GetExtension(modelPath).ToLowerInvariant();
            bool isWmo = ext == ".wmo";
            
            if (isWmo)
            {
                // WMO: Read MOHD chunk for bounding box
                // MOHD is typically after MVER, starts around offset 20-40
                // Format: ... boundingBox1 (3 floats), boundingBox2 (3 floats) at offset 28 in MOHD data
                int mohdOffset = FindChunkOffset(data, "MOHD");
                if (mohdOffset >= 0 && mohdOffset + 8 + 52 <= data.Length)
                {
                    int dataStart = mohdOffset + 8; // Skip chunk ID + size
                    // boundingBox1 starts at offset 28 from MOHD data
                    boundsMin = new float[3];
                    boundsMax = new float[3];
                    boundsMin[0] = BitConverter.ToSingle(data, dataStart + 28);
                    boundsMin[1] = BitConverter.ToSingle(data, dataStart + 32);
                    boundsMin[2] = BitConverter.ToSingle(data, dataStart + 36);
                    boundsMax[0] = BitConverter.ToSingle(data, dataStart + 40);
                    boundsMax[1] = BitConverter.ToSingle(data, dataStart + 44);
                    boundsMax[2] = BitConverter.ToSingle(data, dataStart + 48);
                }
            }
            else
            {
                // MDX/M2: Read header for bounding box
                // MDX header has bounding box around offset 60-84 (depends on version)
                // Try reading at common offset for Alpha MDX
                if (data.Length >= 88)
                {
                    // Alpha MDX: Header starts with "MDLX" or similar
                    // Bounding box is typically at a fixed offset in header
                    // For simplicity, search for reasonable float values
                    int bbOffset = 64; // Common offset for Alpha MDX bounding box
                    if (data.Length >= bbOffset + 24)
                    {
                        boundsMin = new float[3];
                        boundsMax = new float[3];
                        boundsMin[0] = BitConverter.ToSingle(data, bbOffset);
                        boundsMin[1] = BitConverter.ToSingle(data, bbOffset + 4);
                        boundsMin[2] = BitConverter.ToSingle(data, bbOffset + 8);
                        boundsMax[0] = BitConverter.ToSingle(data, bbOffset + 12);
                        boundsMax[1] = BitConverter.ToSingle(data, bbOffset + 16);
                        boundsMax[2] = BitConverter.ToSingle(data, bbOffset + 20);
                        
                        // Sanity check: bounds should be reasonable (not NaN or huge)
                        if (float.IsNaN(boundsMin[0]) || float.IsNaN(boundsMax[0]) ||
                            Math.Abs(boundsMin[0]) > 10000 || Math.Abs(boundsMax[0]) > 10000)
                        {
                            boundsMin = null!;
                            boundsMax = null!;
                        }
                    }
                }
            }
            
            if (boundsMin != null && boundsMax != null)
            {
                var result = (boundsMin, boundsMax);
                _modelBoundsCache[modelPath.ToLowerInvariant()] = result;
                return result;
            }
        }
        catch { }
        
        _modelBoundsCache[modelPath.ToLowerInvariant()] = null;
        return null;
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
    /// Extract bounding box from MDX file via AlphaMpqReader (per-asset MPQ).
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
            var data = AlphaMpqReader.ReadFromMpq(mdxMpqPath);
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
    /// Extract bounding box from WMO file via AlphaMpqReader (per-asset MPQ).
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
            var data = AlphaMpqReader.ReadFromMpq(wmoMpqPath);
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
