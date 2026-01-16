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
                         var wdlBytes = AlphaMpqReader.ReadFromMpq(wdlMpqDiscovered);
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
                        shadowsDir, masksDir, mdnmNames, monmNames, allTextures, groundEffectService, wdlTile);
                }
                else
                {
                    // LK format: Read ADT from MPQ as separate file
                    var adtMpqPath = $"World\\Maps\\{mapName}\\{mapName}_{x}_{y}.adt";
                    var adtBytes = mpqService.ReadFile(adtMpqPath);
                    
                    if (adtBytes == null || adtBytes.Length == 0)
                    {
                        Interlocked.Increment(ref tilesSkipped);
                        return;
                    }

                    // Extract terrain data using LK ADT parsing
                    sample = await ExtractFromLkAdt(adtBytes, tileIndex, tileName, outputDir,
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
            
            // Stitch heightmaps into full world map (WebP, max 16384x16384)
            var heightmapOutput = Path.Combine(stitchedDir, $"{mapName}_full_heightmap.webp");
            var heightmapBounds = StitchHeightmapsToWebP(imagesDir, mapName, heightmapOutput, 16384, progress);
            if (heightmapBounds.HasValue)
            {
                progress?.Report($"Created full heightmap: {heightmapOutput} ({heightmapBounds.Value.width}x{heightmapBounds.Value.height})");
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
        WdlParser.WdlTile? wdlTile = null)
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
                
                string name = nameId < mdnmNames.Count ? Path.GetFileNameWithoutExtension(mdnmNames[(int)nameId]) : "";
                objects.Add(new VlmObjectPlacement(name, nameId, uniqueId, px, py, pz, rx, ry, rz, scale / 1024f, "m2"));
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
                
                string name = nameId < monmNames.Count ? Path.GetFileNameWithoutExtension(monmNames[(int)nameId]) : "";
                objects.Add(new VlmObjectPlacement(name, nameId, uniqueId, px, py, pz, rx, ry, rz, scale / 1024f, "wmo"));
            }
        }
        catch { }

        if (heights.Count == 0)
            return null;

        var heightmapPath = await GenerateHeightmap(heights, tileName, outputDir);
        
        return new VlmTerrainData(
            tileName,
            heights.ToArray(),
            chunkPositions,
            holes,
            heightmapPath,  // HeightmapPath
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
            heightMax == float.MinValue ? 0 : heightMax
        );
    }

    /// <summary>
    /// Extract terrain data from LK ADT bytes (3.3.5+ format).
    /// </summary>
    private async Task<VlmTerrainData?> ExtractFromLkAdt(
        byte[] adtBytes, int tileIndex, string tileName,
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
            int mcinRelOffset = BitConverter.ToInt32(adtBytes, mhdrDataStart); // MCIN offset relative to MHDR data

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

            // Extract textures from MTEX
            int mtexOffset = FindLkChunk(adtBytes, "MTEX");
            if (mtexOffset >= 0)
            {
                int mtexSize = BitConverter.ToInt32(adtBytes, mtexOffset + 4);
                int pos = mtexOffset + 8;
                int end = pos + mtexSize;
                while (pos < end)
                {
                    int nul = Array.IndexOf(adtBytes, (byte)0, pos, end - pos);
                    if (nul == -1) nul = end;
                    int len = nul - pos;
                    if (len > 0)
                    {
                        var texName = System.Text.Encoding.UTF8.GetString(adtBytes, pos, len);
                        textures.Add(texName);
                        textureCollector.TryAdd(texName, 0);
                    }
                    pos = nul + 1;
                }
            }

            // Process each MCNK chunk
            for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
            {
                int mcnkOffset = mcnkOffsets[chunkIndex];
                if (mcnkOffset <= 0 || mcnkOffset + 128 > adtBytes.Length)
                    continue;

                try
                {
                    // Read MCNK header (128 bytes in LK)
                    int headerStart = mcnkOffset + 8; // skip MCNK fourcc + size
                    
                    // Get position from header
                    float baseX = BitConverter.ToSingle(adtBytes, headerStart + 12); // offsetX
                    float baseY = BitConverter.ToSingle(adtBytes, headerStart + 16); // offsetY
                    float baseZ = BitConverter.ToSingle(adtBytes, headerStart + 20); // offsetZ
                    
                    chunkPositions[chunkIndex * 3] = baseX;
                    chunkPositions[chunkIndex * 3 + 1] = baseY;
                    chunkPositions[chunkIndex * 3 + 2] = baseZ;
                    
                    // Get holes from header
                    int holesValue = BitConverter.ToInt32(adtBytes, headerStart + 60); // holes field
                    holes[chunkIndex] = holesValue;

                    // Find MCVT subchunk (heights)
                    int mcvtRelOffset = BitConverter.ToInt32(adtBytes, headerStart + 24);
                    if (mcvtRelOffset > 0)
                    {
                        int mcvtAbs = headerStart + mcvtRelOffset + 8; // skip subchunk header
                        if (mcvtAbs + 145 * 4 <= adtBytes.Length)
                        {
                            var chunkHeights = new float[145];
                            for (int h = 0; h < 145; h++)
                            {
                                chunkHeights[h] = baseZ + BitConverter.ToSingle(adtBytes, mcvtAbs + h * 4);
                                if (chunkHeights[h] < heightMin) heightMin = chunkHeights[h];
                                if (chunkHeights[h] > heightMax) heightMax = chunkHeights[h];
                            }
                            heights.Add(new VlmChunkHeights(chunkIndex, chunkHeights));
                        }
                    }

                    // Find MCNR subchunk (normals)
                    int mcnrRelOffset = BitConverter.ToInt32(adtBytes, headerStart + 28);
                    sbyte[]? normalsArray = null;
                    if (mcnrRelOffset > 0)
                    {
                        int mcnrAbs = headerStart + mcnrRelOffset + 8; // skip subchunk header
                        if (mcnrAbs + 145 * 3 <= adtBytes.Length)
                        {
                            normalsArray = new sbyte[145 * 3];
                            for (int n = 0; n < 145 * 3; n++)
                                normalsArray[n] = (sbyte)adtBytes[mcnrAbs + n];
                        }
                    }

                    // Find MCSH subchunk (shadow map - 64x64 bits = 512 bytes)
                    string? chunkShadowPath = null;
                    int mcshRelOffset = BitConverter.ToInt32(adtBytes, headerStart + 36); // MCSH offset in header
                    if (mcshRelOffset > 0)
                    {
                        int mcshAbs = headerStart + mcshRelOffset;
                        // Verify it's actually MCSH chunk
                        if (mcshAbs + 8 <= adtBytes.Length)
                        {
                            string mcshFourCC = System.Text.Encoding.ASCII.GetString(adtBytes, mcshAbs, 4);
                            if (mcshFourCC == "HSMC") // Reversed on disk
                            {
                                int mcshSize = BitConverter.ToInt32(adtBytes, mcshAbs + 4);
                                int mcshDataStart = mcshAbs + 8;
                                
                                if (mcshSize >= 512 && mcshDataStart + 512 <= adtBytes.Length)
                                {
                                    try
                                    {
                                        var mcshBuf = new byte[512];
                                        Array.Copy(adtBytes, mcshDataStart, mcshBuf, 0, 512);
                                        
                                        var shadow = ShadowMapService.ReadShadow(mcshBuf);
                                        var shadowPng = ShadowMapService.ToPng(shadow);
                                        var shadowFileName = $"{tileName}_c{chunkIndex}.png";
                                        File.WriteAllBytes(Path.Combine(shadowsDir, shadowFileName), shadowPng);
                                        shadowPaths.Add($"shadows/{shadowFileName}");
                                        chunkShadowPath = $"shadows/{shadowFileName}";
                                        
                                        // Store raw shadow bits (first 64 bytes = 512 bits)
                                        var rawShadowBytes = mcshBuf.Take(64).ToArray();
                                        shadowBits.Add(new VlmChunkShadowBits(chunkIndex, Convert.ToBase64String(rawShadowBytes)));
                                    }
                                    catch { }
                                }
                            }
                        }
                    }

                    // Create chunk layer entry with normals and shadow path
                    chunkLayers.Add(new VlmChunkLayers(
                        chunkIndex, 
                        Array.Empty<VlmTextureLayer>(), 
                        chunkShadowPath,
                        normalsArray,
                        null, // mccv colors 
                        0, // area id
                        0  // flags
                    ));
                }
                catch { /* Skip problematic chunks */ }
            }

            var heightmapPath = await GenerateHeightmap(heights, tileName, outputDir);
            
            return new VlmTerrainData(
                tileName,
                heights.ToArray(),
                chunkPositions,
                holes,
                heightmapPath,  // HeightmapPath
                shadowPaths.Count > 0 ? shadowPaths.ToArray() : null,
                shadowBits.Count > 0 ? shadowBits.ToArray() : null,
                null, // alphaPaths
                null, // LiquidMaskPath
                null, // LiquidHeightPath
                0f,   // LiquidMinHeight
                0f,   // LiquidMaxHeight
                textures,
                chunkLayers.Count > 0 ? chunkLayers.ToArray() : null,
                liquids.Count > 0 ? liquids.ToArray() : null,
                objects,
                wdlHeights,
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

        const int Size = 256;
        var heightsDict = chunkHeights.ToDictionary(k => k.ChunkIndex, v => v.Heights);
        
        // First pass: collect all heights to find per-tile min/max
        float minZ = float.MaxValue, maxZ = float.MinValue;
        foreach (var kvp in heightsDict)
        {
            if (kvp.Value != null)
            {
                foreach (var h in kvp.Value)
                {
                    if (!float.IsNaN(h) && !float.IsInfinity(h))
                    {
                        if (h < minZ) minZ = h;
                        if (h > maxZ) maxZ = h;
                    }
                }
            }
        }
        
        if (minZ >= maxZ) { minZ = 0; maxZ = 1; }
        float range = maxZ - minZ;

        using var rawMap = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.L16>(Size, Size);
        
        // Alpha MCVT format: 81 outer (9x9) then 64 inner (8x8) per chunk
        for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
        {
            if (!heightsDict.TryGetValue(chunkIdx, out var hData) || hData == null || hData.Length < 145)
                continue;
            
            int chunkY = chunkIdx / 16;
            int chunkX = chunkIdx % 16;
            int baseX = chunkX * 16;
            int baseY = chunkY * 16;
            
            // Place 9x9 outer vertices at even positions
            for (int oy = 0; oy < 9; oy++)
            {
                for (int ox = 0; ox < 9; ox++)
                {
                    int px = baseX + ox * 2;
                    int py = baseY + oy * 2;
                    if (px < Size && py < Size)
                    {
                        float z = hData[oy * 9 + ox];
                        float norm = Math.Clamp((z - minZ) / range, 0f, 1f);
                        rawMap[px, py] = new SixLabors.ImageSharp.PixelFormats.L16((ushort)(norm * 65535));
                    }
                }
            }
            
            // Place 8x8 inner vertices at odd positions
            for (int iy = 0; iy < 8; iy++)
            {
                for (int ix = 0; ix < 8; ix++)
                {
                    int px = baseX + ix * 2 + 1;
                    int py = baseY + iy * 2 + 1;
                    if (px < Size && py < Size)
                    {
                        float z = hData[81 + iy * 8 + ix];
                        float norm = Math.Clamp((z - minZ) / range, 0f, 1f);
                        rawMap[px, py] = new SixLabors.ImageSharp.PixelFormats.L16((ushort)(norm * 65535));
                    }
                }
            }
        }
        
        // Fill gaps with nearest neighbor interpolation
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                if (rawMap[x, y].PackedValue == 0)
                {
                    // Find nearest non-zero neighbor
                    ushort nearest = 0;
                    float minDist = float.MaxValue;
                    for (int dy = -2; dy <= 2; dy++)
                    {
                        for (int dx = -2; dx <= 2; dx++)
                        {
                            if (dy == 0 && dx == 0) continue;
                            int nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < Size && ny >= 0 && ny < Size)
                            {
                                var val = rawMap[nx, ny].PackedValue;
                                if (val > 0)
                                {
                                    float dist = dx * dx + dy * dy;
                                    if (dist < minDist)
                                    {
                                        minDist = dist;
                                        nearest = val;
                                    }
                                }
                            }
                        }
                    }
                    if (nearest > 0)
                        rawMap[x, y] = new SixLabors.ImageSharp.PixelFormats.L16(nearest);
                }
            }
        }
        
        var filename = $"{tileName}_heightmap.png";
        var imagesDir = Path.Combine(outputDir, "images");
        Directory.CreateDirectory(imagesDir);
        var path = Path.Combine(imagesDir, filename);
        await rawMap.SaveAsPngAsync(path);
        
        return $"images/{filename}";
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
            bmp.Save(pngPath, System.Drawing.Imaging.ImageFormat.Png);
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error converting {blpPath}: {ex.Message}");
            return false;
        }
    }
    
    private (int minX, int minY, int maxX, int maxY, int width, int height)? StitchHeightmapsToWebP(
        string imagesDir, string mapName, string outputPath, int maxSize, IProgress<string>? progress)
    {
        try
        {
            // Find all heightmap tiles
            var pattern = $"{mapName}_*_*_heightmap.png";
            var files = Directory.GetFiles(imagesDir, pattern);
            if (files.Length == 0) return null;
            
            // Parse tile coordinates
            var tiles = new List<(int x, int y, string path)>();
            foreach (var file in files)
            {
                var name = Path.GetFileNameWithoutExtension(file);
                var parts = name.Replace($"{mapName}_", "").Replace("_heightmap", "").Split('_');
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
            int fullWidth = tilesWide * 256;
            int fullHeight = tilesHigh * 256;
            
            // Calculate scale to fit within maxSize
            float scale = 1.0f;
            if (fullWidth > maxSize || fullHeight > maxSize)
            {
                scale = Math.Min((float)maxSize / fullWidth, (float)maxSize / fullHeight);
            }
            
            int outputWidth = (int)(fullWidth * scale);
            int outputHeight = (int)(fullHeight * scale);
            int tileSize = (int)(256 * scale);
            
            progress?.Report($"Stitching {tiles.Count} heightmaps into {outputWidth}x{outputHeight} WebP...");
            
            using var canvas = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.L16>(outputWidth, outputHeight);
            
            foreach (var (x, y, path) in tiles)
            {
                try
                {
                    using var tile = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.L16>(path);
                    
                    // Resize tile if needed
                    if (scale < 1.0f)
                    {
                        tile.Mutate(ctx => ctx.Resize(tileSize, tileSize));
                    }
                    
                    int destX = (x - minX) * tileSize;
                    int destY = (y - minY) * tileSize;
                    
                    canvas.Mutate(ctx => ctx.DrawImage(tile, new SixLabors.ImageSharp.Point(destX, destY), 1.0f));
                }
                catch { /* Skip failed tiles */ }
            }
            
            // Save as WebP
            var encoder = new SixLabors.ImageSharp.Formats.Webp.WebpEncoder
            {
                Quality = 95,
                FileFormat = SixLabors.ImageSharp.Formats.Webp.WebpFileFormatType.Lossless
            };
            canvas.Save(outputPath, encoder);
            
            return (minX, minY, maxX, maxY, outputWidth, outputHeight);
        }
        catch (Exception ex)
        {
            progress?.Report($"Error stitching heightmaps: {ex.Message}");
            return null;
        }
    }
}
