using System.Text.Json;
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
        
        foreach (var tryPath in wdtPaths)
        {
            // Try flat file first
            if (File.Exists(tryPath))
            {
                wdtPath = tryPath;
                progress?.Report($"Found WDT: {wdtPath}");
                break;
            }
            
            // Try per-asset MPQ (file.wdt.MPQ)
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

        if (wdtPath == null)
        {
            progress?.Report($"WDT not found for map '{mapName}'.");
            progress?.Report($"Searched paths:");
            foreach (var p in wdtPaths)
                progress?.Report($"  - {p} (and {p}.MPQ)");
            progress?.Report("If your WDT is inside a larger MPQ archive (terrain.mpq, misc.mpq), extract it first.");
            return new VlmExportResult(0, 0, 0, outputDir);
        }

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

        var existingTiles = wdt.GetExistingAdtsNumbers();
        var adtOffsets = wdt.GetAdtOffsetsInMain();
        var mdnmNames = wdt.GetMdnmFileNames();
        var monmNames = wdt.GetMonmFileNames();
        
        progress?.Report($"Found {existingTiles.Count} tiles in WDT");

        int tilesExported = 0;
        int tilesSkipped = 0;
        var allTextures = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var tileIndex in existingTiles.Take(limit))
        {
            int x = tileIndex % 64;
            int y = tileIndex / 64;
            var tileName = $"{mapName}_{x}_{y}";

            try
            {
                int adtOffset = tileIndex < adtOffsets.Count ? adtOffsets[tileIndex] : 0;
                if (adtOffset <= 0)
                {
                    tilesSkipped++;
                    continue;
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
                    tilesSkipped++;
                    continue;
                }

                // Try to find minimap
                var minimapPath = FindMinimapTile(dataPath, mapName, x, y);
                string? imageRelPath = null;
                
                if (minimapPath != null)
                {
                    var imageFileName = $"{tileName}.png";
                    var outputImagePath = Path.Combine(imagesDir, imageFileName);
                    
                    if (ConvertBlpToPng(minimapPath, outputImagePath))
                    {
                        imageRelPath = $"images/{imageFileName}";
                    }
                }

                // Extract terrain data using AdtAlpha's methods
                var sample = await ExtractFromAdtAlpha(adtAlpha, wdtPath, adtOffset, tileIndex, tileName, outputDir,
                    shadowsDir, masksDir, mdnmNames, monmNames, allTextures);

                if (sample == null)
                {
                    tilesSkipped++;
                    continue;
                }

                var finalSample = new VlmTrainingSample(
                    imageRelPath ?? "",
                    null,
                    sample
                );

                var jsonPath = Path.Combine(datasetDir, $"{tileName}.json");
                var json = JsonSerializer.Serialize(finalSample, _jsonOptions);
                await File.WriteAllTextAsync(jsonPath, json);

                tilesExported++;
                if (tilesExported % 10 == 0)
                    progress?.Report($"Exported {tilesExported} tiles...");
            }
            catch (Exception ex)
            {
                progress?.Report($"Error processing {tileName}: {ex.Message}");
                tilesSkipped++;
            }
        }

        var textureDbPath = Path.Combine(outputDir, "texture_database.json");
        var textureDb = new { count = allTextures.Count, textures = allTextures.ToList() };
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
            foreach (var texture in allTextures)
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
                        Path.Combine(dataPath, texture),
                        Path.Combine(dataPath, "pickles", texture) // Just in case
                    };
                    
                    bool converted = false;
                    foreach (var path in candidates)
                    {
                         if (ConvertBlpToPng(path, pngPath))
                         {
                             converted = true;
                             break;
                         }
                         // Also try .MPQ suffix
                         if (ConvertBlpToPng(path + ".MPQ", pngPath))
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

        if (generateDepth && tilesExported > 0)
        {
            progress?.Report("Generating depth maps with DepthAnything3...");
            var depthService = new DepthMapService();
            var depthCount = await depthService.GenerateDepthMapsAsync(imagesDir, depthsDir, progress);
            
            if (depthCount > 0)
            {
                await UpdateJsonWithDepthPaths(datasetDir, progress);
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
        HashSet<string> textureCollector)
    {
        var heights = new List<VlmChunkHeights>();
        var chunkPositions = new float[256 * 3];
        var holes = new int[256];
        var textures = new List<string>();
        var chunkLayers = new List<VlmChunkLayers>();
        var liquids = new List<VlmLiquidData>();
        var objects = new List<VlmObjectPlacement>();
        var shadowPaths = new List<string>();
        var alphaPaths = new List<string>();
        
        float heightMin = float.MaxValue;
        float heightMax = float.MinValue;

        // Get textures from MTEX
        var mtexNames = adt.GetMtexTextureNames();
        if (mtexNames != null)
        {
            textures.AddRange(mtexNames);
            foreach (var t in mtexNames) textureCollector.Add(t);
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
                    
                    // Extract shadow from McshData (512 bytes = 64x64 bits)
                    var mcshBuf = mcnk.McshData;
                    if (mcshBuf.Length >= 512)
                    {
                        try
                        {
                            var shadow = ShadowMapService.ReadShadow(mcshBuf);
                            var shadowPng = ShadowMapService.ToPng(shadow);
                            var shadowFileName = $"{tileName}_c{chunkIndex}.png";
                            File.WriteAllBytes(Path.Combine(shadowsDir, shadowFileName), shadowPng);
                            shadowPaths.Add($"shadows/{shadowFileName}");
                        }
                        catch { }
                    }
                    
                    // Extract alpha layers from McalData + MclyData
                    var mcalBuf = mcnk.McalData;
                    var mclyBuf = mcnk.MclyData;
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
                                
                                // Advance offset (2048 for uncompressed 4-bit, varies for compressed)
                                alphaOffset += isCompressed ? 4096 : 2048; // Approximate
                            }
                        }
                        catch { }
                    }
                    
                    // Store layer info for this chunk
                    var layerList = new List<VlmTextureLayer>();
                    for (int layer = 0; layer < nLayers && layer < 4 && layer * 16 + 15 < mclyBuf.Length; layer++)
                    {
                        uint textureId = BitConverter.ToUInt32(mclyBuf, layer * 16);
                        uint flags = BitConverter.ToUInt32(mclyBuf, layer * 16 + 4);
                        uint alphaoffs = BitConverter.ToUInt32(mclyBuf, layer * 16 + 8);
                        uint effectId = BitConverter.ToUInt32(mclyBuf, layer * 16 + 12);
                        layerList.Add(new VlmTextureLayer(textureId, flags, alphaoffs, effectId));
                    }
                    chunkLayers.Add(new VlmChunkLayers(chunkIndex, layerList.ToArray()));

                    // Extract Liquid Data (MCLQ - Legacy)
                    var mclqData = mcnk.MclqData;
                    if (mclqData != null && mclqData.Length > 0)
                    {
                        var liquid = LiquidService.ExtractMCLQ(mclqData, chunkIndex);
                        if (liquid != null)
                        {
                            // Save heightmap PNG if exists
                            if (liquid.Heights != null)
                            {
                                var liquidsDir = Path.Combine(outputDir, "liquids");
                                Directory.CreateDirectory(liquidsDir);
                                
                                var heightPng = LiquidService.GenerateHeightPng(liquid.Heights, liquid.MinHeight, liquid.MaxHeight);
                                var heightPath = Path.Combine(liquidsDir, $"{tileName}_c{chunkIndex}_liq_h.png");
                                await File.WriteAllBytesAsync(heightPath, heightPng);
                                
                                // Update liquid record with mask path (optional, currently using convention)
                            }
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

        return new VlmTerrainData(
            tileName,
            heights.ToArray(),
            chunkPositions,
            holes,
            shadowPaths.Count > 0 ? shadowPaths.ToArray() : null,
            alphaPaths.Count > 0 ? alphaPaths.ToArray() : null,
            null, // LiquidMaskPath
            null, // LiquidHeightPath
            0f,   // LiquidMinHeight
            0f,   // LiquidMaxHeight
            textures,
            chunkLayers.Count > 0 ? chunkLayers.ToArray() : null,
            liquids.Count > 0 ? liquids.ToArray() : null,
            objects,
            heightMin == float.MaxValue ? 0 : heightMin,
            heightMax == float.MinValue ? 0 : heightMax
        );
    }

    private string? FindMinimapTile(string clientPath, string mapName, int x, int y)
    {
        var patterns = new[]
        {
            $"textures/minimap/{mapName.ToLower()}/map{x:D2}_{y:D2}.blp",
            $"Textures/Minimap/{mapName}/map{x:D2}_{y:D2}.blp",
            $"textures/Minimap/{mapName}/map{x:D2}_{y:D2}.blp",
            $"World/Minimaps/{mapName}/map{x:D2}_{y:D2}.blp", // Release style
            $"World/Minimaps/{mapName}/map{x}_{y}.blp"        // Alternate release style
        };

        foreach (var pattern in patterns)
        {
            var fullPath = Path.Combine(clientPath, pattern);
            if (File.Exists(fullPath)) return fullPath;
            
            var mpqPath = fullPath + ".MPQ";
            if (File.Exists(mpqPath)) return mpqPath;
        }

        return null;
    }

    private bool ConvertBlpToPng(string blpPath, string pngPath)
    {
        try
        {
            byte[]? blpData;
            // Check for MPQ pseudo-path
            if (blpPath.EndsWith(".MPQ", StringComparison.OrdinalIgnoreCase))
            {
                blpData = AlphaMpqReader.ReadFromMpq(blpPath);
            }
            else if (File.Exists(blpPath))
            {
                blpData = File.ReadAllBytes(blpPath);
            }
            else
            {
                return false;
            }

            if (blpData == null || blpData.Length == 0)
                return false;

            using var ms = new MemoryStream(blpData);
            using var blp = new SereniaBLPLib.BlpFile(ms);
            using var bmp = blp.GetBitmap(0);
            bmp.Save(pngPath, System.Drawing.Imaging.ImageFormat.Png);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
