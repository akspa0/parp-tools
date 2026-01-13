using System.Text.Json;
using WoWMapConverter.Core.Formats.Alpha;
using WoWMapConverter.Core.Services;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// VLM Dataset Exporter - extracts ADT data for VLM training.
/// Outputs structured JSON + image files for terrain reconstruction.
/// </summary>
public class VlmDatasetExporter
{
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    /// <summary>
    /// Export VLM training dataset for a map.
    /// </summary>
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

        // Load WDT
        var wdtPath = Path.Combine(clientPath, "World", "Maps", mapName, $"{mapName}.wdt");
        if (!File.Exists(wdtPath))
        {
            // Try MPQ variant
            var wdtData = AlphaMpqReader.ReadWithMpqFallback(wdtPath);
            if (wdtData == null)
            {
                progress?.Report($"WDT not found: {wdtPath}");
                return new VlmExportResult(0, 0, 0, outputDir);
            }
            // Write temp WDT
            var tempWdt = Path.Combine(outputDir, $"{mapName}.wdt");
            await File.WriteAllBytesAsync(tempWdt, wdtData);
            wdtPath = tempWdt;
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

        var existingTiles = wdt.GetExistingTileIndices();
        var adtOffsets = wdt.GetAdtOffsets();
        var mdnmNames = wdt.GetMdnmNames();
        var monmNames = wdt.GetMonmNames();
        
        progress?.Report($"Found {existingTiles.Count} tiles in WDT");

        int tilesExported = 0;
        int tilesSkipped = 0;
        var allTextures = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Read raw WDT for embedded ADT extraction
        var wdtBytes = await File.ReadAllBytesAsync(wdtPath);

        foreach (var tileIndex in existingTiles.Take(limit))
        {
            var (x, y) = WdtAlpha.IndexToCoords(tileIndex);
            var tileName = $"{mapName}_{x}_{y}";

            try
            {
                // Extract ADT data from WDT
                int adtOffset = tileIndex < adtOffsets.Count ? adtOffsets[tileIndex] : 0;
                if (adtOffset <= 0)
                {
                    tilesSkipped++;
                    continue;
                }

                // Try to find minimap
                var minimapPath = FindMinimapTile(clientPath, mapName, x, y);
                string? imageRelPath = null;
                
                if (minimapPath != null)
                {
                    var imageFileName = $"{tileName}.png";
                    var outputImagePath = Path.Combine(imagesDir, imageFileName);
                    
                    if (ConvertMinimapToPng(minimapPath, outputImagePath))
                    {
                        imageRelPath = $"images/{imageFileName}";
                    }
                }

                // Parse ADT chunk data
                var sample = ExtractAdtData(
                    wdtBytes, adtOffset, tileName,
                    shadowsDir, masksDir, liquidsDir,
                    mdnmNames, monmNames, allTextures);

                if (sample == null)
                {
                    tilesSkipped++;
                    continue;
                }

                // Create final sample with image path
                var finalSample = new VlmTrainingSample(
                    imageRelPath ?? "",
                    null,  // Depth path (DepthAnything3 integration later)
                    sample
                );

                // Write JSON
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

        // Write texture database
        var textureDbPath = Path.Combine(outputDir, "texture_database.json");
        var textureDb = new { count = allTextures.Count, textures = allTextures.ToList() };
        await File.WriteAllTextAsync(textureDbPath, JsonSerializer.Serialize(textureDb, _jsonOptions));

        // Generate depth maps if requested
        if (generateDepth && tilesExported > 0)
        {
            progress?.Report("Generating depth maps with DepthAnything3...");
            var depthService = new DepthMapService();
            var depthCount = await depthService.GenerateDepthMapsAsync(imagesDir, depthsDir, progress);
            
            // Update JSON files with depth paths
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
                
                // Check if depth file exists
                var depthAbsPath = Path.Combine(Path.GetDirectoryName(datasetDir)!, depthRelPath);
                if (File.Exists(depthAbsPath))
                {
                    var updatedSample = sample with { DepthPath = depthRelPath };
                    var updatedJson = JsonSerializer.Serialize(updatedSample, _jsonOptions);
                    await File.WriteAllTextAsync(jsonPath, updatedJson);
                    updated++;
                }
            }
            catch { /* Skip failed files */ }
        }
        
        progress?.Report($"Updated {updated} JSON files with depth paths");
    }

    private string? FindMinimapTile(string clientPath, string mapName, int x, int y)
    {
        // Try various minimap path patterns
        var patterns = new[]
        {
            $"textures/minimap/{mapName.ToLower()}/map{x:D2}_{y:D2}.blp",
            $"Textures/Minimap/{mapName}/map{x:D2}_{y:D2}.blp",
            $"textures/Minimap/{mapName}/map{x:D2}_{y:D2}.blp"
        };

        foreach (var pattern in patterns)
        {
            var fullPath = Path.Combine(clientPath, pattern);
            if (File.Exists(fullPath)) return fullPath;
            
            // Try MPQ
            var mpqPath = fullPath + ".MPQ";
            if (File.Exists(mpqPath)) return mpqPath;
        }

        return null;
    }

    private bool ConvertMinimapToPng(string blpPath, string pngPath)
    {
        try
        {
            byte[]? blpData;
            if (blpPath.EndsWith(".MPQ", StringComparison.OrdinalIgnoreCase))
            {
                blpData = AlphaMpqReader.ReadFromMpq(blpPath);
            }
            else
            {
                blpData = File.ReadAllBytes(blpPath);
            }

            if (blpData == null || blpData.Length == 0)
                return false;

            // Use SereniaBLPLib
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

    private VlmTerrainData? ExtractAdtData(
        byte[] wdtData, int adtOffset, string tileName,
        string shadowsDir, string masksDir, string liquidsDir,
        List<string> mdnmNames, List<string> monmNames,
        HashSet<string> textureCollector)
    {
        // Parse MCNK chunks from embedded ADT
        // This is a simplified parser - full implementation would use AdtAlpha format classes
        
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

        int pos = adtOffset;
        int chunkIndex = 0;

        // Scan for chunks
        while (pos + 8 < wdtData.Length && chunkIndex < 256)
        {
            if (pos + 8 > wdtData.Length) break;

            var tag = System.Text.Encoding.ASCII.GetString(wdtData, pos, 4);
            var tagRev = new string(tag.Reverse().ToArray());
            var size = BitConverter.ToInt32(wdtData, pos + 4);

            if (size < 0 || pos + 8 + size > wdtData.Length)
                break;

            if (tagRev == "MCNK")
            {
                // Parse MCNK
                var mcnkData = new byte[size];
                Array.Copy(wdtData, pos + 8, mcnkData, 0, size);
                
                var chunkData = ParseMcnk(mcnkData, chunkIndex, tileName, 
                    shadowsDir, masksDir, textureCollector,
                    ref heightMin, ref heightMax);
                
                if (chunkData.heights != null)
                {
                    heights.Add(new VlmChunkHeights(chunkIndex, chunkData.heights));
                    chunkPositions[chunkIndex * 3] = chunkData.posX;
                    chunkPositions[chunkIndex * 3 + 1] = chunkData.posY;
                    chunkPositions[chunkIndex * 3 + 2] = chunkData.posZ;
                    holes[chunkIndex] = chunkData.holes;
                    
                    if (chunkData.shadowPath != null)
                        shadowPaths.Add(chunkData.shadowPath);
                    alphaPaths.AddRange(chunkData.alphaPaths);
                    
                    if (chunkData.layers.Length > 0)
                        chunkLayers.Add(new VlmChunkLayers(chunkIndex, chunkData.layers));
                }

                chunkIndex++;
            }
            else if (tagRev == "MTEX")
            {
                // Parse texture list
                var mtexData = new byte[size];
                Array.Copy(wdtData, pos + 8, mtexData, 0, size);
                textures.AddRange(ParseMtex(mtexData));
                foreach (var t in textures) textureCollector.Add(t);
            }
            else if (tagRev == "MDDF")
            {
                // Parse M2 placements
                var mddfData = new byte[size];
                Array.Copy(wdtData, pos + 8, mddfData, 0, size);
                objects.AddRange(ParseMddf(mddfData, mdnmNames));
            }
            else if (tagRev == "MODF")
            {
                // Parse WMO placements
                var modfData = new byte[size];
                Array.Copy(wdtData, pos + 8, modfData, 0, size);
                objects.AddRange(ParseModf(modfData, monmNames));
            }

            pos += 8 + size;
            if ((size & 1) == 1) pos++; // Padding
        }

        if (heights.Count == 0)
            return null;

        return new VlmTerrainData(
            tileName,
            heights.ToArray(),
            chunkPositions,
            holes,
            shadowPaths.Count > 0 ? shadowPaths.ToArray() : null,
            alphaPaths.Count > 0 ? alphaPaths.ToArray() : null,
            textures,
            chunkLayers.Count > 0 ? chunkLayers.ToArray() : null,
            liquids.Count > 0 ? liquids.ToArray() : null,
            objects,
            heightMin == float.MaxValue ? 0 : heightMin,
            heightMax == float.MinValue ? 0 : heightMax
        );
    }

    private (float[] heights, float posX, float posY, float posZ, int holes, 
             string? shadowPath, string[] alphaPaths, VlmTextureLayer[] layers) 
        ParseMcnk(byte[] data, int chunkIndex, string tileName,
                  string shadowsDir, string masksDir, HashSet<string> textureCollector,
                  ref float heightMin, ref float heightMax)
    {
        // Default return
        var emptyResult = (
            heights: (float[]?)null, posX: 0f, posY: 0f, posZ: 0f, holes: 0,
            shadowPath: (string?)null, alphaPaths: Array.Empty<string>(), 
            layers: Array.Empty<VlmTextureLayer>()
        );

        if (data.Length < 128) return emptyResult;

        // MCNK header (Alpha format = 100 bytes, WotLK = 128 bytes)
        // Try to detect format
        int headerSize = 128;
        if (data.Length < 128) headerSize = Math.Min(data.Length, 100);

        // Read key offsets (positions vary by format)
        int ofsHeight = headerSize >= 36 ? BitConverter.ToInt32(data, 32) : 0;
        int ofsLayer = headerSize >= 40 ? BitConverter.ToInt32(data, 36) : 0;
        int ofsAlpha = headerSize >= 48 ? BitConverter.ToInt32(data, 44) : 0;
        int sizeAlpha = headerSize >= 52 ? BitConverter.ToInt32(data, 48) : 0;
        int ofsShadow = headerSize >= 56 ? BitConverter.ToInt32(data, 52) : 0;
        int sizeShadow = headerSize >= 60 ? BitConverter.ToInt32(data, 56) : 0;
        int holesVal = headerSize >= 72 ? BitConverter.ToInt32(data, 68) : 0;

        // Position (at offset 104 in WotLK format)
        float posZ = headerSize >= 108 ? BitConverter.ToSingle(data, 104) : 0;
        float posX = headerSize >= 112 ? BitConverter.ToSingle(data, 108) : 0;
        float posY = headerSize >= 116 ? BitConverter.ToSingle(data, 112) : 0;

        // Heights (MCVT - 145 floats)
        float[]? heights = null;
        if (ofsHeight > 0 && ofsHeight + 145 * 4 <= data.Length)
        {
            heights = new float[145];
            for (int i = 0; i < 145; i++)
            {
                heights[i] = BitConverter.ToSingle(data, ofsHeight + i * 4);
                if (heights[i] < heightMin) heightMin = heights[i];
                if (heights[i] > heightMax) heightMax = heights[i];
            }
        }

        // Shadows (MCSH)
        string? shadowPath = null;
        if (ofsShadow > 0 && sizeShadow >= 512 && ofsShadow + sizeShadow <= data.Length)
        {
            var shadowData = new byte[sizeShadow];
            Array.Copy(data, ofsShadow, shadowData, 0, sizeShadow);
            var shadow = ShadowMapService.ReadShadow(shadowData);
            var shadowPng = ShadowMapService.ToPng(shadow);
            
            var shadowFileName = $"{tileName}_c{chunkIndex}.png";
            File.WriteAllBytes(Path.Combine(shadowsDir, shadowFileName), shadowPng);
            shadowPath = $"shadows/{shadowFileName}";
        }

        // Alpha maps (MCAL) and layers (MCLY)
        var alphaPaths = new List<string>();
        var layers = new List<VlmTextureLayer>();
        
        // TODO: Parse MCLY and MCAL when we have proper chunk offsets
        // For now, just return empty

        return (heights, posX, posY, posZ, holesVal, shadowPath, alphaPaths.ToArray(), layers.ToArray());
    }

    private List<string> ParseMtex(byte[] data)
    {
        var textures = new List<string>();
        int start = 0;
        
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                {
                    var name = System.Text.Encoding.ASCII.GetString(data, start, i - start);
                    if (!string.IsNullOrWhiteSpace(name))
                        textures.Add(name);
                }
                start = i + 1;
            }
        }
        
        return textures;
    }

    private List<VlmObjectPlacement> ParseMddf(byte[] data, List<string> names)
    {
        var objects = new List<VlmObjectPlacement>();
        const int entrySize = 36;  // ENTRY_MDDF size
        
        for (int i = 0; i + entrySize <= data.Length; i += entrySize)
        {
            uint nameId = BitConverter.ToUInt32(data, i);
            uint uniqueId = BitConverter.ToUInt32(data, i + 4);
            float x = BitConverter.ToSingle(data, i + 8);
            float y = BitConverter.ToSingle(data, i + 12);
            float z = BitConverter.ToSingle(data, i + 16);
            float rotX = BitConverter.ToSingle(data, i + 20);
            float rotY = BitConverter.ToSingle(data, i + 24);
            float rotZ = BitConverter.ToSingle(data, i + 28);
            ushort scale = BitConverter.ToUInt16(data, i + 32);

            string name = nameId < names.Count ? Path.GetFileNameWithoutExtension(names[(int)nameId]) : "";
            
            objects.Add(new VlmObjectPlacement(
                name, nameId, uniqueId,
                x, y, z, rotX, rotY, rotZ,
                scale / 1024f, "m2"
            ));
        }
        
        return objects;
    }

    private List<VlmObjectPlacement> ParseModf(byte[] data, List<string> names)
    {
        var objects = new List<VlmObjectPlacement>();
        const int entrySize = 64;  // ENTRY_MODF size
        
        for (int i = 0; i + entrySize <= data.Length; i += entrySize)
        {
            uint nameId = BitConverter.ToUInt32(data, i);
            uint uniqueId = BitConverter.ToUInt32(data, i + 4);
            float x = BitConverter.ToSingle(data, i + 8);
            float y = BitConverter.ToSingle(data, i + 12);
            float z = BitConverter.ToSingle(data, i + 16);
            float rotX = BitConverter.ToSingle(data, i + 20);
            float rotY = BitConverter.ToSingle(data, i + 24);
            float rotZ = BitConverter.ToSingle(data, i + 28);
            ushort scale = BitConverter.ToUInt16(data, i + 60);

            string name = nameId < names.Count ? Path.GetFileNameWithoutExtension(names[(int)nameId]) : "";
            
            objects.Add(new VlmObjectPlacement(
                name, nameId, uniqueId,
                x, y, z, rotX, rotY, rotZ,
                scale / 1024f, "wmo"
            ));
        }
        
        return objects;
    }
}
