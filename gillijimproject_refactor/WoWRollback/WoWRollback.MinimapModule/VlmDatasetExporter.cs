using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WoWRollback.Core.Services.Archive;
using WoWRollback.Core.Services.Minimap;
using WoWRollback.Core.Services.Parsers;
using WoWRollback.Core.Models.ADT;
using WoWRollback.MinimapModule.Models;
using WoWRollback.MinimapModule.Services;

namespace WoWRollback.MinimapModule;

/// <summary>
/// Exports VLM training datasets from game archives.
/// </summary>
public sealed class VlmDatasetExporter
{
    private readonly ILogger<VlmDatasetExporter>? _logger;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public VlmDatasetExporter(ILogger<VlmDatasetExporter>? logger = null)
    {
        _logger = logger;
    }

    /// <summary>
    /// Export VLM training dataset for a single map.
    /// </summary>
    /// <param name="source">Archive source (MPQ or CASC)</param>
    /// <param name="resolver">Minimap file resolver with MD5 translate support</param>
    /// <param name="mapName">Map name (e.g., "development")</param>
    /// <param name="outputDir">Output directory for dataset</param>
    /// <param name="progress">Progress reporter</param>
    public async Task<VlmExportResult> ExportMapAsync(
        IArchiveSource source,
        IMinimapFileResolver resolver,
        string mapName,
        string outputDir,
        IProgress<string>? progress = null,
        int limit = int.MaxValue,
        string? listfilePath = null) // Added listfile path argument
    {
        progress?.Report($"Starting VLM export for map: {mapName}");
        _logger?.LogInformation("Starting VLM export for map: {MapName}", mapName);
        
        // Initialize Listfile Service if provided
        ListfileService? listfileService = null;
        if (!string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
        {
             listfileService = new ListfileService(_logger as ILogger<ListfileService>); // Logger cast might need care or ignore
             listfileService.Load(listfilePath);
        }
        Func<uint, string?>? nameResolver = listfileService != null ? listfileService.Resolve : null;
        
        var imagesDir = Path.Combine(outputDir, "images");
        var datasetDir = Path.Combine(outputDir, "dataset");
        var masksDir = Path.Combine(outputDir, "masks");
        
        Directory.CreateDirectory(imagesDir);
        Directory.CreateDirectory(datasetDir);
        Directory.CreateDirectory(masksDir);

        // Load WDL if present (Global heightmap)
        WdlParser.WdlData? wdlData = null;
        try 
        {
            // WDL is usually at World\Maps\{mapName}\{mapName}.wdl
            var wdlPath = $"World\\Maps\\{mapName}\\{mapName}.wdl";
            if (source.FileExists(wdlPath))
            {
                using var wdlStream = source.OpenFile(wdlPath);
                if (wdlStream != null)
                {
                    using var ms = new MemoryStream();
                    wdlStream.CopyTo(ms);
                    wdlData = WdlParser.Parse(ms.ToArray());
                    _logger?.LogInformation("Loaded WDL data for {MapName}", mapName);
                }
            }
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Failed to load WDL for {MapName}", mapName);
        }

        // Check for Alpha Monolithic WDT
        // If present, we need to extract ADT stats from it.
        byte[]? wdtDataBytes = null;
        int[]? wdtAllOffsets = null; // 4096 offsets
        try
        {
            var wdtPath = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            if (source.FileExists(wdtPath))
            {
                using var wdtStream = source.OpenFile(wdtPath);
                if (wdtStream != null)
                {
                    using var ms = new MemoryStream();
                    wdtStream.CopyTo(ms);
                    wdtDataBytes = ms.ToArray();
                    
                    // Parse MAIN chunk to get offsets
                    // WDT structure: MVER, MPHD, MAIN...
                    // Loop chunks to find MAIN
                    int pos = 0;
                    while (pos + 8 <= wdtDataBytes.Length)
                    {
                         var tag = System.Text.Encoding.ASCII.GetString(wdtDataBytes, pos, 4);
                         var size = BitConverter.ToInt32(wdtDataBytes, pos + 4);
                         var tagRev = new string(tag.Reverse().ToArray());
                         
                         if (tagRev == "MAIN")
                         {
                             // Parse offsets (64x64 grid, 16 bytes per entry, first 4 bytes is offset)
                             wdtAllOffsets = new int[4096];
                             for (int i = 0; i < 4096; i++)
                             {
                                 int entryPos = pos + 8 + i * 16;
                                 if (entryPos + 4 <= wdtDataBytes.Length)
                                     wdtAllOffsets[i] = BitConverter.ToInt32(wdtDataBytes, entryPos);
                             }
                             break;
                         }
                         pos += 8 + size;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Failed to load WDT for {MapName}", mapName);
        }

        int tilesExported = 0;
        int tilesSkipped = 0;
        int tilesChecked = 0;
        int tilesResolved = 0;
        var allTextures = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Scan all possible tiles (0-63, 0-63)
        for (int x = 0; x < 64; x++)
        {
            if (tilesExported >= limit) break;
            for (int y = 0; y < 64; y++)
            {
                if (tilesExported >= limit) break;
                tilesChecked++;
                try
                {
                    // Try to resolve minimap tile
                    if (!resolver.TryResolveTile(mapName, x, y, out var blpPath) || string.IsNullOrEmpty(blpPath))
                    {
                        continue;
                    }
                    
                    tilesResolved++;
                    if (tilesResolved <= 3)
                    {
                        progress?.Report($"Found minimap: {blpPath}");
                    }

                    // Try to load the minimap BLP
                    Stream? blpStream = null;
                    try
                    {
                        blpStream = source.OpenFile(blpPath);
                    }
                    catch
                    {
                        tilesSkipped++;
                        continue;
                    }

                    if (blpStream == null)
                    {
                        tilesSkipped++;
                        continue;
                    }

                    // Convert BLP to PNG
                    var imageFileName = $"{mapName}_{x}_{y}.png";
                    var outputImagePath = Path.Combine(imagesDir, imageFileName);
                    if (!BlpConverter.ConvertToPng(blpStream, outputImagePath))
                    {
                        blpStream.Dispose();
                        tilesSkipped++;
                        continue;
                    }
                    blpStream.Dispose();

                    // Retrieve ADT metadata string for embedding
                    var adtTileName = $"{mapName}_{x}_{y}";
                    var sample = TryExtractAdtMetadata(source, mapName, x, y, imageFileName, masksDir, wdtDataBytes, wdtAllOffsets, wdlData, allTextures, nameResolver);
                    
                    if (sample == null)
                    {
                        // Fallback validation failed - skip if no ADT data found? 
                        // Or create stub? For VLM training we likely need the ADT data.
                        _logger?.LogWarning("No ADT data found for {Tile}", adtTileName);
                        continue;
                    }

                    // Write metadata JSON
                    var datasetPath = Path.Combine(datasetDir, $"{adtTileName}.json");
                    var json = JsonSerializer.Serialize(sample, JsonOptions);
                    await File.WriteAllTextAsync(datasetPath, json);

                    tilesExported++;

                    if (tilesExported % 10 == 0)
                    {
                        progress?.Report($"Exported {tilesExported} tiles...");
                    }
                }
                catch (Exception ex)
                {
                    _logger?.LogWarning(ex, "Failed to process tile {X}_{Y}", x, y);
                    tilesSkipped++;
                }
            }
        }

        // Write texture database
        var textureDbPath = Path.Combine(outputDir, "texture_database.json");
        var textureDb = new { count = allTextures.Count, textures = allTextures.ToList() };
        await File.WriteAllTextAsync(textureDbPath, JsonSerializer.Serialize(textureDb, JsonOptions));

        progress?.Report($"Export complete: {tilesExported} tiles exported, {tilesSkipped} skipped (checked {tilesChecked}, resolved {tilesResolved})");
        _logger?.LogInformation("VLM export complete: {Exported} tiles, {Skipped} skipped, {Textures} unique textures, {Resolved} resolved from {Checked} checked",
            tilesExported, tilesSkipped, allTextures.Count, tilesResolved, tilesChecked);

        return new VlmExportResult(tilesExported, tilesSkipped, allTextures.Count, outputDir);
    }

    private VlmTrainingSample? TryExtractAdtMetadata(
        IArchiveSource source,
        string mapName,
        int x, int y,
        string imageRelativePath,
        string masksOutputDirectory,
        byte[]? wdtData,
        int[]? wdtOffsets,
        WdlParser.WdlData? wdlData,
        HashSet<string> textureCollector,
        Func<uint, string?>? nameResolver)
    {
        // Try to load ADT from archive OR from WDT
        byte[]? adtDataBytes = null;
        var adtPath = $"World\\Maps\\{mapName}\\{mapName}_{x}_{y}.adt";

        // 1. Try separate ADT file (LK+)
        try
        {
            if (source.FileExists(adtPath))
            {
               using var stream = source.OpenFile(adtPath);
               if (stream != null)
               {
                   using var ms = new MemoryStream();
                   stream.CopyTo(ms);
                   adtDataBytes = ms.ToArray();
               }
            }
        }
        catch { /* ignore */ }

        // 2. If no ADT, try extraction from WDT (Alpha/Monolithic)
        if ((adtDataBytes == null || adtDataBytes.Length == 0) && wdtData != null && wdtOffsets != null)
        {
            int tileIdx = y * 64 + x;
            if (tileIdx < wdtOffsets.Length)
            {
                int offset = wdtOffsets[tileIdx];
                if (offset > 0 && offset < wdtData.Length)
                {
                    // How much to read? Until next offset or end?
                    // Alpha chunks are contiguous?
                    // MCNKs usually follow. We need to parse until we hit something else or end?
                    // Actually we can just pass the slice from offset to end, and Parse will stop when chunks end or become invalid?
                    // Or we assume it ends at next tile's offset?
                    // Ideally we pass everything from offset.
                    
                    // Simple heuristic: Copy from offset to end. AdtParser will stop when it sees garbage or EOF.
                    // However, we might read into next tile. AdtParser stops if ReadChunkId fails/invalid.
                    int length = wdtData.Length - offset;
                    adtDataBytes = new byte[length];
                    Array.Copy(wdtData, offset, adtDataBytes, 0, length);
                }
            }
        }
        
        if (adtDataBytes == null || adtDataBytes.Length == 0)
            return null;

        try
        {
            // Parse ADT using Core parser
            var adtData = AdtParser.Parse(adtDataBytes, mapName, x, y, nameResolver);
            
            var textures = new List<string>();
            var layers = new List<VlmTextureLayer>();
            var layerMaskPaths = new List<string>();
            var objects = new List<ObjectPlacement>();
            float heightMin = float.MaxValue;
            float heightMax = float.MinValue;
            
            // Extract WDL heights if available
            short[]? wdlHeights = null;
            if (wdlData != null)
            {
                var tileIndex = y * 64 + x;
                if (tileIndex >= 0 && tileIndex < wdlData.Tiles.Length)
                {
                    var wdlTile = wdlData.Tiles[tileIndex];
                    if (wdlTile != null && wdlTile.HasData)
                    {
                        wdlHeights = new short[17 * 17];
                        for (int r = 0; r < 17; r++)
                        {
                            for (int c = 0; c < 17; c++)
                            {
                                 wdlHeights[r * 17 + c] = wdlTile.Height17[r, c];
                            }
                        }
                    }
                }
            }
            
            string? alphaMapsBase64 = null;
            string? shadowMapBase64 = null;

            // Collect textures
            if (adtData.Textures != null)
            {
                textures.AddRange(adtData.Textures);
                foreach (var t in adtData.Textures)
                    textureCollector.Add(t);
            }

            // Data for mesh generation
            var heights = new float[256][];
            var chunkPositions = new (float x, float y, float z)[256];
            var holes = new int[256];

            // ---------------------------------------------------------
            // 1. Process Chunks (MCNK) to build mesh data and layer info
            // ---------------------------------------------------------
            using (var alphaMs = new MemoryStream())
            using (var shadowMs = new MemoryStream())
            {
                foreach (var chunk in adtData.Chunks)
                {
                    int gridIndex = chunk.IndexY * 16 + chunk.IndexX;
                    
                    // Extract Layers
                    if (chunk.Layers != null)
                    {
                        foreach (var layer in chunk.Layers)
                        {
                            layers.Add(new VlmTextureLayer(
                                (uint)layer.TextureId,
                                layer.Flags,
                                (uint)layer.AlphaOffset,
                                (uint)layer.EffectId
                            ));
                        }
                    }

                    // Collect Heightmap Data for Mesh
                    if (chunk.Heights != null && chunk.Heights.Length > 0)
                    {
                        foreach (var h in chunk.Heights)
                        {
                            if (h < heightMin) heightMin = h;
                            if (h > heightMax) heightMax = h;
                        }
                        
                        if (gridIndex >= 0 && gridIndex < 256)
                        {
                            heights[gridIndex] = chunk.Heights;
                            chunkPositions[gridIndex] = (chunk.PositionX, chunk.PositionY, chunk.PositionZ);
                            holes[gridIndex] = chunk.Holes;
                        }
                    }
                    
                    // Collect Alpha Maps (MCAL)
                    if (chunk.AlphaMap != null)
                    {
                         // Generate PNG masks
                         var masks = AlphaMapGenerator.GenerateAlphaMasks(chunk);
                         foreach (var kvp in masks)
                         {
                             int layerIndex = kvp.Key;
                             byte[] pngBytes = kvp.Value;
                             
                             // Save to disk
                             // Naming: map_x_y_chunkIdx_layerIdx.png
                             var maskFilename = $"{mapName}_{x}_{y}_c{gridIndex}_l{layerIndex}.png";
                             var maskPath = Path.Combine(masksOutputDirectory, maskFilename);
                             File.WriteAllBytes(maskPath, pngBytes);
                             
                             layerMaskPaths.Add($"masks/{maskFilename}");
                         }
                         
                         // Keep raw binary too
                         alphaMs.Write(BitConverter.GetBytes(gridIndex));
                         alphaMs.Write(BitConverter.GetBytes(chunk.AlphaMap.Length));
                         alphaMs.Write(chunk.AlphaMap);
                    }


                    
                    // Collect Shadows (MCSH)
                    if (chunk.ShadowMap != null)
                    {
                         shadowMs.Write(BitConverter.GetBytes(gridIndex));
                         shadowMs.Write(BitConverter.GetBytes(chunk.ShadowMap.Length));
                         shadowMs.Write(chunk.ShadowMap);
                    }
                }
                
                if (alphaMs.Length > 0)
                    alphaMapsBase64 = Convert.ToBase64String(alphaMs.ToArray());
                    
                if (shadowMs.Length > 0)
                    shadowMapBase64 = Convert.ToBase64String(shadowMs.ToArray());
            }

            // ---------------------------------------------------------
            // 2. Generate Mesh (OBJ/MTL) Content (Legacy, optional)
            // ---------------------------------------------------------
            string materialName = $"{mapName}_{x}_{y}";
            var (objContent, mtlContent) = TerrainMeshExporter.GenerateObjStrings(
                heights, 
                chunkPositions, 
                holes, 
                materialName, 
                $"images/{imageRelativePath}"); 

            // ---------------------------------------------------------
            // 3. Build Compact Height Data
            // ---------------------------------------------------------
            var compactHeights = new List<VlmChunkHeights>();
            var chunkLayersList = new List<VlmChunkLayers>();
            var chunkPosFlat = new float[256 * 3];
            var holesFlat = new int[256];
            
            foreach (var chunk in adtData.Chunks)
            {
                int gridIndex = chunk.IndexY * 16 + chunk.IndexX;
                if (gridIndex < 0 || gridIndex >= 256) continue;
                
                // Heights
                if (chunk.Heights != null && chunk.Heights.Length > 0)
                {
                    compactHeights.Add(new VlmChunkHeights(gridIndex, chunk.Heights));
                }
                
                // Positions
                chunkPosFlat[gridIndex * 3 + 0] = chunk.PositionX;
                chunkPosFlat[gridIndex * 3 + 1] = chunk.PositionY;
                chunkPosFlat[gridIndex * 3 + 2] = chunk.PositionZ;
                
                // Holes
                holesFlat[gridIndex] = chunk.Holes;
                
                // Per-chunk layers with MCAL
                if (chunk.Layers != null && chunk.Layers.Count > 0)
                {
                    var layerArr = chunk.Layers.Select(l => new VlmTextureLayer(
                        (uint)l.TextureId, l.Flags, (uint)l.AlphaOffset, (uint)l.EffectId
                    )).ToArray();
                    
                    string? mcalB64 = null;
                    if (chunk.AlphaMap != null && chunk.AlphaMap.Length > 0)
                    {
                        mcalB64 = Convert.ToBase64String(chunk.AlphaMap);
                    }
                    
                    chunkLayersList.Add(new VlmChunkLayers(gridIndex, layerArr, mcalB64));
                }
            }

            // ---------------------------------------------------------
            // 4. Extract Objects
            // ---------------------------------------------------------
            if (adtData.M2Objects != null)
            {
                foreach (var m2 in adtData.M2Objects)
                {
                    objects.Add(new ObjectPlacement(
                        Path.GetFileNameWithoutExtension(m2.ModelName), 
                        m2.NameId,
                        m2.UniqueId,
                        m2.Position.X, m2.Position.Y, m2.Position.Z,
                        m2.Rotation.X, m2.Rotation.Y, m2.Rotation.Z,
                        m2.Scale,
                        "m2"));
                }
            }

            if (adtData.WmoObjects != null)
            {
                foreach (var wmo in adtData.WmoObjects)
                {
                    objects.Add(new ObjectPlacement(
                        Path.GetFileNameWithoutExtension(wmo.WmoName), 
                        wmo.NameId,
                        wmo.UniqueId,
                        wmo.Position.X, wmo.Position.Y, wmo.Position.Z,
                        wmo.Rotation.X, wmo.Rotation.Y, wmo.Rotation.Z,
                        wmo.Scale,
                        "wmo"));
                }
            }

            if (heightMin == float.MaxValue) heightMin = 0;
            if (heightMax == float.MinValue) heightMax = 0;

            // Construct Final Sample with enhanced structure
            return new VlmTrainingSample(
                $"images/{imageRelativePath}",
                new VlmTerrainData(
                    $"{mapName}_{x}_{y}",
                    compactHeights.Count > 0 ? compactHeights.ToArray() : null,
                    chunkPosFlat,
                    holesFlat,
                    objContent,
                    mtlContent,
                    alphaMapsBase64,
                    shadowMapBase64,
                    layerMaskPaths,
                    textures,
                    null, // TexturesExtracted - TODO: implement tileset extraction
                    chunkLayersList.Count > 0 ? chunkLayersList.ToArray() : null,
                    objects,
                    wdlHeights,
                    heightMin,
                    heightMax
                )
            );
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Failed to parse ADT {AdtPath}", adtPath);
            return null;
        }
    }
}

/// <summary>
/// Result of VLM dataset export.
/// </summary>
public record VlmExportResult(
    int TilesExported,
    int TilesSkipped,
    int UniqueTextures,
    string OutputDirectory);
