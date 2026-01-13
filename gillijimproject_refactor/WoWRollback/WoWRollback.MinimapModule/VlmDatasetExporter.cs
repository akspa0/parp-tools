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
        int limit = int.MaxValue)
    {
        progress?.Report($"Starting VLM export for map: {mapName}");
        _logger?.LogInformation("Starting VLM export for map: {MapName}", mapName);

        var imagesDir = Path.Combine(outputDir, "images");
        var datasetDir = Path.Combine(outputDir, "dataset");
        var masksDir = Path.Combine(outputDir, "masks");
        
        Directory.CreateDirectory(imagesDir);
        Directory.CreateDirectory(datasetDir);
        Directory.CreateDirectory(masksDir);

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
                    var sample = TryExtractAdtMetadata(source, mapName, x, y, imageFileName, masksDir, allTextures);
                    
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
        HashSet<string> textureCollector)
    {
        // Try to load ADT from archive
        var adtPath = $"World\\Maps\\{mapName}\\{mapName}_{x}_{y}.adt";
        
        byte[]? adtDataBytes = null;
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
        catch
        {
            return null;
        }

        if (adtDataBytes == null || adtDataBytes.Length == 0)
            return null;

        try
        {
            // Parse ADT using Core parser
            var adtData = AdtParser.Parse(adtDataBytes, mapName, x, y);
            
            var textures = new List<string>();
            var layers = new List<VlmTextureLayer>();
            var layerMaskPaths = new List<string>();
            var objects = new List<ObjectPlacement>();
            float heightMin = float.MaxValue;
            float heightMax = float.MinValue;
            
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
            // 2. Generate Mesh (OBJ/MTL) Content
            // ---------------------------------------------------------
            string materialName = $"{mapName}_{x}_{y}";
            var (objContent, mtlContent) = TerrainMeshExporter.GenerateObjStrings(
                heights, 
                chunkPositions, 
                holes, 
                materialName, 
                $"images/{imageRelativePath}"); 

            // ---------------------------------------------------------
            // 3. Extract Objects
            // ---------------------------------------------------------
            if (adtData.M2Objects != null)
            {
                foreach (var m2 in adtData.M2Objects)
                {
                    // Convert rotation to Euler degrees? Or keep as radians?
                    // ObjectPlacement usually expects degrees.
                    // AdtParser gives Euler radians (if direct read from MDDF) or degrees?
                    // MDDF stores (X,Y,Z) in degrees or radians? 
                    // Usually WoW stores Euler angles in degrees in MDDF? No, likely Radians.
                    // Let's assume radians for now and convert if needed. 
                    // Actually, let's keep raw values. VLM can learn whatever distribution.
                    
                    objects.Add(new ObjectPlacement(
                        Path.GetFileNameWithoutExtension(m2.ModelName), 
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
                        wmo.UniqueId,
                        wmo.Position.X, wmo.Position.Y, wmo.Position.Z,
                        wmo.Rotation.X, wmo.Rotation.Y, wmo.Rotation.Z,
                        wmo.Scale,
                        "wmo"));
                }
            }

            if (heightMin == float.MaxValue) heightMin = 0;
            if (heightMax == float.MinValue) heightMax = 0;

            // Construct Final Sample
            return new VlmTrainingSample(
                $"images/{imageRelativePath}",
                new VlmTerrainData(
                    $"{mapName}_{x}_{y}",
                    objContent,
                    mtlContent,
                    alphaMapsBase64,
                    shadowMapBase64,
                    layerMaskPaths,
                    textures,
                    layers,
                    objects,
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
