using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WoWRollback.Core.Services.Archive;
using WoWRollback.Core.Services.Minimap;
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
        IProgress<string>? progress = null)
    {
        progress?.Report($"Starting VLM export for map: {mapName}");
        _logger?.LogInformation("Starting VLM export for map: {MapName}", mapName);

        var imagesDir = Path.Combine(outputDir, "images");
        var metadataDir = Path.Combine(outputDir, "metadata");
        
        Directory.CreateDirectory(imagesDir);
        Directory.CreateDirectory(metadataDir);

        int tilesExported = 0;
        int tilesSkipped = 0;
        int tilesChecked = 0;
        int tilesResolved = 0;
        var allTextures = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Scan all possible tiles (0-63, 0-63)
        for (int x = 0; x < 64; x++)
        {
            for (int y = 0; y < 64; y++)
            {
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
                    var outputImagePath = Path.Combine(imagesDir, $"{mapName}_{x}_{y}.png");
                    if (!BlpConverter.ConvertToPng(blpStream, outputImagePath))
                    {
                        blpStream.Dispose();
                        tilesSkipped++;
                        continue;
                    }
                    blpStream.Dispose();

                    // Try to load ADT and extract metadata
                    var sample = TryExtractAdtMetadata(source, mapName, x, y, allTextures);
                    
                    // If no ADT, create minimal sample with just terrain summary
                    if (sample == null)
                    {
                        sample = new VlmTrainingSample(
                            $"{mapName}_{x}_{y}",
                            new List<ChunkTextureInfo>(),
                            new List<ObjectPlacement>(),
                            new TerrainSummary(0, 0, false));
                    }

                    // Write metadata JSON
                    var metadataPath = Path.Combine(metadataDir, $"{mapName}_{x}_{y}.json");
                    var json = JsonSerializer.Serialize(sample, JsonOptions);
                    await File.WriteAllTextAsync(metadataPath, json);

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
            tilesExported, tilesSkipped, allTextures.Count);

        return new VlmExportResult(tilesExported, tilesSkipped, allTextures.Count, outputDir);
    }

    private VlmTrainingSample? TryExtractAdtMetadata(
        IArchiveSource source,
        string mapName,
        int x, int y,
        HashSet<string> textureCollector)
    {
        // Try to load ADT from archive
        var adtPath = $"World\\Maps\\{mapName}\\{mapName}_{x}_{y}.adt";
        
        Stream? adtStream = null;
        try
        {
            if (!source.FileExists(adtPath))
                return null;
            adtStream = source.OpenFile(adtPath);
        }
        catch
        {
            return null;
        }

        if (adtStream == null)
            return null;

        try
        {
            using var ms = new MemoryStream();
            adtStream.CopyTo(ms);
            adtStream.Dispose();
            ms.Position = 0;
            
            var terrain = new Warcraft.NET.Files.ADT.Terrain.Wotlk.Terrain(ms.ToArray());
            
            var textures = new List<ChunkTextureInfo>();
            var objects = new List<ObjectPlacement>();
            float heightMin = float.MaxValue;
            float heightMax = float.MinValue;
            bool hasWater = false;

            // Extract chunk-level texture info
            if (terrain.Chunks != null)
            {
                foreach (var chunk in terrain.Chunks)
                {
                    if (chunk?.Header == null) continue;

                    var chunkX = (int)chunk.Header.MapIndexX;
                    var chunkY = (int)chunk.Header.MapIndexY;

                    // Extract texture layers
                    var layers = new List<string>();
                    if (chunk.TextureLayers?.Layers != null && terrain.Textures?.Filenames != null)
                    {
                        foreach (var layer in chunk.TextureLayers.Layers)
                        {
                            if (layer.TextureID < terrain.Textures.Filenames.Count)
                            {
                                var texPath = terrain.Textures.Filenames[(int)layer.TextureID];
                                layers.Add(texPath);
                                textureCollector.Add(texPath);
                            }
                        }
                    }

                    textures.Add(new ChunkTextureInfo(new[] { chunkX, chunkY }, layers.ToArray()));

                    // Extract height bounds
                    if (chunk.Heightmap?.Vertices != null)
                    {
                        foreach (var vertex in chunk.Heightmap.Vertices)
                        {
                            var absoluteHeight = chunk.Header.MapTilePosition.Z + vertex;
                            if (absoluteHeight < heightMin) heightMin = absoluteHeight;
                            if (absoluteHeight > heightMax) heightMax = absoluteHeight;
                        }
                    }

                    // Check for water
                    if (chunk.Header.LiquidSize > 8)
                    {
                        hasWater = true;
                    }
                }
            }

            // Extract object placements
            if (terrain.Models?.Filenames != null && terrain.ModelPlacementInfo?.MDDFEntries != null)
            {
                foreach (var m2 in terrain.ModelPlacementInfo.MDDFEntries)
                {
                    var name = m2.NameId < terrain.Models.Filenames.Count
                        ? Path.GetFileNameWithoutExtension(terrain.Models.Filenames[(int)m2.NameId])
                        : $"m2_{m2.NameId}";
                    objects.Add(new ObjectPlacement(name, m2.Position.X, m2.Position.Y, m2.Position.Z, "m2"));
                }
            }

            if (terrain.WorldModelObjects?.Filenames != null && terrain.WorldModelObjectPlacementInfo?.MODFEntries != null)
            {
                foreach (var wmo in terrain.WorldModelObjectPlacementInfo.MODFEntries)
                {
                    var name = wmo.NameId < terrain.WorldModelObjects.Filenames.Count
                        ? Path.GetFileNameWithoutExtension(terrain.WorldModelObjects.Filenames[(int)wmo.NameId])
                        : $"wmo_{wmo.NameId}";
                    objects.Add(new ObjectPlacement(name, wmo.Position.X, wmo.Position.Y, wmo.Position.Z, "wmo"));
                }
            }

            if (heightMin == float.MaxValue) heightMin = 0;
            if (heightMax == float.MinValue) heightMax = 0;

            return new VlmTrainingSample(
                $"{mapName}_{x}_{y}",
                textures,
                objects,
                new TerrainSummary(heightMin, heightMax, hasWater));
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
