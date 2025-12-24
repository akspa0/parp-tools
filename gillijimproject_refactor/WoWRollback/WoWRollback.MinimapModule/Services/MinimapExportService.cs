using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WoWRollback.MinimapModule.Models;

namespace WoWRollback.MinimapModule.Services;

/// <summary>
/// Main service for generating VLM training datasets from minimap + ADT data.
/// </summary>
public class MinimapExportService
{
    private readonly ILogger<MinimapExportService> _logger;
    private readonly AdtMetadataExtractor _metadataExtractor;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public MinimapExportService(
        ILogger<MinimapExportService> logger,
        AdtMetadataExtractor metadataExtractor)
    {
        _logger = logger;
        _metadataExtractor = metadataExtractor;
    }

    /// <summary>
    /// Generate VLM training dataset from ADT + minimap directories.
    /// </summary>
    /// <param name="adtDirectory">Directory containing ADT files (e.g., World\Maps\development)</param>
    /// <param name="minimapDirectory">Directory containing minimap BLP files (e.g., World\Minimaps\development)</param>
    /// <param name="mapName">Map name (e.g., "development")</param>
    /// <param name="outputDirectory">Output directory for dataset</param>
    public async Task<DatasetExportResult> GenerateDatasetAsync(
        string adtDirectory,
        string minimapDirectory,
        string mapName,
        string outputDirectory)
    {
        _logger.LogInformation("Starting VLM dataset export for map: {MapName}", mapName);
        _logger.LogInformation("ADT directory: {AdtDir}", adtDirectory);
        _logger.LogInformation("Minimap directory: {MinimapDir}", minimapDirectory);

        var imagesDir = Path.Combine(outputDirectory, "images");
        var metadataDir = Path.Combine(outputDirectory, "metadata");
        var texturesDir = Path.Combine(outputDirectory, "textures");

        Directory.CreateDirectory(imagesDir);
        Directory.CreateDirectory(metadataDir);
        Directory.CreateDirectory(texturesDir);

        int tilesProcessed = 0;
        int tilesSkipped = 0;
        var allTextures = new HashSet<string>();

        // Scan for ADT tiles (0-63, 0-63)
        for (int x = 0; x < 64; x++)
        {
            for (int y = 0; y < 64; y++)
            {
                var adtPath = Path.Combine(adtDirectory, $"{mapName}_{x}_{y}.adt");
                var minimapPath = Path.Combine(minimapDirectory, $"map{x:D2}_{y:D2}.blp");

                // Check if both ADT and minimap exist
                bool hasAdt = File.Exists(adtPath) || 
                             File.Exists(Path.Combine(adtDirectory, $"{mapName}_{x}_{y}_obj0.adt"));
                bool hasMinimap = File.Exists(minimapPath);

                if (!hasAdt || !hasMinimap)
                {
                    continue;
                }

                try
                {
                    // Extract metadata from ADT
                    var sample = _metadataExtractor.ExtractMetadata(adtDirectory, mapName, x, y);
                    if (sample == null)
                    {
                        tilesSkipped++;
                        continue;
                    }

                    // Convert minimap BLP to PNG
                    var outputImagePath = Path.Combine(imagesDir, $"{mapName}_{x}_{y}.png");
                    if (!BlpConverter.ConvertToPng(minimapPath, outputImagePath))
                    {
                        _logger.LogWarning("Failed to convert minimap: {Minimap}", minimapPath);
                        tilesSkipped++;
                        continue;
                    }

                    // Write metadata JSON
                    var metadataPath = Path.Combine(metadataDir, $"{mapName}_{x}_{y}.json");
                    var json = JsonSerializer.Serialize(sample, JsonOptions);
                    await File.WriteAllTextAsync(metadataPath, json);

                    // Track unique textures
                    foreach (var chunk in sample.Textures)
                    {
                        foreach (var layer in chunk.Layers)
                        {
                            allTextures.Add(layer);
                        }
                    }

                    tilesProcessed++;

                    if (tilesProcessed % 10 == 0)
                    {
                        _logger.LogInformation("Progress: {Count} tiles processed", tilesProcessed);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to process tile {X}_{Y}", x, y);
                    tilesSkipped++;
                }
            }
        }

        // Write texture database
        var textureDbPath = Path.Combine(outputDirectory, "texture_database.json");
        var textureDb = new TextureDatabase(allTextures.Count, allTextures.ToList());
        await File.WriteAllTextAsync(textureDbPath, JsonSerializer.Serialize(textureDb, JsonOptions));

        _logger.LogInformation("Dataset export complete: {Processed} tiles, {Skipped} skipped, {Textures} unique textures",
            tilesProcessed, tilesSkipped, allTextures.Count);

        return new DatasetExportResult(
            tilesProcessed,
            tilesSkipped,
            allTextures.Count,
            outputDirectory);
    }
}

/// <summary>
/// Result of dataset export operation.
/// </summary>
public record DatasetExportResult(
    int TilesProcessed,
    int TilesSkipped,
    int UniqueTextures,
    string OutputDirectory);

/// <summary>
/// Texture database for VLM training.
/// </summary>
public record TextureDatabase(
    int Count,
    List<string> Textures);
