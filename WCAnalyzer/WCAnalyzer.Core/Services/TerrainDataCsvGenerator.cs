using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Utilities;

namespace WCAnalyzer.Core.Services;

/// <summary>
/// Service for generating CSV reports for terrain data.
/// </summary>
public class TerrainDataCsvGenerator
{
    private readonly ILogger<TerrainDataCsvGenerator> _logger;

    /// <summary>
    /// Creates a new instance of the TerrainDataCsvGenerator class.
    /// </summary>
    /// <param name="logger">The logging service to use.</param>
    public TerrainDataCsvGenerator(ILogger<TerrainDataCsvGenerator> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// The directory to write CSV files to.
    /// </summary>
    public string CsvDirectory { get; private set; }

    /// <summary>
    /// Generates all CSV reports.
    /// </summary>
    /// <param name="results">The ADT analysis results.</param>
    /// <param name="outputDirectory">The output directory.</param>
    /// <returns>A task.</returns>
    public async Task GenerateAllCsvAsync(List<AdtAnalysisResult> results, string outputDirectory)
    {
        if (results == null)
            throw new ArgumentNullException(nameof(results));
        if (string.IsNullOrEmpty(outputDirectory))
            throw new ArgumentException("Output directory cannot be null or empty.", nameof(outputDirectory));

        // Create output directory if it doesn't exist
        if (!Directory.Exists(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        // Create CSV directory
        CsvDirectory = Path.Combine(outputDirectory, "csv");
        if (!Directory.Exists(CsvDirectory))
            Directory.CreateDirectory(CsvDirectory);

        _logger.LogInformation("Generating terrain data CSV reports in {CsvDirectory}...", CsvDirectory);

        // Try to generate heightmap CSV, but continue if it fails
        try
        {
            await GenerateHeightmapCsvAsync(results);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating heightmap CSV, continuing with other reports");
        }

        // Try to generate normal vectors CSV, but continue if it fails
        try
        {
            await GenerateNormalVectorsCsvAsync(results);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating normal vectors CSV, continuing with other reports");
        }

        // Try to generate texture layers CSV, but continue if it fails
        try
        {
            await GenerateTextureLayersCsvAsync(results);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating texture layers CSV, continuing with other reports");
        }

        // Try to generate alpha maps CSV, but continue if it fails
        try
        {
            await GenerateAlphaMapsCsvAsync(results);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating alpha maps CSV, continuing with other reports");
        }

        // Generate model placements CSV - these are high priority
        _logger.LogInformation("Generating model and WMO placement CSV files...");
        await GenerateModelPlacementsCsvAsync(results);
        await GenerateWmoPlacementsCsvAsync(results);

        _logger.LogInformation("CSV report generation complete.");
    }

    /// <summary>
    /// Generates a CSV file containing heightmap data.
    /// </summary>
    /// <param name="results">The ADT analysis results.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public async Task GenerateHeightmapCsvAsync(List<AdtAnalysisResult> results)
    {
        var filePath = Path.Combine(CsvDirectory, "heightmap.csv");
        _logger.LogDebug("Generating heightmap CSV: {FilePath}", filePath);

        using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
        {
            // Write headers
            await writer.WriteLineAsync("FileName,ChunkX,ChunkY,AreaId,Flags,Holes,X,Y,Z");

            // Write data
            foreach (var result in results)
            {
                foreach (var chunk in result.TerrainChunks)
                {
                    for (int y = 0; y < 17; y++)
                    {
                        for (int x = 0; x < 17; x++)
                        {
                            var height = chunk.Heights[y * 17 + x];
                            var worldX = chunk.WorldPosition.X + (x * (533.33333f / 16));
                            var worldY = chunk.WorldPosition.Y + (y * (533.33333f / 16));
                            var worldZ = chunk.WorldPosition.Z + height;

                            await writer.WriteLineAsync($"{result.FileName},{chunk.Position.X},{chunk.Position.Y},{chunk.AreaId},{chunk.Flags},{chunk.Holes},{worldX},{worldY},{worldZ}");
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Generates a CSV file containing normal vector data.
    /// </summary>
    /// <param name="results">The ADT analysis results.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public async Task GenerateNormalVectorsCsvAsync(List<AdtAnalysisResult> results)
    {
        var filePath = Path.Combine(CsvDirectory, "normal_vectors.csv");
        _logger.LogDebug("Generating normal vectors CSV: {FilePath}", filePath);

        using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
        {
            // Write headers
            await writer.WriteLineAsync("FileName,ChunkX,ChunkY,X,Y,Z,NormalX,NormalY,NormalZ");

            // Write data
            foreach (var result in results)
            {
                foreach (var chunk in result.TerrainChunks)
                {
                    for (int y = 0; y < 17; y++)
                    {
                        for (int x = 0; x < 17; x++)
                        {
                            var height = chunk.Heights[y * 17 + x];
                            var normal = chunk.Normals[y * 17 + x];
                            var worldX = chunk.WorldPosition.X + (x * (533.33333f / 16));
                            var worldY = chunk.WorldPosition.Y + (y * (533.33333f / 16));
                            var worldZ = chunk.WorldPosition.Z + height;

                            await writer.WriteLineAsync($"{result.FileName},{chunk.Position.X},{chunk.Position.Y},{worldX},{worldY},{worldZ},{normal.X},{normal.Y},{normal.Z}");
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Generates a CSV file containing texture layer data.
    /// </summary>
    /// <param name="results">The ADT analysis results.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public async Task GenerateTextureLayersCsvAsync(List<AdtAnalysisResult> results)
    {
        var filePath = Path.Combine(CsvDirectory, "texture_layers.csv");
        _logger.LogDebug("Generating texture layers CSV: {FilePath}", filePath);

        using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
        {
            // Write headers
            await writer.WriteLineAsync("FileName,ChunkX,ChunkY,LayerIndex,TextureId,TextureName,Flags,EffectId,AlphaMapOffset,AlphaMapSize");

            // Write data
            foreach (var result in results)
            {
                foreach (var chunk in result.TerrainChunks)
                {
                    for (int i = 0; i < chunk.TextureLayers.Count; i++)
                    {
                        var layer = chunk.TextureLayers[i];
                        await writer.WriteLineAsync($"{result.FileName},{chunk.Position.X},{chunk.Position.Y},{i},{layer.TextureId},{layer.TextureName},{layer.Flags},{layer.EffectId},{layer.AlphaMapOffset},{layer.AlphaMapSize}");
                    }
                }
            }
        }
    }

    /// <summary>
    /// Generates a CSV file containing alpha map data.
    /// </summary>
    /// <param name="results">The ADT analysis results.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public async Task GenerateAlphaMapsCsvAsync(List<AdtAnalysisResult> results)
    {
        var filePath = Path.Combine(CsvDirectory, "alpha_maps.csv");
        _logger.LogDebug("Generating alpha maps CSV: {FilePath}", filePath);

        using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
        {
            // Write headers
            await writer.WriteLineAsync("FileName,ChunkX,ChunkY,LayerIndex,AlphaMapX,AlphaMapY,AlphaValue");

            // Write data
            foreach (var result in results)
            {
                foreach (var chunk in result.TerrainChunks)
                {
                    for (int layerIndex = 0; layerIndex < chunk.TextureLayers.Count; layerIndex++)
                    {
                        var layer = chunk.TextureLayers[layerIndex];
                        if (layer.AlphaMap != null && layer.AlphaMap.Length > 0)
                        {
                            int alphaMapSize = (int)Math.Sqrt(layer.AlphaMap.Length);
                            for (int y = 0; y < alphaMapSize; y++)
                            {
                                for (int x = 0; x < alphaMapSize; x++)
                                {
                                    var alphaValue = layer.AlphaMap[y * alphaMapSize + x];
                                    await writer.WriteLineAsync($"{result.FileName},{chunk.Position.X},{chunk.Position.Y},{layerIndex},{x},{y},{alphaValue}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Generates a CSV file containing model placement data.
    /// </summary>
    /// <param name="results">The ADT analysis results.</param>
    /// <returns>A task.</returns>
    public async Task GenerateModelPlacementsCsvAsync(List<AdtAnalysisResult> results)
    {
        var filePath = Path.Combine(CsvDirectory, "model_placements.csv");
        _logger.LogInformation("Generating model placements CSV: {FilePath}", filePath);

        // Count total model placements
        int totalModelPlacements = results.Sum(r => r.ModelPlacements.Count);
        _logger.LogInformation("Total model placements to export: {Count}", totalModelPlacements);

        try
        {
            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                // Write headers
                await writer.WriteLineAsync("FileName,XCoord,YCoord,UniqueId,NameId,Name,PositionX,PositionY,PositionZ,RotationX,RotationY,RotationZ,Scale,Flags");

                // Write data
                int exportedCount = 0;
                foreach (var result in results)
                {
                    foreach (var model in result.ModelPlacements)
                    {
                        await writer.WriteLineAsync($"{result.FileName},{result.XCoord},{result.YCoord},{model.UniqueId},{model.NameId},{model.Name},{model.Position.X},{model.Position.Y},{model.Position.Z},{model.Rotation.X},{model.Rotation.Y},{model.Rotation.Z},{model.Scale},{model.Flags}");
                        exportedCount++;
                    }
                }
                
                _logger.LogInformation("Successfully exported {Count} model placements to CSV", exportedCount);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating model placements CSV file");
            throw; // Re-throw to allow the caller to handle the exception
        }
    }

    /// <summary>
    /// Generates a CSV file containing WMO placement data.
    /// </summary>
    /// <param name="results">The ADT analysis results.</param>
    /// <returns>A task.</returns>
    public async Task GenerateWmoPlacementsCsvAsync(List<AdtAnalysisResult> results)
    {
        var filePath = Path.Combine(CsvDirectory, "wmo_placements.csv");
        _logger.LogInformation("Generating WMO placements CSV: {FilePath}", filePath);

        // Count total WMO placements
        int totalWmoPlacements = results.Sum(r => r.WmoPlacements.Count);
        _logger.LogInformation("Total WMO placements to export: {Count}", totalWmoPlacements);

        try
        {
            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                // Write headers
                await writer.WriteLineAsync("FileName,XCoord,YCoord,UniqueId,NameId,Name,PositionX,PositionY,PositionZ,RotationX,RotationY,RotationZ,DoodadSet,NameSet,Scale,Flags");

                // Write data
                int exportedCount = 0;
                foreach (var result in results)
                {
                    foreach (var wmo in result.WmoPlacements)
                    {
                        await writer.WriteLineAsync($"{result.FileName},{result.XCoord},{result.YCoord},{wmo.UniqueId},{wmo.NameId},{wmo.Name},{wmo.Position.X},{wmo.Position.Y},{wmo.Position.Z},{wmo.Rotation.X},{wmo.Rotation.Y},{wmo.Rotation.Z},{wmo.DoodadSet},{wmo.NameSet},{wmo.Scale},{wmo.Flags}");
                        exportedCount++;
                    }
                }
                
                _logger.LogInformation("Successfully exported {Count} WMO placements to CSV", exportedCount);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating WMO placements CSV file");
            throw; // Re-throw to allow the caller to handle the exception
        }
    }
}