using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Utilities;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Generates JSON reports from ADT analysis results.
    /// </summary>
    public class JsonReportGenerator
    {
        private readonly ILogger<JsonReportGenerator> _logger;
        private readonly JsonSerializerOptions _jsonOptions;

        /// <summary>
        /// Creates a new instance of the JsonReportGenerator class.
        /// </summary>
        /// <param name="logger">The logger to use.</param>
        public JsonReportGenerator(ILogger<JsonReportGenerator> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true
            };
        }

        /// <summary>
        /// Generates all JSON reports for the analysis results.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="summary">The analysis summary.</param>
        /// <param name="outputDirectory">The directory to write reports to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task GenerateAllReportsAsync(List<AdtAnalysisResult> results, AnalysisSummary summary, string outputDirectory)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));
            if (summary == null)
                throw new ArgumentNullException(nameof(summary));
            if (string.IsNullOrEmpty(outputDirectory))
                throw new ArgumentException("Output directory cannot be null or empty.", nameof(outputDirectory));

            // Create output directory if it doesn't exist
            if (!Directory.Exists(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            // Create a JSON subdirectory
            var jsonDirectory = Path.Combine(outputDirectory, "json");
            if (!Directory.Exists(jsonDirectory))
                Directory.CreateDirectory(jsonDirectory);

            _logger.LogInformation("Generating JSON reports in {JsonDirectory}...", jsonDirectory);

            try
            {
                // Generate summary JSON
                await GenerateSummaryAsync(summary, jsonDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating summary JSON, continuing with other reports");
            }

            try
            {
                // Generate area IDs JSON
                await GenerateAreaIdsAsync(summary, jsonDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating area IDs JSON, continuing with other reports");
            }

            try
            {
                // Generate results JSON
                await GenerateResultsAsync(results, jsonDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating results JSON, continuing with other reports");
            }

            try
            {
                // Generate texture references JSON
                await GenerateTextureReferencesAsync(results, jsonDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating texture references JSON, continuing with other reports");
            }

            try
            {
                // Generate model references JSON
                await GenerateModelReferencesAsync(results, jsonDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating model references JSON, continuing with other reports");
            }

            try
            {
                // Generate WMO references JSON
                await GenerateWmoReferencesAsync(results, jsonDirectory);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating WMO references JSON, continuing with other reports");
            }

            // These are high priority - show detailed logging
            _logger.LogInformation("Generating model and WMO placement JSON files...");
            
            try
            {
                // Generate model placements JSON
                await GenerateModelPlacementsAsync(results, jsonDirectory);
                _logger.LogInformation("Generated model placements JSON successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating model placements JSON");
            }

            try
            {
                // Generate WMO placements JSON
                await GenerateWmoPlacementsAsync(results, jsonDirectory);
                _logger.LogInformation("Generated WMO placements JSON successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating WMO placements JSON");
            }

            _logger.LogInformation("JSON report generation complete.");
        }

        /// <summary>
        /// Generates a JSON file containing the analysis summary.
        /// </summary>
        /// <param name="summary">The analysis summary.</param>
        /// <param name="outputDirectory">The directory to write the file to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateSummaryAsync(AnalysisSummary summary, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "summary.json");
            _logger.LogDebug("Generating summary JSON: {FilePath}", filePath);

            var summaryJson = new
            {
                StartTime = summary.StartTime,
                EndTime = summary.EndTime,
                Duration = summary.Duration.TotalSeconds,
                TotalFiles = summary.TotalFiles,
                ProcessedFiles = summary.ProcessedFiles,
                FailedFiles = summary.FailedFiles,
                TotalTextureReferences = summary.TotalTextureReferences,
                TotalModelReferences = summary.TotalModelReferences,
                TotalWmoReferences = summary.TotalWmoReferences,
                TotalModelPlacements = summary.TotalModelPlacements,
                TotalWmoPlacements = summary.TotalWmoPlacements,
                TotalTerrainChunks = summary.TotalTerrainChunks,
                MaxUniqueId = summary.MaxUniqueId
            };

            var json = JsonSerializer.Serialize(summaryJson, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }

        /// <summary>
        /// Generates a JSON file containing area IDs.
        /// </summary>
        /// <param name="summary">The analysis summary.</param>
        /// <param name="outputDirectory">The directory to write the file to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateAreaIdsAsync(AnalysisSummary summary, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "area_ids.json");
            _logger.LogDebug("Generating area IDs JSON: {FilePath}", filePath);

            var areaIdsJson = summary.AreaIdMap.Select(area => new
            {
                AreaId = area.Key,
                Count = area.Value,
            }).OrderBy(a => a.AreaId).ToList();

            var json = JsonSerializer.Serialize(areaIdsJson, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }

        /// <summary>
        /// Generates a JSON file containing the analysis results.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the file to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateResultsAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "results.json");
            _logger.LogDebug("Generating results JSON: {FilePath}", filePath);

            var resultsJson = results.Select(r => new
            {
                FileName = r.FileName,
                FilePath = r.FilePath,
                XCoord = r.XCoord,
                YCoord = r.YCoord,
                AdtVersion = r.AdtVersion,
                TextureReferencesCount = r.TextureReferences.Count,
                ModelReferencesCount = r.ModelReferences.Count,
                WmoReferencesCount = r.WmoReferences.Count,
                ModelPlacementsCount = r.ModelPlacements.Count,
                WmoPlacementsCount = r.WmoPlacements.Count,
                TerrainChunksCount = r.TerrainChunks.Count,
                UniqueIdsCount = r.UniqueIds.Count
            }).ToList();

            var json = JsonSerializer.Serialize(resultsJson, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }

        /// <summary>
        /// Generates a JSON file containing texture references.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the file to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateTextureReferencesAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "texture_references.json");
            _logger.LogDebug("Generating texture references JSON: {FilePath}", filePath);

            var textureReferencesJson = results
                .SelectMany(r => r.TextureReferences.Select(t => new
                {
                    FileName = r.FileName,
                    TexturePath = t.OriginalPath,
                    NormalizedPath = t.NormalizedPath,
                    ExistsInListfile = t.ExistsInListfile,
                    IsValid = t.IsValid,
                    RepairedPath = t.RepairedPath
                }))
                .GroupBy(t => t.TexturePath)
                .Select(g => new
                {
                    TexturePath = g.Key,
                    Count = g.Count(),
                    UsedIn = g.Select(t => t.FileName).Distinct().ToList()
                })
                .OrderByDescending(t => t.Count)
                .ToList();

            var json = JsonSerializer.Serialize(textureReferencesJson, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }

        /// <summary>
        /// Generates a JSON file containing model references.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the file to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateModelReferencesAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "model_references.json");
            _logger.LogDebug("Generating model references JSON: {FilePath}", filePath);

            var modelReferencesJson = results
                .SelectMany(r => r.ModelReferences.Select(m => new
                {
                    FileName = r.FileName,
                    ModelPath = m.OriginalPath,
                    NormalizedPath = m.NormalizedPath,
                    ExistsInListfile = m.ExistsInListfile,
                    IsValid = m.IsValid,
                    RepairedPath = m.RepairedPath
                }))
                .GroupBy(m => m.ModelPath)
                .Select(g => new
                {
                    ModelPath = g.Key,
                    Count = g.Count(),
                    UsedIn = g.Select(m => m.FileName).Distinct().ToList()
                })
                .OrderByDescending(m => m.Count)
                .ToList();

            var json = JsonSerializer.Serialize(modelReferencesJson, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }

        /// <summary>
        /// Generates a JSON file containing WMO references.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the file to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateWmoReferencesAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "wmo_references.json");
            _logger.LogDebug("Generating WMO references JSON: {FilePath}", filePath);

            var wmoReferencesJson = results
                .SelectMany(r => r.WmoReferences.Select(w => new
                {
                    FileName = r.FileName,
                    WmoPath = w.OriginalPath,
                    NormalizedPath = w.NormalizedPath,
                    ExistsInListfile = w.ExistsInListfile,
                    IsValid = w.IsValid,
                    RepairedPath = w.RepairedPath
                }))
                .GroupBy(w => w.WmoPath)
                .Select(g => new
                {
                    WmoPath = g.Key,
                    Count = g.Count(),
                    UsedIn = g.Select(w => w.FileName).Distinct().ToList()
                })
                .OrderByDescending(w => w.Count)
                .ToList();

            var json = JsonSerializer.Serialize(wmoReferencesJson, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }

        /// <summary>
        /// Generates a JSON file for model placements.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The output directory.</param>
        /// <returns>A task.</returns>
        private async Task GenerateModelPlacementsAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "model_placements.json");
            _logger.LogDebug("Generating model placements JSON: {FilePath}", filePath);

            var modelPlacementsJson = results.Select(r => new
            {
                FileName = r.FileName,
                XCoord = r.XCoord,
                YCoord = r.YCoord,
                ModelPlacements = r.ModelPlacements.Select(m => new
                {
                    UniqueId = m.UniqueId,
                    NameId = m.NameId,
                    Name = m.Name,
                    Position = new { X = m.Position.X, Y = m.Position.Y, Z = m.Position.Z },
                    Rotation = new { X = m.Rotation.X, Y = m.Rotation.Y, Z = m.Rotation.Z },
                    Scale = m.Scale,
                    Flags = m.Flags
                }).ToList()
            }).ToList();

            var json = JsonSerializer.Serialize(modelPlacementsJson, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }

        /// <summary>
        /// Generates a JSON file for WMO placements.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The output directory.</param>
        /// <returns>A task.</returns>
        private async Task GenerateWmoPlacementsAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "wmo_placements.json");
            _logger.LogDebug("Generating WMO placements JSON: {FilePath}", filePath);

            var wmoPlacementsJson = results.Select(r => new
            {
                FileName = r.FileName,
                XCoord = r.XCoord,
                YCoord = r.YCoord,
                WmoPlacements = r.WmoPlacements.Select(w => new
                {
                    UniqueId = w.UniqueId,
                    NameId = w.NameId,
                    Name = w.Name,
                    Position = new { X = w.Position.X, Y = w.Position.Y, Z = w.Position.Z },
                    Rotation = new { X = w.Rotation.X, Y = w.Rotation.Y, Z = w.Rotation.Z },
                    DoodadSet = w.DoodadSet,
                    NameSet = w.NameSet,
                    Flags = w.Flags
                }).ToList()
            }).ToList();

            var json = JsonSerializer.Serialize(wmoPlacementsJson, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);
        }
    }
} 