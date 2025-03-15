using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.UniqueIdAnalysis
{
    /// <summary>
    /// Generates CSV reports from UniqueID analysis results.
    /// </summary>
    public class CsvReportGenerator
    {
        private readonly ILogger<CsvReportGenerator>? _logger;
        private string _csvDirectory = string.Empty;

        /// <summary>
        /// Creates a new instance of the CsvReportGenerator class.
        /// </summary>
        /// <param name="logger">The logger to use.</param>
        public CsvReportGenerator(ILogger<CsvReportGenerator>? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Generates all CSV reports from the analysis results.
        /// </summary>
        /// <param name="result">The analysis result to generate reports from.</param>
        /// <param name="outputDirectory">The directory to write reports to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task GenerateAllCsvReportsAsync(UniqueIdAnalysisResult result, string outputDirectory)
        {
            // Create CSV directory
            _csvDirectory = Path.Combine(outputDirectory, "csv");
            if (!Directory.Exists(_csvDirectory))
            {
                Directory.CreateDirectory(_csvDirectory);
            }
            
            _logger?.LogInformation("Generating CSV reports in {CsvDirectory}...", _csvDirectory);
            
            // Generate summary CSV
            await GenerateSummaryCsvAsync(result);
            
            // Generate clusters CSV
            await GenerateClustersCsvAsync(result);
            
            // Generate assets CSV
            await GenerateAssetsCsvAsync(result);
            
            // Generate placements CSV (all placements across all clusters)
            await GeneratePlacementsCsvAsync(result);
            
            _logger?.LogInformation("CSV report generation complete.");
        }

        /// <summary>
        /// Generates a summary CSV file with overview information.
        /// </summary>
        /// <param name="result">The analysis result.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateSummaryCsvAsync(UniqueIdAnalysisResult result)
        {
            var filePath = Path.Combine(_csvDirectory, "summary.csv");
            _logger?.LogDebug("Generating summary CSV: {FilePath}", filePath);
            
            try
            {
                using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
                {
                    // Write header
                    await writer.WriteLineAsync("Metric,Value");
                    
                    // Write summary information
                    await writer.WriteLineAsync($"AnalysisDate,{result.AnalysisTime:yyyy-MM-dd HH:mm:ss}");
                    await writer.WriteLineAsync($"TotalAdtFiles,{result.TotalAdtFiles}");
                    await writer.WriteLineAsync($"TotalUniqueIds,{result.TotalUniqueIds}");
                    await writer.WriteLineAsync($"MinUniqueId,{result.MinUniqueId}");
                    await writer.WriteLineAsync($"MaxUniqueId,{result.MaxUniqueId}");
                    await writer.WriteLineAsync($"TotalAssets,{result.TotalAssets}");
                    await writer.WriteLineAsync($"TotalClusters,{result.Clusters.Count}");
                }
                
                _logger?.LogInformation("Generated summary CSV: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating summary CSV file");
            }
        }

        /// <summary>
        /// Generates a CSV file containing information about all clusters.
        /// </summary>
        /// <param name="result">The analysis result.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateClustersCsvAsync(UniqueIdAnalysisResult result)
        {
            var filePath = Path.Combine(_csvDirectory, "clusters.csv");
            _logger?.LogDebug("Generating clusters CSV: {FilePath}", filePath);
            
            try
            {
                using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
                {
                    // Write header
                    await writer.WriteLineAsync("ClusterId,MinId,MaxId,Count,Size,Density,AdtCount,AssetCount");
                    
                    // Write clusters information
                    int clusterIndex = 1;
                    foreach (var cluster in result.Clusters.OrderBy(c => c.MinId))
                    {
                        await writer.WriteLineAsync(
                            $"{clusterIndex++}," +
                            $"{cluster.MinId}," +
                            $"{cluster.MaxId}," +
                            $"{cluster.Count}," +
                            $"{cluster.MaxId - cluster.MinId + 1}," +
                            $"{cluster.Density:F6}," +
                            $"{cluster.AdtFiles.Count}," +
                            $"{cluster.Assets.Count}");
                    }
                }
                
                _logger?.LogInformation("Generated clusters CSV: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating clusters CSV file");
            }
        }

        /// <summary>
        /// Generates a CSV file containing information about all assets in clusters.
        /// </summary>
        /// <param name="result">The analysis result.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateAssetsCsvAsync(UniqueIdAnalysisResult result)
        {
            var filePath = Path.Combine(_csvDirectory, "assets.csv");
            _logger?.LogDebug("Generating assets CSV: {FilePath}", filePath);
            
            try
            {
                using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
                {
                    // Write header
                    await writer.WriteLineAsync("ClusterId,AssetPath,Type,Count");
                    
                    // Write assets information for each cluster
                    int clusterIndex = 1;
                    foreach (var cluster in result.Clusters.OrderBy(c => c.MinId))
                    {
                        // Group assets by path and type
                        var assetGroups = cluster.Assets
                            .GroupBy(a => new { a.AssetPath, a.Type })
                            .Select(g => new { 
                                AssetPath = g.Key.AssetPath, 
                                Type = g.Key.Type, 
                                Count = g.Count()
                            })
                            .OrderByDescending(a => a.Count);
                        
                        foreach (var asset in assetGroups)
                        {
                            // Escape any commas in the asset path
                            string escapedPath = asset.AssetPath.Replace("\"", "\"\"");
                            
                            await writer.WriteLineAsync(
                                $"{clusterIndex}," +
                                $"\"{escapedPath}\"," +
                                $"{asset.Type}," +
                                $"{asset.Count}");
                        }
                        
                        clusterIndex++;
                    }
                }
                
                _logger?.LogInformation("Generated assets CSV: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating assets CSV file");
            }
        }

        /// <summary>
        /// Generates a CSV file containing all asset placements across all clusters.
        /// </summary>
        /// <param name="result">The analysis result.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GeneratePlacementsCsvAsync(UniqueIdAnalysisResult result)
        {
            var filePath = Path.Combine(_csvDirectory, "placements.csv");
            _logger?.LogDebug("Generating placements CSV: {FilePath}", filePath);
            
            try
            {
                using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
                {
                    // Write header
                    await writer.WriteLineAsync("ClusterId,UniqueId,AssetPath,Type,MapName,AdtFile,PositionX,PositionY,PositionZ,RotationX,RotationY,RotationZ,Scale");
                    
                    // Write all placements from all clusters
                    int clusterIndex = 1;
                    foreach (var cluster in result.Clusters.OrderBy(c => c.MinId))
                    {
                        foreach (var asset in cluster.Assets.OrderBy(a => a.UniqueId))
                        {
                            // Escape any commas in the asset path and file names
                            string escapedPath = asset.AssetPath.Replace("\"", "\"\"");
                            string escapedMapName = asset.MapName.Replace("\"", "\"\"");
                            string escapedAdtFile = asset.AdtFile.Replace("\"", "\"\"");
                            
                            await writer.WriteLineAsync(
                                $"{clusterIndex}," +
                                $"{asset.UniqueId}," +
                                $"\"{escapedPath}\"," +
                                $"{asset.Type}," +
                                $"\"{escapedMapName}\"," +
                                $"\"{escapedAdtFile}\"," +
                                $"{asset.PositionX:F6}," +
                                $"{asset.PositionY:F6}," +
                                $"{asset.PositionZ:F6}," +
                                $"{asset.RotationX:F6}," +
                                $"{asset.RotationY:F6}," +
                                $"{asset.RotationZ:F6}," +
                                $"{asset.Scale:F6}");
                        }
                        
                        clusterIndex++;
                    }
                }
                
                _logger?.LogInformation("Generated placements CSV: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating placements CSV file");
            }
        }
    }
} 