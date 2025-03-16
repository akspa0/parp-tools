using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Utilities;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for generating reports from ADT analysis results.
    /// </summary>
    public class ReportGenerator
    {
        private readonly ILogger<ReportGenerator> _logger;
        private readonly TerrainDataCsvGenerator _terrainDataCsvGenerator;
        private readonly JsonReportGenerator? _jsonReportGenerator;
        private readonly MarkdownReportGenerator? _markdownReportGenerator;
        private readonly PM4ObjExporter? _pm4ObjExporter;

        /// <summary>
        /// Creates a new instance of the ReportGenerator class.
        /// </summary>
        /// <param name="logger">The logger to use.</param>
        /// <param name="terrainDataCsvGenerator">The terrain data CSV generator to use.</param>
        /// <param name="jsonReportGenerator">Optional JSON report generator to use.</param>
        /// <param name="markdownReportGenerator">Optional Markdown report generator to use.</param>
        /// <param name="pm4ObjExporter">Optional PM4 OBJ exporter to use.</param>
        public ReportGenerator(
            ILogger<ReportGenerator> logger,
            TerrainDataCsvGenerator terrainDataCsvGenerator,
            JsonReportGenerator? jsonReportGenerator = null,
            MarkdownReportGenerator? markdownReportGenerator = null,
            PM4ObjExporter? pm4ObjExporter = null)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _terrainDataCsvGenerator = terrainDataCsvGenerator ?? throw new ArgumentNullException(nameof(terrainDataCsvGenerator));
            _jsonReportGenerator = jsonReportGenerator;
            _markdownReportGenerator = markdownReportGenerator;
            _pm4ObjExporter = pm4ObjExporter;
        }

        /// <summary>
        /// Generates all reports for the analysis results.
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
            {
                Directory.CreateDirectory(outputDirectory);
            }

            _logger.LogInformation("Generating reports in {OutputDirectory}", outputDirectory);

            // Generate terrain data CSV reports
            await _terrainDataCsvGenerator.GenerateAllCsvAsync(results, outputDirectory);

            // Generate JSON reports if available
            if (_jsonReportGenerator != null)
            {
                await _jsonReportGenerator.GenerateAllReportsAsync(results, summary, outputDirectory);
            }

            // Generate Markdown reports if available
            if (_markdownReportGenerator != null)
            {
                await _markdownReportGenerator.GenerateReportsAsync(results, summary, outputDirectory);
            }

            // Generate summary report
            await GenerateSummaryReportAsync(summary, outputDirectory);

            _logger.LogInformation("Report generation complete.");
        }

        /// <summary>
        /// Generates a summary report.
        /// </summary>
        /// <param name="summary">The analysis summary.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateSummaryReportAsync(AnalysisSummary summary, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "summary.txt");
            _logger.LogInformation("Generating summary report: {FilePath}", filePath);

            using var writer = new StreamWriter(filePath, false);
            
            await writer.WriteLineAsync($"# ADT Analysis Summary - {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync($"Total Files: {summary.TotalFiles}");
            await writer.WriteLineAsync($"Processed Files: {summary.ProcessedFiles}");
            await writer.WriteLineAsync($"Failed Files: {summary.FailedFiles}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync($"Total Texture References: {summary.TotalTextureReferences}");
            await writer.WriteLineAsync($"Total Model References: {summary.TotalModelReferences}");
            await writer.WriteLineAsync($"Total WMO References: {summary.TotalWmoReferences}");
            await writer.WriteLineAsync($"Total Terrain Chunks: {summary.TotalTerrainChunks}");
            await writer.WriteLineAsync($"Total Model Placements: {summary.TotalModelPlacements}");
            await writer.WriteLineAsync($"Total WMO Placements: {summary.TotalWmoPlacements}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync($"Missing References: {summary.MissingReferences}");
            await writer.WriteLineAsync($"Files Not In Listfile: {summary.FilesNotInListfile}");
            await writer.WriteLineAsync($"Duplicate IDs: {summary.DuplicateIds}");
            await writer.WriteLineAsync($"Maximum Unique ID: {summary.MaxUniqueId}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync($"Analysis Duration: {summary.Duration.TotalSeconds:F2} seconds");
            await writer.WriteLineAsync($"Start Time: {summary.StartTime}");
            await writer.WriteLineAsync($"End Time: {summary.EndTime}");
        }

        /// <summary>
        /// Generates reports for PM4 analysis results.
        /// </summary>
        /// <param name="analysisResults">The PM4 analysis results to report on.</param>
        /// <param name="outputDirectory">The output directory for the reports.</param>
        /// <param name="exportCsv">Whether to export CSV files.</param>
        /// <param name="exportObj">Whether to export OBJ files.</param>
        /// <param name="exportConsolidatedObj">Whether to export a single consolidated OBJ file. Defaults to true when exportObj is true.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GeneratePM4ReportsAsync(
            IEnumerable<PM4AnalysisResult> analysisResults, 
            string outputDirectory, 
            bool exportCsv = true, 
            bool exportObj = true, 
            bool exportConsolidatedObj = true)
        {
            if (analysisResults == null)
                throw new ArgumentNullException(nameof(analysisResults));

            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new ArgumentException("Output directory cannot be null or whitespace.", nameof(outputDirectory));

            // Ensure output directory exists
            Directory.CreateDirectory(outputDirectory);

            var resultsList = analysisResults.ToList();

            if (resultsList.Count == 0)
            {
                _logger.LogWarning("No PM4 analysis results to report on.");
                return;
            }

            _logger.LogInformation("Generating reports for {Count} PM4 analysis results", resultsList.Count);

            // Generate consolidated report
            await GenerateConsolidatedPM4ReportAsync(resultsList, outputDirectory);

            // Generate CSV reports if requested
            if (exportCsv)
            {
                await GeneratePM4CsvReportsAsync(resultsList, outputDirectory);
            }

            // Generate OBJ files if requested
            if (exportObj && _pm4ObjExporter != null)
            {
                // Export individual OBJ files for each PM4
                await _pm4ObjExporter.ExportAllToObjAsync(resultsList);
                
                // Export consolidated OBJ file if requested
                if (exportConsolidatedObj)
                {
                    await _pm4ObjExporter.ExportToConsolidatedObjAsync(resultsList);
                    _logger.LogInformation("Generated consolidated OBJ file containing all PM4 models");
                }
            }
            else if (exportObj)
            {
                _logger.LogWarning("OBJ export was requested but PM4ObjExporter is not available");
            }

            _logger.LogInformation("PM4 report generation completed");
        }

        private async Task GenerateConsolidatedPM4ReportAsync(List<PM4AnalysisResult> results, string outputDirectory)
        {
            var summary = new System.Text.StringBuilder();
            summary.AppendLine("# PM4 Analysis - Comprehensive Report");
            summary.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            summary.AppendLine($"Files Analyzed: {results.Count}");
            
            // Create table of contents
            summary.AppendLine("\n## Table of Contents");
            summary.AppendLine("1. [Executive Summary](#executive-summary)");
            summary.AppendLine("2. [Chunk Statistics](#chunk-statistics)");
            summary.AppendLine("3. [Geometry Statistics](#geometry-statistics)");
            summary.AppendLine("4. [Position Data Analysis](#position-data-analysis)");
            summary.AppendLine("   - [Valid Positions](#valid-positions)");
            summary.AppendLine("   - [Special Entries](#special-entries)");
            summary.AppendLine("   - [Flag Analysis](#flag-analysis)");
            summary.AppendLine("5. [File-by-File Analysis](#file-by-file-analysis)");
            if (results.Any(r => r.Errors.Count > 0))
            {
                summary.AppendLine("6. [Error Report](#error-report)");
            }
            
            // Executive Summary
            summary.AppendLine("\n## Executive Summary");
            int filesWithErrors = results.Count(r => r.Errors.Count > 0);
            summary.AppendLine($"- **Files Analyzed**: {results.Count}");
            summary.AppendLine($"- **Files with Errors**: {filesWithErrors}");
            summary.AppendLine($"- **Files with Vertex Data**: {results.Count(r => r.HasVertexPositions)}");
            summary.AppendLine($"- **Files with Position Data**: {results.Count(r => r.HasPositionData)}");
            
            // Calculate statistics for executive summary
            int totalVertices = results.Sum(r => r.PM4File?.VertexPositionsChunk?.Vertices.Count ?? 0);
            int totalIndices = results.Sum(r => r.PM4File?.VertexIndicesChunk?.Indices.Count ?? 0);
            int triangleCount = totalIndices / 3;
            int totalPositionEntries = results.Sum(r => r.PM4File?.PositionDataChunk?.Entries.Count ?? 0);
            int totalPosRecords = results.Sum(r => r.PM4File?.PositionDataChunk?.Entries.Count(e => !e.IsSpecialEntry) ?? 0);
            int totalCmdRecords = results.Sum(r => r.PM4File?.PositionDataChunk?.Entries.Count(e => e.IsSpecialEntry) ?? 0);
            
            summary.AppendLine($"- **Total Vertices**: {totalVertices:N0}");
            summary.AppendLine($"- **Total Triangles**: {triangleCount:N0}");
            summary.AppendLine($"- **Total Position Entries**: {totalPositionEntries:N0}");
            summary.AppendLine($"  - Position Records: {totalPosRecords:N0}");
            summary.AppendLine($"  - Command Records: {totalCmdRecords:N0}");
            
            // Chunk statistics
            summary.AppendLine("\n## Chunk Statistics");
            summary.AppendLine("| Chunk Type | Files Count | Percentage | Description |");
            summary.AppendLine("|------------|-------------|------------|-------------|");
            summary.AppendLine($"| MSHD | {results.Count(r => r.HasShadowData)} | {(float)results.Count(r => r.HasShadowData) / results.Count * 100:F1}% | Shadow Data |");
            summary.AppendLine($"| MSPV | {results.Count(r => r.HasVertexPositions)} | {(float)results.Count(r => r.HasVertexPositions) / results.Count * 100:F1}% | Vertex Positions |");
            summary.AppendLine($"| MSPI | {results.Count(r => r.HasVertexIndices)} | {(float)results.Count(r => r.HasVertexIndices) / results.Count * 100:F1}% | Vertex Indices |");
            summary.AppendLine($"| MSCN | {results.Count(r => r.HasNormalCoordinates)} | {(float)results.Count(r => r.HasNormalCoordinates) / results.Count * 100:F1}% | Normal Coordinates |");
            summary.AppendLine($"| MSLK | {results.Count(r => r.HasLinks)} | {(float)results.Count(r => r.HasLinks) / results.Count * 100:F1}% | Links |");
            summary.AppendLine($"| MSVT | {results.Count(r => r.HasVertexData)} | {(float)results.Count(r => r.HasVertexData) / results.Count * 100:F1}% | Vertex Data |");
            summary.AppendLine($"| MSVI | {results.Count(r => r.HasVertexInfo)} | {(float)results.Count(r => r.HasVertexInfo) / results.Count * 100:F1}% | Vertex Info |");
            summary.AppendLine($"| MSUR | {results.Count(r => r.HasSurfaceData)} | {(float)results.Count(r => r.HasSurfaceData) / results.Count * 100:F1}% | Surface Data |");
            summary.AppendLine($"| MPRL | {results.Count(r => r.HasPositionData)} | {(float)results.Count(r => r.HasPositionData) / results.Count * 100:F1}% | Position Data |");
            summary.AppendLine($"| MPRR | {results.Count(r => r.HasValuePairs)} | {(float)results.Count(r => r.HasValuePairs) / results.Count * 100:F1}% | Value Pairs |");
            summary.AppendLine($"| MDBH | {results.Count(r => r.HasBuildingData)} | {(float)results.Count(r => r.HasBuildingData) / results.Count * 100:F1}% | Building Data |");
            summary.AppendLine($"| MDOS | {results.Count(r => r.HasSimpleData)} | {(float)results.Count(r => r.HasSimpleData) / results.Count * 100:F1}% | Simple Data |");
            summary.AppendLine($"| MDSF | {results.Count(r => r.HasFinalData)} | {(float)results.Count(r => r.HasFinalData) / results.Count * 100:F1}% | Final Data |");
            
            // Geometry statistics
            summary.AppendLine("\n## Geometry Statistics");
            
            // Vertex statistics
            summary.AppendLine("\n### Vertex Data (MSPV)");
            summary.AppendLine($"- **Total Vertices**: {totalVertices:N0}");
            
            var filesWithMostVertices = results
                .Where(r => r.PM4File?.VertexPositionsChunk != null)
                .OrderByDescending(r => r.PM4File?.VertexPositionsChunk?.Vertices.Count ?? 0)
                .Take(5)
                .ToList();
            
            if (filesWithMostVertices.Any())
            {
                summary.AppendLine("\n#### Files with Most Vertices");
                summary.AppendLine("| File | Vertex Count |");
                summary.AppendLine("|------|--------------|");
                
                foreach (var file in filesWithMostVertices)
                {
                    summary.AppendLine($"| {file.FileName} | {file.PM4File?.VertexPositionsChunk?.Vertices.Count:N0} |");
                }
            }
            
            // Triangle statistics
            summary.AppendLine("\n### Triangle Data (MSPI)");
            summary.AppendLine($"- **Total Triangles**: {triangleCount:N0}");
            
            var filesWithMostTriangles = results
                .Where(r => r.PM4File?.VertexIndicesChunk != null)
                .OrderByDescending(r => r.PM4File?.VertexIndicesChunk?.Indices.Count ?? 0)
                .Take(5)
                .ToList();
            
            if (filesWithMostTriangles.Any())
            {
                summary.AppendLine("\n#### Files with Most Triangles");
                summary.AppendLine("| File | Triangle Count |");
                summary.AppendLine("|------|---------------|");
                
                foreach (var file in filesWithMostTriangles)
                {
                    int tCount = (file.PM4File?.VertexIndicesChunk?.Indices.Count ?? 0) / 3;
                    summary.AppendLine($"| {file.FileName} | {tCount:N0} |");
                }
            }
            
            // Position Data Analysis
            summary.AppendLine("\n## Position Data Analysis");
            summary.AppendLine($"- **Total Position Entries**: {totalPositionEntries:N0}");
            summary.AppendLine($"- **Position Records**: {totalPosRecords:N0} ({(totalPosRecords > 0 ? (totalPosRecords / (float)totalPositionEntries * 100) : 0):F1}%)");
            summary.AppendLine($"- **Command Records**: {totalCmdRecords:N0} ({(totalCmdRecords > 0 ? (totalCmdRecords / (float)totalPositionEntries * 100) : 0):F1}%)");
            
            var filesWithMostPositions = results
                .Where(r => r.PM4File?.PositionDataChunk != null)
                .OrderByDescending(r => r.PM4File?.PositionDataChunk?.Entries.Count ?? 0)
                .Take(5)
                .ToList();
            
            if (filesWithMostPositions.Any())
            {
                summary.AppendLine("\n#### Files with Most Position Entries");
                summary.AppendLine("| File | Total Entries | Position Records | Command Records |");
                summary.AppendLine("|------|---------------|-----------------|-----------------|");
                
                foreach (var file in filesWithMostPositions)
                {
                    int total = file.PM4File?.PositionDataChunk?.Entries.Count ?? 0;
                    int posCount = file.PM4File?.PositionDataChunk?.Entries.Count(e => !e.IsSpecialEntry) ?? 0;
                    int cmdCount = file.PM4File?.PositionDataChunk?.Entries.Count(e => e.IsSpecialEntry) ?? 0;
                    summary.AppendLine($"| {file.FileName} | {total:N0} | {posCount:N0} | {cmdCount:N0} |");
                }
            }
            
            // Position Records Analysis
            summary.AppendLine("\n### Position Records");
            
            var positionFiles = results
                .Where(r => r.PM4File?.PositionDataChunk?.Entries.Any(e => !e.IsSpecialEntry) == true)
                .ToList();
            
            // Analyze coordinate ranges if any position records exist
            if (positionFiles.Any())
            {
                var coordRanges = new List<(float MinX, float MaxX, float MinY, float MaxY, float MinZ, float MaxZ)>();
                
                foreach (var file in positionFiles)
                {
                    var positions = file.PM4File!.PositionDataChunk!.Entries.Where(e => !e.IsSpecialEntry).ToList();
                    if (positions.Any())
                    {
                        float minX = positions.Min(p => p.CoordinateX);
                        float maxX = positions.Max(p => p.CoordinateX);
                        float minY = positions.Min(p => p.CoordinateY);
                        float maxY = positions.Max(p => p.CoordinateY);
                        float minZ = positions.Min(p => p.CoordinateZ);
                        float maxZ = positions.Max(p => p.CoordinateZ);
                        
                        coordRanges.Add((minX, maxX, minY, maxY, minZ, maxZ));
                    }
                }
                
                if (coordRanges.Any())
                {
                    float overallMinX = coordRanges.Min(r => r.MinX);
                    float overallMaxX = coordRanges.Max(r => r.MaxX);
                    float overallMinY = coordRanges.Min(r => r.MinY);
                    float overallMaxY = coordRanges.Max(r => r.MaxY);
                    float overallMinZ = coordRanges.Min(r => r.MinZ);
                    float overallMaxZ = coordRanges.Max(r => r.MaxZ);
                    
                    summary.AppendLine("\n#### Coordinate Range Analysis");
                    summary.AppendLine("| Axis | Minimum | Maximum | Range |");
                    summary.AppendLine("|------|---------|---------|-------|");
                    summary.AppendLine($"| X | {overallMinX:F2} | {overallMaxX:F2} | {overallMaxX - overallMinX:F2} |");
                    summary.AppendLine($"| Y | {overallMinY:F2} | {overallMaxY:F2} | {overallMaxY - overallMinY:F2} |");
                    summary.AppendLine($"| Z | {overallMinZ:F2} | {overallMaxZ:F2} | {overallMaxZ - overallMinZ:F2} |");
                }
            }
            
            // Sample position records from different files
            summary.AppendLine("\n#### Sample Position Records");
            
            int sampleFileCount = 0;
            foreach (var result in positionFiles.Take(5))
            {
                sampleFileCount++;
                string fileName = result.FileName ?? "Unknown";
                summary.AppendLine($"\n##### Sample {sampleFileCount}: {fileName}");
                summary.AppendLine("| Index | X | Y | Z |");
                summary.AppendLine("|-------|-----|-----|-----|");
                
                var positions = result.PM4File!.PositionDataChunk!.Entries.Where(e => !e.IsSpecialEntry).Take(10).ToList();
                foreach (var pos in positions)
                {
                    summary.AppendLine($"| {pos.Index} | {pos.CoordinateX:F2} | {pos.CoordinateY:F2} | {pos.CoordinateZ:F2} |");
                }
                
                int totalCount = result.PM4File.PositionDataChunk.Entries.Count(e => !e.IsSpecialEntry);
                if (totalCount > 10)
                {
                    summary.AppendLine($"_Showing 10 of {totalCount} position records_");
                }
            }
            
            // Command Records Analysis
            summary.AppendLine("\n### Command Records");
            
            var commandFiles = results
                .Where(r => r.PM4File?.PositionDataChunk?.Entries.Any(e => e.IsSpecialEntry) == true)
                .ToList();
            
            // Example command records from different files
            summary.AppendLine("\n#### Sample Command Records");
            
            sampleFileCount = 0;
            foreach (var result in commandFiles.Take(5))
            {
                sampleFileCount++;
                string fileName = result.FileName ?? "Unknown";
                summary.AppendLine($"\n##### Sample {sampleFileCount}: {fileName}");
                summary.AppendLine("| Index | Special Value (Hex) | As Float | Y Value |");
                summary.AppendLine("|-------|--------------|---------|---------|");
                
                var commands = result.PM4File!.PositionDataChunk!.Entries.Where(e => e.IsSpecialEntry).Take(10).ToList();
                foreach (var cmd in commands)
                {
                    float asFloat = BitConverter.Int32BitsToSingle(cmd.SpecialValue);
                    summary.AppendLine($"| {cmd.Index} | 0x{cmd.SpecialValue:X8} | {asFloat:F2} | {cmd.CoordinateY:F2} |");
                }
                
                int totalCount = result.PM4File.PositionDataChunk.Entries.Count(e => e.IsSpecialEntry);
                if (totalCount > 10)
                {
                    summary.AppendLine($"_Showing 10 of {totalCount} command records_");
                }
            }
            
            // Special value distribution analysis
            summary.AppendLine("\n### Special Value Analysis");
            
            // Special value distribution
            var specialDistribution = results
                .SelectMany(r => r.PM4File?.PositionDataChunk?.Entries.Where(e => e.IsSpecialEntry) ?? Array.Empty<MPRLChunk.ServerPositionData>())
                .GroupBy(e => e.SpecialValue)
                .OrderByDescending(g => g.Count())
                .Take(20)
                .Select(g => new
                {
                    SpecialValue = g.Key,
                    AsFloat = BitConverter.Int32BitsToSingle(g.Key),
                    Count = g.Count()
                })
                .ToList();
                
            summary.AppendLine("\n#### Special Value Distribution");
            summary.AppendLine("| Special Value (Hex) | As Float | Count | Percentage |");
            summary.AppendLine("|--------------|---------|-------|------------|");
            
            int totalSpecialEntries = results.Sum(r => r.PM4File?.PositionDataChunk?.Entries.Count(e => e.IsSpecialEntry) ?? 0);
            foreach (var special in specialDistribution)
            {
                double percentage = (double)special.Count / totalSpecialEntries * 100;
                summary.AppendLine($"| 0x{special.SpecialValue:X8} | {special.AsFloat:F2} | {special.Count:N0} | {percentage:F2}% |");
            }
            
            // Sample special entries from different files
            summary.AppendLine("\n#### Sample Special Entries");
            summary.AppendLine("| File | Index | Special Value (Hex) | As Float | X | Y | Z | Value1 | Value2 | Value3 |");
            summary.AppendLine("|------|-------|--------------|---------|-----|-----|-----|--------|--------|--------|");

            int entriesShown = 0;
            foreach (var result in results.Take(5))
            {
                var entries = result.PM4File?.PositionDataChunk?.Entries.Where(e => e.IsSpecialEntry).Take(2).ToList();
                if (entries != null && entries.Count > 0)
                {
                    string fileName = Path.GetFileName(result.FileName ?? "unknown");
                    foreach (var entry in entries)
                    {
                        float asFloat = BitConverter.Int32BitsToSingle(entry.SpecialValue);
                        summary.AppendLine($"| {fileName} | {entry.Index} | 0x{entry.SpecialValue:X8} | {asFloat:F6} | " +
                                          $"{entry.CoordinateX:F6} | {entry.CoordinateY:F6} | {entry.CoordinateZ:F6} | " +
                                          $"{entry.Value1:F6} | {entry.Value2:F6} | {entry.Value3:F6} |");
                        entriesShown++;
                        if (entriesShown >= 10) break;
                    }
                }
                if (entriesShown >= 10) break;
            }
            
            // Command-Position Pair Analysis
            summary.AppendLine("\n### Command-Position Pair Analysis");
            
            // Find patterns where commands are followed by positions
            var allPairs = new List<(MPRLChunk.ServerPositionData Command, MPRLChunk.ServerPositionData Position)>();
            
            foreach (var result in results.Where(r => r.PM4File?.PositionDataChunk != null))
            {
                var entries = result.PM4File!.PositionDataChunk!.Entries;
                for (int i = 0; i < entries.Count - 1; i++)
                {
                    if (entries[i].IsSpecialEntry && !entries[i + 1].IsSpecialEntry)
                    {
                        allPairs.Add((entries[i], entries[i + 1]));
                    }
                }
            }
            
            if (allPairs.Any())
            {
                summary.AppendLine($"- Total Command-Position Pairs: {allPairs.Count:N0}");
                
                var groupedPairs = allPairs
                    .GroupBy(p => p.Command.SpecialValue)
                    .OrderByDescending(g => g.Count())
                    .Take(15)
                    .ToList();
                    
                summary.AppendLine("\n#### Most Common Command-Position Pairs");
                summary.AppendLine("| Command (Hex) | Command (Dec) | Count | Percentage | Example Position |");
                summary.AppendLine("|--------------|--------------|-------|------------|------------------|");
                
                foreach (var group in groupedPairs)
                {
                    double percentage = (double)group.Count() / allPairs.Count * 100;
                    var sample = group.First();
                    string posValue = $"({sample.Position.CoordinateX:F2}, {sample.Position.CoordinateY:F2}, {sample.Position.CoordinateZ:F2})";
                    summary.AppendLine($"| 0x{group.Key:X8} | {group.Key} | {group.Count():N0} | {percentage:F2}% | {posValue} |");
                }
            }
            
            // File-by-File Analysis
            summary.AppendLine("\n## File-by-File Analysis");
            
            // Sort files by name for easy reference
            var sortedResults = results.OrderBy(r => r.FileName).ToList();
            
            // Create a file index
            summary.AppendLine("\n### File Index");
            for (int i = 0; i < sortedResults.Count; i++)
            {
                var result = sortedResults[i];
                summary.AppendLine($"{i+1}. [{result.FileName}](#{result.FileName?.Replace(".", "").Replace(" ", "-").ToLower()})");
            }
            
            // Generate detailed analysis for each file
            foreach (var result in sortedResults)
            {
                string anchorName = result.FileName?.Replace(".", "").Replace(" ", "-").ToLower();
                summary.AppendLine($"\n### {result.FileName} <a name=\"{anchorName}\"></a>");
                
                // Basic file info
                summary.AppendLine("#### File Information");
                summary.AppendLine($"- **Path**: {result.FilePath}");
                summary.AppendLine($"- **Version**: {result.Version}");
                
                // Chunk presence
                summary.AppendLine("\n#### Chunks Present");
                List<string> presentChunks = new List<string>();
                if (result.HasShadowData) presentChunks.Add("MSHD (Shadow Data)");
                if (result.HasVertexPositions) presentChunks.Add("MSPV (Vertex Positions)");
                if (result.HasVertexIndices) presentChunks.Add("MSPI (Vertex Indices)");
                if (result.HasNormalCoordinates) presentChunks.Add("MSCN (Normal Coordinates)");
                if (result.HasLinks) presentChunks.Add("MSLK (Links)");
                if (result.HasVertexData) presentChunks.Add("MSVT (Vertex Data)");
                if (result.HasVertexInfo) presentChunks.Add("MSVI (Vertex Info)");
                if (result.HasSurfaceData) presentChunks.Add("MSUR (Surface Data)");
                if (result.HasPositionData) presentChunks.Add("MPRL (Position Data)");
                if (result.HasValuePairs) presentChunks.Add("MPRR (Value Pairs)");
                if (result.HasBuildingData) presentChunks.Add("MDBH (Building Data)");
                if (result.HasSimpleData) presentChunks.Add("MDOS (Simple Data)");
                if (result.HasFinalData) presentChunks.Add("MDSF (Final Data)");
                
                for (int i = 0; i < presentChunks.Count; i++)
                {
                    summary.AppendLine($"- {presentChunks[i]}");
                }
                
                // Detailed statistics for this file
                if (result.PM4File?.VertexPositionsChunk != null)
                {
                    int vCount = result.PM4File.VertexPositionsChunk.Vertices.Count;
                    summary.AppendLine($"\n#### Vertex Data: {vCount:N0} vertices");
                    
                    if (vCount > 0)
                    {
                        // Show first few vertices
                        summary.AppendLine("\nSample Vertices:");
                        summary.AppendLine("| Index | X | Y | Z |");
                        summary.AppendLine("|-------|-----|-----|-----|");
                        
                        for (int i = 0; i < Math.Min(5, vCount); i++)
                        {
                            var v = result.PM4File.VertexPositionsChunk.Vertices[i];
                            summary.AppendLine($"| {i} | {v.X:F2} | {v.Y:F2} | {v.Z:F2} |");
                        }
                        
                        if (vCount > 5)
                        {
                            summary.AppendLine($"_Showing 5 of {vCount} vertices_");
                        }
                    }
                }
                
                if (result.PM4File?.VertexIndicesChunk != null)
                {
                    int iCount = result.PM4File.VertexIndicesChunk.Indices.Count;
                    int tCount = iCount / 3;
                    summary.AppendLine($"\n#### Triangle Data: {tCount:N0} triangles ({iCount:N0} indices)");
                    
                    if (tCount > 0)
                    {
                        // Show first few triangles
                        summary.AppendLine("\nSample Triangles:");
                        summary.AppendLine("| Index | Vertex 1 | Vertex 2 | Vertex 3 |");
                        summary.AppendLine("|-------|----------|----------|----------|");
                        
                        for (int i = 0; i < Math.Min(5, tCount); i++)
                        {
                            int baseIdx = i * 3;
                            if (baseIdx + 2 < iCount)
                            {
                                var i1 = result.PM4File.VertexIndicesChunk.Indices[baseIdx];
                                var i2 = result.PM4File.VertexIndicesChunk.Indices[baseIdx + 1];
                                var i3 = result.PM4File.VertexIndicesChunk.Indices[baseIdx + 2];
                                summary.AppendLine($"| {i} | {i1} | {i2} | {i3} |");
                            }
                        }
                        
                        if (tCount > 5)
                        {
                            summary.AppendLine($"_Showing 5 of {tCount} triangles_");
                        }
                    }
                }
                
                if (result.PM4File?.PositionDataChunk != null)
                {
                    var entries = result.PM4File.PositionDataChunk.Entries;
                    int totalEntries = entries.Count;
                    int filePositionCount = entries.Count(e => !e.IsSpecialEntry);
                    int fileCommandCount = entries.Count(e => e.IsSpecialEntry);
                    
                    summary.AppendLine($"\n#### Position Data: {totalEntries:N0} entries ({filePositionCount:N0} positions, {fileCommandCount:N0} commands)");
                    
                    // Show position record statistics if any
                    if (filePositionCount > 0)
                    {
                        var filePosRecords = entries.Where(e => !e.IsSpecialEntry).ToList();
                        
                        // Coordinate ranges
                        var minX = filePosRecords.Min(p => p.CoordinateX);
                        var maxX = filePosRecords.Max(p => p.CoordinateX);
                        var minY = filePosRecords.Min(p => p.CoordinateY);
                        var maxY = filePosRecords.Max(p => p.CoordinateY);
                        var minZ = filePosRecords.Min(p => p.CoordinateZ);
                        var maxZ = filePosRecords.Max(p => p.CoordinateZ);
                        
                        summary.AppendLine("\nPosition Record Coordinate Ranges:");
                        summary.AppendLine("| Axis | Minimum | Maximum | Range |");
                        summary.AppendLine("|------|---------|---------|-------|");
                        summary.AppendLine($"| X | {minX:F2} | {maxX:F2} | {maxX - minX:F2} |");
                        summary.AppendLine($"| Y | {minY:F2} | {maxY:F2} | {maxY - minY:F2} |");
                        summary.AppendLine($"| Z | {minZ:F2} | {maxZ:F2} | {maxZ - minZ:F2} |");
                        
                        // Show sample position records
                        summary.AppendLine("\nSample Position Records:");
                        summary.AppendLine("| Index | X | Y | Z |");
                        summary.AppendLine("|-------|-----|-----|-----|");
                        
                        for (int i = 0; i < Math.Min(5, filePositionCount); i++)
                        {
                            var pos = filePosRecords[i];
                            summary.AppendLine($"| {pos.Index} | {pos.CoordinateX:F2} | {pos.CoordinateY:F2} | {pos.CoordinateZ:F2} |");
                        }
                        
                        if (filePositionCount > 5)
                        {
                            summary.AppendLine($"_Showing 5 of {filePositionCount} position records_");
                        }
                    }
                    
                    // Show command record statistics if any
                    if (fileCommandCount > 0)
                    {
                        var fileCmdRecords = entries.Where(e => e.IsSpecialEntry).ToList();
                        
                        // Show sample command records
                        summary.AppendLine("\nSample Special Entries:");
                        summary.AppendLine("| Index | Special Value (Hex) | As Float | X | Y | Z | Value1 | Value2 | Value3 |");
                        summary.AppendLine("|-------|--------------|---------|-----|-----|-----|--------|--------|--------|");

                        for (int i = 0; i < Math.Min(10, fileCmdRecords.Count); i++)
                        {
                            var entry = fileCmdRecords[i];
                            float asFloat = BitConverter.Int32BitsToSingle(entry.SpecialValue);
                            summary.AppendLine($"| {entry.Index} | 0x{entry.SpecialValue:X8} | {asFloat:F6} | " +
                                              $"{entry.CoordinateX:F6} | {entry.CoordinateY:F6} | {entry.CoordinateZ:F6} | " +
                                              $"{entry.Value1:F6} | {entry.Value2:F6} | {entry.Value3:F6} |");
                        }

                        if (fileCmdRecords.Count > 10)
                        {
                            summary.AppendLine($"_Showing top 10 of {fileCmdRecords.Count} special entries_");
                        }
                    }
                }
                
                // Show errors if any
                if (result.Errors.Count > 0)
                {
                    summary.AppendLine("\n#### Errors");
                    foreach (var error in result.Errors)
                    {
                        summary.AppendLine($"- {error}");
                    }
                }
            }
            
            // Error Report
            if (filesWithErrors > 0)
            {
                summary.AppendLine("\n## Error Report <a name=\"error-report\"></a>");
                summary.AppendLine($"Total Files with Errors: {filesWithErrors}");
                
                summary.AppendLine("\n| File | Error Count |");
                summary.AppendLine("|------|-------------|");
                foreach (var result in results.Where(r => r.Errors.Count > 0).OrderByDescending(r => r.Errors.Count))
                {
                    summary.AppendLine($"| [{result.FileName}](#{result.FileName?.Replace(".", "").Replace(" ", "-").ToLower()}) | {result.Errors.Count} |");
                }
            }
            
            // Write the comprehensive report
            await File.WriteAllTextAsync(Path.Combine(outputDirectory, "pm4_comprehensive_report.md"), summary.ToString());
        }
        
        private async Task GeneratePM4CsvReportsAsync(List<PM4AnalysisResult> results, string outputDirectory)
        {
            var csvDir = Path.Combine(outputDirectory, "pm4_csv_reports");
            Directory.CreateDirectory(csvDir);
            
            // Generate CSV for vertex positions (MSPV)
            using (var vertexWriter = new StreamWriter(Path.Combine(csvDir, "vertices.csv")))
            {
                // Write header
                await vertexWriter.WriteLineAsync("FileName,VertexIndex,PositionX,PositionY,PositionZ");
                
                // Write data
                foreach (var result in results)
                {
                    if (result.PM4File?.VertexPositionsChunk?.Vertices != null)
                    {
                        string fileName = result.FileName?.Replace(",", "_") ?? "Unknown";
                        var vertices = result.PM4File.VertexPositionsChunk.Vertices;
                        
                        for (int i = 0; i < vertices.Count; i++)
                        {
                            var vertex = vertices[i];
                            string line = String.Format(
                                "{0},{1},{2},{3},{4}",
                                fileName,
                                i,
                                vertex.X.ToString("F6", System.Globalization.CultureInfo.InvariantCulture),
                                vertex.Y.ToString("F6", System.Globalization.CultureInfo.InvariantCulture),
                                vertex.Z.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)
                            );
                            await vertexWriter.WriteLineAsync(line);
                        }
                    }
                }
            }
            
            // Generate CSV for position data (MPRL)
            using (var positionWriter = new StreamWriter(Path.Combine(csvDir, "positions.csv")))
            {
                // Write header
                await positionWriter.WriteLineAsync("FileName,PositionIndex,Type,X,Y,Z,Flag");
                
                // Write data
                foreach (var result in results)
                {
                    if (result.PM4File?.PositionDataChunk?.Entries != null)
                    {
                        string fileName = result.FileName?.Replace(",", "_") ?? "Unknown";
                        var positions = result.PM4File.PositionDataChunk.Entries;
                        
                        for (int i = 0; i < positions.Count; i++)
                        {
                            var pos = positions[i];
                            string entryType = pos.IsSpecialEntry ? "Valid" : "Special";
                            string x = pos.IsSpecialEntry ? pos.CoordinateX.ToString("F6", System.Globalization.CultureInfo.InvariantCulture) : 
                                      (float.IsNaN(pos.Value1) ? "NaN" : pos.Value1.ToString("F6", System.Globalization.CultureInfo.InvariantCulture));
                            string y = pos.CoordinateY.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
                            string z = pos.IsSpecialEntry ? pos.CoordinateZ.ToString("F6", System.Globalization.CultureInfo.InvariantCulture) : 
                                      (float.IsNaN(pos.Value3) ? "NaN" : pos.Value3.ToString("F6", System.Globalization.CultureInfo.InvariantCulture));
                            
                            string line = String.Format(
                                "{0},{1},{2},{3},{4},{5},{6}",
                                fileName,
                                i,
                                entryType,
                                x,
                                y,
                                z,
                                pos.IsSpecialEntry
                            );
                            await positionWriter.WriteLineAsync(line);
                        }
                    }
                }
            }
            
            // Generate CSV for triangles (derived from vertex indices)
            using (var triangleWriter = new StreamWriter(Path.Combine(csvDir, "triangles.csv")))
            {
                // Write header
                await triangleWriter.WriteLineAsync("FileName,TriangleIndex,Vertex1,Vertex2,Vertex3");
                
                // Write data
                foreach (var result in results)
                {
                    if (result.PM4File?.VertexIndicesChunk?.Indices != null && 
                        result.PM4File.VertexIndicesChunk.Indices.Count >= 3)
                    {
                        string fileName = result.FileName?.Replace(",", "_") ?? "Unknown";
                        var indices = result.PM4File.VertexIndicesChunk.Indices;
                        
                        int triangleCount = indices.Count / 3;
                        for (int i = 0; i < triangleCount; i++)
                        {
                            int baseIndex = i * 3;
                            string line = String.Format(
                                "{0},{1},{2},{3},{4}",
                                fileName,
                                i,
                                indices[baseIndex],
                                indices[baseIndex + 1],
                                indices[baseIndex + 2]
                            );
                            await triangleWriter.WriteLineAsync(line);
                        }
                    }
                }
            }
            
            // Generate CSV for file summary
            using (var fileSummaryWriter = new StreamWriter(Path.Combine(csvDir, "file_summary.csv")))
            {
                // Write header
                await fileSummaryWriter.WriteLineAsync("FileName,Version,MSHD,MSPV,MSPI,MSCN,MSLK,MSVT,MSVI,MSUR,MPRL,MPRR,MDBH,MDOS,MDSF,VertexCount,TriangleCount,ErrorCount");
                
                // Write data
                foreach (var result in results)
                {
                    int vertexCount = result.PM4File?.VertexPositionsChunk?.Vertices.Count ?? 0;
                    int triangleCount = (result.PM4File?.VertexIndicesChunk?.Indices.Count ?? 0) / 3;
                    
                    string line = String.Format(
                        "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}",
                        result.FileName?.Replace(",", "_") ?? "Unknown",
                        result.Version,
                        result.HasShadowData ? "1" : "0",
                        result.HasVertexPositions ? "1" : "0",
                        result.HasVertexIndices ? "1" : "0",
                        result.HasNormalCoordinates ? "1" : "0",
                        result.HasLinks ? "1" : "0",
                        result.HasVertexData ? "1" : "0",
                        result.HasVertexInfo ? "1" : "0",
                        result.HasSurfaceData ? "1" : "0",
                        result.HasPositionData ? "1" : "0",
                        result.HasValuePairs ? "1" : "0",
                        result.HasBuildingData ? "1" : "0",
                        result.HasSimpleData ? "1" : "0",
                        result.HasFinalData ? "1" : "0",
                        vertexCount,
                        triangleCount,
                        result.Errors.Count
                    );
                    await fileSummaryWriter.WriteLineAsync(line);
                }
            }
        }
    }
}