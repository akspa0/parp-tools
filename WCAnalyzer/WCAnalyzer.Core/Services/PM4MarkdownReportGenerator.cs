using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Generates Markdown reports for PM4 file analysis.
    /// </summary>
    public class PM4MarkdownReportGenerator
    {
        private readonly ILogger? _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4MarkdownReportGenerator"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        public PM4MarkdownReportGenerator(ILogger? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Generates a comprehensive Markdown report for multiple PM4 files.
        /// </summary>
        /// <param name="results">The list of PM4 analysis results</param>
        /// <param name="outputPath">The path to write the report to</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task GenerateComprehensiveMultiFileReportAsync(List<PM4AnalysisResult> results, string outputPath)
        {
            if (results == null || results.Count == 0)
                throw new ArgumentException("Results list cannot be null or empty.", nameof(results));
            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));

            var report = new StringBuilder();

            // Report Header
            report.AppendLine("# PM4 Files Comprehensive Analysis Report");
            report.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            report.AppendLine();
            
            // Table of Contents
            report.AppendLine("## Table of Contents");
            report.AppendLine("- [Summary](#summary)");
            report.AppendLine("- [Chunk Statistics](#chunk-statistics)");
            report.AppendLine("- [Special Value Analysis](#special-value-analysis)");
            report.AppendLine("- [Coordinate Bounds](#coordinate-bounds)");
            report.AppendLine("- [Individual File Reports](#individual-file-reports)");
            
            foreach (var result in results)
            {
                string fileAnchor = result.FileName?.Replace(".", "-").Replace(" ", "-").ToLowerInvariant() ?? "unknown";
                report.AppendLine($"  - [{result.FileName}](#{fileAnchor})");
            }
            
            report.AppendLine();
            
            // Summary Section
            report.AppendLine("## Summary <a name=\"summary\"></a>");
            report.AppendLine($"Total Files Analyzed: {results.Count}");
            
            int totalVertices = results.Sum(r => r.PM4File?.VertexPositionsChunk?.Vertices.Count ?? 0);
            int totalTriangles = results.Sum(r => (r.PM4File?.VertexIndicesChunk?.Indices.Count ?? 0) / 3);
            int totalPositions = results.Sum(r => r.PM4File?.PositionDataChunk?.Entries.Count(e => !e.IsSpecialEntry) ?? 0);
            int totalSpecial = results.Sum(r => r.PM4File?.PositionDataChunk?.Entries.Count(e => e.IsSpecialEntry) ?? 0);
            int totalErrors = results.Sum(r => r.Errors.Count);
            
            report.AppendLine($"- Total Vertices: {totalVertices:N0}");
            report.AppendLine($"- Total Triangles: {totalTriangles:N0}");
            report.AppendLine($"- Total Position Records: {totalPositions:N0}");
            report.AppendLine($"- Total Special Records: {totalSpecial:N0}");
            report.AppendLine($"- Total Errors: {totalErrors}");
            report.AppendLine();
            
            // File Summary Table
            report.AppendLine("### Files Summary");
            report.AppendLine("| File Name | Version | Vertices | Triangles | Positions | Special Values | Errors |");
            report.AppendLine("|-----------|---------|----------|-----------|-----------|----------------|--------|");
            
            foreach (var result in results)
            {
                int vertexCount = result.PM4File?.VertexPositionsChunk?.Vertices.Count ?? 0;
                int triangleCount = (result.PM4File?.VertexIndicesChunk?.Indices.Count ?? 0) / 3;
                int positionCount = result.PM4File?.PositionDataChunk?.Entries.Count(e => !e.IsSpecialEntry) ?? 0;
                int specialCount = result.PM4File?.PositionDataChunk?.Entries.Count(e => e.IsSpecialEntry) ?? 0;
                
                report.AppendLine($"| {result.FileName} | {result.Version} | {vertexCount:N0} | {triangleCount:N0} | {positionCount:N0} | {specialCount:N0} | {result.Errors.Count} |");
            }
            report.AppendLine();
            
            // Chunk Statistics
            report.AppendLine("## Chunk Statistics <a name=\"chunk-statistics\"></a>");
            
            var chunkCounts = new Dictionary<string, int>
            {
                { "MSHD (Shadow Data)", results.Count(r => r.HasShadowData) },
                { "MSPV (Vertex Positions)", results.Count(r => r.HasVertexPositions) },
                { "MSPI (Vertex Indices)", results.Count(r => r.HasVertexIndices) },
                { "MSCN (Normal Coordinates)", results.Count(r => r.HasNormalCoordinates) },
                { "MSLK (Links)", results.Count(r => r.HasLinks) },
                { "MSVT (Vertex Data)", results.Count(r => r.HasVertexData) },
                { "MSVI (Vertex Info)", results.Count(r => r.HasVertexInfo) },
                { "MSUR (Surface Data)", results.Count(r => r.HasSurfaceData) },
                { "MPRL (Position Data)", results.Count(r => r.HasPositionData) },
                { "MPRR (Value Pairs)", results.Count(r => r.HasValuePairs) },
                { "MDBH (Building Data)", results.Count(r => r.HasBuildingData) },
                { "MDOS (Simple Data)", results.Count(r => r.HasSimpleData) },
                { "MDSF (Final Data)", results.Count(r => r.HasFinalData) }
            };
            
            report.AppendLine("| Chunk Type | Count | Percentage |");
            report.AppendLine("|------------|-------|------------|");
            
            foreach (var chunk in chunkCounts.OrderByDescending(c => c.Value))
            {
                double percentage = results.Count > 0 ? (chunk.Value * 100.0 / results.Count) : 0;
                report.AppendLine($"| {chunk.Key} | {chunk.Value} | {percentage:F1}% |");
            }
            report.AppendLine();
            
            // Special Value Analysis
            report.AppendLine("## Special Value Analysis <a name=\"special-value-analysis\"></a>");
            
            var allSpecialValues = new Dictionary<uint, int>();
            foreach (var result in results)
            {
                var specialEntries = result.PM4File?.PositionDataChunk?.Entries.Where(e => e.IsSpecialEntry) ?? Enumerable.Empty<MPRLEntry>();
                foreach (var entry in specialEntries)
                {
                    if (allSpecialValues.ContainsKey(entry.SpecialValue))
                    {
                        allSpecialValues[entry.SpecialValue]++;
                    }
                    else
                    {
                        allSpecialValues[entry.SpecialValue] = 1;
                    }
                }
            }
            
            if (allSpecialValues.Count > 0)
            {
                report.AppendLine("### Most Common Special Values");
                report.AppendLine("| Special Value | Hex Value | Occurrences | Percentage |");
                report.AppendLine("|---------------|-----------|-------------|------------|");
                
                foreach (var value in allSpecialValues.OrderByDescending(v => v.Value).Take(15))
                {
                    double percentage = totalSpecial > 0 ? (value.Value * 100.0 / totalSpecial) : 0;
                    report.AppendLine($"| {value.Key} | 0x{value.Key:X8} | {value.Value:N0} | {percentage:F1}% |");
                }
                
                if (allSpecialValues.Count > 15)
                {
                    report.AppendLine($"\n_Showing top 15 of {allSpecialValues.Count} distinct special values_");
                }
            }
            else
            {
                report.AppendLine("No special values found in the analyzed files.");
            }
            report.AppendLine();
            
            // Coordinate Bounds
            report.AppendLine("## Coordinate Bounds <a name=\"coordinate-bounds\"></a>");
            
            var allVertices = results
                .SelectMany(r => r.PM4File?.VertexPositionsChunk?.Vertices ?? Enumerable.Empty<MSPVVertex>())
                .ToList();
                
            var allPositions = results
                .SelectMany(r => r.PM4File?.PositionDataChunk?.Entries.Where(e => !e.IsSpecialEntry) ?? Enumerable.Empty<MPRLEntry>())
                .ToList();
            
            if (allVertices.Count > 0)
            {
                report.AppendLine("### Vertex Coordinate Ranges");
                report.AppendLine("| Axis | Minimum | Maximum | Range |");
                report.AppendLine("|------|---------|---------|-------|");
                
                float minX = allVertices.Min(v => v.X);
                float maxX = allVertices.Max(v => v.X);
                float minY = allVertices.Min(v => v.Y);
                float maxY = allVertices.Max(v => v.Y);
                float minZ = allVertices.Min(v => v.Z);
                float maxZ = allVertices.Max(v => v.Z);
                
                report.AppendLine($"| X | {minX:F2} | {maxX:F2} | {maxX - minX:F2} |");
                report.AppendLine($"| Y | {minY:F2} | {maxY:F2} | {maxY - minY:F2} |");
                report.AppendLine($"| Z | {minZ:F2} | {maxZ:F2} | {maxZ - minZ:F2} |");
            }
            
            if (allPositions.Count > 0)
            {
                report.AppendLine("\n### Position Record Coordinate Ranges");
                report.AppendLine("| Axis | Minimum | Maximum | Range |");
                report.AppendLine("|------|---------|---------|-------|");
                
                float minX = allPositions.Min(p => p.CoordinateX);
                float maxX = allPositions.Max(p => p.CoordinateX);
                float minY = allPositions.Min(p => p.CoordinateY);
                float maxY = allPositions.Max(p => p.CoordinateY);
                float minZ = allPositions.Min(p => p.CoordinateZ);
                float maxZ = allPositions.Max(p => p.CoordinateZ);
                
                report.AppendLine($"| X | {minX:F2} | {maxX:F2} | {maxX - minX:F2} |");
                report.AppendLine($"| Y | {minY:F2} | {maxY:F2} | {maxY - minY:F2} |");
                report.AppendLine($"| Z | {minZ:F2} | {maxZ:F2} | {maxZ - minZ:F2} |");
            }
            report.AppendLine();
            
            // Individual File Reports
            report.AppendLine("## Individual File Reports <a name=\"individual-file-reports\"></a>");
            
            foreach (var result in results)
            {
                if (result.PM4File != null)
                {
                    string fileAnchor = result.FileName?.Replace(".", "-").Replace(" ", "-").ToLowerInvariant() ?? "unknown";
                    report.AppendLine($"### {result.FileName} <a name=\"{fileAnchor}\"></a>");
                    await GenerateIndividualFileReportAsync(result, report);
                    report.AppendLine("<hr>");
                }
            }
            
            // Write the report to file
            await File.WriteAllTextAsync(outputPath, report.ToString());
            _logger?.LogInformation("Generated comprehensive multi-file Markdown report: {FilePath}", outputPath);
        }

        /// <summary>
        /// Generates a comprehensive Markdown report for a PM4 file.
        /// </summary>
        /// <param name="pm4File">The PM4 file to analyze</param>
        /// <param name="outputPath">The path to write the report to</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task GenerateComprehensiveReportAsync(PM4File pm4File, string outputPath)
        {
            if (pm4File == null)
                throw new ArgumentNullException(nameof(pm4File));
            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));

            var report = new StringBuilder();

            // File Information
            report.AppendLine("# PM4 File Analysis Report");
            report.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            report.AppendLine($"\n## File Information");
            report.AppendLine($"- File Name: {pm4File.FileName ?? "Unknown"}");
            report.AppendLine($"- File Path: {pm4File.FilePath ?? "Unknown"}");
            report.AppendLine($"- File Size: {pm4File.FileSize:N0} bytes");
            report.AppendLine($"- Version: {pm4File.Version?.Version ?? 0}");

            // Chunk Information
            report.AppendLine("\n## Chunk Information");
            report.AppendLine("| Chunk Type | Present | Size (bytes) |");
            report.AppendLine("|------------|---------|--------------|");
            
            // Add logic to display chunk information
            if (pm4File.ShadowDataChunk != null)
                report.AppendLine($"| MSHD (Shadow Data) | Yes | {pm4File.ShadowDataChunk.Size:N0} |");
            if (pm4File.VertexPositionsChunk != null)
                report.AppendLine($"| MSPV (Vertex Positions) | Yes | {pm4File.VertexPositionsChunk.Size:N0} |");
            if (pm4File.VertexIndicesChunk != null)
                report.AppendLine($"| MSPI (Vertex Indices) | Yes | {pm4File.VertexIndicesChunk.Size:N0} |");
            if (pm4File.NormalCoordinatesChunk != null)
                report.AppendLine($"| MSCN (Normal Coordinates) | Yes | {pm4File.NormalCoordinatesChunk.Size:N0} |");
            if (pm4File.LinksChunk != null)
                report.AppendLine($"| MSLK (Links) | Yes | {pm4File.LinksChunk.Size:N0} |");
            if (pm4File.VertexDataChunk != null)
                report.AppendLine($"| MSVT (Vertex Data) | Yes | {pm4File.VertexDataChunk.Size:N0} |");
            if (pm4File.VertexInfoChunk != null)
                report.AppendLine($"| MSVI (Vertex Info) | Yes | {pm4File.VertexInfoChunk.Size:N0} |");
            if (pm4File.SurfaceDataChunk != null)
                report.AppendLine($"| MSUR (Surface Data) | Yes | {pm4File.SurfaceDataChunk.Size:N0} |");
            if (pm4File.PositionDataChunk != null)
                report.AppendLine($"| MPRL (Position Data) | Yes | {pm4File.PositionDataChunk.Size:N0} |");
            if (pm4File.ValuePairsChunk != null)
                report.AppendLine($"| MPRR (Value Pairs) | Yes | {pm4File.ValuePairsChunk.Size:N0} |");
            if (pm4File.BuildingDataChunk != null)
                report.AppendLine($"| MDBH (Building Data) | Yes | {pm4File.BuildingDataChunk.Size:N0} |");
            if (pm4File.SimpleDataChunk != null)
                report.AppendLine($"| MDOS (Simple Data) | Yes | {pm4File.SimpleDataChunk.Size:N0} |");
            if (pm4File.FinalDataChunk != null)
                report.AppendLine($"| MDSF (Final Data) | Yes | {pm4File.FinalDataChunk.Size:N0} |");

            // Position Data Analysis
            if (pm4File.PositionDataChunk != null)
            {
                report.AppendLine("\n## Position Data Analysis");
                await GeneratePositionDataReportAsync(pm4File.PositionDataChunk, report);
            }

            // Vertex Data Analysis
            if (pm4File.VertexPositionsChunk != null)
            {
                report.AppendLine("\n## Vertex Data Analysis");
                await GenerateVertexDataReportAsync(pm4File.VertexPositionsChunk, report);
            }
            
            // Triangle Data Analysis
            if (pm4File.VertexIndicesChunk != null)
            {
                report.AppendLine("\n## Triangle Data Analysis");
                await GenerateTriangleDataReportAsync(pm4File.VertexIndicesChunk, report);
            }
            
            // Errors section
            if (pm4File.Errors.Count > 0)
            {
                report.AppendLine("\n## Errors");
                foreach (var error in pm4File.Errors)
                {
                    report.AppendLine($"- {error}");
                }
            }

            // Write the report to file
            await File.WriteAllTextAsync(outputPath, report.ToString());
            _logger?.LogInformation("Generated comprehensive Markdown report: {FilePath}", outputPath);
        }

        /// <summary>
        /// Generates an individual file report as part of a multi-file report.
        /// </summary>
        private async Task GenerateIndividualFileReportAsync(PM4AnalysisResult result, StringBuilder report)
        {
            if (result.PM4File == null) return;
            
            // Basic information
            report.AppendLine($"Version: {result.Version}");
            report.AppendLine($"File Path: {result.FilePath ?? "Unknown"}");
            report.AppendLine();
            
            // Chunk information
            report.AppendLine("#### Chunks Present");
            report.AppendLine("| Chunk Type | Present | Size (bytes) |");
            report.AppendLine("|------------|---------|--------------|");
            
            // Add logic to display chunk information
            if (result.HasShadowData)
                report.AppendLine($"| MSHD (Shadow Data) | Yes | {result.PM4File.ShadowDataChunk?.Size ?? 0:N0} |");
            if (result.HasVertexPositions)
                report.AppendLine($"| MSPV (Vertex Positions) | Yes | {result.PM4File.VertexPositionsChunk?.Size ?? 0:N0} |");
            if (result.HasVertexIndices)
                report.AppendLine($"| MSPI (Vertex Indices) | Yes | {result.PM4File.VertexIndicesChunk?.Size ?? 0:N0} |");
            if (result.HasNormalCoordinates)
                report.AppendLine($"| MSCN (Normal Coordinates) | Yes | {result.PM4File.NormalCoordinatesChunk?.Size ?? 0:N0} |");
            if (result.HasLinks)
                report.AppendLine($"| MSLK (Links) | Yes | {result.PM4File.LinksChunk?.Size ?? 0:N0} |");
            if (result.HasVertexData)
                report.AppendLine($"| MSVT (Vertex Data) | Yes | {result.PM4File.VertexDataChunk?.Size ?? 0:N0} |");
            if (result.HasVertexInfo)
                report.AppendLine($"| MSVI (Vertex Info) | Yes | {result.PM4File.VertexInfoChunk?.Size ?? 0:N0} |");
            if (result.HasSurfaceData)
                report.AppendLine($"| MSUR (Surface Data) | Yes | {result.PM4File.SurfaceDataChunk?.Size ?? 0:N0} |");
            if (result.HasPositionData)
                report.AppendLine($"| MPRL (Position Data) | Yes | {result.PM4File.PositionDataChunk?.Size ?? 0:N0} |");
            if (result.HasValuePairs)
                report.AppendLine($"| MPRR (Value Pairs) | Yes | {result.PM4File.ValuePairsChunk?.Size ?? 0:N0} |");
            if (result.HasBuildingData)
                report.AppendLine($"| MDBH (Building Data) | Yes | {result.PM4File.BuildingDataChunk?.Size ?? 0:N0} |");
            if (result.HasSimpleData)
                report.AppendLine($"| MDOS (Simple Data) | Yes | {result.PM4File.SimpleDataChunk?.Size ?? 0:N0} |");
            if (result.HasFinalData)
                report.AppendLine($"| MDSF (Final Data) | Yes | {result.PM4File.FinalDataChunk?.Size ?? 0:N0} |");
            
            // Data summaries
            int vertexCount = result.PM4File.VertexPositionsChunk?.Vertices.Count ?? 0;
            int triangleCount = (result.PM4File.VertexIndicesChunk?.Indices.Count ?? 0) / 3;
            int positionCount = result.PM4File.PositionDataChunk?.Entries.Count(e => !e.IsSpecialEntry) ?? 0;
            int specialCount = result.PM4File.PositionDataChunk?.Entries.Count(e => e.IsSpecialEntry) ?? 0;
            
            report.AppendLine();
            report.AppendLine("#### Data Summary");
            report.AppendLine($"- Vertices: {vertexCount:N0}");
            report.AppendLine($"- Triangles: {triangleCount:N0}");
            report.AppendLine($"- Position Records: {positionCount:N0}");
            report.AppendLine($"- Special Records: {specialCount:N0}");
            
            // Position data summary if present
            if (result.PM4File.PositionDataChunk != null && result.PM4File.PositionDataChunk.Entries.Count > 0)
            {
                var positionEntries = result.PM4File.PositionDataChunk.Entries.Where(e => !e.IsSpecialEntry).ToList();
                var specialEntries = result.PM4File.PositionDataChunk.Entries.Where(e => e.IsSpecialEntry).ToList();
                
                if (positionEntries.Count > 0)
                {
                    report.AppendLine();
                    report.AppendLine("#### Position Data");
                    
                    // Coordinate ranges
                    float minX = positionEntries.Min(p => p.CoordinateX);
                    float maxX = positionEntries.Max(p => p.CoordinateX);
                    float minY = positionEntries.Min(p => p.CoordinateY);
                    float maxY = positionEntries.Max(p => p.CoordinateY);
                    float minZ = positionEntries.Min(p => p.CoordinateZ);
                    float maxZ = positionEntries.Max(p => p.CoordinateZ);
                    
                    report.AppendLine("Coordinate Ranges:");
                    report.AppendLine($"- X: {minX:F2} to {maxX:F2} (Range: {maxX - minX:F2})");
                    report.AppendLine($"- Y: {minY:F2} to {maxY:F2} (Range: {maxY - minY:F2})");
                    report.AppendLine($"- Z: {minZ:F2} to {maxZ:F2} (Range: {maxZ - minZ:F2})");
                }
                
                if (specialEntries.Count > 0)
                {
                    report.AppendLine();
                    report.AppendLine("#### Special Values");
                    
                    // Group by special value and count occurrences
                    var specialValueCounts = specialEntries
                        .GroupBy(e => e.SpecialValue)
                        .Select(g => new { SpecialValue = g.Key, Count = g.Count() })
                        .OrderByDescending(x => x.Count)
                        .Take(10)
                        .ToList();
                    
                    if (specialValueCounts.Count > 0)
                    {
                        report.AppendLine("Most Common Special Values:");
                        report.AppendLine("| Special Value | Hex Value | Count |");
                        report.AppendLine("|---------------|-----------|-------|");
                        
                        foreach (var value in specialValueCounts)
                        {
                            report.AppendLine($"| {value.SpecialValue} | 0x{value.SpecialValue:X8} | {value.Count} |");
                        }
                        
                        if (specialValueCounts.Count < specialEntries.GroupBy(e => e.SpecialValue).Count())
                        {
                            report.AppendLine($"\n_Showing top {specialValueCounts.Count} of {specialEntries.GroupBy(e => e.SpecialValue).Count()} distinct special values_");
                        }
                    }
                }
            }
            
            // Errors if any
            if (result.Errors.Count > 0)
            {
                report.AppendLine();
                report.AppendLine("#### Errors");
                foreach (var error in result.Errors)
                {
                    report.AppendLine($"- {error}");
                }
            }
        }

        /// <summary>
        /// Generates the position data section of the report.
        /// </summary>
        private async Task GeneratePositionDataReportAsync(MPRLChunk positionChunk, StringBuilder report)
        {
            var entries = positionChunk.Entries;
            int totalEntries = entries.Count;
            var positionRecords = entries.Where(e => !e.IsSpecialEntry).ToList();
            var specialRecords = entries.Where(e => e.IsSpecialEntry).ToList();

            report.AppendLine($"### Overview");
            report.AppendLine($"- Total Entries: {totalEntries:N0}");
            report.AppendLine($"- Position Records: {positionRecords.Count:N0}");
            report.AppendLine($"- Special Records: {specialRecords.Count:N0}");

            if (positionRecords.Count > 0)
            {
                report.AppendLine("\n### Position Records Analysis");
                
                // Calculate coordinate ranges
                var minX = positionRecords.Min(p => p.CoordinateX);
                var maxX = positionRecords.Max(p => p.CoordinateX);
                var minY = positionRecords.Min(p => p.CoordinateY);
                var maxY = positionRecords.Max(p => p.CoordinateY);
                var minZ = positionRecords.Min(p => p.CoordinateZ);
                var maxZ = positionRecords.Max(p => p.CoordinateZ);

                report.AppendLine("\n#### Coordinate Ranges");
                report.AppendLine("| Axis | Minimum | Maximum | Range |");
                report.AppendLine("|------|---------|---------|--------|");
                report.AppendLine($"| X | {minX:F2} | {maxX:F2} | {maxX - minX:F2} |");
                report.AppendLine($"| Y | {minY:F2} | {maxY:F2} | {maxY - minY:F2} |");
                report.AppendLine($"| Z | {minZ:F2} | {maxZ:F2} | {maxZ - minZ:F2} |");

                report.AppendLine("\n#### Sample Position Records");
                report.AppendLine("| Index | X | Y | Z |");
                report.AppendLine("|-------|-----|-----|-----|");
                foreach (var pos in positionRecords.Take(5))
                {
                    report.AppendLine($"| {pos.Index} | {pos.CoordinateX:F2} | {pos.CoordinateY:F2} | {pos.CoordinateZ:F2} |");
                }
            }

            if (specialRecords.Count > 0)
            {
                report.AppendLine("\n### Special Records Analysis");
                
                // Group by special value and count occurrences
                var specialValueCounts = specialRecords
                    .GroupBy(e => e.SpecialValue)
                    .Select(g => new { SpecialValue = g.Key, Count = g.Count() })
                    .OrderByDescending(x => x.Count)
                    .ToList();
                
                report.AppendLine("\n#### Special Value Distribution");
                report.AppendLine("| Special Value | Hex Value | Count | Percentage |");
                report.AppendLine("|---------------|-----------|-------|------------|");
                
                foreach (var value in specialValueCounts.Take(15))
                {
                    double percentage = (value.Count * 100.0) / specialRecords.Count;
                    report.AppendLine($"| {value.SpecialValue} | 0x{value.SpecialValue:X8} | {value.Count} | {percentage:F1}% |");
                }
                
                if (specialValueCounts.Count > 15)
                {
                    report.AppendLine($"\n_Showing top 15 of {specialValueCounts.Count} distinct special values_");
                }
                
                report.AppendLine("\n#### Sample Special Records");
                report.AppendLine("| Index | Command (Hex) | Y Coordinate | Special Value |");
                report.AppendLine("|-------|---------------|--------------|---------------|");
                foreach (var cmd in specialRecords.Take(10))
                {
                    report.AppendLine($"| {cmd.Index} | 0x{cmd.SpecialValue:X8} | {cmd.CoordinateY:F2} | {cmd.SpecialValue} |");
                }

                if (specialRecords.Count > 10)
                {
                    report.AppendLine($"\n_Showing top 10 of {specialRecords.Count} special records_");
                }
            }
        }

        /// <summary>
        /// Generates the vertex data section of the report.
        /// </summary>
        private async Task GenerateVertexDataReportAsync(MSPVChunk vertexChunk, StringBuilder report)
        {
            var vertices = vertexChunk.Vertices;
            
            report.AppendLine($"### Overview");
            report.AppendLine($"- Total Vertices: {vertices.Count:N0}");

            if (vertices.Count > 0)
            {
                // Calculate coordinate ranges
                var minX = vertices.Min(v => v.X);
                var maxX = vertices.Max(v => v.X);
                var minY = vertices.Min(v => v.Y);
                var maxY = vertices.Max(v => v.Y);
                var minZ = vertices.Min(v => v.Z);
                var maxZ = vertices.Max(v => v.Z);

                report.AppendLine("\n#### Coordinate Ranges");
                report.AppendLine("| Axis | Minimum | Maximum | Range |");
                report.AppendLine("|------|---------|---------|--------|");
                report.AppendLine($"| X | {minX:F2} | {maxX:F2} | {maxX - minX:F2} |");
                report.AppendLine($"| Y | {minY:F2} | {maxY:F2} | {maxY - minY:F2} |");
                report.AppendLine($"| Z | {minZ:F2} | {maxZ:F2} | {maxZ - minZ:F2} |");

                report.AppendLine("\n#### Sample Vertices");
                report.AppendLine("| Index | X | Y | Z |");
                report.AppendLine("|-------|-----|-----|-----|");
                for (int i = 0; i < Math.Min(5, vertices.Count); i++)
                {
                    var vertex = vertices[i];
                    report.AppendLine($"| {i} | {vertex.X:F2} | {vertex.Y:F2} | {vertex.Z:F2} |");
                }
            }
        }
        
        /// <summary>
        /// Generates the triangle data section of the report.
        /// </summary>
        private async Task GenerateTriangleDataReportAsync(MSPIChunk indexChunk, StringBuilder report)
        {
            var indices = indexChunk.Indices;
            int triangleCount = indices.Count / 3;
            
            report.AppendLine($"### Overview");
            report.AppendLine($"- Total Indices: {indices.Count:N0}");
            report.AppendLine($"- Total Triangles: {triangleCount:N0}");

            if (triangleCount > 0)
            {
                report.AppendLine("\n#### Sample Triangles");
                report.AppendLine("| Index | Vertex 1 | Vertex 2 | Vertex 3 |");
                report.AppendLine("|-------|----------|----------|----------|");
                
                for (int i = 0; i < Math.Min(5, triangleCount); i++)
                {
                    int baseIndex = i * 3;
                    if (baseIndex + 2 < indices.Count)
                    {
                        report.AppendLine($"| {i} | {indices[baseIndex]} | {indices[baseIndex + 1]} | {indices[baseIndex + 2]} |");
                    }
                }
            }
        }
    }
} 