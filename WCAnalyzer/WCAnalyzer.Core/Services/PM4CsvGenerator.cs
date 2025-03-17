using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;
using WCAnalyzer.Core.Utilities;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for generating CSV reports from PM4 analysis results.
    /// </summary>
    public class PM4CsvGenerator
    {
        private readonly ILogger<PM4CsvGenerator> _logger;
        private readonly string _outputDirectory;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4CsvGenerator"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        /// <param name="outputDirectory">Optional output directory for the reports</param>
        public PM4CsvGenerator(ILogger<PM4CsvGenerator>? logger = null, string? outputDirectory = null)
        {
            _logger = logger ?? NullLogger<PM4CsvGenerator>.Instance;
            _outputDirectory = outputDirectory ?? Path.Combine(Environment.CurrentDirectory, "output");

            // Create output directory if it doesn't exist
            if (!Directory.Exists(_outputDirectory))
            {
                Directory.CreateDirectory(_outputDirectory);
            }
        }

        /// <summary>
        /// Generates all CSV reports from a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GenerateAllCsvReportsAsync(PM4AnalysisResult result)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (string.IsNullOrEmpty(result.FileName))
            {
                _logger.LogWarning("PM4 analysis result has no filename, using 'unknown'");
                result.FileName = "unknown";
            }

            _logger.LogInformation("Generating CSV reports for {FileName}", result.FileName);

            var outputDir = Path.Combine(_outputDirectory, Path.GetFileNameWithoutExtension(result.FileName));
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            try
            {
                // Generate vertex positions report
                await GenerateVertexPositionsReportAsync(result, outputDir);

                // Generate vertex indices report
                await GenerateVertexIndicesReportAsync(result, outputDir);

                // Generate position data report
                await GeneratePositionDataReportAsync(result, outputDir);

                // Generate position references report
                await GeneratePositionReferencesReportAsync(result, outputDir);

                // Generate vertex data report (MSVT)
                await GenerateVertexDataReportAsync(result, outputDir);

                // Generate links report
                await GenerateLinksReportAsync(result, outputDir);

                // Generate summary report
                await GenerateSummaryReportAsync(result, outputDir);

                // Generate unknown chunks report
                await GenerateUnknownChunksReportAsync(result, outputDir);

                // Export to OBJ format if vertex data is available (either MSPV or MSVT)
                if (result.HasVertexPositions || result.HasVertexData)
                {
                    _logger.LogInformation("Exporting vertex data to OBJ format");
                    var objExporter = new PM4ObjExporter();
                    await objExporter.ExportToDirectoryAsync(result, outputDir);
                }

                _logger.LogInformation("Successfully generated all CSV reports for {FileName}", result.FileName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating CSV reports for {FileName}", result.FileName);
                throw;
            }
        }

        /// <summary>
        /// Generates a CSV report of vertex positions from a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateVertexPositionsReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (!result.HasVertexPositions)
            {
                _logger.LogInformation("No vertex positions data in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "vertex_positions.csv");
            _logger.LogInformation("Generating vertex positions report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Index,X,Y,Z");

            var positions = result.PM4Data.VertexPositions;
            for (int i = 0; i < positions.Count; i++)
            {
                var pos = positions[i];
                csv.AppendLine($"{i},{pos.X.ToString("F8", CultureInfo.InvariantCulture)},{pos.Y.ToString("F8", CultureInfo.InvariantCulture)},{pos.Z.ToString("F8", CultureInfo.InvariantCulture)}");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger.LogInformation("Vertex positions report generated with {Count} entries", positions.Count);
        }

        /// <summary>
        /// Generates a CSV report of vertex indices from a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateVertexIndicesReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (!result.HasVertexIndices)
            {
                _logger.LogInformation("No vertex indices data in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "vertex_indices.csv");
            _logger.LogInformation("Generating vertex indices report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Triangle,Index1,Index2,Index3");

            var indices = result.PM4Data.VertexIndices;
            int triangleCount = indices.Count / 3;

            for (int i = 0; i < triangleCount; i++)
            {
                int baseIndex = i * 3;
                if (baseIndex + 2 < indices.Count)
                {
                    csv.AppendLine($"{i},{indices[baseIndex]},{indices[baseIndex + 1]},{indices[baseIndex + 2]}");
                }
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger.LogInformation("Vertex indices report generated with {Count} triangles", triangleCount);
        }

        /// <summary>
        /// Generates a CSV report of position data from a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GeneratePositionDataReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (!result.HasPositionData)
            {
                _logger.LogInformation("No position data in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "position_data.csv");
            _logger.LogInformation("Generating position data report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Index,Value0x00,Value0x04,Value0x08,Value0x0C,Value0x10");

            var positions = result.PM4Data.PositionData;
            for (int i = 0; i < positions.Count; i++)
            {
                var pos = positions[i];
                csv.AppendLine($"{i},{pos.Value0x00},{pos.Value0x04},{pos.Value0x08.ToString("F8", CultureInfo.InvariantCulture)},{pos.Value0x0C.ToString("F8", CultureInfo.InvariantCulture)},{pos.Value0x10}");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger.LogInformation("Position data report generated with {Count} entries", positions.Count);
        }

        /// <summary>
        /// Generates a CSV report of position references from a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GeneratePositionReferencesReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (!result.HasPositionReferences)
            {
                _logger.LogInformation("No position references in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "position_references.csv");
            _logger.LogInformation("Generating position references report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Index,Value0x00,Value0x04");

            var references = result.PM4Data.PositionReferences;
            for (int i = 0; i < references.Count; i++)
            {
                var reference = references[i];
                
                // Output the actual values as they are without trying to convert them to floats
                // since they're 16-bit values and not IEEE 754 format
                csv.AppendLine($"{i},{reference.Value0x00},{reference.Value0x04}");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger.LogInformation("Position references report generated with {Count} entries", references.Count);
        }

        /// <summary>
        /// Generates a CSV report of links from a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateLinksReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (!result.HasLinks)
            {
                _logger.LogInformation("No links data in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "links.csv");
            _logger.LogInformation("Generating links report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Index,SourceIndex,TargetIndex");

            var links = result.PM4Data.Links;
            for (int i = 0; i < links.Count; i++)
            {
                var link = links[i];
                csv.AppendLine($"{i},{link.SourceIndex},{link.TargetIndex}");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger.LogInformation("Links report generated with {Count} entries", links.Count);
        }

        /// <summary>
        /// Generates all CSV reports from multiple PM4 analysis results.
        /// </summary>
        /// <param name="results">The PM4 analysis results to report on.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GenerateAllCsvReportsAsync(IEnumerable<PM4AnalysisResult> results)
        {
            foreach (var result in results)
            {
                await GenerateAllCsvReportsAsync(result);
            }
        }

        /// <summary>
        /// Generates reports for a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <param name="outputDirectory">Optional output directory for the reports.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task GenerateReportsAsync(PM4AnalysisResult result, string? outputDirectory = null)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            string effectiveOutputDir = outputDirectory ?? _outputDirectory;

            if (string.IsNullOrEmpty(result.FileName))
            {
                _logger.LogWarning("PM4 analysis result has no filename, using 'unknown'");
                result.FileName = "unknown";
            }

            _logger.LogInformation("Generating reports for {FileName} in directory {OutputDirectory}", result.FileName, effectiveOutputDir);

            var fileOutputDir = Path.Combine(effectiveOutputDir, Path.GetFileNameWithoutExtension(result.FileName));
            if (!Directory.Exists(fileOutputDir))
            {
                Directory.CreateDirectory(fileOutputDir);
            }

            try
            {
                // Generate vertex positions report
                await GenerateVertexPositionsReportAsync(result, fileOutputDir);

                // Generate vertex indices report
                await GenerateVertexIndicesReportAsync(result, fileOutputDir);

                // Generate position data report
                await GeneratePositionDataReportAsync(result, fileOutputDir);

                // Generate position references report
                await GeneratePositionReferencesReportAsync(result, fileOutputDir);

                // Generate vertex data report (MSVT)
                await GenerateVertexDataReportAsync(result, fileOutputDir);

                // Generate links report
                await GenerateLinksReportAsync(result, fileOutputDir);
                
                // Generate summary report
                await GenerateSummaryReportAsync(result, fileOutputDir);

                // Generate unknown chunks report
                await GenerateUnknownChunksReportAsync(result, fileOutputDir);
                
                // Export to OBJ format if vertex data is available (either MSPV or MSVT)
                if (result.HasVertexPositions || result.HasVertexData)
                {
                    _logger.LogInformation("Exporting vertex data to OBJ format");
                    var objExporter = new PM4ObjExporter();
                    await objExporter.ExportToDirectoryAsync(result, fileOutputDir);
                }

                _logger.LogInformation("Successfully generated all reports for {FileName}", result.FileName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating reports for {FileName}", result.FileName);
                throw;
            }
        }

        /// <summary>
        /// Generates a CSV report for position data from a PM4 chunk.
        /// </summary>
        /// <param name="positionData">The position data.</param>
        /// <param name="outputPath">The output path for the CSV file.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GeneratePositionDataCsvAsync(List<PositionData> positionData, string outputPath)
        {
            if (positionData == null)
            {
                throw new ArgumentNullException(nameof(positionData));
            }

            if (string.IsNullOrEmpty(outputPath))
            {
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
            }

            try
            {
                using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
                
                // Write header
                await writer.WriteLineAsync("Index,Value0x00,Value0x04,Value0x08,Value0x0C,Value0x10,Value0x14,Value0x18,Value0x1C");
                
                // Write data
                for (int i = 0; i < positionData.Count; i++)
                {
                    var entry = positionData[i];
                    
                    await writer.WriteLineAsync(
                        $"{i}," +
                        $"{entry.Value0x00}," +
                        $"{entry.Value0x04}," +
                        $"{entry.Value0x08.ToString("F8", CultureInfo.InvariantCulture)}," +
                        $"{entry.Value0x0C.ToString("F8", CultureInfo.InvariantCulture)}," +
                        $"{entry.Value0x10.ToString("F8", CultureInfo.InvariantCulture)}," +
                        $"{entry.Value0x14.ToString("F8", CultureInfo.InvariantCulture)}," +
                        $"{entry.Value0x18.ToString("F8", CultureInfo.InvariantCulture)}," +
                        $"{entry.Value0x1C.ToString("F8", CultureInfo.InvariantCulture)}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating position data CSV: {Path}", outputPath);
                throw;
            }
        }

        /// <summary>
        /// Generates a CSV report for vertex positions from a PM4 chunk.
        /// </summary>
        /// <param name="vertexPositions">The vertex positions.</param>
        /// <param name="outputPath">The output path for the CSV file.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GenerateVertexPositionsCsvAsync(List<Vector3> vertexPositions, string outputPath)
        {
            if (vertexPositions == null)
            {
                throw new ArgumentNullException(nameof(vertexPositions));
            }

            if (string.IsNullOrEmpty(outputPath))
            {
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
            }

            try
            {
                using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
                
                // Write header
                await writer.WriteLineAsync("Index,X,Y,Z");
                
                // Write data
                for (int i = 0; i < vertexPositions.Count; i++)
                {
                    var vertex = vertexPositions[i];
                    
                    await writer.WriteLineAsync(
                        $"{i}," +
                        $"{vertex.X.ToString("F8", CultureInfo.InvariantCulture)}," +
                        $"{vertex.Y.ToString("F8", CultureInfo.InvariantCulture)}," +
                        $"{vertex.Z.ToString("F8", CultureInfo.InvariantCulture)}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating vertex positions CSV: {Path}", outputPath);
                throw;
            }
        }

        /// <summary>
        /// Generates a CSV report for vertex indices from a PM4 chunk.
        /// </summary>
        /// <param name="vertexIndices">The vertex indices.</param>
        /// <param name="outputPath">The output path for the CSV file.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GenerateVertexIndicesCsvAsync(List<int> vertexIndices, string outputPath)
        {
            if (vertexIndices == null)
            {
                throw new ArgumentNullException(nameof(vertexIndices));
            }

            if (string.IsNullOrEmpty(outputPath))
            {
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
            }

            try
            {
                using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
                
                // Write header
                await writer.WriteLineAsync("Index,Value");
                
                // Write data
                for (int i = 0; i < vertexIndices.Count; i++)
                {
                    await writer.WriteLineAsync($"{i},{vertexIndices[i]}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating vertex indices CSV: {Path}", outputPath);
                throw;
            }
        }

        /// <summary>
        /// Generates a CSV file with link data.
        /// </summary>
        /// <param name="links">List of links.</param>
        /// <param name="outputPath">Output file path.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task GenerateLinksCsvAsync(List<LinkData> links, string outputPath)
        {
            if (links == null || links.Count == 0)
            {
                _logger?.LogWarning("No link data to export to CSV file");
                return;
            }

            try
            {
                // Ensure directory exists
                var directory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                _logger?.LogInformation("Generating links CSV file: {OutputPath}", outputPath);

                // Write CSV file
                using (var writer = new StreamWriter(outputPath, false, Encoding.UTF8))
                {
                    // Write header
                    await writer.WriteLineAsync("Index,SourceIndex,TargetIndex,Value0x00,Value0x04,Value0x08,Value0x0C");
                    
                    // Write data
                    for (int i = 0; i < links.Count; i++)
                    {
                        var link = links[i];
                        await writer.WriteLineAsync(
                            $"{i}," +
                            $"{link.SourceIndex}," +
                            $"{link.TargetIndex}," +
                            $"{link.Value0x00}," +
                            $"{link.Value0x04}," +
                            $"{link.Value0x08}," +
                            $"{link.Value0x0C}");
                    }
                }

                _logger?.LogInformation("Successfully generated links CSV file: {OutputPath}", outputPath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating links CSV file: {OutputPath}", outputPath);
                throw;
            }
        }

        /// <summary>
        /// Generates a CSV report for position references from a PM4 chunk.
        /// </summary>
        /// <param name="positionReferences">The position references.</param>
        /// <param name="outputPath">The output path for the CSV file.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GeneratePositionReferencesCsvAsync(List<PositionReference> positionReferences, string outputPath)
        {
            if (positionReferences == null)
            {
                throw new ArgumentNullException(nameof(positionReferences));
            }

            if (string.IsNullOrEmpty(outputPath))
            {
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
            }

            try
            {
                using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
                
                // Write header
                await writer.WriteLineAsync("Index,Value0x00,Value0x04");
                
                // Write data
                for (int i = 0; i < positionReferences.Count; i++)
                {
                    var reference = positionReferences[i];
                    
                    // Output the actual values as they are without trying to convert them to floats
                    // since they're 16-bit values and not IEEE 754 format
                    await writer.WriteLineAsync($"{i},{reference.Value0x00},{reference.Value0x04}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating position references CSV: {Path}", outputPath);
                throw;
            }
        }

        /// <summary>
        /// Generates a CSV report of vertex data (MSVT chunk) from a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateVertexDataReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (!result.HasVertexData)
            {
                _logger.LogInformation("No vertex data (MSVT) in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "vertex_data_msvt.csv");
            _logger.LogInformation("Generating vertex data (MSVT) report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Index,RawY,RawX,RawZ,WorldX,WorldY,WorldZ");

            var vertices = result.PM4Data.VertexData;
            for (int i = 0; i < vertices.Count; i++)
            {
                var vertex = vertices[i];
                csv.AppendLine($"{i}," +
                    $"{vertex.RawY.ToString("F8", CultureInfo.InvariantCulture)}," +
                    $"{vertex.RawX.ToString("F8", CultureInfo.InvariantCulture)}," +
                    $"{vertex.RawZ.ToString("F8", CultureInfo.InvariantCulture)}," +
                    $"{vertex.WorldX.ToString("F8", CultureInfo.InvariantCulture)}," +
                    $"{vertex.WorldY.ToString("F8", CultureInfo.InvariantCulture)}," +
                    $"{vertex.WorldZ.ToString("F8", CultureInfo.InvariantCulture)}");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger.LogInformation("Vertex data (MSVT) report generated with {Count} entries", vertices.Count);
        }

        /// <summary>
        /// Generates a report of unknown chunks from a PM4 analysis result.
        /// </summary>
        /// <param name="result">The PM4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateUnknownChunksReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (result.PM4Data.UnknownChunks == null || result.PM4Data.UnknownChunks.Count == 0)
            {
                _logger?.LogInformation("No unknown chunks data in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "unknown_chunks.csv");
            _logger?.LogInformation("Generating unknown chunks report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("ChunkName,OriginalChunkName,Size,Position,HexPreview");

            foreach (var chunk in result.PM4Data.UnknownChunks)
            {
                csv.AppendLine($"{chunk.ChunkName},{chunk.OriginalChunkName},{chunk.Size},{chunk.Position},\"{chunk.HexPreview}\"");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger?.LogInformation("Unknown chunks report generated with {Count} entries", result.PM4Data.UnknownChunks.Count);
        }

        private async Task GenerateSummaryReportAsync(PM4AnalysisResult result, string outputDir)
        {
            var outputPath = Path.Combine(outputDir, "summary.txt");
            _logger?.LogInformation("Generating summary report at {OutputPath}", outputPath);

            var summary = new StringBuilder();
            summary.AppendLine($"PM4 Analysis Summary for {result.FileName}");
            summary.AppendLine($"==================================");
            summary.AppendLine();
            summary.AppendLine($"File: {result.FilePath}");
            summary.AppendLine($"Version: {result.PM4Data.Version}");
            summary.AppendLine();
            
            summary.AppendLine("Data present in the file:");
            summary.AppendLine($"- Vertex Positions: {(result.HasVertexPositions ? "Yes" : "No")} ({result.PM4Data.VertexPositions.Count} entries)");
            summary.AppendLine($"- Vertex Indices: {(result.HasVertexIndices ? "Yes" : "No")} ({result.PM4Data.VertexIndices.Count} entries)");
            summary.AppendLine($"- Position Data: {(result.HasPositionData ? "Yes" : "No")} ({result.PM4Data.PositionData.Count} entries)");
            summary.AppendLine($"- Position References: {(result.HasPositionReference ? "Yes" : "No")} ({result.PM4Data.PositionReferences.Count} entries)");
            summary.AppendLine($"- Vertex Data (MSVT): {(result.HasVertexData ? "Yes" : "No")} ({result.PM4Data.VertexData.Count} entries)");
            summary.AppendLine($"- Links: {(result.HasLinks ? "Yes" : "No")} ({result.PM4Data.Links.Count} entries)");
            summary.AppendLine();
            
            // Include unknown chunks information
            if (result.PM4Data.UnknownChunks?.Count > 0)
            {
                summary.AppendLine($"Unknown Chunks Found: {result.PM4Data.UnknownChunks.Count}");
                summary.AppendLine("-------------------------");
                summary.AppendLine("Chunk ID | Original ID | Size (bytes) | Position | Data Preview (hex)");
                summary.AppendLine("---------|-------------|--------------|----------|------------------");
                
                foreach (var chunk in result.PM4Data.UnknownChunks)
                {
                    // Try to identify float values in hex and display them in standard notation
                    string previewText = chunk.HexPreview;
                    
                    summary.AppendLine($"{chunk.ChunkName,-9}| {chunk.OriginalChunkName,-11}| {chunk.Size,-12}| {chunk.Position,-8}| {previewText}");
                }
                
                summary.AppendLine();
                summary.AppendLine("NOTE: Unknown chunks may contain important data that we don't yet have parsers for.");
                summary.AppendLine();
            }
            
            summary.AppendLine("CSV Reports Generated:");
            if (result.HasVertexPositions)
                summary.AppendLine("- vertex_positions.csv");
            if (result.HasVertexIndices)
                summary.AppendLine("- vertex_indices.csv");
            if (result.HasPositionData)
                summary.AppendLine("- position_data.csv");
            if (result.HasPositionReference)
                summary.AppendLine("- position_references.csv");
            if (result.HasVertexData)
                summary.AppendLine("- vertex_data_msvt.csv");
            if (result.HasLinks)
                summary.AppendLine("- links.csv");
            
            if (result.HasErrors)
            {
                summary.AppendLine();
                summary.AppendLine("Errors encountered during parsing:");
                foreach (var error in result.Errors)
                {
                    summary.AppendLine($"- {error}");
                }
            }
            
            await File.WriteAllTextAsync(outputPath, summary.ToString());
            _logger?.LogInformation("Summary report generated");
        }
    }
} 