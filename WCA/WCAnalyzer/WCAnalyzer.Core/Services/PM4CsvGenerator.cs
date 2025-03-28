using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
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
        private readonly ILogger? _logger;
        private readonly string _outputDirectory;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4CsvGenerator"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        /// <param name="outputDirectory">Optional output directory for the reports</param>
        public PM4CsvGenerator(ILogger? logger = null, string? outputDirectory = null)
        {
            _logger = logger;
            _outputDirectory = outputDirectory ?? Path.Combine(Directory.GetCurrentDirectory(), "csv_reports");
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

            try
            {
                _logger?.LogInformation("Generating CSV reports for {FileName}", result.FileName);

                // Create the output directory if it doesn't exist
                var csvDir = Path.Combine(_outputDirectory, "pm4_csv");
                Directory.CreateDirectory(csvDir);

                // Only generate reports if the result has a valid PM4File
                if (result.PM4File == null)
                {
                    _logger?.LogWarning("Cannot generate CSV reports for {FileName}: PM4File is null", result.FileName);
                    return;
                }

                // Generate the vertex positions report
                await GenerateVertexPositionsReportAsync(result, csvDir);

                // Generate the vertex indices report
                await GenerateVertexIndicesReportAsync(result, csvDir);

                // Generate the position data report
                await GeneratePositionDataReportAsync(result, csvDir);

                _logger?.LogInformation("CSV report generation completed for {FileName}", result.FileName);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating CSV reports for {FileName}", result.FileName);
            }
        }

        /// <summary>
        /// Generates a CSV report for vertex positions.
        /// </summary>
        private async Task GenerateVertexPositionsReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (result.PM4File?.VertexPositionsChunk == null || result.PM4File.VertexPositionsChunk.Vertices.Count == 0)
            {
                _logger?.LogInformation("No vertex positions found in {FileName}", result.FileName);
                return;
            }

            var fileName = Path.GetFileNameWithoutExtension(result.FileName) + "_vertices.csv";
            var filePath = Path.Combine(outputDir, fileName);

            try
            {
                using var writer = new StreamWriter(filePath);
                
                // Write header
                await writer.WriteLineAsync("Index,X,Y,Z");
                
                // Write data
                for (int i = 0; i < result.PM4File.VertexPositionsChunk.Vertices.Count; i++)
                {
                    var vertex = result.PM4File.VertexPositionsChunk.Vertices[i];
                    await writer.WriteLineAsync($"{i},{vertex.X.ToString(CultureInfo.InvariantCulture)},{vertex.Y.ToString(CultureInfo.InvariantCulture)},{vertex.Z.ToString(CultureInfo.InvariantCulture)}");
                }
                
                _logger?.LogInformation("Generated vertex positions report: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating vertex positions report: {FilePath}", filePath);
            }
        }

        /// <summary>
        /// Generates a CSV report for vertex indices.
        /// </summary>
        private async Task GenerateVertexIndicesReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (result.PM4File?.VertexIndicesChunk == null || result.PM4File.VertexIndicesChunk.Indices.Count == 0)
            {
                _logger?.LogInformation("No vertex indices found in {FileName}", result.FileName);
                return;
            }

            var fileName = Path.GetFileNameWithoutExtension(result.FileName) + "_triangles.csv";
            var filePath = Path.Combine(outputDir, fileName);

            try
            {
                using var writer = new StreamWriter(filePath);
                
                // Write header
                await writer.WriteLineAsync("TriangleIndex,Vertex1,Vertex2,Vertex3");
                
                // Write data
                int triangleCount = result.PM4File.VertexIndicesChunk.Indices.Count / 3;
                for (int i = 0; i < triangleCount; i++)
                {
                    int baseIndex = i * 3;
                    if (baseIndex + 2 < result.PM4File.VertexIndicesChunk.Indices.Count)
                    {
                        await writer.WriteLineAsync($"{i},{result.PM4File.VertexIndicesChunk.Indices[baseIndex]},{result.PM4File.VertexIndicesChunk.Indices[baseIndex + 1]},{result.PM4File.VertexIndicesChunk.Indices[baseIndex + 2]}");
                    }
                }
                
                _logger?.LogInformation("Generated vertex indices report: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating vertex indices report: {FilePath}", filePath);
            }
        }

        /// <summary>
        /// Generates a CSV report for position data.
        /// </summary>
        private async Task GeneratePositionDataReportAsync(PM4AnalysisResult result, string outputDir)
        {
            if (result.PM4File?.PositionDataChunk == null || result.PM4File.PositionDataChunk.Entries.Count == 0)
            {
                _logger?.LogInformation("No position data found in {FileName}", result.FileName);
                return;
            }

            var baseFileName = Path.GetFileNameWithoutExtension(result.FileName);
            
            // Create two separate reports - one for position records and one for command records
            await GeneratePositionRecordsReportAsync(result.PM4File.PositionDataChunk, baseFileName, outputDir);
            await GenerateCommandRecordsReportAsync(result.PM4File.PositionDataChunk, baseFileName, outputDir);
            
            // Generate a sequence report showing the pattern of commands and positions
            await GenerateSequenceReportAsync(result.PM4File.PositionDataChunk, baseFileName, outputDir);
        }

        /// <summary>
        /// Generates a CSV report for position records (regular 3D positions).
        /// </summary>
        private async Task GeneratePositionRecordsReportAsync(MPRLChunk positionChunk, string baseFileName, string outputDir)
        {
            var positionRecords = positionChunk.Entries.Where(e => !e.IsSpecialEntry).ToList();
            if (positionRecords.Count == 0)
            {
                _logger?.LogInformation("No position records found in {FileName}", baseFileName);
                return;
            }

            var fileName = baseFileName + "_position_records.csv";
            var filePath = Path.Combine(outputDir, fileName);

            try
            {
                using var writer = new StreamWriter(filePath);
                
                // Write header with all possible values
                await writer.WriteLineAsync("Index,X,Y,Z,Value1,Value2,Value3,IsSpecial");
                
                // Write data with all values
                foreach (var entry in positionRecords)
                {
                    await writer.WriteLineAsync(
                        $"{entry.Index}," +
                        $"{entry.CoordinateX.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.CoordinateY.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.CoordinateZ.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value1.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value2.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value3.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.IsSpecialEntry}"
                    );
                }
                
                _logger?.LogInformation("Generated position records report: {FilePath} ({Count} entries)", filePath, positionRecords.Count);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating position records report: {FilePath}", filePath);
            }
        }

        /// <summary>
        /// Generates a CSV report for command records.
        /// </summary>
        private async Task GenerateCommandRecordsReportAsync(MPRLChunk positionChunk, string baseFileName, string outputDir)
        {
            var specialRecords = positionChunk.Entries.Where(e => e.IsSpecialEntry).ToList();
            if (specialRecords.Count == 0)
            {
                _logger?.LogInformation("No special entries found in {FileName}", baseFileName);
                return;
            }

            var fileName = baseFileName + "_special_entries.csv";
            var filePath = Path.Combine(outputDir, fileName);

            try
            {
                using var writer = new StreamWriter(filePath);
                
                // Write header with all possible values
                await writer.WriteLineAsync("Index,SpecialValueHex,SpecialValueDec,AsFloat,X,Y,Z,Value1,Value2,Value3,IsSpecial");
                
                // Write data with all values
                foreach (var entry in specialRecords)
                {
                    string specialHex = $"0x{entry.SpecialValue:X8}";
                    // Reinterpret the bit pattern as a float
                    float asFloat = BitConverter.Int32BitsToSingle(entry.SpecialValue);
                    await writer.WriteLineAsync(
                        $"{entry.Index}," +
                        $"{specialHex}," +
                        $"{entry.SpecialValue}," +
                        $"{asFloat.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.CoordinateX.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.CoordinateY.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.CoordinateZ.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value1.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value2.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value3.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.IsSpecialEntry}"
                    );
                }
                
                _logger?.LogInformation("Generated special entries report: {FilePath} ({Count} entries)", filePath, specialRecords.Count);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating special entries report: {FilePath}", filePath);
            }
        }

        /// <summary>
        /// Generates a CSV report showing the sequence of commands and positions.
        /// </summary>
        private async Task GenerateSequenceReportAsync(MPRLChunk positionChunk, string baseFileName, string outputDir)
        {
            if (positionChunk.Entries.Count == 0)
            {
                return;
            }

            var fileName = baseFileName + "_sequence.csv";
            var filePath = Path.Combine(outputDir, fileName);

            try
            {
                using var writer = new StreamWriter(filePath);
                
                // Write header
                await writer.WriteLineAsync("Index,Type,Value1,Value2,Value3,CommandHex,X,Y,Z");
                
                // Write data
                foreach (var entry in positionChunk.Entries)
                {
                    string type = entry.IsSpecialEntry ? "Special" : "Position";
                    string value1 = float.IsNaN(entry.Value1) ? "NaN" : entry.Value1.ToString(CultureInfo.InvariantCulture);
                    string value3 = float.IsNaN(entry.Value3) ? "NaN" : entry.Value3.ToString(CultureInfo.InvariantCulture);
                    string commandHex = entry.IsSpecialEntry ? $"0x{entry.SpecialValue:X8}" : "";
                    
                    await writer.WriteLineAsync($"{entry.Index},{type},{value1},{entry.Value2.ToString(CultureInfo.InvariantCulture)},{value3},{commandHex},{entry.CoordinateX.ToString(CultureInfo.InvariantCulture)},{entry.CoordinateY.ToString(CultureInfo.InvariantCulture)},{entry.CoordinateZ.ToString(CultureInfo.InvariantCulture)}");
                }
                
                _logger?.LogInformation("Generated sequence report: {FilePath} ({Count} entries)", filePath, positionChunk.Entries.Count);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating sequence report: {FilePath}", filePath);
            }
        }

        private async Task GenerateAllEntriesReportAsync(MPRLChunk positionChunk, string baseFileName, string outputDir)
        {
            if (positionChunk.Entries.Count == 0)
            {
                _logger?.LogInformation("No entries found in {FileName}", baseFileName);
                return;
            }

            var fileName = baseFileName + "_all_entries.csv";
            var filePath = Path.Combine(outputDir, fileName);

            try
            {
                using var writer = new StreamWriter(filePath);
                
                // Write header with all possible values
                await writer.WriteLineAsync("Index,EntryType,X,Y,Z,Value1,Value2,Value3,SpecialValueHex,SpecialValueDec,AsFloat,IsSpecial");
                
                // Write data with all values
                foreach (var entry in positionChunk.Entries)
                {
                    string type = entry.IsSpecialEntry ? "Special" : "Position";
                    string specialHex = entry.IsSpecialEntry ? $"0x{entry.SpecialValue:X8}" : "";
                    string specialDec = entry.IsSpecialEntry ? entry.SpecialValue.ToString() : "";
                    string asFloat = entry.IsSpecialEntry ? BitConverter.Int32BitsToSingle(entry.SpecialValue).ToString(CultureInfo.InvariantCulture) : "";
                    
                    await writer.WriteLineAsync(
                        $"{entry.Index}," +
                        $"{type}," +
                        $"{entry.CoordinateX.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.CoordinateY.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.CoordinateZ.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value1.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value2.ToString(CultureInfo.InvariantCulture)}," +
                        $"{entry.Value3.ToString(CultureInfo.InvariantCulture)}," +
                        $"{specialHex}," +
                        $"{specialDec}," +
                        $"{asFloat}," +
                        $"{entry.IsSpecialEntry}"
                    );
                }
                
                _logger?.LogInformation("Generated all entries report: {FilePath} ({Count} entries)", filePath, positionChunk.Entries.Count);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating all entries report: {FilePath}", filePath);
            }
        }

        public async Task GenerateReportsAsync(MPRLChunk positionChunk, string baseFileName, string outputDir)
        {
            if (positionChunk == null)
                throw new ArgumentNullException(nameof(positionChunk));
            if (string.IsNullOrEmpty(baseFileName))
                throw new ArgumentException("Base file name cannot be null or empty.", nameof(baseFileName));
            if (string.IsNullOrEmpty(outputDir))
                throw new ArgumentException("Output directory cannot be null or empty.", nameof(outputDir));

            // Create output directory if it doesn't exist
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            // Generate position records report
            await GeneratePositionRecordsReportAsync(positionChunk, baseFileName, outputDir);

            // Generate command records report
            await GenerateCommandRecordsReportAsync(positionChunk, baseFileName, outputDir);
            
            // Generate all entries report (combined)
            await GenerateAllEntriesReportAsync(positionChunk, baseFileName, outputDir);
        }
    }
} 