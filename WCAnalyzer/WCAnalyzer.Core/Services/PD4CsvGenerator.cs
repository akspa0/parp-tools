using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PD4;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for generating CSV reports from PD4 analysis results.
    /// </summary>
    public class PD4CsvGenerator
    {
        private readonly ILogger<PD4CsvGenerator>? _logger;
        private readonly string _outputDirectory;

        /// <summary>
        /// Initializes a new instance of the <see cref="PD4CsvGenerator"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        /// <param name="outputDirectory">Optional output directory for the reports</param>
        public PD4CsvGenerator(ILogger<PD4CsvGenerator>? logger = null, string? outputDirectory = null)
        {
            _logger = logger;
            _outputDirectory = outputDirectory ?? Path.Combine(Environment.CurrentDirectory, "output");

            // Create output directory if it doesn't exist
            if (!Directory.Exists(_outputDirectory))
            {
                Directory.CreateDirectory(_outputDirectory);
            }
        }

        /// <summary>
        /// Generates all CSV reports from a PD4 analysis result.
        /// </summary>
        /// <param name="result">The PD4 analysis result to report on.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GenerateAllCsvReportsAsync(PD4AnalysisResult result)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (string.IsNullOrEmpty(result.FileName))
            {
                _logger?.LogWarning("PD4 analysis result has no filename, using 'unknown'");
                result.FileName = "unknown";
            }

            _logger?.LogInformation("Generating CSV reports for {FileName}", result.FileName);

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

                // Generate texture names report
                await GenerateTextureNamesReportAsync(result, outputDir);

                // Generate material data report
                await GenerateMaterialDataReportAsync(result, outputDir);

                // Generate summary report
                await GenerateSummaryReportAsync(result, outputDir);

                _logger?.LogInformation("Successfully generated all CSV reports for {FileName}", result.FileName);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating CSV reports for {FileName}", result.FileName);
                throw;
            }
        }

        /// <summary>
        /// Generates a CSV report of vertex positions from a PD4 analysis result.
        /// </summary>
        /// <param name="result">The PD4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateVertexPositionsReportAsync(PD4AnalysisResult result, string outputDir)
        {
            if (!result.HasVertexPositions)
            {
                _logger?.LogInformation("No vertex positions data in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "vertex_positions.csv");
            _logger?.LogInformation("Generating vertex positions report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Index,X,Y,Z");

            var positions = result.PD4Data.VertexPositions;
            for (int i = 0; i < positions.Count; i++)
            {
                var pos = positions[i];
                csv.AppendLine($"{i},{pos.X.ToString(CultureInfo.InvariantCulture)},{pos.Y.ToString(CultureInfo.InvariantCulture)},{pos.Z.ToString(CultureInfo.InvariantCulture)}");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger?.LogInformation("Vertex positions report generated with {Count} entries", positions.Count);
        }

        /// <summary>
        /// Generates a CSV report of vertex indices from a PD4 analysis result.
        /// </summary>
        /// <param name="result">The PD4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateVertexIndicesReportAsync(PD4AnalysisResult result, string outputDir)
        {
            if (!result.HasVertexIndices)
            {
                _logger?.LogInformation("No vertex indices data in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "vertex_indices.csv");
            _logger?.LogInformation("Generating vertex indices report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Triangle,Index1,Index2,Index3");

            var indices = result.PD4Data.VertexIndices;
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
            _logger?.LogInformation("Vertex indices report generated with {Count} triangles", triangleCount);
        }

        /// <summary>
        /// Generates a CSV report of texture names from a PD4 analysis result.
        /// </summary>
        /// <param name="result">The PD4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateTextureNamesReportAsync(PD4AnalysisResult result, string outputDir)
        {
            if (!result.HasTextureNames)
            {
                _logger?.LogInformation("No texture names in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "texture_names.csv");
            _logger?.LogInformation("Generating texture names report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Index,TextureName");

            var textureNames = result.PD4Data.TextureNames;
            for (int i = 0; i < textureNames.Count; i++)
            {
                var textureName = textureNames[i];
                csv.AppendLine($"{i},{textureName}");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger?.LogInformation("Texture names report generated with {Count} entries", textureNames.Count);
        }

        /// <summary>
        /// Generates a CSV report of material data from a PD4 analysis result.
        /// </summary>
        /// <param name="result">The PD4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateMaterialDataReportAsync(PD4AnalysisResult result, string outputDir)
        {
            if (!result.HasMaterialData)
            {
                _logger?.LogInformation("No material data in {FileName}, skipping report", result.FileName);
                return;
            }

            var outputPath = Path.Combine(outputDir, "material_data.csv");
            _logger?.LogInformation("Generating material data report at {OutputPath}", outputPath);

            var csv = new StringBuilder();
            csv.AppendLine("Index,TextureIndex,Flags,Value1,Value2");

            var materials = result.PD4Data.MaterialData;
            for (int i = 0; i < materials.Count; i++)
            {
                var material = materials[i];
                csv.AppendLine($"{i},{material.TextureIndex},{material.Flags:X8},{material.Value1},{material.Value2}");
            }

            await File.WriteAllTextAsync(outputPath, csv.ToString());
            _logger?.LogInformation("Material data report generated with {Count} entries", materials.Count);
        }

        /// <summary>
        /// Generates a summary report for a PD4 analysis result.
        /// </summary>
        /// <param name="result">The PD4 analysis result to report on.</param>
        /// <param name="outputDir">The output directory for the report.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateSummaryReportAsync(PD4AnalysisResult result, string outputDir)
        {
            var outputPath = Path.Combine(outputDir, "summary.txt");
            _logger?.LogInformation("Generating summary report at {OutputPath}", outputPath);

            var summary = new StringBuilder();
            summary.AppendLine($"PD4 Analysis Summary for {result.FileName}");
            summary.AppendLine($"==================================");
            summary.AppendLine();
            summary.AppendLine($"File: {result.FilePath}");
            summary.AppendLine($"Version: {result.PD4Data.Version}");
            summary.AppendLine();
            
            summary.AppendLine("Data present in the file:");
            summary.AppendLine($"- Vertex Positions: {(result.HasVertexPositions ? "Yes" : "No")} ({result.PD4Data.VertexPositions.Count} entries)");
            summary.AppendLine($"- Vertex Indices: {(result.HasVertexIndices ? "Yes" : "No")} ({result.PD4Data.VertexIndices.Count} entries)");
            summary.AppendLine($"- Texture Names: {(result.HasTextureNames ? "Yes" : "No")} ({result.PD4Data.TextureNames.Count} entries)");
            summary.AppendLine($"- Material Data: {(result.HasMaterialData ? "Yes" : "No")} ({result.PD4Data.MaterialData.Count} entries)");
            summary.AppendLine();
            
            summary.AppendLine("CSV Reports Generated:");
            if (result.HasVertexPositions)
                summary.AppendLine("- vertex_positions.csv");
            if (result.HasVertexIndices)
                summary.AppendLine("- vertex_indices.csv");
            if (result.HasTextureNames)
                summary.AppendLine("- texture_names.csv");
            if (result.HasMaterialData)
                summary.AppendLine("- material_data.csv");
            
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

        /// <summary>
        /// Generates all CSV reports from multiple PD4 analysis results.
        /// </summary>
        /// <param name="results">The PD4 analysis results to report on.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GenerateAllCsvReportsAsync(IEnumerable<PD4AnalysisResult> results)
        {
            foreach (var result in results)
            {
                await GenerateAllCsvReportsAsync(result);
            }
        }

        /// <summary>
        /// Generates all CSV reports from a PD4 analysis result with a specified output directory.
        /// </summary>
        /// <param name="result">The PD4 analysis result to report on.</param>
        /// <param name="outputDirectory">The output directory for the reports.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task GenerateAllCsvReportsAsync(PD4AnalysisResult result, string outputDirectory)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));
                
            if (string.IsNullOrEmpty(outputDirectory))
                throw new ArgumentNullException(nameof(outputDirectory));

            if (string.IsNullOrEmpty(result.FileName))
            {
                _logger?.LogWarning("PD4 analysis result has no filename, using 'unknown'");
                result.FileName = "unknown";
            }

            _logger?.LogInformation("Generating CSV reports for {FileName} in directory {OutputDirectory}", result.FileName, outputDirectory);

            var outputDir = Path.Combine(outputDirectory, Path.GetFileNameWithoutExtension(result.FileName));
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

                // Generate texture names report
                await GenerateTextureNamesReportAsync(result, outputDir);

                // Generate material data report
                await GenerateMaterialDataReportAsync(result, outputDir);

                // Generate summary report
                await GenerateSummaryReportAsync(result, outputDir);

                _logger?.LogInformation("Successfully generated all CSV reports for {FileName}", result.FileName);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating CSV reports for {FileName}", result.FileName);
                throw;
            }
        }
    }
} 