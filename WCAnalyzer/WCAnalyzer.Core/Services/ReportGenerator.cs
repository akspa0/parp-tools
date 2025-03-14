using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Utilities;

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

        /// <summary>
        /// Creates a new instance of the ReportGenerator class.
        /// </summary>
        /// <param name="logger">The logger to use.</param>
        /// <param name="terrainDataCsvGenerator">The terrain data CSV generator to use.</param>
        /// <param name="jsonReportGenerator">Optional JSON report generator to use.</param>
        /// <param name="markdownReportGenerator">Optional Markdown report generator to use.</param>
        public ReportGenerator(
            ILogger<ReportGenerator> logger,
            TerrainDataCsvGenerator terrainDataCsvGenerator,
            JsonReportGenerator? jsonReportGenerator = null,
            MarkdownReportGenerator? markdownReportGenerator = null)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _terrainDataCsvGenerator = terrainDataCsvGenerator ?? throw new ArgumentNullException(nameof(terrainDataCsvGenerator));
            _jsonReportGenerator = jsonReportGenerator;
            _markdownReportGenerator = markdownReportGenerator;
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
    }
}