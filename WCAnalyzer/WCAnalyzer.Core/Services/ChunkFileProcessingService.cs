using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PD4;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Utilities;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for processing PM4 and PD4 files.
    /// </summary>
    public class ChunkFileProcessingService
    {
        private readonly ILogger<ChunkFileProcessingService> _logger;
        private readonly PM4Parser _pm4Parser;
        private readonly PD4Parser _pd4Parser;

        /// <summary>
        /// Initializes a new instance of the <see cref="ChunkFileProcessingService"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="pm4Parser">The PM4 parser.</param>
        /// <param name="pd4Parser">The PD4 parser.</param>
        public ChunkFileProcessingService(
            ILogger<ChunkFileProcessingService> logger,
            PM4Parser pm4Parser,
            PD4Parser pd4Parser)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _pm4Parser = pm4Parser ?? throw new ArgumentNullException(nameof(pm4Parser));
            _pd4Parser = pd4Parser ?? throw new ArgumentNullException(nameof(pd4Parser));
        }

        /// <summary>
        /// Processes all PM4 files in a directory and creates analysis results.
        /// </summary>
        /// <param name="directoryPath">The directory path containing PM4 files.</param>
        /// <param name="searchPattern">The search pattern to use (default is "*.pm4").</param>
        /// <param name="searchOption">The search option to use (default is TopDirectoryOnly).</param>
        /// <returns>A dictionary mapping file names to their analysis results.</returns>
        public async Task<Dictionary<string, PM4AnalysisResult>> ProcessPM4DirectoryAsync(
            string directoryPath,
            string searchPattern = "*.pm4",
            SearchOption searchOption = SearchOption.TopDirectoryOnly)
        {
            if (string.IsNullOrEmpty(directoryPath))
                throw new ArgumentNullException(nameof(directoryPath));

            if (!Directory.Exists(directoryPath))
                throw new DirectoryNotFoundException($"Directory not found: {directoryPath}");

            _logger.LogInformation("Processing PM4 files in directory: {DirectoryPath}", directoryPath);

            var results = new Dictionary<string, PM4AnalysisResult>();
            var filePaths = Directory.GetFiles(directoryPath, searchPattern, searchOption);

            foreach (var filePath in filePaths)
            {
                try
                {
                    _logger.LogDebug("Processing PM4 file: {FilePath}", filePath);
                    var result = await Task.Run(() => _pm4Parser.ParseFile(filePath));
                    results[Path.GetFileName(filePath)] = result;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing PM4 file: {FilePath}", filePath);
                }
            }

            _logger.LogInformation("Processed {Count} PM4 files from directory: {DirectoryPath}", results.Count, directoryPath);
            return results;
        }

        /// <summary>
        /// Processes all PD4 files in a directory and creates analysis results.
        /// </summary>
        /// <param name="directoryPath">The directory path containing PD4 files.</param>
        /// <param name="searchPattern">The search pattern to use (default is "*.pd4").</param>
        /// <param name="searchOption">The search option to use (default is TopDirectoryOnly).</param>
        /// <returns>A dictionary mapping file names to their analysis results.</returns>
        public async Task<Dictionary<string, PD4AnalysisResult>> ProcessPD4DirectoryAsync(
            string directoryPath,
            string searchPattern = "*.pd4",
            SearchOption searchOption = SearchOption.TopDirectoryOnly)
        {
            if (string.IsNullOrEmpty(directoryPath))
                throw new ArgumentNullException(nameof(directoryPath));

            if (!Directory.Exists(directoryPath))
                throw new DirectoryNotFoundException($"Directory not found: {directoryPath}");

            _logger.LogInformation("Processing PD4 files in directory: {DirectoryPath}", directoryPath);

            var results = new Dictionary<string, PD4AnalysisResult>();
            var filePaths = Directory.GetFiles(directoryPath, searchPattern, searchOption);

            foreach (var filePath in filePaths)
            {
                try
                {
                    _logger.LogDebug("Processing PD4 file: {FilePath}", filePath);
                    var result = await Task.Run(() => _pd4Parser.ParseFile(filePath));
                    results[Path.GetFileName(filePath)] = result;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing PD4 file: {FilePath}", filePath);
                }
            }

            _logger.LogInformation("Processed {Count} PD4 files from directory: {DirectoryPath}", results.Count, directoryPath);
            return results;
        }

        /// <summary>
        /// Returns all chunk types found in PM4 files.
        /// </summary>
        /// <param name="results">The PM4 analysis results.</param>
        /// <returns>A set of chunk signatures found in the files.</returns>
        public HashSet<string> GetAllPM4ChunkTypes(IEnumerable<PM4AnalysisResult> results)
        {
            var chunkTypes = new HashSet<string>();

            foreach (var result in results)
            {
                if (result.PM4File == null)
                    continue;

                // Add all chunk types present in this file
                if (result.HasVersion) chunkTypes.Add("MVER");
                if (result.HasCRC) chunkTypes.Add("MCRC");
                if (result.HasShadowData) chunkTypes.Add("MSHD");
                if (result.HasVertexPositions) chunkTypes.Add("MSPV");
                if (result.HasVertexIndices) chunkTypes.Add("MSPI");
                if (result.HasNormalCoordinates) chunkTypes.Add("MSCN");
                if (result.HasLinks) chunkTypes.Add("MSLK");
                if (result.HasVertexData) chunkTypes.Add("MSVT");
                if (result.HasVertexInfo) chunkTypes.Add("MSVI");
                if (result.HasSurfaceData) chunkTypes.Add("MSUR");
                if (result.HasPositionData) chunkTypes.Add("MPRL");
                if (result.HasPositionReference) chunkTypes.Add("MPRR");
                if (result.HasDestructibleBuildingHeader) chunkTypes.Add("MDBH");
                if (result.HasObjectData) chunkTypes.Add("MDOS");
                if (result.HasServerFlagData) chunkTypes.Add("MDSF");
            }

            return chunkTypes;
        }

        /// <summary>
        /// Returns all chunk types found in PD4 files.
        /// </summary>
        /// <param name="results">The PD4 analysis results.</param>
        /// <returns>A set of chunk signatures found in the files.</returns>
        public HashSet<string> GetAllPD4ChunkTypes(IEnumerable<PD4AnalysisResult> results)
        {
            var chunkTypes = new HashSet<string>();

            foreach (var result in results)
            {
                if (result.PD4File == null)
                    continue;

                // Add all chunk types present in this file
                if (result.HasVersion) chunkTypes.Add("MVER");
                if (result.HasCRC) chunkTypes.Add("MCRC");
                if (result.HasShadowData) chunkTypes.Add("MSHD");
                if (result.HasVertexPositions) chunkTypes.Add("MSPV");
                if (result.HasVertexIndices) chunkTypes.Add("MSPI");
                if (result.HasNormalCoordinates) chunkTypes.Add("MSCN");
                if (result.HasLinks) chunkTypes.Add("MSLK");
                if (result.HasVertexData) chunkTypes.Add("MSVT");
                if (result.HasVertexInfo) chunkTypes.Add("MSVI");
                if (result.HasSurfaceData) chunkTypes.Add("MSUR");
            }

            return chunkTypes;
        }

        /// <summary>
        /// Creates summary reports for all PM4 files.
        /// </summary>
        /// <param name="results">The PM4 analysis results.</param>
        /// <param name="outputDirectory">The directory to save the reports to.</param>
        /// <returns>A dictionary mapping file names to their report file paths.</returns>
        public async Task<Dictionary<string, string>> CreatePM4SummaryReportsAsync(
            Dictionary<string, PM4AnalysisResult> results,
            string outputDirectory)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            if (string.IsNullOrEmpty(outputDirectory))
                throw new ArgumentNullException(nameof(outputDirectory));

            if (!Directory.Exists(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            var reportPaths = new Dictionary<string, string>();

            foreach (var kvp in results)
            {
                var fileName = kvp.Key;
                var result = kvp.Value;
                var reportPath = Path.Combine(outputDirectory, $"{Path.GetFileNameWithoutExtension(fileName)}_summary.txt");

                await File.WriteAllTextAsync(reportPath, result.GetSummary());
                reportPaths[fileName] = reportPath;
            }

            return reportPaths;
        }

        /// <summary>
        /// Creates detailed reports for all PM4 files.
        /// </summary>
        /// <param name="results">The PM4 analysis results.</param>
        /// <param name="outputDirectory">The directory to save the reports to.</param>
        /// <returns>A dictionary mapping file names to their report file paths.</returns>
        public async Task<Dictionary<string, string>> CreatePM4DetailedReportsAsync(
            Dictionary<string, PM4AnalysisResult> results,
            string outputDirectory)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            if (string.IsNullOrEmpty(outputDirectory))
                throw new ArgumentNullException(nameof(outputDirectory));

            if (!Directory.Exists(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            var reportPaths = new Dictionary<string, string>();

            foreach (var kvp in results)
            {
                var fileName = kvp.Key;
                var result = kvp.Value;
                var reportPath = Path.Combine(outputDirectory, $"{Path.GetFileNameWithoutExtension(fileName)}_detailed.txt");

                await File.WriteAllTextAsync(reportPath, result.GetDetailedReport());
                reportPaths[fileName] = reportPath;
            }

            return reportPaths;
        }

        /// <summary>
        /// Creates summary reports for all PD4 files.
        /// </summary>
        /// <param name="results">The PD4 analysis results.</param>
        /// <param name="outputDirectory">The directory to save the reports to.</param>
        /// <returns>A dictionary mapping file names to their report file paths.</returns>
        public async Task<Dictionary<string, string>> CreatePD4SummaryReportsAsync(
            Dictionary<string, PD4AnalysisResult> results,
            string outputDirectory)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            if (string.IsNullOrEmpty(outputDirectory))
                throw new ArgumentNullException(nameof(outputDirectory));

            if (!Directory.Exists(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            var reportPaths = new Dictionary<string, string>();

            foreach (var kvp in results)
            {
                var fileName = kvp.Key;
                var result = kvp.Value;
                var reportPath = Path.Combine(outputDirectory, $"{Path.GetFileNameWithoutExtension(fileName)}_summary.txt");

                await File.WriteAllTextAsync(reportPath, result.GetSummary());
                reportPaths[fileName] = reportPath;
            }

            return reportPaths;
        }

        /// <summary>
        /// Creates detailed reports for all PD4 files.
        /// </summary>
        /// <param name="results">The PD4 analysis results.</param>
        /// <param name="outputDirectory">The directory to save the reports to.</param>
        /// <returns>A dictionary mapping file names to their report file paths.</returns>
        public async Task<Dictionary<string, string>> CreatePD4DetailedReportsAsync(
            Dictionary<string, PD4AnalysisResult> results,
            string outputDirectory)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            if (string.IsNullOrEmpty(outputDirectory))
                throw new ArgumentNullException(nameof(outputDirectory));

            if (!Directory.Exists(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            var reportPaths = new Dictionary<string, string>();

            foreach (var kvp in results)
            {
                var fileName = kvp.Key;
                var result = kvp.Value;
                var reportPath = Path.Combine(outputDirectory, $"{Path.GetFileNameWithoutExtension(fileName)}_detailed.txt");

                await File.WriteAllTextAsync(reportPath, result.GetDetailedReport());
                reportPaths[fileName] = reportPath;
            }

            return reportPaths;
        }
    }
} 