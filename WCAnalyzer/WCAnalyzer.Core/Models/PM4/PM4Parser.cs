using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using WCAnalyzer.Core.Models.PM4.Chunks;
using WCAnalyzer.Core.Services;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;

namespace WCAnalyzer.Core.Models.PM4
{
    /// <summary>
    /// Parser for PM4 files.
    /// </summary>
    public class PM4Parser
    {
        private readonly ILogger<PM4Parser> _logger;
        private readonly Services.PM4Parser _serviceParser;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4Parser"/> class.
        /// </summary>
        /// <param name="logger">Logger instance.</param>
        public PM4Parser(ILogger<PM4Parser>? logger = null)
        {
            _logger = logger ?? NullLogger<PM4Parser>.Instance;
            _serviceParser = new Services.PM4Parser(null);
        }

        /// <summary>
        /// Parses a PM4 file from the specified path.
        /// </summary>
        /// <param name="filePath">Path to the PM4 file.</param>
        /// <returns>Analysis result containing parsed data.</returns>
        public async Task<PM4AnalysisResult> ParseAsync(string filePath)
        {
            throw new NotImplementedException("This method is being replaced by the newer implementation. Use ParseFile instead.");
        }
        
        /// <summary>
        /// Parses a PM4 file from the specified path.
        /// </summary>
        /// <param name="filePath">Path to the PM4 file.</param>
        /// <returns>Analysis result containing parsed data.</returns>
        public PM4AnalysisResult ParseFile(string filePath)
        {
            return _serviceParser.ParseFile(filePath);
        }

        /// <summary>
        /// Parses multiple PM4 files.
        /// </summary>
        /// <param name="filePaths">The paths to the PM4 files.</param>
        /// <returns>The analysis results.</returns>
        public IEnumerable<PM4AnalysisResult> ParseFiles(IEnumerable<string> filePaths)
        {
            if (filePaths == null)
                throw new ArgumentNullException(nameof(filePaths));

            var results = new List<PM4AnalysisResult>();
            foreach (var filePath in filePaths)
            {
                try
                {
                    var result = ParseFile(filePath);
                    results.Add(result);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error parsing PM4 file: {FilePath}", filePath);
                    results.Add(new PM4AnalysisResult
                    {
                        FilePath = filePath,
                        FileName = Path.GetFileName(filePath),
                        Success = false,
                        Errors = new List<string> { $"Exception: {ex.Message}" }
                    });
                }
            }

            return results;
        }

        /// <summary>
        /// Logs details about the PM4 file result.
        /// </summary>
        /// <param name="result">The PM4 analysis result.</param>
        private void LogResultDetails(PM4AnalysisResult result)
        {
            if (result == null)
                return;
                
            _logger.LogDebug("PM4 File Details:");
            _logger.LogDebug("  - File: {FileName}", result.FileName);
            _logger.LogDebug("  - Success: {Success}", result.Success);
            
            if (result.HasErrors)
            {
                _logger.LogDebug("  - Errors: {ErrorCount}", result.Errors.Count);
                foreach (var error in result.Errors)
                {
                    _logger.LogDebug("    - {Error}", error);
                }
            }
            
            if (result.PM4Data != null)
            {
                _logger.LogDebug("  - Version: {Version}", result.PM4Data.Version);
                LogDataDetails(result.PM4Data);
            }
        }
        
        /// <summary>
        /// Logs details about the PM4 data.
        /// </summary>
        /// <param name="data">The PM4 data.</param>
        private void LogDataDetails(PM4Data data)
        {
            if (data == null)
                return;
                
            _logger.LogDebug("  - Vertex Positions: {Count}", data.VertexPositions?.Count ?? 0);
            _logger.LogDebug("  - Vertex Indices: {Count}", data.VertexIndices?.Count ?? 0);
            _logger.LogDebug("  - Links: {Count}", data.Links?.Count ?? 0);
            _logger.LogDebug("  - Position Data: {Count}", data.PositionData?.Count ?? 0);
            _logger.LogDebug("  - Position References: {Count}", data.PositionReferences?.Count ?? 0);
        }
    }
} 