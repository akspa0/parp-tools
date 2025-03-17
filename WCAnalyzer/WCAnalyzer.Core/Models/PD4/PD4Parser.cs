using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using WCAnalyzer.Core.Models.PD4.Chunks;
using WCAnalyzer.Core.Services;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;

namespace WCAnalyzer.Core.Models.PD4
{
    /// <summary>
    /// Parser for PD4 files.
    /// </summary>
    public class PD4Parser
    {
        private readonly ILogger<PD4Parser> _logger;
        private readonly Services.PD4Parser _serviceParser;

        /// <summary>
        /// Initializes a new instance of the <see cref="PD4Parser"/> class.
        /// </summary>
        /// <param name="logger">Logger instance.</param>
        public PD4Parser(ILogger<PD4Parser>? logger = null)
        {
            _logger = logger ?? NullLogger<PD4Parser>.Instance;
            _serviceParser = new Services.PD4Parser(null);
        }

        /// <summary>
        /// Parses a PD4 file from the specified path.
        /// </summary>
        /// <param name="filePath">Path to the PD4 file.</param>
        /// <returns>Analysis result containing parsed data.</returns>
        public async Task<PD4AnalysisResult> ParseAsync(string filePath)
        {
            throw new NotImplementedException("This method is being replaced by the newer implementation. Use ParseFile instead.");
        }
        
        /// <summary>
        /// Parses a PD4 file from the specified path.
        /// </summary>
        /// <param name="filePath">Path to the PD4 file.</param>
        /// <returns>Analysis result containing parsed data.</returns>
        public PD4AnalysisResult ParseFile(string filePath)
        {
            return _serviceParser.ParseFile(filePath);
        }

        /// <summary>
        /// Parses multiple PD4 files.
        /// </summary>
        /// <param name="filePaths">The paths to the PD4 files.</param>
        /// <returns>The analysis results.</returns>
        public IEnumerable<PD4AnalysisResult> ParseFiles(IEnumerable<string> filePaths)
        {
            if (filePaths == null)
                throw new ArgumentNullException(nameof(filePaths));

            var results = new List<PD4AnalysisResult>();
            foreach (var filePath in filePaths)
            {
                try
                {
                    var result = ParseFile(filePath);
                    results.Add(result);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error parsing PD4 file: {FilePath}", filePath);
                    results.Add(new PD4AnalysisResult
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
        /// Logs details about the PD4 file result.
        /// </summary>
        /// <param name="result">The PD4 analysis result.</param>
        private void LogResultDetails(PD4AnalysisResult result)
        {
            if (result == null)
                return;
                
            _logger.LogDebug("PD4 File Details:");
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
            
            if (result.PD4Data != null)
            {
                _logger.LogDebug("  - Version: {Version}", result.PD4Data.Version);
                LogDataDetails(result.PD4Data);
            }
        }
        
        /// <summary>
        /// Logs details about the PD4 data.
        /// </summary>
        /// <param name="data">The PD4 data.</param>
        private void LogDataDetails(PD4Data data)
        {
            if (data == null)
                return;
                
            _logger.LogDebug("  - Vertex Positions: {Count}", data.VertexPositions?.Count ?? 0);
            _logger.LogDebug("  - Vertex Indices: {Count}", data.VertexIndices?.Count ?? 0);
            _logger.LogDebug("  - Texture Names: {Count}", data.TextureNames?.Count ?? 0);
            _logger.LogDebug("  - Material Data: {Count}", data.MaterialData?.Count ?? 0);
        }
    }
} 