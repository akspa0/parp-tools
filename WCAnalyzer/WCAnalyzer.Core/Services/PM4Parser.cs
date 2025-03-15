using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using System.Linq;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for parsing PM4 files and generating analysis results.
    /// </summary>
    public class PM4Parser
    {
        private readonly ILogger<PM4Parser>? _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4Parser"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        public PM4Parser(ILogger<PM4Parser>? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Parses a PM4 file and returns an analysis result.
        /// </summary>
        /// <param name="filePath">Path to the PM4 file.</param>
        /// <returns>A PM4AnalysisResult containing the analysis data.</returns>
        public PM4AnalysisResult ParseFile(string filePath)
        {
            try
            {
                _logger?.LogInformation("Starting to parse PM4 file: {FilePath}", filePath);

                if (!File.Exists(filePath))
                {
                    _logger?.LogError("PM4 file not found: {FilePath}", filePath);
                    var errorResult = new PM4AnalysisResult
                    {
                        FileName = Path.GetFileName(filePath),
                        FilePath = filePath,
                        Errors = new List<string> { $"File not found: {filePath}" }
                    };
                    return errorResult;
                }

                // Load the PM4 file - passing null for the logger since we can't easily create a proper ILogger<PM4File>
                var pm4File = PM4File.FromFile(filePath, null);
                
                // If there's a logger available, log any errors from PM4File
                if (_logger != null && pm4File.Errors.Count > 0)
                {
                    foreach (var error in pm4File.Errors)
                    {
                        _logger.LogWarning("PM4File reported error: {Error}", error);
                    }
                }

                // Generate analysis result
                var result = PM4AnalysisResult.FromPM4File(pm4File);
                
                _logger?.LogInformation("Successfully parsed PM4 file: {FilePath}", filePath);
                
                return result;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error parsing PM4 file: {FilePath}", filePath);
                var errorResult = new PM4AnalysisResult
                {
                    FileName = Path.GetFileName(filePath),
                    FilePath = filePath,
                    Errors = new List<string> { $"Error parsing PM4 file: {ex.Message}" }
                };
                return errorResult;
            }
        }

        /// <summary>
        /// Parses multiple PM4 files asynchronously and returns analysis results.
        /// </summary>
        /// <param name="filePaths">List of paths to PM4 files.</param>
        /// <returns>A list of PM4AnalysisResult containing the analysis data for each file.</returns>
        public async Task<List<PM4AnalysisResult>> ParseFilesAsync(IEnumerable<string> filePaths)
        {
            var results = new List<PM4AnalysisResult>();
            
            foreach (var filePath in filePaths)
            {
                // Use Task.Run to offload CPU-bound work to a thread pool thread
                var result = await Task.Run(() => ParseFile(filePath));
                results.Add(result);
            }
            
            _logger?.LogInformation("Completed parsing {Count} PM4 files", results.Count);
            return results;
        }

        /// <summary>
        /// Parses all PM4 files in a directory and returns analysis results.
        /// </summary>
        /// <param name="directoryPath">Path to the directory containing PM4 files.</param>
        /// <param name="searchPattern">File search pattern (default: *.pm4).</param>
        /// <param name="searchOption">Directory search options (default: AllDirectories).</param>
        /// <returns>A list of PM4AnalysisResult containing the analysis data for each file.</returns>
        public async Task<List<PM4AnalysisResult>> ParseDirectoryAsync(
            string directoryPath, 
            string searchPattern = "*.pm4", 
            SearchOption searchOption = SearchOption.AllDirectories)
        {
            try
            {
                _logger?.LogInformation("Searching for PM4 files in directory: {DirectoryPath}", directoryPath);
                
                if (!Directory.Exists(directoryPath))
                {
                    _logger?.LogError("Directory not found: {DirectoryPath}", directoryPath);
                    throw new DirectoryNotFoundException($"Directory not found: {directoryPath}");
                }

                var filePaths = Directory.GetFiles(directoryPath, searchPattern, searchOption);
                _logger?.LogInformation("Found {Count} PM4 files in directory", filePaths.Length);

                return await ParseFilesAsync(filePaths);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error parsing PM4 files in directory: {DirectoryPath}", directoryPath);
                throw;
            }
        }

        /// <summary>
        /// Combines multiple PM4 analysis results into a single summary.
        /// </summary>
        /// <param name="results">The list of PM4 analysis results to combine.</param>
        /// <returns>A string containing the combined summary.</returns>
        public string GenerateSummary(IEnumerable<PM4AnalysisResult> results)
        {
            var summary = new System.Text.StringBuilder();
            var resultsList = results as List<PM4AnalysisResult> ?? new List<PM4AnalysisResult>(results);
            
            summary.AppendLine($"PM4 Analysis Summary");
            summary.AppendLine($"===================");
            summary.AppendLine($"Files Analyzed: {resultsList.Count}");
            
            int filesWithErrors = resultsList.Count(r => r.Errors.Count > 0);
            summary.AppendLine($"Files with Errors: {filesWithErrors}");
            
            summary.AppendLine("\nChunk Statistics:");
            summary.AppendLine($"- Files with Shadow Data: {resultsList.Count(r => r.HasShadowData)}");
            summary.AppendLine($"- Files with Vertex Positions: {resultsList.Count(r => r.HasVertexPositions)}");
            summary.AppendLine($"- Files with Vertex Indices: {resultsList.Count(r => r.HasVertexIndices)}");
            summary.AppendLine($"- Files with Normal Coordinates: {resultsList.Count(r => r.HasNormalCoordinates)}");
            summary.AppendLine($"- Files with Links: {resultsList.Count(r => r.HasLinks)}");
            summary.AppendLine($"- Files with Vertex Data: {resultsList.Count(r => r.HasVertexData)}");
            summary.AppendLine($"- Files with Vertex Indices 2: {resultsList.Count(r => r.HasVertexIndices2)}");
            summary.AppendLine($"- Files with Surface Data: {resultsList.Count(r => r.HasSurfaceData)}");
            summary.AppendLine($"- Files with Position Data: {resultsList.Count(r => r.HasPositionData)}");
            summary.AppendLine($"- Files with Value Pairs: {resultsList.Count(r => r.HasValuePairs)}");
            summary.AppendLine($"- Files with Building Data: {resultsList.Count(r => r.HasBuildingData)}");
            summary.AppendLine($"- Files with Simple Data: {resultsList.Count(r => r.HasSimpleData)}");
            summary.AppendLine($"- Files with Final Data: {resultsList.Count(r => r.HasFinalData)}");
            
            if (filesWithErrors > 0)
            {
                summary.AppendLine("\nFiles with Errors:");
                foreach (var result in resultsList.Where(r => r.Errors.Count > 0))
                {
                    summary.AppendLine($"- {result.FileName}:");
                    foreach (var error in result.Errors)
                    {
                        summary.AppendLine($"  - {error}");
                    }
                }
            }
            
            return summary.ToString();
        }

        /// <summary>
        /// Resolves file references in the PM4 analysis result using a listfile.
        /// </summary>
        /// <param name="result">The PM4 analysis result to enhance.</param>
        /// <param name="listfile">Dictionary mapping FileDataIDs to file names.</param>
        public void ResolveReferences(PM4AnalysisResult result, Dictionary<uint, string> listfile)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));
            
            if (listfile == null || listfile.Count == 0)
            {
                _logger?.LogInformation("No listfile provided for reference resolution");
                return;
            }
            
            // PM4 files don't actually contain model FileDataIDs to resolve
            // The structure previously assumed to have FileDataIDs actually contains vertex coordinates and flag data
            
            _logger?.LogInformation("PM4 files do not contain FileDataIDs to resolve - this is server-side collision mesh data");
            
            // The file contains server-side collision and navigation data, not model references
            result.ResolvedFileNames.Clear();
        }

        /// <summary>
        /// Asynchronously parses PM4 files from a directory and resolves references using a listfile.
        /// </summary>
        /// <param name="directoryPath">The directory containing PM4 files.</param>
        /// <param name="listfilePath">Path to the listfile for reference resolution.</param>
        /// <param name="searchPattern">File search pattern.</param>
        /// <param name="searchOption">Directory search options.</param>
        /// <returns>A list of PM4 analysis results.</returns>
        public async Task<List<PM4AnalysisResult>> ParseDirectoryWithReferencesAsync(
            string directoryPath,
            string listfilePath,
            string searchPattern = "*.pm4",
            SearchOption searchOption = SearchOption.AllDirectories)
        {
            // First load the listfile
            Dictionary<uint, string> listfile = new Dictionary<uint, string>();
            try
            {
                if (File.Exists(listfilePath))
                {
                    _logger?.LogInformation("Loading listfile from {ListfilePath}", listfilePath);
                    
                    var lines = await File.ReadAllLinesAsync(listfilePath);
                    foreach (var line in lines)
                    {
                        var parts = line.Split(';');
                        if (parts.Length >= 2 && uint.TryParse(parts[0], out uint fileId))
                        {
                            listfile[fileId] = parts[1];
                        }
                    }
                    
                    _logger?.LogInformation("Loaded {Count} entries from listfile", listfile.Count);
                }
                else
                {
                    _logger?.LogWarning("Listfile not found at {ListfilePath}", listfilePath);
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error loading listfile from {ListfilePath}", listfilePath);
            }

            // Parse the PM4 files
            var results = await ParseDirectoryAsync(directoryPath, searchPattern, searchOption);

            // Resolve references for each result
            if (listfile.Count > 0)
            {
                foreach (var result in results)
                {
                    ResolveReferences(result, listfile);
                }
            }

            return results;
        }
    }
} 