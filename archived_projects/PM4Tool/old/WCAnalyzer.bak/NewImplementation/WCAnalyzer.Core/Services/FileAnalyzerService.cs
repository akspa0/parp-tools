using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.ADT;
using WCAnalyzer.Core.Files.PD4;
using WCAnalyzer.Core.Files.PM4;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for analyzing various Warcraft file formats
    /// </summary>
    public class FileAnalyzerService
    {
        private readonly ILogger? _logger;
        
        /// <summary>
        /// Creates a new FileAnalyzerService
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public FileAnalyzerService(ILogger? logger = null)
        {
            _logger = logger;
        }
        
        /// <summary>
        /// Analyzes a file and returns the appropriate parsed file object
        /// </summary>
        /// <param name="filePath">Path to the file</param>
        /// <returns>Parsed file object or null if the file type is not supported</returns>
        public object? AnalyzeFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
            {
                _logger?.LogError("File path is null or empty");
                return null;
            }
            
            if (!File.Exists(filePath))
            {
                _logger?.LogError($"File not found: {filePath}");
                return null;
            }
            
            try
            {
                string extension = Path.GetExtension(filePath).ToLowerInvariant();
                
                switch (extension)
                {
                    case ".adt":
                        return AnalyzeADTFile(filePath);
                    
                    case ".pm4":
                        return AnalyzePM4File(filePath);
                    
                    case ".pd4":
                        return AnalyzePD4File(filePath);
                    
                    // Add support for other file types as needed
                    // case ".wmo":
                    //     return AnalyzeWMOFile(filePath);
                    // case ".m2":
                    //     return AnalyzeM2File(filePath);
                    // etc.
                    
                    default:
                        _logger?.LogError($"Unsupported file format: {extension}");
                        return null;
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, $"Error analyzing file: {filePath}");
                return null;
            }
        }
        
        /// <summary>
        /// Analyzes an ADT file
        /// </summary>
        /// <param name="filePath">Path to the ADT file</param>
        /// <returns>Parsed ADT file</returns>
        public ADTFile AnalyzeADTFile(string filePath)
        {
            _logger?.LogInformation($"Analyzing ADT file: {filePath}");
            
            ADTFile adtFile = new ADTFile(_logger);
            adtFile.Parse(filePath);
            
            // Log any errors
            foreach (string error in adtFile.GetErrors())
            {
                _logger?.LogError($"ADT parsing error: {error}");
            }
            
            return adtFile;
        }
        
        /// <summary>
        /// Analyzes a PM4 file
        /// </summary>
        /// <param name="filePath">Path to the PM4 file</param>
        /// <returns>Parsed PM4 file</returns>
        public PM4File AnalyzePM4File(string filePath)
        {
            _logger?.LogInformation($"Analyzing PM4 file: {filePath}");
            
            PM4File pm4File = new PM4File(_logger);
            pm4File.Parse(filePath);
            
            // Log any errors
            foreach (string error in pm4File.GetErrors())
            {
                _logger?.LogError($"PM4 parsing error: {error}");
            }
            
            return pm4File;
        }
        
        /// <summary>
        /// Analyzes a PD4 file
        /// </summary>
        /// <param name="filePath">Path to the PD4 file</param>
        /// <returns>Parsed PD4 file</returns>
        public PD4File AnalyzePD4File(string filePath)
        {
            _logger?.LogInformation($"Analyzing PD4 file: {filePath}");
            
            PD4File pd4File = new PD4File(_logger);
            pd4File.Parse(filePath);
            
            // Log any errors
            foreach (string error in pd4File.GetErrors())
            {
                _logger?.LogError($"PD4 parsing error: {error}");
            }
            
            return pd4File;
        }
        
        /// <summary>
        /// Analyzes multiple ADT files asynchronously
        /// </summary>
        /// <param name="filePaths">Paths to ADT files</param>
        /// <returns>Dictionary of file paths to parsed ADT files</returns>
        public async Task<Dictionary<string, ADTFile>> AnalyzeADTFilesAsync(IEnumerable<string> filePaths)
        {
            Dictionary<string, ADTFile> results = new Dictionary<string, ADTFile>();
            
            await Task.Run(() => {
                foreach (string filePath in filePaths)
                {
                    try
                    {
                        ADTFile adtFile = AnalyzeADTFile(filePath);
                        results[filePath] = adtFile;
                    }
                    catch (Exception ex)
                    {
                        _logger?.LogError(ex, $"Error analyzing ADT file: {filePath}");
                    }
                }
            });
            
            return results;
        }
        
        /// <summary>
        /// Analyzes multiple PM4 files asynchronously
        /// </summary>
        /// <param name="filePaths">Paths to PM4 files</param>
        /// <returns>Dictionary of file paths to parsed PM4 files</returns>
        public async Task<Dictionary<string, PM4File>> AnalyzePM4FilesAsync(IEnumerable<string> filePaths)
        {
            Dictionary<string, PM4File> results = new Dictionary<string, PM4File>();
            
            await Task.Run(() => {
                foreach (string filePath in filePaths)
                {
                    try
                    {
                        PM4File pm4File = AnalyzePM4File(filePath);
                        results[filePath] = pm4File;
                    }
                    catch (Exception ex)
                    {
                        _logger?.LogError(ex, $"Error analyzing PM4 file: {filePath}");
                    }
                }
            });
            
            return results;
        }
        
        /// <summary>
        /// Analyzes multiple PD4 files asynchronously
        /// </summary>
        /// <param name="filePaths">Paths to PD4 files</param>
        /// <returns>Dictionary of file paths to parsed PD4 files</returns>
        public async Task<Dictionary<string, PD4File>> AnalyzePD4FilesAsync(IEnumerable<string> filePaths)
        {
            Dictionary<string, PD4File> results = new Dictionary<string, PD4File>();
            
            await Task.Run(() => {
                foreach (string filePath in filePaths)
                {
                    try
                    {
                        PD4File pd4File = AnalyzePD4File(filePath);
                        results[filePath] = pd4File;
                    }
                    catch (Exception ex)
                    {
                        _logger?.LogError(ex, $"Error analyzing PD4 file: {filePath}");
                    }
                }
            });
            
            return results;
        }
    }
} 