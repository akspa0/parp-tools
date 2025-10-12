using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT
{
    /// <summary>
    /// Represents an ADT (terrain) file
    /// </summary>
    public class ADTFile : IFileParser<ADTFile>
    {
        /// <summary>
        /// Logger instance
        /// </summary>
        private readonly ILogger? _logger;
        
        /// <summary>
        /// List of errors encountered during parsing
        /// </summary>
        private readonly List<string> _errors = new List<string>();
        
        /// <summary>
        /// Chunks in this ADT file
        /// </summary>
        public List<IChunk> Chunks { get; } = new List<IChunk>();
        
        /// <summary>
        /// The file path this ADT was loaded from
        /// </summary>
        public string? FilePath { get; private set; }
        
        /// <summary>
        /// The MVER chunk containing version information
        /// </summary>
        public MverChunk? VersionChunk { get; private set; }
        
        /// <summary>
        /// The MHDR chunk containing header information
        /// </summary>
        public MhdrChunk? HeaderChunk { get; private set; }
        
        /// <summary>
        /// Creates a new ADT file parser
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public ADTFile(ILogger? logger = null)
        {
            _logger = logger;
        }
        
        /// <summary>
        /// Parse an ADT file from a file path
        /// </summary>
        /// <param name="filePath">Path to the ADT file</param>
        /// <returns>Parsed ADT file</returns>
        public ADTFile Parse(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
            {
                _errors.Add("File path is null or empty");
                return this;
            }
            
            if (!File.Exists(filePath))
            {
                _errors.Add($"File not found: {filePath}");
                return this;
            }
            
            try
            {
                FilePath = filePath;
                byte[] data = File.ReadAllBytes(filePath);
                return Parse(data);
            }
            catch (Exception ex)
            {
                _errors.Add($"Error reading file {filePath}: {ex.Message}");
                return this;
            }
        }
        
        /// <summary>
        /// Parse an ADT file from binary data
        /// </summary>
        /// <param name="data">Binary file data</param>
        /// <returns>Parsed ADT file</returns>
        public ADTFile Parse(byte[] data)
        {
            if (data == null)
            {
                _errors.Add("ADT data is null");
                return this;
            }
            
            try
            {
                Chunks.Clear();
                
                using (MemoryStream ms = new MemoryStream(data))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    ADTChunkFactory factory = new ADTChunkFactory(_logger);
                    Chunks.AddRange(factory.ReadAllChunks(reader));
                }
                
                // Extract key chunks
                ExtractKeyChunks();
                
                // Validate ADT structure
                ValidateADTStructure();
            }
            catch (Exception ex)
            {
                _errors.Add($"Error parsing ADT data: {ex.Message}");
            }
            
            return this;
        }
        
        /// <summary>
        /// Extract important chunks for easy reference
        /// </summary>
        private void ExtractKeyChunks()
        {
            foreach (var chunk in Chunks)
            {
                if (chunk is MverChunk mver)
                {
                    VersionChunk = mver;
                }
                else if (chunk is MhdrChunk mhdr)
                {
                    HeaderChunk = mhdr;
                }
            }
        }
        
        /// <summary>
        /// Validate that the ADT has the required structure
        /// </summary>
        private void ValidateADTStructure()
        {
            // Check version chunk
            if (VersionChunk == null)
            {
                _errors.Add("Missing MVER chunk");
            }
            else if (VersionChunk.Version != 18)
            {
                _errors.Add($"Unexpected ADT version: {VersionChunk.Version} (expected 18)");
            }
            
            // Check header chunk
            if (HeaderChunk == null)
            {
                _errors.Add("Missing MHDR chunk");
            }
        }
        
        /// <summary>
        /// Get the list of errors encountered during parsing
        /// </summary>
        /// <returns>List of error messages</returns>
        public List<string> GetErrors()
        {
            return new List<string>(_errors);
        }
        
        /// <summary>
        /// Returns a string representation of this ADT file
        /// </summary>
        public override string ToString()
        {
            return $"ADT File: {Path.GetFileName(FilePath ?? "Unknown")} ({Chunks.Count} chunks)";
        }
    }
} 