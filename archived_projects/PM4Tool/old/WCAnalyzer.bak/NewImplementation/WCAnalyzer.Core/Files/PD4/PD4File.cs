using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PD4
{
    /// <summary>
    /// Represents a PD4 file
    /// </summary>
    public class PD4File
    {
        private readonly ILogger? _logger;
        private readonly ChunkFactory _chunkFactory;
        private readonly List<string> _errors = new List<string>();
        private readonly List<IChunk> _chunks = new List<IChunk>();
        
        /// <summary>
        /// All chunks in the file
        /// </summary>
        public IReadOnlyList<IChunk> Chunks => _chunks.AsReadOnly();
        
        /// <summary>
        /// MVER chunk (version information)
        /// </summary>
        public IChunk? VersionChunk => _chunks.FirstOrDefault(c => c.Signature == "MVER");
        
        /// <summary>
        /// Creates a new PD4 file
        /// </summary>
        /// <param name="chunkFactory">Chunk factory to use for creating chunks</param>
        /// <param name="logger">Optional logger</param>
        public PD4File(ChunkFactory chunkFactory, ILogger? logger = null)
        {
            _logger = logger;
            _chunkFactory = chunkFactory ?? throw new ArgumentNullException(nameof(chunkFactory));
        }
        
        /// <summary>
        /// Creates a new PD4 file using the default chunk factory
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public PD4File(ILogger? logger = null)
            : this(new PD4ChunkFactory(logger), logger)
        {
        }
        
        /// <summary>
        /// Parses a PD4 file from the specified path
        /// </summary>
        /// <param name="filePath">Path to the PD4 file</param>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        public bool Parse(string filePath)
        {
            try
            {
                if (string.IsNullOrEmpty(filePath))
                {
                    AddError("File path is null or empty");
                    return false;
                }
                
                if (!File.Exists(filePath))
                {
                    AddError($"File not found: {filePath}");
                    return false;
                }
                
                byte[] fileData = File.ReadAllBytes(filePath);
                return Parse(fileData);
            }
            catch (Exception ex)
            {
                AddError($"Error reading file: {ex.Message}");
                return false;
            }
        }
        
        /// <summary>
        /// Parses a PD4 file from the specified data
        /// </summary>
        /// <param name="fileData">Raw file data</param>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        public bool Parse(byte[] fileData)
        {
            try
            {
                // Clear previous state
                _errors.Clear();
                _chunks.Clear();
                
                if (fileData == null)
                {
                    AddError("File data is null");
                    return false;
                }
                
                if (fileData.Length < 8)
                {
                    AddError("File data is too short");
                    return false;
                }
                
                using (MemoryStream ms = new MemoryStream(fileData))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    // Parse chunks until end of file
                    while (ms.Position < ms.Length - 8)
                    {
                        // Read chunk signature (4 bytes)
                        byte[] signatureBytes = reader.ReadBytes(4);
                        string signature = Encoding.ASCII.GetString(signatureBytes);
                        
                        // Read chunk size (4 bytes)
                        uint chunkSize = reader.ReadUInt32();
                        
                        // Ensure we don't read past the end of the file
                        if (ms.Position + chunkSize > ms.Length)
                        {
                            AddError($"Chunk {signature} extends beyond end of file");
                            return false;
                        }
                        
                        // Read chunk data
                        byte[] chunkData = reader.ReadBytes((int)chunkSize);
                        
                        // Create and parse the chunk
                        IChunk chunk = _chunkFactory.CreateChunk(signature, chunkData);
                        chunk.Parse();
                        
                        // Add to list
                        _chunks.Add(chunk);
                        
                        // Add any chunk errors
                        foreach (string error in chunk.GetErrors())
                        {
                            AddError($"Chunk {signature}: {error}");
                        }
                    }
                }
                
                return true;
            }
            catch (Exception ex)
            {
                AddError($"Error parsing file: {ex.Message}");
                return false;
            }
        }
        
        /// <summary>
        /// Gets all parsing errors
        /// </summary>
        /// <returns>List of error messages</returns>
        public IEnumerable<string> GetErrors()
        {
            return _errors.AsReadOnly();
        }
        
        /// <summary>
        /// Adds an error message
        /// </summary>
        /// <param name="error">Error message</param>
        private void AddError(string error)
        {
            _errors.Add(error);
            _logger?.LogError(error);
        }
    }
} 