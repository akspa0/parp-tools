using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;
using NewPM4Reader.Interfaces;
using NewPM4Reader.PM4.Chunks;
using NewPM4Reader.Utilities;

namespace NewPM4Reader.PM4
{
    /// <summary>
    /// Represents a PM4 file that contains multiple chunks.
    /// </summary>
    public class PM4File
    {
        private readonly ILogger? _logger;

        /// <summary>
        /// Gets the dictionary of chunks in the PM4 file, grouped by signature.
        /// A signature can have multiple chunks.
        /// </summary>
        public Dictionary<string, List<IPM4Chunk>> ChunksBySignature { get; } = new Dictionary<string, List<IPM4Chunk>>();

        /// <summary>
        /// Gets a flat list of all chunks in the file.
        /// </summary>
        public IEnumerable<IPM4Chunk> AllChunks => ChunksBySignature.Values.SelectMany(list => list);

        /// <summary>
        /// Gets a list of unique chunk signatures in the file.
        /// </summary>
        public IEnumerable<string> ChunkSignatures => ChunksBySignature.Keys;

        /// <summary>
        /// Gets the version of the PM4 file from the MVER chunk.
        /// </summary>
        public uint Version
        {
            get
            {
                if (ChunksBySignature.TryGetValue("REVM", out var chunks) && 
                    chunks.Count > 0 && 
                    chunks[0] is REVM revm)
                {
                    return revm.Version;
                }
                return 0;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4File"/> class.
        /// </summary>
        /// <param name="logger">Optional logger for logging operations.</param>
        public PM4File(ILogger? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Gets all chunks of a specific type.
        /// </summary>
        /// <typeparam name="T">The type of chunk to get.</typeparam>
        /// <returns>A list of chunks of the specified type.</returns>
        public List<T> GetChunks<T>() where T : IPM4Chunk
        {
            return AllChunks.OfType<T>().ToList();
        }

        /// <summary>
        /// Gets chunks with the specified signature.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <returns>A list of chunks with the specified signature, or an empty list if not found.</returns>
        public List<IPM4Chunk> GetChunksBySignature(string signature)
        {
            if (ChunksBySignature.TryGetValue(signature, out var chunks))
            {
                return chunks;
            }
            return new List<IPM4Chunk>();
        }

        /// <summary>
        /// Loads a PM4 file from the specified path.
        /// </summary>
        /// <param name="filePath">The path to the PM4 file.</param>
        public void Load(string filePath)
        {
            _logger?.LogInformation("Loading PM4 file: {FilePath}", filePath);
            
            using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            Load(stream);
        }

        /// <summary>
        /// Loads a PM4 file from the specified stream.
        /// </summary>
        /// <param name="stream">The stream containing PM4 data.</param>
        public void Load(Stream stream)
        {
            _logger?.LogDebug("Reading PM4 data from stream");
            
            ChunksBySignature.Clear();
            
            // Read the entire file into memory for reliable seeking
            byte[] fileData;
            using (var memStream = new MemoryStream())
            {
                stream.CopyTo(memStream);
                fileData = memStream.ToArray();
            }
            
            _logger?.LogDebug("Loaded {Size} bytes into memory", fileData.Length);
            
            using var ms = new MemoryStream(fileData);
            using var reader = new BinaryReader(ms);
            
            while (ms.Position < ms.Length - 8) // Need at least 8 bytes for header
            {
                long startPos = ms.Position;
                
                try
                {
                    // Read chunk signature and size
                    var signatureBytes = reader.ReadBytes(4);
                    var signature = Encoding.ASCII.GetString(signatureBytes);
                    
                    // Check if we've reached the end of valid data
                    if (string.IsNullOrWhiteSpace(signature) || signature.Contains("\0"))
                    {
                        _logger?.LogDebug("Reached end of valid chunk data at position {Position}", startPos);
                        break;
                    }
                    
                    var chunkSize = reader.ReadInt32();
                    
                    _logger?.LogDebug("Found chunk {Signature} with size {Size} at position {Position}", 
                        signature, chunkSize, startPos);
                    
                    // Validate the chunk size
                    if (chunkSize < 0 || chunkSize > ms.Length - ms.Position)
                    {
                        _logger?.LogWarning("Invalid chunk size {Size} for chunk {Signature} at position {Position}", 
                            chunkSize, signature, startPos);
                        break;
                    }
                    
                    // Read the chunk data
                    byte[] chunkData = reader.ReadBytes(chunkSize);
                    
                    // Create chunk based on signature
                    IPM4Chunk chunk;
                    using (var chunkStream = new MemoryStream(chunkData))
                    using (var chunkReader = new BinaryReader(chunkStream))
                    {
                        chunk = CreateChunk(signature, chunkData, chunkReader);
                    }
                    
                    // Add the chunk to the appropriate list
                    if (!ChunksBySignature.TryGetValue(signature, out var chunkList))
                    {
                        chunkList = new List<IPM4Chunk>();
                        ChunksBySignature[signature] = chunkList;
                    }
                    
                    chunkList.Add(chunk);
                    _logger?.LogDebug("Added chunk {Signature} (index {Index})", signature, chunkList.Count - 1);
                }
                catch (Exception ex)
                {
                    _logger?.LogError(ex, "Error reading chunk at position {Position}", startPos);
                    break;
                }
            }
            
            _logger?.LogInformation("Finished loading PM4 file. Found {ChunkCount} chunks of {SignatureCount} different types", 
                AllChunks.Count(), ChunksBySignature.Count);
        }
        
        /// <summary>
        /// Creates a chunk instance from its signature and data.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="data">The raw chunk data.</param>
        /// <param name="reader">A BinaryReader positioned at the start of the chunk data.</param>
        /// <returns>The created chunk instance.</returns>
        private IPM4Chunk CreateChunk(string signature, byte[] data, BinaryReader reader)
        {
            try
            {
                switch (signature)
                {
                    // Version chunk
                    case "REVM": // MVER reversed
                        return new REVM(reader);
                    
                    // Index/ID chunks
                    case "IBDM": // MDBI reversed
                        return new IBDM(reader);
                    
                    // File path chunks
                    case "FBDM": // MDBF reversed
                        return new FBDM(reader);
                    
                    // Header chunks
                    case "HBDM": // MDBH reversed
                        return new HBDM(reader);
                    
                    // Vertex position data
                    case "VPSM": // MSPV reversed
                        return new VPSM(reader);
                    
                    // Index/face data
                    case "IPSM": // MSPI reversed
                        return new IPSM(reader);
                    
                    // Other map/mesh related chunks
                    case "MSPL": // LPSM reversed - Map Splitter
                        return new MSPL(reader);
                    
                    // For unknown/unimplemented chunks, read as raw data
                    default:
                        _logger?.LogInformation("Unknown chunk type: {Signature}. Treating as raw data.", signature);
                        return new UnknownChunk(signature, data);
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error creating chunk of type {Signature}", signature);
                return new UnknownChunk(signature, data);
            }
        }
    }
} 