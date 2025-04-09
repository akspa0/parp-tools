using System;
using System.IO;
using Microsoft.Extensions.Logging;
using NewPM4Reader.Interfaces;
using NewPM4Reader.PM4.Chunks;

namespace NewPM4Reader.PM4
{
    /// <summary>
    /// Factory class for creating PM4 chunk objects based on signature.
    /// </summary>
    public class ChunkFactory
    {
        private readonly ILogger? _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="ChunkFactory"/> class.
        /// </summary>
        /// <param name="logger">Optional logger for logging operations.</param>
        public ChunkFactory(ILogger? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Creates a chunk based on its signature.
        /// </summary>
        /// <param name="signature">The chunk signature (4 characters).</param>
        /// <param name="reader">The binary reader containing the chunk data.</param>
        /// <param name="dataSize">The size of the chunk data.</param>
        /// <returns>An instance of <see cref="IPM4Chunk"/> based on the signature.</returns>
        public IPM4Chunk CreateChunk(string signature, BinaryReader reader, int dataSize)
        {
            _logger?.LogDebug("Creating chunk of type {Signature} with data size {Size}", signature, dataSize);

            // Store the current position to revert if needed
            long startPosition = reader.BaseStream.Position;

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
                    
                    // Vertex data
                    case "VVSM":
                        return new VVSM(reader);
                    
                    // Normal data
                    case "NVSM":
                        return new NVSM(reader);
                    
                    // Texture coordinate data
                    case "TVSM":
                        return new TVSM(reader);
                    
                    // Index/face data
                    case "IPSM": // MSPI reversed
                        return new IPSM(reader);
                    
                    // Index data
                    case "IVSM":
                        return new IVSM(reader);
                    
                    // Shadow data
                    case "DHSM":
                        return new DHSM(reader);
                    
                    // RUSM chunk
                    case "RUSM":
                        return new RUSM(reader);
                    
                    // KLSM chunk
                    case "KLSM":
                        return new KLSM(reader);
                    
                    // Other map/mesh related chunks
                    case "MSPL": // LPSM reversed - Map Splitter
                        return new MSPL(reader);
                    
                    // For unknown/unimplemented chunks, read as raw data
                    default:
                        _logger?.LogInformation("Unknown chunk type: {Signature}. Treating as raw data.", signature);
                        
                        // For unknown chunks, read raw data
                        byte[] data = reader.ReadBytes(dataSize);
                        return new UnknownChunk(signature, data);
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error creating chunk of type {Signature}", signature);
                
                // Revert to the start position
                reader.BaseStream.Seek(startPosition, SeekOrigin.Begin);
                
                // Read as raw data for error recovery
                byte[] data = reader.ReadBytes(dataSize);
                return new UnknownChunk(signature, data);
            }
        }
    }
} 